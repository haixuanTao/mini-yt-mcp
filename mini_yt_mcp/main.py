"""Main script to orchestrate YouTube video download and audio analysis for dance generation."""

import argparse
import sys
from pathlib import Path
from typing import Optional
import yt_dlp
import re

from .audio_analyzer import AudioAnalyzer
from .csv_move import CSVMove
from .downloader import YouTubeDownloader


def search_youtube_video(search_query: str) -> str:
    """Search for a YouTube video and return the URL of the first result.

    Args:
        search_query: Search terms for YouTube

    Returns:
        YouTube URL of the first search result

    Raises:
        Exception: If search fails or no results found
    """
    print(f"Searching YouTube for: '{search_query}'")

    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search for videos using ytsearch prefix
            search_query_formatted = f"ytsearch1:{search_query}"
            search_results = ydl.extract_info(search_query_formatted, download=False)

            if search_results and 'entries' in search_results and search_results['entries']:
                first_result = search_results['entries'][0]
                video_url = first_result.get('url') or first_result.get('webpage_url')

                # If we got an ID, construct the URL
                if 'id' in first_result and not video_url:
                    video_url = f"https://www.youtube.com/watch?v={first_result['id']}"

                video_title = first_result.get('title', 'Unknown Title')

                print(f"Found: '{video_title}'")
                print(f"URL: {video_url}")
                return video_url
            else:
                raise Exception("No search results found")

    except Exception as e:
        raise Exception(f"Search failed: {e}")


def is_youtube_url(text: str) -> bool:
    """Check if the text is a YouTube URL."""
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=',
        r'https?://(?:www\.)?youtu\.be/',
        r'https?://(?:www\.)?youtube\.com/embed/',
        r'https?://(?:www\.)?youtube\.com/v/',
    ]

    return any(re.match(pattern, text) for pattern in youtube_patterns)


def extract_video_id(youtube_url: str) -> str:
    """Extract YouTube video ID from URL.

    Args:
        youtube_url: YouTube video URL

    Returns:
        Video ID string

    Raises:
        Exception: If video ID cannot be extracted
    """
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    raise Exception(f"Could not extract video ID from URL: {youtube_url}")


def find_existing_download(video_id: str, download_dir: str) -> Optional[str]:
    """Find existing downloaded file for a video ID.

    Args:
        video_id: YouTube video ID
        download_dir: Directory to search in

    Returns:
        Path to existing file if found, None otherwise
    """
    download_path = Path(download_dir)
    if not download_path.exists():
        return None

    # Common audio/video extensions
    extensions = ['.wav', '.mp3', '.m4a', '.aac', '.mp4', '.webm', '.mkv', '.avi']

    # Look for files containing the video ID
    for file_path in download_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # Check if filename contains the video ID
            if video_id in file_path.name:
                return str(file_path)

    return None


def main():
    """Main entry point for the audio dance analyzer."""
    parser = argparse.ArgumentParser(
        description="Download YouTube videos and generate dance moves from audio analysis, or play local CSV files"
    )
    parser.add_argument("input", help="YouTube video URL, search terms, or local CSV file path")
    parser.add_argument(
        "--quality", default="best", help="Video quality (default: best)"
    )
    parser.add_argument(
        "--download-dir",
        default="downloads",
        help="Directory to save downloaded videos (default: downloads)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save analysis results (default: output)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if video file already exists",
    )
    parser.add_argument(
        "--export-csv", action="store_true", help="Export dance moves to CSV format"
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Play dance moves on Reachy Mini robot after analysis",
    )
    parser.add_argument(
        "--robot-scale",
        type=float,
        default=0.5,
        help="Movement scale factor for robot (default: 0.5)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback during robot movements",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0,
        help="Start time in audio file for robot playback (seconds)",
    )
    parser.add_argument(
        "--robot-duration", type=float, help="Limit robot playback duration in seconds"
    )

    args = parser.parse_args()

    # Determine input type and process accordingly
    input_path = Path(args.input)

    # Check if input is a local CSV file
    if input_path.exists() and input_path.suffix.lower() == '.csv':
        print(f"Processing local CSV file: {args.input}")

        # Skip to robot movement directly
        if args.robot:
            print("\n=== Starting Robot Movement ===")
            try:
                # Load dance moves from CSV
                print(f"Loading dance moves from: {args.input}")
                dance_move = CSVMove(
                    str(args.input), args.robot_scale, audio_sync=not args.no_audio
                )
                print(f"Dance duration: {dance_move.duration:.2f} seconds")

                try:
                    print("Connecting to Reachy Mini robot...")
                    from reachy_mini import ReachyMini

                    def robot_play_function(dance_move_obj):
                        """Function to play dance moves on robot."""
                        with ReachyMini() as mini:
                            mini.play_move(dance_move_obj)

                    if dance_move.audio_player and not args.no_audio:
                        print("Playing dance with synchronized audio...")
                        if args.start_time > 0:
                            print(f"Starting audio from {args.start_time:.1f}s")

                        # Play with synchronized audio
                        dance_move.play_with_audio(
                            robot_play_function, args.start_time, args.robot_duration
                        )
                        print("Synchronized dance complete!")

                    else:
                        print("Playing dance without audio...")
                        with ReachyMini() as mini:
                            mini.play_move(dance_move)
                        print("Dance complete!")

                except ImportError:
                    print("Error: reachy_mini not installed. Robot movement disabled.")
                    print("Install reachy_mini package to enable robot functionality.")
                except Exception as e:
                    print(f"Error connecting to robot: {e}")
                    print("Robot movement failed.")

            except Exception as e:
                print(f"Error loading dance moves: {e}")
                print("Robot movement failed.")
        else:
            print("CSV file loaded. Use --robot flag to play on robot.")

        return  # Exit early for CSV files

    # Process YouTube URL or search string
    youtube_url = args.input

    # Check if input is a YouTube URL or search string
    if not is_youtube_url(args.input):
        print(f"Input '{args.input}' is not a YouTube URL, treating as search terms...")
        try:
            youtube_url = search_youtube_video(args.input)
        except Exception as e:
            print(f"Search failed: {e}")
            sys.exit(1)
    else:
        print(f"Processing YouTube URL: {args.input}")

    # Initialize components for YouTube processing
    downloader = YouTubeDownloader(args.download_dir)
    analyzer = AudioAnalyzer(args.output_dir)

    # Get video info
    video_info = downloader.get_video_info(youtube_url)
    if video_info:
        print(f"Video title: {video_info['title']}")
        print(f"Duration: {video_info['duration']} seconds")
        print(f"Uploader: {video_info['uploader']}")

    # Check for cached download first
    try:
        video_id = extract_video_id(youtube_url)
        print(f"Video ID: {video_id}")

        # Check if file already exists
        existing_file = find_existing_download(video_id, args.download_dir)
        if existing_file:
            print(f"Found cached download: {existing_file}")
            video_path = existing_file
        else:
            # Download audio if not cached
            if not args.skip_download:
                print("\nDownloading audio...")
                video_path = downloader.download_video(youtube_url, args.quality)

                if not video_path:
                    print("Failed to download audio")
                    sys.exit(1)

                print(f"Audio downloaded: {video_path}")
            else:
                # Look for any existing audio/video file as fallback
                download_dir = Path(args.download_dir)
                audio_files = list(download_dir.glob("*"))
                audio_files = [
                    f
                    for f in audio_files
                    if f.suffix.lower()
                    in [".wav", ".mp3", ".m4a", ".aac", ".mp4", ".webm", ".mkv", ".avi"]
                ]

                if not audio_files:
                    print("No audio or video files found in download directory")
                    sys.exit(1)

                video_path = str(audio_files[0])  # Use first audio/video file found
                print(f"Using existing file: {video_path}")

    except Exception as e:
        print(f"Error with video ID extraction: {e}")
        # Fallback to original download logic without caching
        if not args.skip_download:
            print("\nDownloading audio (without caching)...")
            video_path = downloader.download_video(youtube_url, args.quality)

            if not video_path:
                print("Failed to download audio")
                sys.exit(1)

            print(f"Audio downloaded: {video_path}")
        else:
            print("Cannot proceed without valid video ID and skip-download enabled")
            sys.exit(1)

    # Analyze audio and generate dance moves
    print("\nAnalyzing audio and generating dance moves...")
    analysis_results = analyzer.analyze_video(video_path)

    # Print summary
    print("\n=== Dance Analysis Summary ===")
    if "error" in analysis_results:
        print(f"Error: {analysis_results['error']}")
        sys.exit(1)

    print(f"Audio duration: {analysis_results['summary']['duration']:.2f} seconds")
    print(f"Tempo: {analysis_results['summary']['tempo']:.1f} BPM")
    print(f"Total beats detected: {analysis_results['summary']['total_beats']}")
    print(f"Total onsets detected: {analysis_results['summary']['total_onsets']}")
    print(f"Key moments identified: {analysis_results['summary']['key_moments']}")
    print(
        f"Dance moves generated: {analysis_results['summary']['dance_moves_generated']}"
    )

    # Export to CSV if requested
    if args.export_csv:
        print("\nExporting dance moves to CSV...")
        csv_path = analyzer.export_dance_csv(analysis_results, video_path)
        print(f"CSV exported: {csv_path}")

    print(f"\nResults saved in: {args.output_dir}")
    print("\nOutput files:")
    print(
        f"- JSON analysis: {Path(args.output_dir) / f'{Path(video_path).stem}_dance_analysis.json'}"
    )
    print(f"- Source audio: {analysis_results['audio_path']}")
    if args.export_csv:
        print(
            f"- CSV data: {Path(args.output_dir) / f'{Path(video_path).stem}_dance_moves.csv'}"
        )

    # Show sample interpolated head movements
    if analysis_results["dance_sequence"]:
        print("\n=== Sample Head Movement Sequence (with interpolation) ===")
        beat_frames = [
            f
            for f in analysis_results["dance_sequence"]
            if f.get("frame_type") == "beat"
        ]
        for i, frame in enumerate(beat_frames[:3]):  # Show first 3 beats
            timestamp = frame["timestamp"]
            coords = frame.get("coordinates", [0, 0, 0, 0, 0, 0])
            print(
                f"Beat {frame.get('beat_number', i + 1)} at {timestamp:.2f}s: {frame.get('original_move', 'unknown')}"
            )
            print(
                f"        XYZ: ({coords[0]:+.1f}, {coords[1]:+.1f}, {coords[2]:+.1f}) cm"
            )
            print(
                f"        RPY: ({coords[3]:+.1f}, {coords[4]:+.1f}, {coords[5]:+.1f}) deg"
            )

        total_frames = len(analysis_results["dance_sequence"])
        beat_count = len(beat_frames)
        interpolated_count = total_frames - beat_count

        print(
            f"\nTotal sequence: {total_frames} frames ({beat_count} beats + {interpolated_count} interpolated)"
        )
        print(
            f"Framerate: ~{total_frames / analysis_results['summary']['duration']:.1f} FPS"
        )

    # Robot movement integration
    if args.robot:
        print("\n=== Starting Robot Movement ===")

        # First check if CSV was exported
        if not args.export_csv:
            print("Exporting dance moves to CSV for robot playback...")
            csv_path = analyzer.export_dance_csv(analysis_results, video_path)
            print(f"CSV exported: {csv_path}")
        else:
            csv_path = (
                Path(args.output_dir) / f"{Path(video_path).stem}_dance_moves.csv"
            )

        try:
            # Load dance moves from CSV
            print(f"Loading dance moves from: {csv_path}")
            dance_move = CSVMove(
                str(csv_path), args.robot_scale, audio_sync=not args.no_audio
            )
            print(f"Dance duration: {dance_move.duration:.2f} seconds")

            try:
                print("Connecting to Reachy Mini robot...")
                from reachy_mini import ReachyMini

                def robot_play_function(dance_move_obj):
                    """Function to play dance moves on robot."""
                    with ReachyMini() as mini:
                        mini.play_move(dance_move_obj)

                if dance_move.audio_player and not args.no_audio:
                    print("Playing dance with synchronized audio...")
                    if args.start_time > 0:
                        print(f"Starting audio from {args.start_time:.1f}s")

                    # Play with synchronized audio
                    dance_move.play_with_audio(
                        robot_play_function, args.start_time, args.robot_duration
                    )
                    print("Synchronized dance complete!")

                else:
                    print("Playing dance without audio...")
                    with ReachyMini() as mini:
                        mini.play_move(dance_move)
                    print("Dance complete!")

            except ImportError:
                print("Error: reachy_mini not installed. Robot movement disabled.")
                print("Install reachy_mini package to enable robot functionality.")
            except Exception as e:
                print(f"Error connecting to robot: {e}")
                print("Robot movement failed.")

        except Exception as e:
            print(f"Error loading dance moves: {e}")
            print("Robot movement failed.")


if __name__ == "__main__":
    main()
