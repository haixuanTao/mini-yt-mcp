"""YouTube video downloader using yt-dlp."""

import os
import re
import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Any


class YouTubeDownloader:
    """Download YouTube videos using yt-dlp."""

    def __init__(self, download_dir: str = "downloads"):
        """Initialize the downloader.

        Args:
            download_dir: Directory to save downloaded videos
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)


    def download_audio(self, url: str, quality: str = "best") -> Optional[str]:
        """Download audio only from a YouTube video.

        Args:
            url: YouTube video URL
            quality: Audio quality preference (best, worst, specific format)

        Returns:
            Path to downloaded audio file, or None if failed
        """
        output_template = str(self.download_dir / "%(title)s [%(id)s].%(ext)s")

        ydl_opts = {
            'format': 'bestaudio/best',  # Download best audio only
            'outtmpl': output_template,
            'extract_flat': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '320',
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown')

                # Download the audio
                ydl.download([url])

                # Find the downloaded audio file
                # Instead of using glob with title (which can have special chars),
                # look for all audio files and match by modification time or filename
                audio_extensions = ['.wav', '.mp3', '.m4a', '.aac']

                # Get all audio files in the download directory
                audio_files = []
                for ext in audio_extensions:
                    audio_files.extend(self.download_dir.glob(f"*{ext}"))

                # Find the most recently created audio file that contains the title
                # (or just return the most recent if title matching fails)
                if audio_files:
                    # Sort by modification time (most recent first)
                    audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                    # Find the best matching file using similarity scoring
                    # Remove special characters from title for comparison
                    title_clean = re.sub(r'[^\w\s-]', '', video_title).strip().lower()

                    best_match = None
                    best_score = 0.0

                    for file in audio_files:
                        file_name = file.stem.lower()

                        # Calculate similarity score between title and filename
                        score = self._calculate_similarity(title_clean, file_name)

                        if score > best_score:
                            best_score = score
                            best_match = file

                    # Return the best match if we found a reasonable similarity (>0.3)
                    # Otherwise return the most recent file
                    if best_match and best_score > 0.3:
                        return str(best_match)
                    else:
                        return str(audio_files[0])

        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None

        return None

    def _calculate_similarity(self, title: str, filename: str) -> float:
        """Calculate similarity score between video title and filename.

        Args:
            title: Cleaned video title (lowercase, no special chars)
            filename: Audio file name (lowercase)

        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not title or not filename:
            return 0.0

        title_words = set(word for word in title.split() if len(word) > 2)
        filename_words = set(word for word in filename.split() if len(word) > 2)

        if not title_words:
            return 0.0

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(title_words.intersection(filename_words))
        union = len(title_words.union(filename_words))

        similarity_score = intersection / union if union > 0 else 0.0

        # Bonus for exact substring matches
        substring_bonus = 0.0
        for title_word in title_words:
            if title_word in filename:
                substring_bonus += 0.1

        # Bonus for similar length (prevents very short matches from scoring too high)
        length_ratio = min(len(title), len(filename)) / max(len(title), len(filename))
        length_bonus = length_ratio * 0.2

        # Combined score with caps at 1.0
        total_score = min(1.0, similarity_score + substring_bonus + length_bonus)

        return total_score

    def download_video(self, url: str, quality: str = "best") -> Optional[str]:
        """Download a YouTube video (kept for backwards compatibility).

        Args:
            url: YouTube video URL
            quality: Video quality preference (best, worst, specific format)

        Returns:
            Path to downloaded video file, or None if failed
        """
        # For audio analysis, just download audio
        return self.download_audio(url, quality)

    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading.

        Args:
            url: YouTube video URL

        Returns:
            Video information dictionary
        """
        ydl_opts = {
            'extract_flat': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'uploader': info.get('uploader'),
                    'view_count': info.get('view_count'),
                    'description': info.get('description'),
                }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None