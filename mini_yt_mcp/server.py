#!/usr/bin/env python3
"""Mini YouTube MCP Server - Model Context Protocol server for YouTube audio analysis and dance generation."""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    JSONRPCError,
    TextContent,
    Tool,
)

from .audio_analyzer import AudioAnalyzer
from .csv_move import CSVMove
from .downloader import YouTubeDownloader
import threading


class YouTubeMCPServer:
    """MCP Server for YouTube audio analysis and dance generation."""

    def __init__(self):
        self.server = Server("mini-yt-mcp")

        # Use persistent cache directory instead of temp
        cache_dir = Path.home() / ".cache" / "mini-yt-mcp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.download_dir = cache_dir / "downloads"
        self.output_dir = cache_dir / "output"
        self.download_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Set up debug log file
        self.log_file = cache_dir / "debug.log"
        self._log("=" * 80)
        self._log("Mini YouTube MCP Server Starting")
        self._log(f"Cache dir: {cache_dir}")
        self._log(f"Download dir: {self.download_dir}")
        self._log(f"Output dir: {self.output_dir}")
        self._log("=" * 80)

        # Initialize components
        self.analyzer = AudioAnalyzer(str(self.output_dir))
        self.downloader = YouTubeDownloader(str(self.download_dir))

        # Track current playback
        self.current_dance_move = None
        self.playback_task = None
        self.stop_event = threading.Event()
        self.current_robot = None

        # Query cache: maps search query -> (video_url, csv_path)
        self.query_cache_file = cache_dir / "query_cache.json"
        self.query_cache = self._load_query_cache()

        # Register tools
        self._register_tools()

    def _log(self, message: str):
        """Log message to both stderr and debug file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}] {message}"
        print(log_message, file=sys.stderr)
        try:
            with open(self.log_file, "a") as f:
                f.write(log_message + "\n")
        except Exception:
            pass  # Ignore logging errors

    def _load_query_cache(self) -> Dict[str, Dict[str, str]]:
        """Load query cache from disk."""
        if self.query_cache_file.exists():
            try:
                with open(self.query_cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self._log(f"Failed to load query cache: {e}")
        return {}

    def _save_query_cache(self):
        """Save query cache to disk."""
        try:
            with open(self.query_cache_file, "w") as f:
                json.dump(self.query_cache, f, indent=2)
        except Exception as e:
            self._log(f"Failed to save query cache: {e}")

    def _register_tools(self):
        """Register all available tools with the MCP server."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools."""
            return [
                # Tool(
                # name="search_youtube",
                # description="Search for YouTube videos by query terms",
                # inputSchema={
                # "type": "object",
                # "properties": {
                # "query": {
                # "type": "string",
                # "description": "Search terms for YouTube videos"
                # }
                # },
                # "required": ["query"]
                # }
                # ),
                # Tool(
                # name="get_video_info",
                # description="Get information about a YouTube video",
                # inputSchema={
                # "type": "object",
                # "properties": {
                # "url": {
                # "type": "string",
                # "description": "YouTube video URL"
                # }
                # },
                # "required": ["url"]
                # }
                # ),
                # Tool(
                # name="download_audio",
                # description="Download audio from a YouTube video",
                # inputSchema={
                # "type": "object",
                # "properties": {
                # "url": {
                # "type": "string",
                # "description": "YouTube video URL"
                # },
                # "quality": {
                # "type": "string",
                # "description": "Audio quality preference",
                # "default": "best"
                # }
                # },
                # "required": ["url"]
                # }
                # ),
                # Tool(
                # name="analyze_audio",
                # description="Analyze audio file and generate dance sequence",
                # inputSchema={
                # "type": "object",
                # "properties": {
                # "audio_path": {
                # "type": "string",
                # "description": "Path to audio file to analyze"
                # }
                # },
                # "required": ["audio_path"]
                # }
                # ),
                Tool(
                    name="play_music",
                    description="Play the searched music",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "search music",
                            },
                        },
                        "required": ["input"],
                    },
                ),
                Tool(
                    name="stop_music",
                    description="Stop the currently playing music",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                # Tool(
                # name="load_dance_from_csv",
                # description="Load and analyze dance moves from CSV file",
                # inputSchema={
                # "type": "object",
                # "properties": {
                # "csv_path": {
                # "type": "string",
                # "description": "Path to CSV file with dance moves",
                # },
                # "scale": {
                # "type": "number",
                # "description": "Scale factor for movements",
                # "default": 0.5,
                # },
                # },
                # "required": ["csv_path"],
                # },
                # ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_youtube":
                    return await self._search_youtube(arguments["query"])

                elif name == "get_video_info":
                    return await self._get_video_info(arguments["url"])

                elif name == "download_audio":
                    return await self._download_audio(
                        arguments["url"], arguments.get("quality", "best")
                    )

                elif name == "analyze_audio":
                    return await self._analyze_audio(arguments["audio_path"])

                elif name == "play_music":
                    return await self._generate_dance_from_youtube(
                        arguments["input"], arguments.get("export_csv", True)
                    )

                elif name == "stop_music":
                    return await self._stop_music()

                elif name == "load_dance_from_csv":
                    return await self._load_dance_from_csv(
                        arguments["csv_path"], arguments.get("scale", 0.5)
                    )

                else:
                    raise JSONRPCError(METHOD_NOT_FOUND, f"Unknown tool: {name}")

            except Exception as e:
                raise JSONRPCError(INTERNAL_ERROR, f"Tool execution failed: {str(e)}")

    async def _search_youtube(self, query: str) -> List[TextContent]:
        """Search for YouTube videos."""
        try:
            from .main import search_youtube_video

            video_url = search_youtube_video(query)
            video_info = self.downloader.get_video_info(video_url)

            result = {"query": query, "found_url": video_url, "video_info": video_info}

            return [
                TextContent(
                    type="text",
                    text=f"🔍 YouTube Search Results\n\n"
                    f"**Query**: {query}\n"
                    f"**Found**: {video_info.get('title', 'Unknown')}\n"
                    f"**URL**: {video_url}\n"
                    f"**Duration**: {video_info.get('duration', 'Unknown')} seconds\n"
                    f"**Uploader**: {video_info.get('uploader', 'Unknown')}\n\n"
                    f"```json\n{json.dumps(result, indent=2)}\n```",
                )
            ]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Search failed: {str(e)}")]

    async def _get_video_info(self, url: str) -> List[TextContent]:
        """Get video information."""
        try:
            video_info = self.downloader.get_video_info(url)

            if video_info:
                return [
                    TextContent(
                        type="text",
                        text=f"📺 Video Information\n\n"
                        f"**Title**: {video_info.get('title', 'Unknown')}\n"
                        f"**Duration**: {video_info.get('duration', 'Unknown')} seconds\n"
                        f"**Uploader**: {video_info.get('uploader', 'Unknown')}\n"
                        f"**Views**: {video_info.get('view_count', 'Unknown'):,}\n"
                        f"**Description**: {video_info.get('description', 'No description')[:200]}...\n\n"
                        f"```json\n{json.dumps(video_info, indent=2)}\n```",
                    )
                ]
            else:
                return [
                    TextContent(type="text", text="❌ Failed to get video information")
                ]

        except Exception as e:
            return [
                TextContent(type="text", text=f"❌ Error getting video info: {str(e)}")
            ]

    async def _download_audio(
        self, url: str, quality: str = "best"
    ) -> List[TextContent]:
        """Download audio from YouTube."""
        try:
            audio_path = self.downloader.download_audio(url, quality)

            if audio_path:
                file_size = Path(audio_path).stat().st_size / 1024 / 1024  # MB
                return [
                    TextContent(
                        type="text",
                        text=f"🎵 Audio Downloaded Successfully\n\n"
                        f"**Path**: {audio_path}\n"
                        f"**Size**: {file_size:.2f} MB\n"
                        f"**Quality**: {quality}\n"
                        f"**Format**: {Path(audio_path).suffix}\n\n"
                        f"Audio is ready for analysis!",
                    )
                ]
            else:
                return [TextContent(type="text", text="❌ Failed to download audio")]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Download failed: {str(e)}")]

    async def _analyze_audio(self, audio_path: str) -> List[TextContent]:
        """Analyze audio and generate dance sequence."""
        try:
            # Check if file exists
            if not Path(audio_path).exists():
                return [
                    TextContent(
                        type="text", text=f"❌ Audio file not found: {audio_path}"
                    )
                ]

            # Analyze the audio
            results = self.analyzer.analyze_video(audio_path)

            if "error" in results:
                return [
                    TextContent(
                        type="text", text=f"❌ Analysis failed: {results['error']}"
                    )
                ]

            # Format results
            summary = results["summary"]

            return [
                TextContent(
                    type="text",
                    text=f"🎼 Audio Analysis Complete\n\n"
                    f"**Duration**: {summary['duration']:.2f} seconds\n"
                    f"**Tempo**: {summary['tempo']:.1f} BPM\n"
                    f"**Beats Detected**: {summary['total_beats']}\n"
                    f"**Onsets Detected**: {summary['total_onsets']}\n"
                    f"**Key Moments**: {summary['key_moments']}\n"
                    f"**Dance Moves Generated**: {summary['dance_moves_generated']}\n\n"
                    f"**Analysis Results**: {results['audio_path']}\n\n"
                    f"```json\n{json.dumps(summary, indent=2)}\n```",
                )
            ]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Analysis failed: {str(e)}")]

    async def _generate_dance_from_youtube(
        self, input_text: str, export_csv: bool = True
    ) -> List[TextContent]:
        """Complete workflow: download and analyze."""
        try:
            from .main import is_youtube_url, search_youtube_video, extract_video_id, find_existing_download

            # Normalize query for cache lookup (lowercase, strip whitespace)
            cache_key = input_text.lower().strip()

            # Check query cache first
            if cache_key in self.query_cache:
                cached = self.query_cache[cache_key]
                cached_csv = cached.get("csv_path")
                cached_url = cached.get("video_url")

                # Verify cached CSV still exists
                if cached_csv and Path(cached_csv).exists():
                    self._log(f"✅ Found cached query result: {cache_key}")
                    self._log(f"   Using CSV: {cached_csv}")

                    # Get video info for display
                    video_info = None
                    if cached_url:
                        video_info = self.downloader.get_video_info(cached_url)

                    # Format response
                    response = "🎉 Music Playing (from cache)!\n\n"
                    if video_info:
                        response += f"**Video**: {video_info.get('title', 'Unknown')}\n"
                        response += f"**Uploader**: {video_info.get('uploader', 'Unknown')}\n"
                    response += f"**CSV**: {Path(cached_csv).name}\n"
                    response += "\n🔊 **Audio playback starting in background**\n"
                    response += "Use `stop_music` to stop playback.\n"

                    # Clear stop event for new playback
                    self.stop_event.clear()

                    # Start playback in background
                    self.playback_task = asyncio.create_task(self._start_playback_background(cached_csv))

                    return [TextContent(type="text", text=response)]
                else:
                    self._log(f"⚠️ Cached CSV not found, re-processing")
                    # Remove invalid cache entry
                    del self.query_cache[cache_key]
                    self._save_query_cache()

            # Determine if input is URL or search terms
            if is_youtube_url(input_text):
                youtube_url = input_text
                self._log(f"Processing YouTube URL: {input_text}")
            else:
                self._log(f"Searching for: '{input_text}'")
                youtube_url = search_youtube_video(input_text)
                self._log(f"Found: {youtube_url}")

            # Get video info
            video_info = self.downloader.get_video_info(youtube_url)

            # Check for cached download first
            audio_path = None
            try:
                video_id = extract_video_id(youtube_url)
                self._log(f"Video ID: {video_id}")

                existing_file = find_existing_download(video_id, str(self.download_dir))
                if existing_file:
                    self._log(f"✅ Found cached audio file: {existing_file}")
                    audio_path = existing_file
            except Exception as e:
                self._log(f"⚠️ Could not check cache: {e}")

            # Download audio if not cached (with retry)
            if not audio_path:
                self._log("📥 Downloading audio...")
                audio_path = self.downloader.download_audio(youtube_url)
                if not audio_path:
                    self._log("⚠️ First download attempt failed, retrying...")
                    audio_path = self.downloader.download_audio(youtube_url)
                    if not audio_path:
                        return [TextContent(type="text", text="❌ Failed to download audio after 2 attempts")]
                self._log(f"✅ Audio downloaded: {audio_path}")

            # Check if CSV already exists for this audio file
            csv_path = None
            expected_csv = Path(self.output_dir) / f"{Path(audio_path).stem}_dance_moves.csv"
            if expected_csv.exists():
                self._log(f"✅ Found cached CSV: {expected_csv}")
                csv_path = str(expected_csv)
                # Load minimal info for summary display
                summary = {
                    "duration": 0,
                    "tempo": 0,
                    "dance_moves_generated": 0
                }
            else:
                # Analyze audio
                self._log("🎼 Analyzing audio...")
                results = self.analyzer.analyze_video(audio_path)

                if "error" in results:
                    return [
                        TextContent(
                            type="text", text=f"❌ Analysis failed: {results['error']}"
                        )
                    ]

                summary = results["summary"]

                # Export CSV if requested
                if export_csv:
                    csv_path = self.analyzer.export_dance_csv(results, audio_path)
                    self._log(f"✅ CSV exported: {csv_path}")

            # Save to query cache
            if csv_path:
                self.query_cache[cache_key] = {
                    "video_url": youtube_url,
                    "csv_path": csv_path
                }
                self._save_query_cache()
                self._log(f"💾 Saved query to cache: {cache_key}")

            # Format response
            response = "🎉 Music Playing!\n\n"

            if video_info:
                response += f"**Video**: {video_info.get('title', 'Unknown')}\n"
                response += f"**Uploader**: {video_info.get('uploader', 'Unknown')}\n"

            if summary["duration"] > 0:
                response += f"**Duration**: {summary['duration']:.2f} seconds\n"
                response += f"**Tempo**: {summary['tempo']:.1f} BPM\n"
            if summary["dance_moves_generated"] > 0:
                response += (
                    f"\n**Dance Moves Generated**: {summary['dance_moves_generated']}\n"
                )

            if csv_path:
                response += f"**CSV Export**: {Path(csv_path).name}\n"

            response += "\n🔊 **Audio playback starting in background**\n"
            response += "Use `stop_music` to stop playback.\n"

            # Clear stop event for new playback
            self.stop_event.clear()

            # Start playback in background (non-blocking)
            self.playback_task = asyncio.create_task(self._start_playback_background(csv_path))

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Workflow failed: {str(e)}")]

    async def _start_playback_background(self, csv_path: str):
        """Start playback in background task."""
        try:
            print("🎵 Loading dance moves from CSV...", file=sys.stderr)
            print(f"🎵 CSV path: {csv_path}", file=sys.stderr)

            # Create dance move in thread (blocking operations) with stop event
            self.current_dance_move = await asyncio.to_thread(
                CSVMove, csv_path, 0.5, True, self.stop_event
            )
            print(f"✅ Dance moves loaded, duration: {self.current_dance_move.duration:.2f}s", file=sys.stderr)

            # Try to connect to robot and play dance moves
            try:
                print("🤖 Attempting to import ReachyMini...", file=sys.stderr)
                from reachy_mini import ReachyMini
                print("✅ ReachyMini imported successfully", file=sys.stderr)

                # Run the entire robot play function in a thread
                def play_dance_with_robot():
                    """Play dance with robot in blocking thread."""
                    try:
                        print("🤖 Thread started - connecting to robot...", file=sys.stderr)

                        # This is the robot play function
                        def robot_play_function(dance_move_obj):
                            print("🤖 Inside robot_play_function", file=sys.stderr)
                            mini = ReachyMini()
                            # Store robot reference for cancellation BEFORE starting movements
                            self.current_robot = mini
                            print("🤖 Robot connected! Starting movements...", file=sys.stderr)
                            print(f"   Robot reference stored: {id(mini)}", file=sys.stderr)
                            try:
                                mini.play_move(dance_move_obj)
                                print("✅ Robot movements complete", file=sys.stderr)
                            finally:
                                print("🤖 Cleaning up robot connection...", file=sys.stderr)
                                mini.client.disconnect()
                                self.current_robot = None

                        if self.current_dance_move.audio_player:
                            print("🔊 Playing dance with synchronized audio...", file=sys.stderr)
                            # This blocks until dance is complete
                            self.current_dance_move.play_with_audio(
                                robot_play_function,
                                0,  # start_time
                                None,  # duration (full duration)
                            )
                        else:
                            print("⚠️ No audio player, playing moves only", file=sys.stderr)
                            robot_play_function(self.current_dance_move)

                        print("✅ Dance complete!", file=sys.stderr)

                    except Exception as e:
                        import traceback
                        print(f"❌ Error in play_dance_with_robot thread: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        raise
                    finally:
                        self.current_robot = None

                # Run the entire dance in a thread
                print("🤖 Starting dance thread...", file=sys.stderr)
                await asyncio.to_thread(play_dance_with_robot)
                print("✅ Dance thread completed!", file=sys.stderr)

            except ImportError as e:
                print(f"⚠️ reachy_mini import failed: {e}", file=sys.stderr)
                print("⚠️ Playing audio only.", file=sys.stderr)
                # Fallback: just play audio
                if self.current_dance_move.audio_player:
                    self.current_dance_move.start_audio()
                    print(
                        f"🔊 Audio playback started for {Path(csv_path).name}",
                        file=sys.stderr,
                    )
            except Exception as e:
                import traceback
                print(f"❌ Robot error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print("⚠️ Falling back to audio only", file=sys.stderr)
                # Fallback: just play audio
                if self.current_dance_move.audio_player:
                    self.current_dance_move.start_audio()

        except Exception as e:
            import traceback
            print(f"❌ Failed to start playback: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.current_dance_move = None

    async def _stop_music(self) -> List[TextContent]:
        """Stop the currently playing music."""
        try:
            print("🛑 Stop music requested", file=sys.stderr)

            stopped_anything = False

            # Set stop event to signal threads to stop
            self.stop_event.set()
            print("🛑 Stop event set", file=sys.stderr)

            # Cancel robot movement if robot is active
            if self.current_robot:
                print("🛑 Cancelling robot movement...", file=sys.stderr)
                try:
                    self.current_robot.cancel_move()
                    print("✅ Robot cancellation signal sent", file=sys.stderr)
                    stopped_anything = True
                except Exception as e:
                    import traceback
                    print(f"⚠️ Failed to cancel robot: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            else:
                print("⚠️ No robot reference available to cancel", file=sys.stderr)

            # Stop audio playback (this is quick)
            if self.current_dance_move and self.current_dance_move.audio_player:
                print("🛑 Stopping audio...", file=sys.stderr)
                self.current_dance_move.stop_audio()
                stopped_anything = True

            # Cancel the background playback task if running
            if self.playback_task and not self.playback_task.done():
                print("🛑 Cancelling playback task...", file=sys.stderr)
                self.playback_task.cancel()
                try:
                    await self.playback_task
                except asyncio.CancelledError:
                    print("✅ Playback task cancelled", file=sys.stderr)
                stopped_anything = True

            # Clear state
            self.current_dance_move = None
            self.playback_task = None
            self.current_robot = None

            if not stopped_anything:
                return [
                    TextContent(
                        type="text",
                        text="ℹ️ No music is currently playing",
                    )
                ]

            return [
                TextContent(
                    type="text",
                    text="🛑 Music and robot stopped successfully\n\n"
                    "Audio playback and robot movements have been stopped.",
                )
            ]
        except Exception as e:
            import traceback
            print(f"❌ Failed to stop music: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return [TextContent(type="text", text=f"❌ Failed to stop music: {str(e)}")]

    async def _load_dance_from_csv(
        self, csv_path: str, scale: float = 0.5
    ) -> List[TextContent]:
        """Load and analyze dance moves from CSV."""
        try:
            # Check if file exists
            if not Path(csv_path).exists():
                return [
                    TextContent(type="text", text=f"❌ CSV file not found: {csv_path}")
                ]

            # Load CSV
            dance_move = CSVMove(csv_path, scale=scale, audio_sync=False)

            response = "💃 Dance CSV Loaded Successfully\n\n"
            response += f"**File**: {Path(csv_path).name}\n"
            response += f"**Duration**: {dance_move.duration:.2f} seconds\n"
            response += f"**Scale Factor**: {scale}\n"

            if hasattr(dance_move, "has_body_yaw") and dance_move.has_body_yaw:
                response += "**Body Yaw**: Enabled\n"

            response += (
                "\n**CSV file contains robot movement data ready for playback!**"
            )

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ CSV loading failed: {str(e)}")]

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Mini YouTube MCP Server")
    parser.add_argument("--version", action="version", version="mini-yt-mcp 0.1.0")

    parser.parse_args()

    server = YouTubeMCPServer()

    print("🚀 Starting Mini YouTube MCP Server...", file=sys.stderr)
    print(
        "🎵 Available tools: play_music, stop_music",
        file=sys.stderr,
    )

    asyncio.run(server.run())


if __name__ == "__main__":
    import sys

    main()
