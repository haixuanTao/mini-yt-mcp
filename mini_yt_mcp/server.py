#!/usr/bin/env python3
"""Mini YouTube MCP Server - Model Context Protocol server for YouTube audio analysis and dance generation."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    JSONRPCError,
)

from .audio_analyzer import AudioAnalyzer
from .downloader import YouTubeDownloader
from .csv_move import CSVMove


class YouTubeMCPServer:
    """MCP Server for YouTube audio analysis and dance generation."""

    def __init__(self):
        self.server = Server("mini-yt-mcp")
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.download_dir = self.temp_dir / "downloads"

        # Initialize components
        self.analyzer = AudioAnalyzer(str(self.output_dir))
        self.downloader = YouTubeDownloader(str(self.download_dir))

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools with the MCP server."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools."""
            return [
                Tool(
                    name="search_youtube",
                    description="Search for YouTube videos by query terms",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search terms for YouTube videos"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_video_info",
                    description="Get information about a YouTube video",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "YouTube video URL"
                            }
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="download_audio",
                    description="Download audio from a YouTube video",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "YouTube video URL"
                            },
                            "quality": {
                                "type": "string",
                                "description": "Audio quality preference",
                                "default": "best"
                            }
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="analyze_audio",
                    description="Analyze audio file and generate dance sequence",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "audio_path": {
                                "type": "string",
                                "description": "Path to audio file to analyze"
                            }
                        },
                        "required": ["audio_path"]
                    }
                ),
                Tool(
                    name="generate_dance_from_youtube",
                    description="Complete workflow: download YouTube audio and generate dance sequence",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "YouTube URL or search terms"
                            },
                            "export_csv": {
                                "type": "boolean",
                                "description": "Export dance moves to CSV format",
                                "default": True
                            }
                        },
                        "required": ["input"]
                    }
                ),
                Tool(
                    name="load_dance_from_csv",
                    description="Load and analyze dance moves from CSV file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "csv_path": {
                                "type": "string",
                                "description": "Path to CSV file with dance moves"
                            },
                            "scale": {
                                "type": "number",
                                "description": "Scale factor for movements",
                                "default": 0.5
                            }
                        },
                        "required": ["csv_path"]
                    }
                )
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
                        arguments["url"],
                        arguments.get("quality", "best")
                    )

                elif name == "analyze_audio":
                    return await self._analyze_audio(arguments["audio_path"])

                elif name == "generate_dance_from_youtube":
                    return await self._generate_dance_from_youtube(
                        arguments["input"],
                        arguments.get("export_csv", True)
                    )

                elif name == "load_dance_from_csv":
                    return await self._load_dance_from_csv(
                        arguments["csv_path"],
                        arguments.get("scale", 0.5)
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

            result = {
                "query": query,
                "found_url": video_url,
                "video_info": video_info
            }

            return [TextContent(
                type="text",
                text=f"🔍 YouTube Search Results\n\n"
                     f"**Query**: {query}\n"
                     f"**Found**: {video_info.get('title', 'Unknown')}\n"
                     f"**URL**: {video_url}\n"
                     f"**Duration**: {video_info.get('duration', 'Unknown')} seconds\n"
                     f"**Uploader**: {video_info.get('uploader', 'Unknown')}\n\n"
                     f"```json\n{json.dumps(result, indent=2)}\n```"
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Search failed: {str(e)}"
            )]

    async def _get_video_info(self, url: str) -> List[TextContent]:
        """Get video information."""
        try:
            video_info = self.downloader.get_video_info(url)

            if video_info:
                return [TextContent(
                    type="text",
                    text=f"📺 Video Information\n\n"
                         f"**Title**: {video_info.get('title', 'Unknown')}\n"
                         f"**Duration**: {video_info.get('duration', 'Unknown')} seconds\n"
                         f"**Uploader**: {video_info.get('uploader', 'Unknown')}\n"
                         f"**Views**: {video_info.get('view_count', 'Unknown'):,}\n"
                         f"**Description**: {video_info.get('description', 'No description')[:200]}...\n\n"
                         f"```json\n{json.dumps(video_info, indent=2)}\n```"
                )]
            else:
                return [TextContent(
                    type="text",
                    text="❌ Failed to get video information"
                )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Error getting video info: {str(e)}"
            )]

    async def _download_audio(self, url: str, quality: str = "best") -> List[TextContent]:
        """Download audio from YouTube."""
        try:
            audio_path = self.downloader.download_audio(url, quality)

            if audio_path:
                file_size = Path(audio_path).stat().st_size / 1024 / 1024  # MB
                return [TextContent(
                    type="text",
                    text=f"🎵 Audio Downloaded Successfully\n\n"
                         f"**Path**: {audio_path}\n"
                         f"**Size**: {file_size:.2f} MB\n"
                         f"**Quality**: {quality}\n"
                         f"**Format**: {Path(audio_path).suffix}\n\n"
                         f"Audio is ready for analysis!"
                )]
            else:
                return [TextContent(
                    type="text",
                    text="❌ Failed to download audio"
                )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Download failed: {str(e)}"
            )]

    async def _analyze_audio(self, audio_path: str) -> List[TextContent]:
        """Analyze audio and generate dance sequence."""
        try:
            # Check if file exists
            if not Path(audio_path).exists():
                return [TextContent(
                    type="text",
                    text=f"❌ Audio file not found: {audio_path}"
                )]

            # Analyze the audio
            results = self.analyzer.analyze_video(audio_path)

            if "error" in results:
                return [TextContent(
                    type="text",
                    text=f"❌ Analysis failed: {results['error']}"
                )]

            # Format results
            summary = results["summary"]

            return [TextContent(
                type="text",
                text=f"🎼 Audio Analysis Complete\n\n"
                     f"**Duration**: {summary['duration']:.2f} seconds\n"
                     f"**Tempo**: {summary['tempo']:.1f} BPM\n"
                     f"**Beats Detected**: {summary['total_beats']}\n"
                     f"**Onsets Detected**: {summary['total_onsets']}\n"
                     f"**Key Moments**: {summary['key_moments']}\n"
                     f"**Dance Moves Generated**: {summary['dance_moves_generated']}\n\n"
                     f"**Analysis Results**: {results['audio_path']}\n\n"
                     f"```json\n{json.dumps(summary, indent=2)}\n```"
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Analysis failed: {str(e)}"
            )]

    async def _generate_dance_from_youtube(self, input_text: str, export_csv: bool = True) -> List[TextContent]:
        """Complete workflow: download and analyze."""
        try:
            from .main import search_youtube_video, is_youtube_url

            # Determine if input is URL or search terms
            if is_youtube_url(input_text):
                youtube_url = input_text
                print(f"Processing YouTube URL: {input_text}")
            else:
                print(f"Searching for: '{input_text}'")
                youtube_url = search_youtube_video(input_text)
                print(f"Found: {youtube_url}")

            # Get video info
            video_info = self.downloader.get_video_info(youtube_url)

            # Download audio
            audio_path = self.downloader.download_audio(youtube_url)
            if not audio_path:
                return [TextContent(
                    type="text",
                    text="❌ Failed to download audio"
                )]

            # Analyze audio
            results = self.analyzer.analyze_video(audio_path)

            if "error" in results:
                return [TextContent(
                    type="text",
                    text=f"❌ Analysis failed: {results['error']}"
                )]

            # Export CSV if requested
            csv_path = None
            if export_csv:
                csv_path = self.analyzer.export_dance_csv(results, audio_path)

            # Format response
            summary = results["summary"]
            response = f"🎉 Complete Dance Generation Workflow\n\n"

            if video_info:
                response += f"**Video**: {video_info.get('title', 'Unknown')}\n"
                response += f"**Uploader**: {video_info.get('uploader', 'Unknown')}\n"

            response += f"**Audio**: {Path(audio_path).name}\n"
            response += f"**Duration**: {summary['duration']:.2f} seconds\n"
            response += f"**Tempo**: {summary['tempo']:.1f} BPM\n"
            response += f"**Dance Moves Generated**: {summary['dance_moves_generated']}\n"

            if csv_path:
                response += f"**CSV Export**: {csv_path}\n"

            response += f"\n**Files Created**:\n"
            response += f"- Audio: {audio_path}\n"
            response += f"- Analysis: {results.get('analysis_path', 'N/A')}\n"
            if csv_path:
                response += f"- CSV: {csv_path}\n"

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Workflow failed: {str(e)}"
            )]

    async def _load_dance_from_csv(self, csv_path: str, scale: float = 0.5) -> List[TextContent]:
        """Load and analyze dance moves from CSV."""
        try:
            # Check if file exists
            if not Path(csv_path).exists():
                return [TextContent(
                    type="text",
                    text=f"❌ CSV file not found: {csv_path}"
                )]

            # Load CSV
            dance_move = CSVMove(csv_path, scale=scale, audio_sync=False)

            response = f"💃 Dance CSV Loaded Successfully\n\n"
            response += f"**File**: {Path(csv_path).name}\n"
            response += f"**Duration**: {dance_move.duration:.2f} seconds\n"
            response += f"**Scale Factor**: {scale}\n"

            if hasattr(dance_move, 'has_body_yaw') and dance_move.has_body_yaw:
                response += f"**Body Yaw**: Enabled\n"

            response += f"\n**CSV file contains robot movement data ready for playback!**"

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ CSV loading failed: {str(e)}"
            )]

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Mini YouTube MCP Server")
    parser.add_argument(
        "--version", action="version", version="mini-yt-mcp 0.1.0"
    )

    args = parser.parse_args()

    server = YouTubeMCPServer()

    print("🚀 Starting Mini YouTube MCP Server...", file=sys.stderr)
    print("🎵 Available tools: search_youtube, get_video_info, download_audio, analyze_audio, generate_dance_from_youtube, load_dance_from_csv", file=sys.stderr)

    asyncio.run(server.run())


if __name__ == "__main__":
    import sys
    main()