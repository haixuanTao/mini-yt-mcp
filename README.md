# Mini YouTube MCP

**Mini YouTube MCP** - A minimal YouTube Model Context Protocol for downloading videos and generating dance moves from audio analysis.

## Features

- **YouTube Integration**: Download videos and audio using yt-dlp with search support
- **Audio Analysis**: Extract musical features (tempo, beats, onsets, energy levels)
- **Dance Generation**: Generate robot dance sequences based on audio characteristics
- **Energy-Based Movement**: Different movement patterns for low, medium, and high energy sections
- **Robot Integration**: Direct support for Reachy Mini robot with synchronized audio playback
- **CSV Export**: Export dance sequences with full movement data and sequence tracking
- **Caching**: Intelligent caching for both downloads and dance analysis

## Usage

### Robot Setup (Optional)

If you want to use the Reachy Mini robot features, start the robot daemon:

```bash
python -m reachy_mini.daemon.app.main
```

This daemon must be running before playing dance moves on the robot.

### Basic usage

**From YouTube URL:**

```bash
python -m mini_yt_mcp.main "https://www.youtube.com/watch?v=VIDEO_ID" --export-csv
```

**From search terms:**

```bash
python -m mini_yt_mcp.main "upbeat dance song" --export-csv
```

**Play on robot from CSV:**

```bash
python -m mini_yt_mcp.main "output/song_dance_moves.csv" --robot --robot-scale 0.5
```

## MCP Server

### Using Claude Desktop

To use this MCP server, you'll need an MCP-compatible client. Claude Desktop is one client you can use:

**Download Claude Desktop:**

- Visit [claude.ai/download](https://claude.ai/download)
- Available for macOS, Windows, and Linux
- Requires a Claude account (free tier available)

### Claude Desktop Configuration

To use this MCP server with Claude Desktop, add the following to your Claude Desktop config file:

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

**Configuration:**

```json
{
  "mcpServers": {
    "mini-yt-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/haixuanTao/mini-yt-mcp",
        "mini-yt-mcp-server"
      ],
      "env": {
        "UV_GIT_LFS": "1"
      }
    }
  }
}
```

**Note:** `UV_GIT_LFS=1` enables Git LFS support in uv. This requires uv 0.9.0+ and only works with HTTPS git URLs.

After updating the config, restart Claude Desktop. You can then use commands like:

- "Play some upbeat dance music"
- "Play https://youtube.com/watch?v=VIDEO_ID"
- "Stop the music"

### Alternative: Moly (Open Source)

[Moly](https://github.com/moxin-org/moly) is an open source AI desktop client built in Rust.

**Installation:**

```bash
# Download from releases
# Visit: https://github.com/moxin-org/moly/releases

# Or build from source
git clone https://github.com/moxin-org/moly.git
cd moly
cargo run --release
```

**configuration**

MCP Tab with:

```json
{
  "servers": {
    "spotify": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/haixuanTao/mini-yt-mcp",
        "mini-yt-mcp-server"
      ],
      "env": {
        "UV_GIT_LFS": "1"
      }
    }
  },
  "enabled": true,
  "dangerous_mode_enabled": true
}
```

### MCP Integration Example

The server works with any MCP-compatible client. Here's what the tools provide:

**Play Music (YouTube URL or search query):**

```json
{
  "tool": "play_music",
  "arguments": {
    "input": "https://youtube.com/watch?v=VIDEO_ID"
  }
}
```

Or with a search query:

```json
{
  "tool": "play_music",
  "arguments": {
    "input": "upbeat dance music 2024"
  }
}
```

**Stop Music:**

```json
{
  "tool": "stop_music",
  "arguments": {}
}
```

### MCP Server Features

- **Async Operation**: Non-blocking audio processing
- **Error Handling**: Comprehensive error reporting
- **File Management**: Automatic temporary file handling
- **Rich Responses**: Detailed JSON results with metadata
- **Tool Validation**: Input schema validation for all tools
