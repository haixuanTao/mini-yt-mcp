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

## Installation

### Prerequisites

**Install Git LFS:**

The reachy_mini dependency contains Git LFS assets. Install Git LFS to ensure compatibility:

macOS:
```bash
brew install git-lfs
git lfs install
```

Ubuntu/Debian:
```bash
sudo apt-get install git-lfs
git lfs install
```

Windows:
```bash
# Download from https://git-lfs.com/
git lfs install
```

### Install the package

**Option 1: Clone and install locally**
```bash
git clone https://github.com/haixuanTao/mini-yt-mcp.git
cd mini-yt-mcp
UV_GIT_LFS=1 uv sync
```

**Option 2: Install directly with uvx**
```bash
UV_GIT_LFS=1 uvx --from git+https://github.com/haixuanTao/mini-yt-mcp mini-yt-mcp-server
```

**Important:** `UV_GIT_LFS=1` enables Git LFS support in uv. This requires:
- Git LFS installed on your system (`brew install git-lfs`)
- uv version 0.9.0 or newer
- HTTPS URLs (SSH URLs don't work with UV_GIT_LFS)

## Usage

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

### Advanced options
```bash
python -m mini_yt_mcp.main "dance music 2024" \
    --export-csv \
    --robot \
    --robot-scale 0.3 \
    --start-time 30 \
    --robot-duration 60 \
    --download-dir "downloads" \
    --output-dir "output"
```

### Options

- `--quality`: Audio quality (default: best)
- `--download-dir`: Directory for downloaded audio (default: downloads)
- `--output-dir`: Directory for analysis results (default: output)
- `--skip-download`: Skip download if audio exists
- `--export-csv`: Export dance moves to CSV format
- `--robot`: Play dance moves on Reachy Mini robot
- `--robot-scale`: Movement scale factor for robot (default: 0.5)
- `--no-audio`: Disable audio playback during robot movements
- `--start-time`: Start time in audio for robot playback (seconds)
- `--robot-duration`: Limit robot playback duration (seconds)

## Output

The tool creates:
- Downloaded audio files in `downloads/`
- Analysis results in `output/`:
  - `{song_name}_dance_analysis.json`: Complete dance sequence with audio features
  - `{song_name}_dance_moves.csv`: CSV with timestamped dance moves and coordinates

## Dance Data Format

### CSV Output
Each row contains dance move data:
- `frame_number`: Frame index in dance sequence
- `timestamp`: Time in seconds
- `move_type`: Type of movement (main_beat, half_beat)
- `energy_level`: Audio energy level (0.0-1.0)
- `sequence_type`: Energy category (low_energy, medium_energy, high_energy)
- `sequence_variation`: Sequence pattern variation number
- `sequence_position`: Position within 8-beat sequence (0-7)
- `sequence_repetition`: Repetition count for tracking
- `head_movement_name`: Descriptive name of the movement
- `x_cm, y_cm, z_cm`: Head position coordinates in centimeters
- `roll_deg, pitch_deg, yaw_deg`: Head rotation angles in degrees
- `body_yaw_deg`: Body rotation angle in degrees

### JSON Output
Complete analysis including:
- Audio features (tempo, beats, onsets, energy levels)
- Musical structure detection
- Complete dance sequence with interpolation
- Movement generation metadata

## Requirements

- Python 3.10+ (required for MCP server)
- yt-dlp (YouTube downloading)
- librosa (audio analysis)
- numpy, scipy, pandas (numerical processing)
- mcp (Model Context Protocol support)
- reachy_mini (optional - for robot integration)

## Energy-Based Movement Patterns

The system generates different movement patterns based on audio energy:

- **Low Energy**: Smooth left-to-right body swaying over 2 beats, gentle head movements
- **Medium Energy**: Alternating body movements with return to center on half-beats
- **High Energy**: Dynamic movements with position holding and varied sequences

## Sequence Tracking

Each dance move includes sequence information for better understanding:
- Sequence type (energy category)
- Pattern variation (A-K sequences per energy level)
- Position within 8-beat cycle
- Repetition count for tracking progression

## MCP Server

Mini YT MCP includes a **Model Context Protocol (MCP) server** that exposes YouTube download and dance generation functionality to compatible clients.

### Starting the MCP Server

```bash
# Start the MCP server
mini-yt-mcp-server

# Or run directly
python -m mini_yt_mcp.server
```

### Available MCP Tools

The server provides these tools for use with MCP-compatible clients:

1. **play_music** - Search for music on YouTube and play it (downloads, analyzes, and plays on robot if available)
2. **stop_music** - Stop the currently playing music

### Installing Claude Desktop

To use this MCP server, you'll need an MCP-compatible client. Claude Desktop is the official client:

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

**Note:** `UV_GIT_LFS=1` enables Git LFS support in uv. This requires uv 0.9.0+ and only works with HTTPS git URLs (not SSH).

After updating the config, restart Claude Desktop. You can then use commands like:
- "Play some upbeat dance music"
- "Play https://youtube.com/watch?v=VIDEO_ID"
- "Stop the music"

### Alternative: Moly (Open Source)

[Moly](https://github.com/moxin-org/moly) is an open source AI desktop client built in Rust. While MCP support is not currently documented, you can check their repository for updates on MCP integration:

**Installation:**
```bash
# Download from releases
# Visit: https://github.com/moxin-org/moly/releases

# Or build from source
git clone https://github.com/moxin-org/moly.git
cd moly
cargo run --release
```

**Note:** MCP server configuration for Moly may differ from Claude Desktop. Please refer to [Moly's documentation](https://github.com/moxin-org/moly) for the latest information on MCP support and configuration.

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