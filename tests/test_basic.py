#!/usr/bin/env python3
"""Basic tests for mini-yt-mcp functionality."""

import tempfile
import os
import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported."""
    try:
        from mini_yt_mcp import audio_analyzer, csv_move, downloader
        print("✅ All core modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_module_initialization():
    """Test that core classes can be initialized."""
    try:
        from mini_yt_mcp.audio_analyzer import AudioAnalyzer
        from mini_yt_mcp.downloader import YouTubeDownloader

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test AudioAnalyzer initialization
            analyzer = AudioAnalyzer(temp_dir)
            assert analyzer.output_dir.exists()

            # Test YouTubeDownloader initialization
            downloader = YouTubeDownloader(temp_dir)
            assert downloader.download_dir.exists()

        print("✅ Core classes initialize correctly")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_help_command():
    """Test that the help command works."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "mini_yt_mcp.main", "--help"
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and "usage:" in result.stdout:
            print("✅ Help command works correctly")
            return True
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Help command test failed: {e}")
        return False

def test_entry_point():
    """Test that the mini-yt-mcp entry point works (if installed)."""
    try:
        result = subprocess.run([
            "mini-yt-mcp", "--help"
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and "usage:" in result.stdout:
            print("✅ Entry point works correctly")
            return True
        else:
            print("⚠️  Entry point not available (may not be installed via pip/uvx)")
            return True  # Not a failure if not installed
    except FileNotFoundError:
        print("⚠️  Entry point not found (may not be installed via pip/uvx)")
        return True  # Not a failure if not installed
    except Exception as e:
        print(f"❌ Entry point test failed: {e}")
        return False

def test_project_structure():
    """Test that required project files exist."""
    required_files = [
        "pyproject.toml",
        "README.md",
        ".gitignore",
        "mini_yt_mcp/__init__.py",
        "mini_yt_mcp/main.py",
        "mini_yt_mcp/audio_analyzer.py",
        "mini_yt_mcp/csv_move.py",
        "mini_yt_mcp/downloader.py"
    ]

    project_root = Path(__file__).parent.parent
    missing_files = []

    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required project files exist")
        return True

def run_all_tests():
    """Run all tests and return overall success."""
    tests = [
        ("Project Structure", test_project_structure),
        ("Module Imports", test_imports),
        ("Module Initialization", test_module_initialization),
        ("Help Command", test_help_command),
        ("Entry Point", test_entry_point),
    ]

    print("🧪 Running Mini YT MCP Tests")
    print("=" * 40)

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        success = test_func()
        results.append(success)

    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        return True
    else:
        print(f"❌ {total - passed} tests failed. ({passed}/{total} passed)")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)