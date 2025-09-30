#!/usr/bin/env python3
"""Example script showing how to use the csv_move.py with audio synchronization."""

from pathlib import Path

# Import our enhanced CSVMove class
from csv_move import CSVMove


def demonstrate_audio_sync():
    """Demonstrate audio synchronized dance moves."""

    # Find the dance CSV file
    csv_files = list(Path("output").glob("*_dance_moves.csv"))

    if not csv_files:
        print("No dance move CSV files found in output folder")
        print("Run the dance generator first:")
        print(
            "python -m mini_yt_mcp.main 'https://youtube.com/watch?v=...' --export-csv"
        )
        return

    csv_path = str(csv_files[0])
    print(f"Using dance file: {Path(csv_path).name}")

    # Load dance moves with audio synchronization
    print("\n=== Loading Dance with Audio ===")
    dance_move = CSVMove(csv_path, scale=0.4, audio_sync=True)

    print(f"Dance duration: {dance_move.duration:.1f} seconds")

    if dance_move.audio_player:
        print(f"Audio file: {Path(dance_move.audio_player.audio_path).name}")

        # Test different playback scenarios
        scenarios = [
            {"name": "First 10 seconds", "start": 0, "duration": 10},
            {"name": "Middle section", "start": 30, "duration": 15},
            {"name": "High energy part", "start": 60, "duration": 20},
        ]

        for scenario in scenarios:
            print(f"\n=== Testing: {scenario['name']} ===")
            print(f"Playing from {scenario['start']}s for {scenario['duration']}s")

            try:
                # Start audio
                dance_move.start_audio(scenario["start"], scenario["duration"])

                # Simulate robot dancing (in real use, this would be robot.play_move(dance_move))
                print("🤖 Robot is dancing to the music...")

                # Show some dance moves during this time
                start_time = scenario["start"]
                end_time = start_time + scenario["duration"]
                sample_times = [
                    start_time + i * (scenario["duration"] / 5) for i in range(6)
                ]
                # Stop audio
                dance_move.stop_audio()

                input("Press Enter to continue to next scenario...")

            except KeyboardInterrupt:
                print("\nDemo interrupted")
                dance_move.stop_audio()
                break

    else:
        print("No matching audio file found")
        print("Make sure the audio file is in the downloads folder")


def show_usage_examples():
    """Show usage examples for csv_move.py."""

    print("\n" + "=" * 60)
    print("CSV Move Usage Examples")
    print("=" * 60)

    examples = [
        {
            "title": "Test CSV loading and audio detection",
            "command": 'python csv_move.py "output/your_dance_moves.csv" --test',
        },
        {
            "title": "Test audio playback only (first 30 seconds)",
            "command": 'python csv_move.py "output/your_dance_moves.csv" --audio-test --duration 30',
        },
        {
            "title": "Play dance with audio on robot",
            "command": 'python csv_move.py "output/your_dance_moves.csv" --scale 0.5',
        },
        {
            "title": "Play from specific time with duration limit",
            "command": 'python csv_move.py "output/your_dance_moves.csv" --start-time 30 --duration 60',
        },
        {
            "title": "Play robot moves without audio",
            "command": 'python csv_move.py "output/your_dance_moves.csv" --no-audio',
        },
        {
            "title": "Test high energy dance section",
            "command": 'python csv_move.py "output/your_dance_moves.csv" --audio-test --start-time 120 --duration 30',
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   {example['command']}")

    print("\n" + "=" * 60)
    print("Audio Requirements:")
    print("- pygame: pip install pygame (recommended)")
    print("- OR system audio player (afplay on macOS, aplay on Linux)")
    print("- Audio file must be in downloads/ folder with matching name")
    print("=" * 60)


if __name__ == "__main__":
    print("🎵 Audio Synchronized Dance Demo 🤖")

    try:
        demonstrate_audio_sync()
        show_usage_examples()

    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")
        show_usage_examples()
