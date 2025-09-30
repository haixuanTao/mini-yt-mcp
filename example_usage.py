#!/usr/bin/env python3
"""Example usage of the audio dance generator."""

from mini_yt_mcp.audio_analyzer import AudioAnalyzer
import numpy as np
import os

def create_test_audio():
    """Create a simple test audio file for demonstration."""
    try:
        import librosa

        # Create a simple test signal: sine waves at different frequencies
        sr = 22050
        duration = 5.0  # 5 seconds
        t = np.linspace(0, duration, int(sr * duration))

        # Create a beat pattern with bass drum (low freq) every 0.5 seconds
        beat_times = np.arange(0, duration, 0.5)
        audio = np.zeros_like(t)

        for beat_time in beat_times:
            # Add a bass drum hit (low frequency burst)
            start_idx = int(beat_time * sr)
            end_idx = min(start_idx + int(0.1 * sr), len(t))
            if end_idx <= len(audio):
                audio[start_idx:end_idx] += np.sin(2 * np.pi * 80 * t[start_idx:end_idx]) * np.exp(-10 * (t[start_idx:end_idx] - beat_time))

        # Add some melody
        melody_freq = 440  # A4
        audio += 0.3 * np.sin(2 * np.pi * melody_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t))

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8

        # Save as WAV file
        import soundfile as sf
        test_audio_path = "test_audio.wav"
        sf.write(test_audio_path, audio, sr)
        return test_audio_path

    except ImportError:
        print("soundfile not available, creating a dummy audio file")
        # Create a minimal WAV file header for testing
        test_audio_path = "test_audio.wav"
        with open(test_audio_path, 'wb') as f:
            # Minimal WAV header (this won't be valid audio, just for testing)
            f.write(b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x22\x56\x00\x00\x01\x00\x08\x00data\x00\x08\x00\x00')
            f.write(b'\x00' * 2048)  # Some audio data
        return test_audio_path

def demonstrate_dance_generation():
    """Demonstrate the dance generation system."""
    print("=== Audio Dance Generator Demo ===\n")

    # Initialize the analyzer
    analyzer = AudioAnalyzer("demo_output")

    print("1. Creating test audio file...")
    test_audio_path = create_test_audio()

    if not os.path.exists(test_audio_path):
        print("Failed to create test audio file")
        return

    print(f"   Test audio created: {test_audio_path}")

    try:
        print("\n2. Analyzing audio features...")
        audio_features = analyzer.analyze_audio_features(test_audio_path)

        print(f"   Duration: {audio_features['duration']:.2f} seconds")
        print(f"   Tempo: {audio_features['tempo']:.1f} BPM")
        print(f"   Beats detected: {len(audio_features['beats'])}")
        print(f"   Onsets detected: {len(audio_features['onsets'])}")

        print("\n3. Detecting key moments...")
        key_moments = analyzer.detect_key_moments(audio_features)
        print(f"   Key moments found: {len(key_moments)}")

        print("\n4. Generating dance moves...")
        dance_sequence = analyzer.generate_dance_sequence(key_moments)
        print(f"   Dance moves generated: {len(dance_sequence)}")

        print("\n5. Sample dance moves:")
        for i, move in enumerate(dance_sequence[:5]):
            timestamp = move['timestamp']
            energy = move['energy_level']
            moves = ', '.join(move['moves'])
            print(f"   {timestamp:.1f}s (energy: {energy:.2f}): {moves}")

        if len(dance_sequence) > 5:
            print(f"   ... and {len(dance_sequence) - 5} more moves")

        print(f"\n6. Analysis complete! Results saved in 'demo_output' directory")

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("This might be due to missing audio processing libraries.")
        print("Try installing: pip install librosa soundfile")

    finally:
        # Clean up test file
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)

if __name__ == "__main__":
    demonstrate_dance_generation()