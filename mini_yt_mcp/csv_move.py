#!/usr/bin/env python3
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class AudioPlayer:
    """Play audio files synchronized with dance moves."""

    def __init__(self, audio_path: str = None):
        """Initialize audio player.

        Args:
            audio_path: Path to audio file to play
        """
        self.audio_path = audio_path
        self.audio_thread = None
        self.stop_audio = threading.Event()

    def find_matching_audio(self, csv_path: str) -> str:
        """Find matching audio file for a given CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            Path to matching audio file or None if not found
        """
        csv_file = Path(csv_path)
        csv_name = csv_file.stem

        # Remove "_dance_moves" suffix if present
        if csv_name.endswith("_dance_moves"):
            audio_name = csv_name[:-12]  # Remove "_dance_moves"
        else:
            audio_name = csv_name

        # Look in downloads folder relative to CSV
        downloads_folder = csv_file.parent.parent / "downloads"
        if not downloads_folder.exists():
            downloads_folder = csv_file.parent / "downloads"

        # Look for audio files with matching name
        audio_extensions = [".wav", ".mp3", ".m4a", ".aac"]

        for ext in audio_extensions:
            audio_file = downloads_folder / f"{audio_name}{ext}"
            if audio_file.exists():
                return str(audio_file)

        # If exact match not found, look for files containing the audio name
        if downloads_folder.exists():
            for audio_file in downloads_folder.glob("*"):
                if audio_file.suffix.lower() in audio_extensions:
                    if audio_name.lower() in audio_file.stem.lower():
                        return str(audio_file)

        return None

    def play_audio_pygame(self, start_time: float = 0, duration: float = None):
        """Play audio using pygame (cross-platform).

        Args:
            start_time: Start time in seconds
            duration: Duration to play in seconds (None for full)
        """
        try:
            import pygame

            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.music.load(self.audio_path)

            pygame.mixer.music.play(start=start_time)

            # Monitor playback and stop when requested
            start_play_time = time.time()

            while pygame.mixer.music.get_busy() and not self.stop_audio.is_set():
                time.sleep(0.1)

                # Stop if duration exceeded
                if duration and (time.time() - start_play_time) >= duration:
                    pygame.mixer.music.stop()
                    break

        except ImportError:
            print("Warning: pygame not available for audio playback")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def play_audio_system(self, start_time: float = 0, duration: float = None):
        """Play audio using system command (macOS/Linux).

        Args:
            start_time: Start time in seconds
            duration: Duration to play in seconds (None for full)
        """
        try:
            import subprocess
            import sys

            if sys.platform == "darwin":  # macOS
                cmd = ["afplay", self.audio_path]
            else:  # Linux
                cmd = ["aplay", self.audio_path]

            # Add start time if supported
            if start_time > 0 and sys.platform == "darwin":
                cmd.extend(["-t", str(start_time)])

            process = subprocess.Popen(cmd)

            # Monitor and stop if needed
            start_play_time = time.time()

            while process.poll() is None and not self.stop_audio.is_set():
                time.sleep(0.1)

                # Stop if duration exceeded
                if duration and (time.time() - start_play_time) >= duration:
                    process.terminate()
                    break

            if process.poll() is None:
                process.terminate()

        except Exception as e:
            print(f"Error playing audio with system command: {e}")

    def start_playback(self, start_time: float = 0, duration: float = None):
        """Start audio playback in background thread.

        Args:
            start_time: Start time in seconds
            duration: Duration to play in seconds (None for full)
        """
        if not self.audio_path:
            # Silenced for MCP usage: print("No audio file specified")
            return

        if not Path(self.audio_path).exists():
            # Silenced for MCP usage: print(f"Audio file not found: {self.audio_path}")
            return

        # Silenced for MCP usage: print(f"Starting audio playback: {Path(self.audio_path).name}")
        # if start_time > 0:
        #     print(f"  Starting at {start_time:.1f}s")
        # if duration:
        #     print(f"  Duration: {duration:.1f}s")

        self.stop_audio.clear()

        # Try pygame first, fallback to system command
        try:
            import pygame

            self.audio_thread = threading.Thread(
                target=self.play_audio_pygame, args=(start_time, duration)
            )
        except ImportError:
            self.audio_thread = threading.Thread(
                target=self.play_audio_system, args=(start_time, duration)
            )

        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop_playback(self):
        """Stop audio playback."""
        if self.audio_thread and self.audio_thread.is_alive():
            # Silenced for MCP usage: print("Stopping audio playback...")
            self.stop_audio.set()
            self.audio_thread.join(timeout=1)


class CSVMove:
    def __init__(self, csv_path: str, scale: float = 0.3, audio_sync: bool = True, stop_event=None):
        df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self.stop_event = stop_event  # Event to signal playback should stop

        # Set up audio player if requested
        self.audio_player = None
        if audio_sync:
            self.audio_player = AudioPlayer()
            audio_path = self.audio_player.find_matching_audio(csv_path)
            if audio_path:
                self.audio_player.audio_path = audio_path
                # Silenced for MCP usage: print(f"Found matching audio: {Path(audio_path).name}")
            else:
                # Silenced for MCP usage: print("No matching audio file found in downloads folder")
                self.audio_player = None

        # Handle new CSV format with timestamp in seconds
        if "timestamp" in df.columns:
            times = df["timestamp"].values - df["timestamp"].values[0]
        else:
            # Fallback for old format
            times = (
                df["timestamp_ms"].values / 1000.0
                - df["timestamp_ms"].values[0] / 1000.0
            )

        self._duration = float(times[-1])

        # New CSV format has direct XYZ RPY coordinates
        if all(
            col in df.columns
            for col in ["x_cm", "y_cm", "z_cm", "roll_deg", "pitch_deg", "yaw_deg"]
        ):
            # Silenced for MCP usage: print("Using new XYZ RPY format")

            # Silenced sequence information for MCP usage
            # if "sequence_type" in df.columns and "sequence_variation" in df.columns:
            #     unique_sequences = df[["sequence_type", "sequence_variation", "sequence_position", "sequence_repetition"]].drop_duplicates()
            #     print(f"\n=== Dance Sequence Information ===")
            #     for _, seq in unique_sequences.iterrows():
            #         seq_type = seq.get("sequence_type", "unknown")
            #         seq_var = seq.get("sequence_variation", "unknown")
            #         seq_pos = seq.get("sequence_position", "unknown")
            #         seq_rep = seq.get("sequence_repetition", "unknown")
            #         print(f"Sequence: {seq_type} variation {seq_var}, position {seq_pos}, repetition {seq_rep}")
            #
            #     # Print summary of movement names if available
            #     if "head_movement_name" in df.columns:
            #         unique_moves = df["head_movement_name"].unique()
            #         print(f"\nMovement patterns detected: {len(unique_moves)} unique movements")
            #         for move in unique_moves[:5]:  # Show first 5 movements
            #             print(f"  - {move}")
            #         if len(unique_moves) > 5:
            #             print(f"  ... and {len(unique_moves) - 5} more")
            #     print(f"=== End Sequence Information ===\n")

            # Extract coordinates directly (already in cm and degrees)
            x_cm = df["x_cm"].values * scale * 0.01  # Convert cm to meters for robot
            y_cm = df["y_cm"].values * scale * 0.01
            z_cm = df["z_cm"].values * scale * 0.01

            roll_rad = np.deg2rad(df["roll_deg"].values) * scale
            pitch_rad = np.deg2rad(df["pitch_deg"].values) * scale
            yaw_rad = np.deg2rad(df["yaw_deg"].values) * scale

            # Check if body_yaw column exists and use it
            if "body_yaw_deg" in df.columns:
                body_yaw_rad = np.deg2rad(df["body_yaw_deg"].values) * scale * 3
                self.has_body_yaw = True
                # Silenced for MCP usage: print("Found body_yaw_deg column - using separate body rotation")
            else:
                body_yaw_rad = yaw_rad * 0.5  # Fallback to old calculation
                self.has_body_yaw = False

            # Create interpolators for all 6 DOF
            self.x_interp = interp1d(
                times,
                x_cm,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.y_interp = interp1d(
                times,
                y_cm,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.z_interp = interp1d(
                times,
                z_cm,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.roll_interp = interp1d(
                times,
                roll_rad,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.pitch_interp = interp1d(
                times,
                pitch_rad,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.yaw_interp = interp1d(
                times,
                yaw_rad,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.body_yaw_interp = interp1d(
                times,
                body_yaw_rad,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )

            self.new_format = True

        else:
            pass  # Silenced for MCP usage: print("Using legacy format")
            # Legacy format handling
            # Use smoothed data if available, else original
            pitch_col = next(
                (col for col in df.columns if "pitch" in col and "smooth" in col),
                "pitch",
            )
            yaw_col = next(
                (col for col in df.columns if "yaw" in col and "smooth" in col), "yaw"
            )
            center_x_col = next(
                (col for col in df.columns if "center_x" in col and "smooth" in col),
                "center_x",
            )
            center_y_col = next(
                (col for col in df.columns if "center_y" in col and "smooth" in col),
                "center_y",
            )
            center_z_col = next(
                (col for col in df.columns if "center_z" in col and "smooth" in col),
                "center_z",
            )

            pitch = np.deg2rad(df[pitch_col].values) * scale
            yaw = np.deg2rad(df[yaw_col].values) * scale

            # Normalize center coordinates (assuming screen coordinates, convert to movement scale)
            center_x = (
                (df[center_x_col].values - df[center_x_col].values[0]) * scale * 0.001
            )
            center_y = (
                (df[center_y_col].values - df[center_y_col].values[0]) * scale * 0.001
            )
            center_z = (
                (df[center_z_col].values - df[center_z_col].values[0]) * scale * 0.1
            )

            self.pitch_interp = interp1d(
                times,
                pitch,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.yaw_interp = interp1d(
                times, yaw, kind="next", bounds_error=False, fill_value="extrapolate"
            )
            self.center_x_interp = interp1d(
                times,
                center_x,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.center_y_interp = interp1d(
                times,
                center_y,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.center_z_interp = interp1d(
                times,
                center_z,
                kind="next",
                bounds_error=False,
                fill_value="extrapolate",
            )

            self.new_format = False

    @property
    def duration(self):
        return self._duration

    def evaluate(self, t):
        # Check if we should stop playback
        if self.stop_event and self.stop_event.is_set():
            raise StopIteration("Playback stopped by user")

        t = np.clip(t, 0, self.duration)

        if self.new_format:
            # New format with full 6-DOF XYZ RPY
            x = float(self.x_interp(t)) * 0.5
            y = float(self.y_interp(t)) * 0.5
            z = float(self.z_interp(t)) * 0.5
            roll = float(self.roll_interp(t)) * 0.5
            pitch = float(self.pitch_interp(t)) * 0.5
            yaw = float(self.yaw_interp(t)) * 0.5

            # Create full rotation matrix with roll, pitch, yaw (XYZ Euler angles)
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)

            # Combined rotation matrix (Z*Y*X order)
            head = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
                    [
                        sy * cp,
                        sy * sp * sr + cy * cr,
                        sy * sp * cr - cy * sr,
                        -y,
                    ],  # Invert Y for robot coordinates
                    [-sp, cp * sr, cp * cr, z],
                    [0, 0, 0, 1],
                ]
            )

            # Enhanced antenna movement based on all rotations
            antennas = np.array(
                [
                    pitch * 0.4 + roll * 0.2,  # Right antenna
                    -pitch * 0.4 - roll * 0.2,  # Left antenna
                ]
            )

            # Use body_yaw from CSV if available, otherwise calculate from head yaw
            if self.has_body_yaw:
                body_yaw = float(self.body_yaw_interp(t)) * 0.5
            else:
                body_yaw = yaw * 0.3 + roll * 0.1  # Fallback calculation

        else:
            # Legacy format
            pitch = float(self.pitch_interp(t))
            yaw = float(self.yaw_interp(t))
            center_x = float(self.center_x_interp(t))
            center_y = float(self.center_y_interp(t))
            center_z = float(self.center_z_interp(t))

            # Head matrix with rotation and translation from center coordinates
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)

            head = np.array(
                [
                    [cy * cp, -sy, cy * sp, center_x * 0.5],  # Reduce X movement
                    [
                        sy * cp,
                        cy,
                        sy * sp,
                        -center_y * 0.5,
                    ],  # Invert Y for robot coordinates
                    [-sp, 0, cp, center_z * 0.5],
                    [0, 0, 0, 1],
                ]
            )

            antennas = np.array([pitch * 0.3, -pitch * 0.3])
            body_yaw = yaw * 0.2

        return head, antennas, body_yaw

    def start_audio(self, start_time: float = 0, duration: float = None):
        """Start synchronized audio playback.

        Args:
            start_time: Start time in audio file (seconds)
            duration: Duration to play (seconds, None for full)
        """
        if self.audio_player:
            self.audio_player.start_playback(start_time, duration)

    def stop_audio(self):
        """Stop audio playback."""
        if self.audio_player:
            self.audio_player.stop_playback()

    def play_with_audio(
        self, robot_play_function, start_time: float = 0, duration: float = None
    ):
        """Play dance moves with synchronized audio.

        Args:
            robot_play_function: Function that plays the moves on robot
            start_time: Start time in audio file (seconds)
            duration: Duration to play (seconds, None for full)
        """
        try:
            # Start audio first
            if self.audio_player:
                self.start_audio(start_time, duration)

            # Busy wait one second
            instant = time.time()
            while time.time() - instant < 1.8:
                time.sleep(0.01)

            # Start robot movements
            robot_play_function(self)

        finally:
            # Always stop audio when done
            self.stop_audio()


def main():
    parser = argparse.ArgumentParser(
        description="Play dance moves from CSV on Reachy Mini robot with synchronized audio"
    )
    parser.add_argument("csv_path", help="CSV file path with dance moves")
    parser.add_argument(
        "--scale", type=float, default=0.5, help="Movement scale factor (default: 0.5)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - just load and analyze CSV without robot",
    )
    parser.add_argument(
        "--duration", type=float, help="Limit playback duration in seconds"
    )
    parser.add_argument(
        "--start-time", type=float, default=0, help="Start time in audio file (seconds)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback (robot moves only)",
    )
    parser.add_argument(
        "--audio-test",
        action="store_true",
        help="Test audio playback only (no robot)",
    )
    args = parser.parse_args()

    print(f"Loading CSV dance moves from: {args.csv_path}")
    move = CSVMove(args.csv_path, args.scale, audio_sync=not args.no_audio)

    print(f"Dance duration: {move.duration:.2f} seconds")

    if args.audio_test:
        # Test audio playback only
        print("Audio test mode - testing audio playback without robot")
        if move.audio_player:
            try:
                print(f"Playing audio from {args.start_time:.1f}s...")
                if args.duration:
                    print(f"Duration: {args.duration:.1f}s")

                move.start_audio(args.start_time, args.duration)

                # Wait for audio to complete
                duration = (
                    args.duration
                    if args.duration
                    else (move.duration - args.start_time)
                )
                time.sleep(duration)

                move.stop_audio()
                print("Audio test complete!")

            except KeyboardInterrupt:
                print("\nAudio test interrupted")
                move.stop_audio()
        else:
            print("No audio file found for testing")

    elif args.test:
        print("Test mode - analyzing movement data:")

        # Sample a few time points to show the movement
        test_times = [
            0,
            move.duration * 0.25,
            move.duration * 0.5,
            move.duration * 0.75,
            move.duration,
        ]

        for t in test_times:
            head, antennas, body_yaw = move.evaluate(t)
            position = head[:3, 3]  # Extract translation
            print(
                f"  t={t:.2f}s: pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), "
                f"antennas=({antennas[0]:.3f}, {antennas[1]:.3f}), body_yaw={body_yaw:.3f}"
            )

        print("CSV dance move loaded successfully!")

        # Test audio finding if enabled
        if move.audio_player:
            print(f"Audio file: {move.audio_player.audio_path}")

    else:
        print("Connecting to Reachy Mini robot...")
        try:
            from reachy_mini import ReachyMini

            def robot_play_function(dance_move):
                """Function to play dance moves on robot."""
                with ReachyMini() as mini:
                    mini.play_move(dance_move)

            if move.audio_player and not args.no_audio:
                print("Playing dance with synchronized audio...")
                if args.start_time > 0:
                    print(f"Starting audio from {args.start_time:.1f}s")

                # Play with synchronized audio
                move.play_with_audio(
                    robot_play_function, args.start_time, args.duration
                )
                print("Synchronized dance complete!")

            else:
                print("Playing dance without audio...")
                with ReachyMini() as mini:
                    mini.play_move(move)
                print("Dance complete!")

        except ImportError:
            print(
                "Error: reachy_mini not installed. Use --test or --audio-test for testing without robot."
            )
        except KeyboardInterrupt:
            print("\nDance interrupted")
            move.stop_audio()
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            print("Use --test or --audio-test for testing without robot.")
            move.stop_audio()


if __name__ == "__main__":
    main()
