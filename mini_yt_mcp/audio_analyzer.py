"""Audio analysis and dance move generation using librosa."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


class DanceMoveGenerator:
    """Generate dance moves based on audio features."""

    def __init__(self):
        """Initialize the head movement generator for dance."""
        # 8-beat sequence patterns - sharp, structured movements
        # Format: [x, y, z, roll, pitch, yaw] where:
        # x, y, z = position relative to neutral (in cm)
        # roll, pitch, yaw = rotation in degrees

        # Define both simple (2-beat repeated) and complex (8-beat) sequence patterns
        self.eight_beat_sequences = {
            "high_energy": [
                # Simple Sequence A: Left/Right snaps (2-beat repeated twice) - closer to medium energy
                [
                    {
                        "name": "Sharp snap left",
                        "coords": [0, 0.5, 0.25, -15, 0, -24],
                    },  # Beat 1
                    {
                        "name": "Sharp snap right",
                        "coords": [0, 0.5, 0.25, 15, 0, 24],
                    },  # Beat 2
                    {
                        "name": "Sharp snap left",
                        "coords": [0, 0.5, 0.25, -15, 0, -24],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Sharp snap right",
                        "coords": [0, 0.5, 0.25, 15, 0, 24],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Sharp snap left",
                        "coords": [0, 0.5, 0.25, -15, 0, -24],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Sharp snap right",
                        "coords": [0, 0.5, 0.25, 15, 0, 24],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Sharp snap left",
                        "coords": [0, 0.5, 0.25, -15, 0, -24],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Sharp snap right",
                        "coords": [0, 0.5, 0.25, 15, 0, 24],
                    },  # Beat 8 (repeat)
                ],
                # Simple Sequence B: Up/Down nods (2-beat repeated twice) - closer to medium energy
                [
                    {
                        "name": "Strong head drop",
                        "coords": [0, 0, -1.9, 0, -22, 0],
                    },  # Beat 1
                    {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},  # Beat 2
                    {
                        "name": "Strong head drop",
                        "coords": [0, 0, -1.9, 0, -22, 0],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Head snap up",
                        "coords": [0, 0, 2.2, 0, 19, 0],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Strong head drop",
                        "coords": [0, 0, -1.9, 0, -22, 0],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Head snap up",
                        "coords": [0, 0, 2.2, 0, 19, 0],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Strong head drop",
                        "coords": [0, 0, -1.9, 0, -22, 0],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Head snap up",
                        "coords": [0, 0, 2.2, 0, 19, 0],
                    },  # Beat 8 (repeat)
                ],
                # Simple Sequence C: Forward/Back thrust (2-beat repeated twice) - closer to medium energy
                [
                    {
                        "name": "Head thrust forward",
                        "coords": [0, 2.6, 0, 0, -9, 0],
                    },  # Beat 1
                    {
                        "name": "Head jerk back",
                        "coords": [0, -1.9, 0, 0, 11, 0],
                    },  # Beat 2
                    {
                        "name": "Head thrust forward",
                        "coords": [0, 2.6, 0, 0, -9, 0],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Head jerk back",
                        "coords": [0, -1.9, 0, 0, 11, 0],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Head thrust forward",
                        "coords": [0, 2.6, 0, 0, -9, 0],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Head jerk back",
                        "coords": [0, -1.9, 0, 0, 11, 0],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Head thrust forward",
                        "coords": [0, 2.6, 0, 0, -9, 0],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Head jerk back",
                        "coords": [0, -1.9, 0, 0, 11, 0],
                    },  # Beat 8 (repeat)
                ],
                # Complex Sequence D: Original side to vertical pattern - closer to medium energy
                [
                    {
                        "name": "Head snap left",
                        "coords": [0, 0, 0, 0, 0, -28],
                    },  # Beat 1
                    {
                        "name": "Head snap right",
                        "coords": [0, 0, 0, 0, 0, 28],
                    },  # Beat 2
                    {
                        "name": "Sharp head drop",
                        "coords": [0, 0, -1.9, 0, -22, 0],
                    },  # Beat 3
                    {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},  # Beat 4
                    {
                        "name": "Head thrust forward",
                        "coords": [0, 2.6, 0, 0, -9, 0],
                    },  # Beat 5
                    {
                        "name": "Head jerk back",
                        "coords": [0, -1.9, 0, 0, 11, 0],
                    },  # Beat 6
                    {
                        "name": "Strong tilt left",
                        "coords": [0, 0, 0, -26, 0, 0],
                    },  # Beat 7
                    {
                        "name": "Strong tilt right",
                        "coords": [0, 0, 0, 26, 0, 0],
                    },  # Beat 8
                ],
                # Sequence B: Aggressive circular pattern - closer to medium energy
                [
                    {
                        "name": "Head whip left",
                        "coords": [0, 0, 0, -24, 0, -27],
                    },  # Beat 1
                    {
                        "name": "Head slam down",
                        "coords": [0, 1.1, -1.9, 0, -26, 0],
                    },  # Beat 2
                    {
                        "name": "Head whip right",
                        "coords": [0, 0, 0, 24, 0, 27],
                    },  # Beat 3
                    {
                        "name": "Head throw back",
                        "coords": [0, -1.5, 1.9, 0, 21, 0],
                    },  # Beat 4
                    {
                        "name": "Diagonal tilt left",
                        "coords": [0, 0.8, 0, -24, -11, -15],
                    },  # Beat 5
                    {
                        "name": "Power nod center",
                        "coords": [0, 0, -1.1, 0, -24, 0],
                    },  # Beat 6
                    {
                        "name": "Diagonal tilt right",
                        "coords": [0, -0.8, 0, 24, -11, 15],
                    },  # Beat 7
                    {
                        "name": "Head explosion up",
                        "coords": [0, 0, 2.6, 0, 24, 0],
                    },  # Beat 8
                ],
                # Sequence C: Forward/back dominant - closer to medium energy
                [
                    {
                        "name": "Head punch forward",
                        "coords": [0, 3, 0, 0, -13, 0],
                    },  # Beat 1
                    {
                        "name": "Head recoil back",
                        "coords": [0, -2.2, 0, 0, 15, 0],
                    },  # Beat 2
                    {
                        "name": "Forward lean left",
                        "coords": [0, 1.9, -0.4, -21, -9, 0],
                    },  # Beat 3
                    {
                        "name": "Back lean right",
                        "coords": [0, -1.9, 0.4, 21, 9, 0],
                    },  # Beat 4
                    {
                        "name": "Strong forward nod",
                        "coords": [0, 2.2, -1.1, 0, -21, 0],
                    },  # Beat 5
                    {
                        "name": "Sharp back tilt",
                        "coords": [0, -1.5, 1.1, 0, 19, 0],
                    },  # Beat 6
                    {
                        "name": "Side thrust left",
                        "coords": [0, 0, 0, -24, 0, -26],
                    },  # Beat 7
                    {
                        "name": "Side thrust right",
                        "coords": [0, 0, 0, 24, 0, 26],
                    },  # Beat 8
                ],
                # Sequence G: Diagonal patterns - reduced energy
                [
                    {
                        "name": "Diagonal up-left",
                        "coords": [0, 0, 2.5, -24, 16, -28],
                    },  # Beat 1
                    {
                        "name": "Diagonal down-right",
                        "coords": [0, 0, -2.5, 24, -16, 28],
                    },  # Beat 2
                    {
                        "name": "Diagonal up-right",
                        "coords": [0, 0, 2.5, 24, 16, 28],
                    },  # Beat 3
                    {
                        "name": "Diagonal down-left",
                        "coords": [0, 0, -2.5, -24, -16, -28],
                    },  # Beat 4
                    {
                        "name": "Sharp diagonal left",
                        "coords": [0, 0, 1.1, -24, 9, -24],
                    },  # Beat 5
                    {
                        "name": "Sharp diagonal right",
                        "coords": [0, 0, -1.1, 24, -9, 24],
                    },  # Beat 6
                    {
                        "name": "Power diagonal up",
                        "coords": [0, 0, 3, 0, 20, 0],
                    },  # Beat 7
                    {
                        "name": "Power diagonal down",
                        "coords": [0, 0, -3, 0, -20, 0],
                    },  # Beat 8
                ],
                [
                    # Sequence H: Rapid fire pattern - reduced energy
                    {
                        "name": "Quick left snap",
                        "coords": [0, 0, 0, 0, 0, -24],
                    },  # Beat 1
                    {
                        "name": "Quick right snap",
                        "coords": [0, 0, 0, 0, 0, 24],
                    },  # Beat 2
                    {
                        "name": "Quick up jerk",
                        "coords": [0, 0, 2.5, 0, 20, 0],
                    },  # Beat 3
                    {
                        "name": "Quick down slam",
                        "coords": [0, 0, -2.5, 0, -20, 0],
                    },  # Beat 4
                    {
                        "name": "Rapid left turn",
                        "coords": [0, 1.5, 0, -20, 0, -28],
                    },  # Beat 5
                    {
                        "name": "Rapid right turn",
                        "coords": [0, -1.5, 0, 20, 0, 28],
                    },  # Beat 6
                    {
                        "name": "Explosive forward",
                        "coords": [0, 4.5, 0, 0, -24, 0],
                    },  # Beat 7
                    {
                        "name": "Snap back center",
                        "coords": [0, -2.5, 0, 0, 16, 0],
                    },  # Beat 8
                ],
                # Sequence I: Spiral pattern - reduced energy
                [
                    {
                        "name": "Spiral start left",
                        "coords": [-1.5, 0, 0.8, -16, 8, -20],
                    },  # Beat 1
                    {
                        "name": "Spiral forward",
                        "coords": [0, 1, 1.5, 0, -12, -8],
                    },  # Beat 2
                    {
                        "name": "Spiral right",
                        "coords": [1.5, 1.5, 0.8, 16, 0, 20],
                    },  # Beat 3
                    {
                        "name": "Spiral back",
                        "coords": [0.8, -2.5, -0.8, 12, 16, 12],
                    },  # Beat 4
                    {
                        "name": "Spiral down-left",
                        "coords": [-2.5, -0.8, -1.5, -20, -8, -24],
                    },  # Beat 5
                    {
                        "name": "Spiral up-forward",
                        "coords": [0.8, 1, 2.5, 8, -16, 8],
                    },  # Beat 6
                    {
                        "name": "Spiral complete right",
                        "coords": [2.5, 0.8, 0, 24, 4, 28],
                    },  # Beat 7
                    {
                        "name": "Spiral reset center",
                        "coords": [0, -1.5, 0.8, 0, 12, 0],
                    },  # Beat 8
                ],
                # Sequence J: Sharp angles - reduced energy
                [
                    {
                        "name": "Sharp left angle",
                        "coords": [-1, 0, 0, -36, 0, -40],
                    },  # Beat 1
                    {
                        "name": "Sharp right angle",
                        "coords": [1, 0, 0, 36, 0, 40],
                    },  # Beat 2
                    {
                        "name": "Sharp up angle",
                        "coords": [0, 0, 1, 0, -36, 0],
                    },  # Beat 3
                    {
                        "name": "Sharp down angle",
                        "coords": [0, 0, -1, 0, 36, 0],
                    },  # Beat 4
                    {
                        "name": "Corner left-up",
                        "coords": [-0.5, 0.5, 2.5, -28, -20, -32],
                    },  # Beat 5
                    {
                        "name": "Corner right-down",
                        "coords": [0.5, -0.5, -2.5, 28, 20, 32],
                    },  # Beat 6
                    {
                        "name": "Angular thrust",
                        "coords": [1.5, 2, -0.8, 16, -28, 16],
                    },  # Beat 7
                    {
                        "name": "Angular recoil",
                        "coords": [-1, -2, 1, -16, 24, -16],
                    },  # Beat 8
                ],
                # Translation-only sequence - High energy (avg ~24.0)
                [
                    {
                        "name": "Sharp right slide",
                        "coords": [2.4, 0, 0, 0, 0, 0],
                    },  # Beat 1
                    {
                        "name": "Sharp left slide",
                        "coords": [-2.4, 0, 0, 0, 0, 0],
                    },  # Beat 2
                    {
                        "name": "Sharp forward thrust",
                        "coords": [0, 2.4, 0, 0, 0, 0],
                    },  # Beat 3
                    {
                        "name": "Sharp backward pull",
                        "coords": [0, -2.4, 0, 0, 0, 0],
                    },  # Beat 4
                    {"name": "Sharp up lift", "coords": [0, 0, 2.4, 0, 0, 0]},  # Beat 5
                    {
                        "name": "Sharp down drop",
                        "coords": [0, 0, -2.4, 0, 0, 0],
                    },  # Beat 6
                    {
                        "name": "Sharp diagonal out",
                        "coords": [1.7, 1.7, 0, 0, 0, 0],
                    },  # Beat 7
                    {
                        "name": "Sharp diagonal in",
                        "coords": [-1.7, -1.7, 0, 0, 0, 0],
                    },  # Beat 8
                ],
            ],
            "medium_energy": [
                # Simple Sequence A: Nod down/up (2-beat repeated twice) - balanced medium energy
                [
                    {
                        "name": "Sharp nod down",
                        "coords": [0, 0, 0, 0, -21, 0],
                    },  # Beat 1
                    {"name": "Head up", "coords": [0, 0, 1.9, 0, 13, 0]},  # Beat 2
                    {
                        "name": "Sharp nod down",
                        "coords": [0, 0, 0, 0, -21, 0],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Head up",
                        "coords": [0, 0, 1.9, 0, 13, 0],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Sharp nod down",
                        "coords": [0, 0, 0, 0, -21, 0],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Head up",
                        "coords": [0, 0, 1.9, 0, 13, 0],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Sharp nod down",
                        "coords": [0, 0, 0, 0, -21, 0],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Head up",
                        "coords": [0, 0, 1.9, 0, 13, 0],
                    },  # Beat 8 (repeat)
                ],
                # Simple Sequence B: Turn left/right (2-beat repeated twice) - balanced medium energy
                [
                    {
                        "name": "Head turn left",
                        "coords": [0, 0, 0, 0, 0, -26],
                    },  # Beat 1
                    {
                        "name": "Head turn right",
                        "coords": [0, 0, 0, 0, 0, 26],
                    },  # Beat 2
                    {
                        "name": "Head turn left",
                        "coords": [0, 0, 0, 0, 0, -26],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Head turn right",
                        "coords": [0, 0, 0, 0, 0, 26],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Head turn left",
                        "coords": [0, 0, 0, 0, 0, -26],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Head turn right",
                        "coords": [0, 0, 0, 0, 0, 26],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Head turn left",
                        "coords": [0, 0, 0, 0, 0, -26],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Head turn right",
                        "coords": [0, 0, 0, 0, 0, 26],
                    },  # Beat 8 (repeat)
                ],
                # Simple Sequence C: Lean left forward/right back (2-beat repeated twice) - slightly reduced medium energy
                [
                    {
                        "name": "Lean left forward",
                        "coords": [0, 1.6, 0, -21, -4, 0],
                    },  # Beat 1
                    {
                        "name": "Lean right back",
                        "coords": [0, -1.6, 0, 21, 4, 0],
                    },  # Beat 2
                    {
                        "name": "Lean left forward",
                        "coords": [0, 1.6, 0, -21, -4, 0],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Lean right back",
                        "coords": [0, -1.6, 0, 21, 4, 0],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Lean left forward",
                        "coords": [0, 1.6, 0, -21, -4, 0],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Lean right back",
                        "coords": [0, -1.6, 0, 21, 4, 0],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Lean left forward",
                        "coords": [0, 1.6, 0, -21, -4, 0],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Lean right back",
                        "coords": [0, -1.6, 0, 21, 4, 0],
                    },  # Beat 8 (repeat)
                ],
                # Complex Sequence D: Original classic nod and turn - perfectly balanced
                [
                    {
                        "name": "Sharp nod down",
                        "coords": [0, 0, 0, 0, -19, 0],
                    },  # Beat 1 - Center
                    {
                        "name": "Head up",
                        "coords": [0, 0, 1.5, 0, 11, 0],
                    },  # Beat 2 - Center
                    {
                        "name": "Head turn left",
                        "coords": [0, 0, 0, 0, 0, -22],
                    },  # Beat 3 - Left
                    {
                        "name": "Head turn right",
                        "coords": [0, 0, 0, 0, 0, 22],
                    },  # Beat 4 - Right
                    {
                        "name": "Lean left forward",
                        "coords": [0, 1.5, 0, -19, -4, 0],
                    },  # Beat 5 - Left
                    {
                        "name": "Lean right back",
                        "coords": [0, -1.5, 0, 19, 4, 0],
                    },  # Beat 6 - Right
                    {
                        "name": "Center nod down",
                        "coords": [0, 0, 0, 0, -15, 0],
                    },  # Beat 7 - Center
                    {
                        "name": "Center lift up",
                        "coords": [0, 0, 1.5, 0, 9, 0],
                    },  # Beat 8 - Center
                ],
                # Sequence B: Rhythmic bob pattern - perfectly balanced
                [
                    {
                        "name": "Bob down center",
                        "coords": [0, 0, -0.8, 0, -22, 0],
                    },  # Beat 1
                    {
                        "name": "Bob up left",
                        "coords": [0, 0, 1.5, -15, 15, -15],
                    },  # Beat 2
                    {
                        "name": "Bob down center",
                        "coords": [0, 0.8, 0, 0, -15, 0],
                    },  # Beat 3
                    {
                        "name": "Bob up right",
                        "coords": [0, 0, 1.5, 15, 15, 15],
                    },  # Beat 4
                    {
                        "name": "Side lean left",
                        "coords": [0, 0, 0, -22, 0, -11],
                    },  # Beat 5
                    {"name": "Center nod", "coords": [0, 0.8, 0, 0, -11, 0]},  # Beat 6
                    {
                        "name": "Side lean right",
                        "coords": [0, 0, 0, 22, 0, 11],
                    },  # Beat 7
                    {"name": "Upward lift", "coords": [0, 0, 2.2, 0, 8, 0]},  # Beat 8
                ],
                # Sequence C: Flow sequence
                [
                    {
                        "name": "Flow start left",
                        "coords": [0, 0.8, 0.8, -15, -8, -19],
                    },  # Beat 1
                    {
                        "name": "Flow down center",
                        "coords": [0, 1.5, -0.8, 0, -19, 0],
                    },  # Beat 2
                    {
                        "name": "Flow right up",
                        "coords": [0, 0.8, 1.5, 15, 11, 19],
                    },  # Beat 3
                    {
                        "name": "Flow back center",
                        "coords": [0, -0.8, 0.8, 0, 15, 0],
                    },  # Beat 4
                    {
                        "name": "Flow diagonal 1",
                        "coords": [0, 1.5, 0, -19, -11, -15],
                    },  # Beat 5
                    {
                        "name": "Flow diagonal 2",
                        "coords": [0, -0.8, 1.5, 11, 15, 11],
                    },  # Beat 6
                    {
                        "name": "Flow circle left",
                        "coords": [0, 0, 0.8, -11, -4, -22],
                    },  # Beat 7
                    {
                        "name": "Flow circle right",
                        "coords": [0, 0, 0.8, 11, -4, 22],
                    },  # Beat 8
                ],
                # Sequence G: Smooth waves
                [
                    {
                        "name": "Wave left start",
                        "coords": [0, 0.8, 0, -13, -6, -15],
                    },  # Beat 1
                    {
                        "name": "Wave center dip",
                        "coords": [0, 0, -0.8, 0, -11, 0],
                    },  # Beat 2
                    {
                        "name": "Wave right rise",
                        "coords": [0, 0.8, 0.8, 13, 8, 15],
                    },  # Beat 3
                    {
                        "name": "Wave back center",
                        "coords": [0, -0.8, 0, 0, 9, 0],
                    },  # Beat 4
                    {
                        "name": "Wave forward left",
                        "coords": [0, 1.5, 0, -11, -8, -11],
                    },  # Beat 5
                    {
                        "name": "Wave up right",
                        "coords": [0, 0, 1.5, 11, 11, 11],
                    },  # Beat 6
                    {
                        "name": "Wave down left",
                        "coords": [0, -0.8, -0.8, -9, -9, -13],
                    },  # Beat 7
                    {
                        "name": "Wave reset center",
                        "coords": [0, 0.8, 0.8, 0, 6, 0],
                    },  # Beat 8
                ],
                # Sequence H: Alternating emphasis
                [
                    {
                        "name": "Emphasis left nod",
                        "coords": [0, 0, 0, -15, -13, -19],
                    },  # Beat 1
                    {
                        "name": "Soft center up",
                        "coords": [0, 0, 0.8, 0, 8, 0],
                    },  # Beat 2
                    {
                        "name": "Emphasis right nod",
                        "coords": [0, 0, 0, 15, -13, 19],
                    },  # Beat 3
                    {
                        "name": "Soft center up",
                        "coords": [0, 0, 0.8, 0, 8, 0],
                    },  # Beat 4
                    {
                        "name": "Strong forward",
                        "coords": [0, 2.2, 0, 0, -16, 0],
                    },  # Beat 5
                    {
                        "name": "Gentle back",
                        "coords": [0, -0.8, 0.8, 0, 6, 0],
                    },  # Beat 6
                    {
                        "name": "Side emphasis left",
                        "coords": [0, 0.8, 0, -19, 0, -15],
                    },  # Beat 7
                    {
                        "name": "Side emphasis right",
                        "coords": [0, 0.8, 0, 19, 0, 15],
                    },  # Beat 8
                ],
                # Sequence I: Figure-8 pattern
                [
                    {
                        "name": "Figure-8 start",
                        "coords": [0, 1, 1, -15, -10, -18],
                    },  # Beat 1
                    {
                        "name": "Figure-8 cross center",
                        "coords": [0, 0, 0, 0, 0, 0],
                    },  # Beat 2
                    {
                        "name": "Figure-8 right loop",
                        "coords": [0, 1, 1, 15, 10, 18],
                    },  # Beat 3
                    {
                        "name": "Figure-8 back cross",
                        "coords": [0, -1, -1, 0, 5, 0],
                    },  # Beat 4
                    {
                        "name": "Figure-8 left down",
                        "coords": [0, 0, -1, -18, -15, -20],
                    },  # Beat 5
                    {
                        "name": "Figure-8 up cross",
                        "coords": [0, 1, 1, 0, 12, 0],
                    },  # Beat 6
                    {
                        "name": "Figure-8 right down",
                        "coords": [0, 0, -1, 18, -15, 20],
                    },  # Beat 7
                    {
                        "name": "Figure-8 complete",
                        "coords": [0, -1, 0, 0, 8, 0],
                    },  # Beat 8
                ],
                # Sequence J: Pulse pattern
                [
                    {
                        "name": "Pulse out left",
                        "coords": [0, 1, 0, -22, 0, -20],
                    },  # Beat 1
                    {
                        "name": "Pulse in center",
                        "coords": [0, -1, 0, 0, 0, 0],
                    },  # Beat 2
                    {
                        "name": "Pulse out right",
                        "coords": [0, 1, 0, 22, 0, 20],
                    },  # Beat 3
                    {
                        "name": "Pulse in center",
                        "coords": [0, -1, 0, 0, 0, 0],
                    },  # Beat 4
                    {
                        "name": "Pulse up forward",
                        "coords": [0, 2, 2, 0, -18, 0],
                    },  # Beat 5
                    {
                        "name": "Pulse down back",
                        "coords": [0, -1, -1, 0, 15, 0],
                    },  # Beat 6
                    {
                        "name": "Pulse diagonal left",
                        "coords": [0, 1, 1, -20, -12, -15],
                    },  # Beat 7
                    {
                        "name": "Pulse diagonal right",
                        "coords": [0, 1, 1, 20, -12, 15],
                    },  # Beat 8
                ],
                # Translation-only sequence - Medium energy (avg ~22.2)
                [
                    {
                        "name": "Smooth right glide",
                        "coords": [2.2, 0, 0, 0, 0, 0],
                    },  # Beat 1
                    {
                        "name": "Smooth left glide",
                        "coords": [-2.2, 0, 0, 0, 0, 0],
                    },  # Beat 2
                    {
                        "name": "Smooth forward glide",
                        "coords": [0, 2.2, 0, 0, 0, 0],
                    },  # Beat 3
                    {
                        "name": "Smooth backward glide",
                        "coords": [0, -2.2, 0, 0, 0, 0],
                    },  # Beat 4
                    {
                        "name": "Smooth up glide",
                        "coords": [0, 0, 2.2, 0, 0, 0],
                    },  # Beat 5
                    {
                        "name": "Smooth down glide",
                        "coords": [0, 0, -2.2, 0, 0, 0],
                    },  # Beat 6
                    {
                        "name": "Smooth diagonal out",
                        "coords": [1.6, 1.6, 0, 0, 0, 0],
                    },  # Beat 7
                    {
                        "name": "Smooth diagonal in",
                        "coords": [-1.6, -1.6, 0, 0, 0, 0],
                    },  # Beat 8
                ],
            ],
            "low_energy": [
                # Simple Sequence A: Gentle nod down/up (2-beat repeated twice) - rotation only
                [
                    {
                        "name": "Gentle nod down",
                        "coords": [0, 0, 0, 0, -22, 0],
                    },  # Beat 1
                    {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},  # Beat 2
                    {
                        "name": "Gentle nod down",
                        "coords": [0, 0, 0, 0, -22, 0],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Gentle nod up",
                        "coords": [0, 0, 0, 0, 18, 0],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Gentle nod down",
                        "coords": [0, 0, 0, 0, -22, 0],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Gentle nod up",
                        "coords": [0, 0, 0, 0, 18, 0],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Gentle nod down",
                        "coords": [0, 0, 0, 0, -22, 0],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Gentle nod up",
                        "coords": [0, 0, 0, 0, 18, 0],
                    },  # Beat 8 (repeat)
                ],
                # Simple Sequence B: Soft turn left/right (2-beat repeated twice)
                [
                    {
                        "name": "Soft turn left",
                        "coords": [0, 0, 0, 0, 0, -18],
                    },  # Beat 1
                    {
                        "name": "Soft turn right",
                        "coords": [0, 0, 0, 0, 0, 18],
                    },  # Beat 2
                    {
                        "name": "Soft turn left",
                        "coords": [0, 0, 0, 0, 0, -18],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Soft turn right",
                        "coords": [0, 0, 0, 0, 0, 18],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Soft turn left",
                        "coords": [0, 0, 0, 0, 0, -18],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Soft turn right",
                        "coords": [0, 0, 0, 0, 0, 18],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Soft turn left",
                        "coords": [0, 0, 0, 0, 0, -18],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Soft turn right",
                        "coords": [0, 0, 0, 0, 0, 18],
                    },  # Beat 8 (repeat)
                ],
                # Simple Sequence C: Light tilt left/right (2-beat repeated twice)
                [
                    {
                        "name": "Light tilt left",
                        "coords": [0, 0, 0, -15, 0, 0],
                    },  # Beat 1
                    {
                        "name": "Light tilt right",
                        "coords": [0, 0, 0, 15, 0, 0],
                    },  # Beat 2
                    {
                        "name": "Light tilt left",
                        "coords": [0, 0, 0, -15, 0, 0],
                    },  # Beat 3 (repeat)
                    {
                        "name": "Light tilt right",
                        "coords": [0, 0, 0, 15, 0, 0],
                    },  # Beat 4 (repeat)
                    {
                        "name": "Light tilt left",
                        "coords": [0, 0, 0, -15, 0, 0],
                    },  # Beat 5 (repeat)
                    {
                        "name": "Light tilt right",
                        "coords": [0, 0, 0, 15, 0, 0],
                    },  # Beat 6 (repeat)
                    {
                        "name": "Light tilt left",
                        "coords": [0, 0, 0, -15, 0, 0],
                    },  # Beat 7 (repeat)
                    {
                        "name": "Light tilt right",
                        "coords": [0, 0, 0, 15, 0, 0],
                    },  # Beat 8 (repeat)
                ],
                # Complex Sequence D: Original gentle sway - rotation only
                [
                    {
                        "name": "Gentle nod down",
                        "coords": [0, 0, 0, 0, -12, 0],
                    },  # Beat 1
                    {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 8, 0]},  # Beat 2
                    {
                        "name": "Soft turn left",
                        "coords": [0, 0, 0, 0, 0, -18],
                    },  # Beat 3
                    {
                        "name": "Soft turn right",
                        "coords": [0, 0, 0, 0, 0, 18],
                    },  # Beat 4
                    {
                        "name": "Light tilt left",
                        "coords": [0, 0, 0, -15, 0, 0],
                    },  # Beat 5
                    {
                        "name": "Light tilt right",
                        "coords": [0, 0, 0, 15, 0, 0],
                    },  # Beat 6
                    {
                        "name": "Gentle pitch forward",
                        "coords": [0, 0, 0, 0, -8, 0],
                    },  # Beat 7
                    {
                        "name": "Gentle pitch back",
                        "coords": [0, 0, 0, 0, 6, 0],
                    },  # Beat 8
                ],
                # Sequence B: Subtle exploration - rotation only
                [
                    {
                        "name": "Gentle pitch down",
                        "coords": [0, 0, 0, 0, -18, 0],
                    },  # Beat 1 - Center
                    {
                        "name": "Gentle tilt left",
                        "coords": [0, 0, 0, -20, 0, -12],
                    },  # Beat 2 - Left
                    {
                        "name": "Minimal turn left",
                        "coords": [0, 0, 0, -15, 0, -15],
                    },  # Beat 3 - Left
                    {
                        "name": "Minimal turn right",
                        "coords": [0, 0, 0, 15, 0, 15],
                    },  # Beat 4 - Right
                    {
                        "name": "Gentle pitch forward",
                        "coords": [0, 0, 0, 0, -16, 0],
                    },  # Beat 5 - Center
                    {
                        "name": "Gentle tilt right",
                        "coords": [0, 0, 0, 15, 0, 12],
                    },  # Beat 6 - Right
                    {
                        "name": "Light roll left",
                        "coords": [0, 0, 0, -22, 0, -8],
                    },  # Beat 7 - Left
                    {
                        "name": "Light roll right",
                        "coords": [0, 0, 0, 22, 0, 8],
                    },  # Beat 8 - Right
                ],
                # Sequence C: Contemplative moves - rotation only
                [
                    {
                        "name": "Thoughtful nod",
                        "coords": [0, 0, 0, 0, -10, 0],
                    },  # Beat 1 - Center
                    {
                        "name": "Curious tilt left",
                        "coords": [0, 0, 0, -8, 0, -12],
                    },  # Beat 2 - Left
                    {
                        "name": "Ponder left turn",
                        "coords": [0, 0, 0, -6, 0, -16],
                    },  # Beat 3 - Left
                    {
                        "name": "Ponder right turn",
                        "coords": [0, 0, 0, 6, 0, 16],
                    },  # Beat 4 - Right
                    {
                        "name": "Meditative tilt right",
                        "coords": [0, 0, 0, 12, -2, 10],
                    },  # Beat 5 - Right
                    {
                        "name": "Peaceful center",
                        "coords": [0, 0, 0, 0, 0, 0],
                    },  # Beat 6 - Center
                    {
                        "name": "Gentle roll left",
                        "coords": [0, 0, 0, -10, 2, -8],
                    },  # Beat 7 - Left
                    {
                        "name": "Gentle roll right",
                        "coords": [0, 0, 0, 10, 2, 8],
                    },  # Beat 8 - Right
                ],
                # Sequence G: Subtle rotations - rotation only
                [
                    {
                        "name": "Gentle turn left",
                        "coords": [0, 0, 0, -6, 0, -14],
                    },  # Beat 1
                    {
                        "name": "Gentle pitch forward",
                        "coords": [0, 0, 0, 0, -8, 0],
                    },  # Beat 2
                    {
                        "name": "Gentle turn right",
                        "coords": [0, 0, 0, 6, 0, 14],
                    },  # Beat 3
                    {
                        "name": "Gentle pitch back",
                        "coords": [0, 0, 0, 0, 6, 0],
                    },  # Beat 4
                    {
                        "name": "Subtle tilt left",
                        "coords": [0, 0, 0, -10, 4, -8],
                    },  # Beat 5
                    {
                        "name": "Subtle nod down",
                        "coords": [0, 0, 0, 0, -10, 0],
                    },  # Beat 6
                    {
                        "name": "Subtle tilt right",
                        "coords": [0, 0, 0, 10, 4, 8],
                    },  # Beat 7
                    {
                        "name": "Return to center",
                        "coords": [0, 0, 0, 0, 0, 0],
                    },  # Beat 8
                ],
                # Sequence H: Breathing pattern - rotation only
                [
                    {"name": "Inhale up", "coords": [0, 0, 0, 0, 10, 0]},  # Beat 1
                    {"name": "Hold gentle", "coords": [0, 0, 0, 0, 8, 0]},  # Beat 2
                    {"name": "Exhale down", "coords": [0, 0, 0, 0, -8, 0]},  # Beat 3
                    {"name": "Rest center", "coords": [0, 0, 0, 0, 0, 0]},  # Beat 4
                    {"name": "Inhale left", "coords": [0, 0, 0, -12, 6, -10]},  # Beat 5
                    {"name": "Hold left", "coords": [0, 0, 0, -8, 8, -12]},  # Beat 6
                    {"name": "Exhale right", "coords": [0, 0, 0, 12, 6, 10]},  # Beat 7
                    {"name": "Rest right", "coords": [0, 0, 0, 8, 0, 12]},  # Beat 8
                ],
                # Sequence I: Gentle rotational circles - rotation only
                [
                    {
                        "name": "Circle start top",
                        "coords": [0, 0, 0, 0, 18, 0],
                    },  # Beat 1
                    {"name": "Circle right", "coords": [0, 0, 0, 8, 16, 10]},  # Beat 2
                    {"name": "Circle bottom", "coords": [0, 0, 0, 0, -18, 0]},  # Beat 3
                    {"name": "Circle left", "coords": [0, 0, 0, -8, 16, -10]},  # Beat 4
                    {
                        "name": "Small circle top",
                        "coords": [0, 0, 0, 0, 16, 0],
                    },  # Beat 5
                    {
                        "name": "Small circle right",
                        "coords": [0, 0, 0, 6, 14, 8],
                    },  # Beat 6
                    {
                        "name": "Small circle bottom",
                        "coords": [0, 0, 0, 0, -16, 0],
                    },  # Beat 7
                    {
                        "name": "Small circle left",
                        "coords": [0, 0, 0, -6, 14, -8],
                    },  # Beat 8
                ],
                # Sequence J: Micro movements - rotation only
                [
                    {
                        "name": "Micro turn left",
                        "coords": [0, 0, 0, -14, 0, -8],
                    },  # Beat 1
                    {
                        "name": "Micro turn right",
                        "coords": [0, 0, 0, 14, 0, 8],
                    },  # Beat 2
                    {"name": "Micro pitch up", "coords": [0, 0, 0, 0, 16, 0]},  # Beat 3
                    {
                        "name": "Micro pitch down",
                        "coords": [0, 0, 0, 0, -16, 0],
                    },  # Beat 4
                    {
                        "name": "Micro pitch forward",
                        "coords": [0, 0, 0, 0, -15, 0],
                    },  # Beat 5
                    {
                        "name": "Micro pitch back",
                        "coords": [0, 0, 0, 0, 15, 0],
                    },  # Beat 6
                    {
                        "name": "Micro tilt left",
                        "coords": [0, 0, 0, -8, 13, -6],
                    },  # Beat 7
                    {
                        "name": "Micro tilt right",
                        "coords": [0, 0, 0, 8, 13, 6],
                    },  # Beat 8
                ],
            ],
        }

        # Current sequence tracking
        self.current_sequence_type = None
        self.current_sequence_index = 0  # Which sequence variation (0, 1, or 2)
        self.sequence_position = 0  # Position within 8-beat sequence (0-7)
        self.sequence_energy_level = "medium_energy"

        # Sequence repetition tracking
        self.sequence_repetition_count = (
            0  # How many times current sequence has been completed
        )
        self.target_repetitions = (
            1  # Target repetitions for current sequence (only once)
        )

        # Global beat counter for body_yaw alternating pattern
        self.global_beat_count = 0
        # Track current body yaw position to hold on half-beats
        self.current_body_yaw = 0.0

    def generate_move(
        self,
        energy_level: float,
        beat_strength: float,
        onset_strength: float,
        tempo_change: bool,
        instrument_features: Dict[str, Any] = None,
        is_main_beat: bool = True,
        energy_category: str = None,
    ) -> Dict[str, Any]:
        """Generate head movement based on audio features using 8-beat sequences.

        Args:
            energy_level: Energy level from 0-1 (for backwards compatibility)
            beat_strength: Beat strength from 0-1
            onset_strength: Onset strength from 0-1
            tempo_change: Whether there's a tempo change
            instrument_features: Dictionary with instrument-specific features
            energy_category: Direct energy category ('low_energy', 'medium_energy', 'high_energy')
                           If provided, this takes precedence over numeric energy_level

        Returns:
            Dictionary with head movement information including XYZ RPY coordinates
        """
        # Use energy_category directly if provided, otherwise convert from numeric energy_level
        if energy_category:
            target_energy = energy_category
        else:
            # Fallback to old logic for backwards compatibility
            if energy_level > 0.67:
                target_energy = "high_energy"
            elif energy_level > 0.33:
                target_energy = "medium_energy"
            else:
                target_energy = "low_energy"

        # Start new 8-beat sequence if needed or energy level changed significantly
        # Only restart if energy change is large or sequence is complete
        energy_change_threshold = 0.4  # Require significant energy change to restart
        current_energy_val = {
            "low_energy": 0.16,
            "medium_energy": 0.5,
            "high_energy": 0.83,
        }[target_energy]
        prev_energy_val = {
            "low_energy": 0.16,
            "medium_energy": 0.5,
            "high_energy": 0.83,
        }.get(self.sequence_energy_level, 0.5)

        # Check if we should change sequence (now based on repetitions)
        # Check if we just completed a full 8-beat sequence (position wrapped from 7 to 0)
        sequence_completed = (
            self.sequence_position == 7
        )  # We're at the last beat of sequence

        should_change_sequence = (
            self.current_sequence_type is None
            or (
                sequence_completed
                and self.sequence_repetition_count >= self.target_repetitions
            )
            or abs(current_energy_val - prev_energy_val) > energy_change_threshold
        )

        if should_change_sequence:
            self.current_sequence_type = target_energy
            self.sequence_energy_level = target_energy
            self.sequence_position = 0

            # Reset repetition tracking for new sequence
            self.sequence_repetition_count = 0
            self.target_repetitions = 1  # New target for this sequence (only once)

            # Randomly select a sequence variation when starting new sequence
            available_sequences = len(self.eight_beat_sequences[target_energy])
            self.current_sequence_index = random.randint(0, available_sequences - 1)

        # Get movement from current position in current sequence variation
        current_sequence = self.eight_beat_sequences[self.current_sequence_type][
            self.current_sequence_index
        ]

        movement = current_sequence[self.sequence_position]

        # Get base coordinates and make movements sharper
        raw_coords = movement["coords"]
        sharpness_multiplier = 1.5  # Make all movements 50% sharper
        sharp_coords = [coord * sharpness_multiplier for coord in raw_coords]

        # Apply intensity modifiers based on beat strength and onset
        intensity = 1.0
        if beat_strength > 0.8:
            intensity *= 1.3  # Extra intensity for strong beats
        if onset_strength > 0.7:
            intensity *= 1.2  # Extra snap for strong onsets

        # Apply intensity to coordinates
        final_coords = [coord * intensity for coord in sharp_coords]

        # Apply instrument-specific modifications for head movements (excluding drum beats for yaw)
        if instrument_features:
            final_coords = self.apply_instrument_modifications(
                final_coords, instrument_features
            )

        # Calculate body_yaw based on drum beats (only for main beats)
        # Pass current sequence energy level to control swing intensity
        body_yaw = self.calculate_body_yaw(
            instrument_features,
            is_main_beat=is_main_beat,
            energy_category=self.current_sequence_type,
        )

        # Clamp extreme values to keep movements realistic
        final_coords = [
            max(-60, min(60, final_coords[0])),  # x position: -60 to 60 cm
            max(-60, min(60, final_coords[1])),  # y position: -60 to 60 cm
            max(-30, min(30, final_coords[2])),  # z position: -30 to 30 cm
            max(-90, min(90, final_coords[3])),  # roll: -90 to 90 degrees
            max(-90, min(90, final_coords[4])),  # pitch: -90 to 90 degrees
            max(-180, min(180, final_coords[5])),  # yaw: -180 to 180 degrees
        ]

        # Clamp body_yaw
        body_yaw = max(-180, min(180, body_yaw))

        # Advance sequence position and check for sequence completion
        old_position = self.sequence_position
        self.sequence_position = (self.sequence_position + 1) % 8

        # If we just wrapped from 7 to 0, we completed a sequence
        if old_position == 7 and self.sequence_position == 0:
            self.sequence_repetition_count += 1

        # Create movement data
        sequence_labels = {
            "high_energy": [
                "A-LeftRight",
                "B-UpDown",
                "C-ForwardBack",
                "D-SideVertical",
                "E-Circular",
                "F-ForwardBack",
                "G-Diagonal",
                "H-RapidFire",
                "I-Spiral",
                "J-SharpAngles",
                "K-Translation",
            ],
            "medium_energy": [
                "A-NodUpDown",
                "B-TurnLeftRight",
                "C-LeanForwardBack",
                "D-Classic",
                "E-Rhythmic",
                "F-Flow",
                "G-Waves",
                "H-Emphasis",
                "I-Figure8",
                "J-Pulse",
                "K-Translation",
            ],
            "low_energy": [
                "A-GentleNod",
                "B-SoftTurn",
                "C-LightTilt",
                "D-Gentle",
                "E-Explore",
                "F-Contemplative",
                "G-Drift",
                "H-Breathing",
                "I-Circles",
                "J-Micro",
            ],
        }

        sequence_label = sequence_labels[self.current_sequence_type][
            self.current_sequence_index
        ]

        move_data = {
            "energy_level": energy_level,
            "beat_strength": beat_strength,
            "onset_strength": onset_strength,
            "tempo_change": tempo_change,
            "sequence_position": self.sequence_position,
            "sequence_type": self.current_sequence_type,
            "sequence_variation": self.current_sequence_index,
            "sequence_repetition": self.sequence_repetition_count,
            "target_repetitions": self.target_repetitions,
            "body_yaw": body_yaw,
            "head_movements": [
                {
                    "name": f"{movement['name']} ({sequence_label} Rep{self.sequence_repetition_count + 1}/{self.target_repetitions} Beat{self.sequence_position + 1}/8)",
                    "coords": final_coords,
                }
            ],
        }

        return move_data

    def calculate_body_yaw(
        self,
        instrument_features: Dict[str, Any],
        is_main_beat: bool = True,
        energy_category: str = "medium_energy",
    ) -> float:
        """Calculate body yaw rotation that swings side-to-side with beat rhythm.

        Args:
            instrument_features: Dictionary with instrument-specific features
            is_main_beat: Whether this is a main beat (True) or half-beat (False)
            energy_category: Current energy category ('low_energy', 'medium_energy', 'high_energy')

        Returns:
            Body yaw rotation in degrees
        """
        # LOW ENERGY: Smooth left-to-right swing over 2 beats
        if energy_category == "low_energy":
            # Increment beat count for tracking
            if is_main_beat:
                self.global_beat_count += 1

            # Create a smooth swing from left to right over 2 beats (4 half-beats)
            # Beat cycle: 0->left, 0.5->center-left, 1->center, 1.5->center-right, 2->right, then repeat
            total_half_beats = self.global_beat_count * 2 + (0 if is_main_beat else 1)
            swing_cycle_position = (total_half_beats % 4) / 4.0  # 0.0 to 1.0 over 2 beats

            # Create smooth left-to-right swing: -15° to +15° over 2 beats
            max_swing = 15.0
            # Use sine wave for smooth movement: starts at left (-15°), goes to right (+15°)
            import math
            body_yaw = -max_swing * math.cos(swing_cycle_position * math.pi)

            self.current_body_yaw = body_yaw
            return body_yaw

        # MEDIUM ENERGY: Move back to center on half beats, alternate on main beats
        if energy_category == "medium_energy":
            if not is_main_beat:
                # Half-beat: move back to center
                body_yaw = 0.0
                self.current_body_yaw = body_yaw
                return body_yaw
            else:
                # Main beat: normal alternating swing pattern
                base_swing_amount = 20
                kick_strength = instrument_features.get("kick_strength", 0.0)
                snare_strength = instrument_features.get("snare_strength", 0.0)
                hihat_strength = instrument_features.get("hihat_strength", 0.0)
                drum_intensity = max(kick_strength, snare_strength, hihat_strength)
                intensity_multiplier = 0.4 + (drum_intensity * 0.6)
                swing_amount = base_swing_amount * intensity_multiplier

                body_yaw = (
                    swing_amount if (self.global_beat_count % 2) == 0 else -swing_amount
                )
                self.current_body_yaw = body_yaw
                self.global_beat_count += 1
                return body_yaw

        # HIGH ENERGY: Keep original logic (hold position on half-beats, swing on main beats)
        if not is_main_beat:
            return self.current_body_yaw

        # High energy main beat logic
        base_swing_amount = 30  # Higher swing for high energy

        # Calculate overall drum intensity to scale the swing
        kick_strength = instrument_features.get("kick_strength", 0.0)
        snare_strength = instrument_features.get("snare_strength", 0.0)
        hihat_strength = instrument_features.get("hihat_strength", 0.0)
        drum_intensity = max(kick_strength, snare_strength, hihat_strength)

        # More aggressive scaling for high energy
        intensity_multiplier = 0.6 + (drum_intensity * 0.4)
        swing_amount = base_swing_amount * intensity_multiplier

        # Determine direction: even beats = right (+), odd beats = left (-)
        body_yaw = swing_amount if (self.global_beat_count % 2) == 0 else -swing_amount

        # Store the new position for half-beats to hold
        self.current_body_yaw = body_yaw
        self.global_beat_count += 1

        return body_yaw

    def apply_instrument_modifications(
        self, coords: List[float], instrument_features: Dict[str, Any]
    ) -> List[float]:
        """Apply instrument-specific modifications to movement coordinates.

        Note: Drum effects are now handled by body_yaw, not head movements.

        Args:
            coords: [x, y, z, roll, pitch, yaw] coordinates
            instrument_features: Dictionary with instrument-specific features

        Returns:
            Modified coordinates with instrument-specific adjustments
        """
        x, y, z, roll, pitch, yaw = coords

        # VOCAL-BASED MODIFICATIONS
        vocal_active = instrument_features.get("vocal_active", False)
        vocal_energy = instrument_features.get("vocal_energy", 0.0)
        vocal_pitch = instrument_features.get("vocal_pitch", 0.0)

        if vocal_active and vocal_energy > 0.2:
            # Vocals affect yaw (head turning) and pitch (up/down nods)
            if vocal_pitch > 200:  # Higher vocal pitch
                pitch += 15 * min(vocal_energy, 1.0)  # Look up for high notes
                yaw += 10 * min(vocal_energy, 1.0)  # Slight turn
            elif vocal_pitch > 100:  # Mid vocal pitch
                yaw += 20 * min(vocal_energy, 1.0)  # Turn more for mid notes
            # Low vocal pitch keeps movement more centered

        # BASS/HARMONIC INSTRUMENT MODIFICATIONS
        bass_energy = instrument_features.get("bass_energy", 0.0)
        mid_energy = instrument_features.get("mid_energy", 0.0)

        # Bass affects position (forward/back movement)
        if bass_energy > 0.3:
            y += 4 * min(bass_energy, 1.0)  # Forward movement for bass

        # Mid-range instruments (guitar, piano) affect pitch
        if mid_energy > 0.3:
            pitch += 12 * min(mid_energy, 1.0)  # Up movement for melodic instruments

        return [x, y, z, roll, pitch, yaw]


class AudioAnalyzer:
    """Analyze audio for key moments and generate dance moves."""

    def __init__(self, output_dir: str = "output"):
        """Initialize the audio analyzer.

        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.dance_generator = DanceMoveGenerator()

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video file.

        Args:
            video_path: Path to video file

        Returns:
            Path to extracted audio file or None if failed
        """
        try:
            import subprocess

            video_name = Path(video_path).stem
            audio_path = self.output_dir / f"{video_name}_audio.wav"

            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "22050",
                "-ac",
                "1",
                "-y",
                str(audio_path),
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            return str(audio_path)

        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def separate_instruments(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate audio into different instrument components.

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Dictionary with separated instrument tracks
        """
        print("Separating audio into instrument tracks...")

        # Use librosa's harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Further separate percussive into different drum components using frequency analysis
        # Create different frequency filters for drum components

        # Low-pass for kick drum (typically 20-120 Hz dominant)
        kick_filtered = librosa.effects.preemphasis(y_percussive, coef=0.0)
        S_kick = librosa.stft(kick_filtered)
        S_kick_filtered = S_kick.copy()
        freqs = librosa.fft_frequencies(sr=sr)
        # Keep only low frequencies for kick
        freq_mask = (freqs >= 20) & (freqs <= 120)
        S_kick_filtered[~freq_mask] *= 0.1  # Heavily attenuate other frequencies
        y_kick = librosa.istft(S_kick_filtered)

        # Band-pass for snare (typically 150-300 Hz fundamental, 2-5kHz for crack)
        S_snare = librosa.stft(y_percussive)
        S_snare_filtered = S_snare.copy()
        snare_mask1 = (freqs >= 150) & (freqs <= 300)
        snare_mask2 = (freqs >= 2000) & (freqs <= 5000)
        freq_mask = snare_mask1 | snare_mask2
        S_snare_filtered[~freq_mask] *= 0.1
        y_snare = librosa.istft(S_snare_filtered)

        # High-pass for hi-hat/cymbals (typically 5kHz+)
        S_hihat = librosa.stft(y_percussive)
        S_hihat_filtered = S_hihat.copy()
        hihat_mask = freqs >= 5000
        S_hihat_filtered[~hihat_mask] *= 0.1
        y_hihat = librosa.istft(S_hihat_filtered)

        # Vocal extraction using spectral subtraction approach
        # Vocals are typically in the center channel and in 80Hz-1kHz range for fundamentals
        S_harmonic = librosa.stft(y_harmonic)

        # Simple vocal isolation: focus on harmonic content in vocal frequency range
        vocal_mask = (freqs >= 80) & (freqs <= 4000)
        S_vocal = S_harmonic.copy()
        S_vocal[~vocal_mask] *= 0.3  # Reduce non-vocal frequencies
        y_vocal = librosa.istft(S_vocal)

        # Other instruments (harmonic content minus estimated vocals)
        # Ensure arrays have same length for subtraction
        min_length = min(len(y_harmonic), len(y_vocal))
        y_other = y_harmonic[:min_length] - (
            y_vocal[:min_length] * 0.5
        )  # Subtract estimated vocals with some overlap

        return {
            "full": y,
            "harmonic": y_harmonic,
            "percussive": y_percussive,
            "kick": y_kick,
            "snare": y_snare,
            "hihat": y_hihat,
            "vocals": y_vocal,
            "other_instruments": y_other,
        }

    def analyze_drum_patterns(
        self, drum_tracks: Dict[str, np.ndarray], sr: int
    ) -> Dict[str, Any]:
        """Analyze drum patterns for each drum component.

        Args:
            drum_tracks: Dictionary with separated drum tracks
            sr: Sample rate

        Returns:
            Dictionary with drum pattern analysis
        """
        drum_analysis = {}

        for drum_type in ["kick", "snare", "hihat"]:
            if drum_type in drum_tracks:
                y_drum = drum_tracks[drum_type]

                # Detect onsets for this specific drum
                onset_frames = librosa.onset.onset_detect(
                    y=y_drum,
                    sr=sr,
                    pre_max=20,
                    post_max=20,
                    pre_avg=100,
                    post_avg=100,
                    delta=0.2,
                    wait=10,
                )
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)

                # Calculate onset strength over time
                onset_strength = librosa.onset.onset_strength(y=y_drum, sr=sr)

                # RMS energy for this drum component
                rms = librosa.feature.rms(y=y_drum)[0]

                drum_analysis[drum_type] = {
                    "onsets": onset_times.tolist(),
                    "onset_strength": onset_strength.tolist(),
                    "rms_energy": rms.tolist(),
                    "hit_count": len(onset_times),
                }

        return drum_analysis

    def analyze_vocal_features(
        self, vocal_track: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """Analyze vocal-specific features.

        Args:
            vocal_track: Separated vocal audio
            sr: Sample rate

        Returns:
            Dictionary with vocal features
        """
        # Pitch detection for vocals
        pitches, magnitudes = librosa.piptrack(y=vocal_track, sr=sr, threshold=0.1)

        # Extract fundamental frequency over time
        pitch_track = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
            pitch_track.append(pitch)

        # Vocal activity detection (where voice is present)
        vocal_rms = librosa.feature.rms(y=vocal_track)[0]
        vocal_threshold = np.percentile(vocal_rms, 60)  # Adaptive threshold
        vocal_activity = vocal_rms > vocal_threshold

        # Formant-like features using MFCCs
        mfcc_vocal = librosa.feature.mfcc(y=vocal_track, sr=sr, n_mfcc=13)

        return {
            "pitch_track": pitch_track,
            "vocal_activity": vocal_activity.tolist(),
            "vocal_rms": vocal_rms.tolist(),
            "mfcc_vocal": mfcc_vocal.tolist(),
            "vocal_presence_ratio": float(np.sum(vocal_activity) / len(vocal_activity)),
        }

    def analyze_harmonic_instruments(
        self, harmonic_track: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """Analyze harmonic instruments like bass, guitar, piano, strings.

        Args:
            harmonic_track: Separated harmonic audio
            sr: Sample rate

        Returns:
            Dictionary with harmonic instrument features
        """
        # Chroma features for harmonic analysis
        chroma = librosa.feature.chroma_stft(y=harmonic_track, sr=sr)

        # Spectral contrast for instrument texture
        spectral_contrast = librosa.feature.spectral_contrast(y=harmonic_track, sr=sr)

        # Low-frequency emphasis for bass detection
        y_bass = librosa.effects.preemphasis(harmonic_track, coef=0.0)
        S_bass = librosa.stft(y_bass)
        freqs = librosa.fft_frequencies(sr=sr)
        bass_mask = (freqs >= 40) & (freqs <= 250)  # Bass frequency range
        S_bass_filtered = S_bass.copy()
        S_bass_filtered[~bass_mask] *= 0.1
        y_bass_filtered = librosa.istft(S_bass_filtered)
        bass_energy = librosa.feature.rms(y=y_bass_filtered)[0]

        # Mid-frequency for guitar/piano (250Hz - 4kHz)
        mid_mask = (freqs >= 250) & (freqs <= 4000)
        S_mid = S_bass.copy()
        S_mid[~mid_mask] *= 0.1
        y_mid_filtered = librosa.istft(S_mid)
        mid_energy = librosa.feature.rms(y=y_mid_filtered)[0]

        # Harmonic content ratio
        harmonic_rms = librosa.feature.rms(y=harmonic_track)[0]
        harmonic_content_ratio = np.mean(harmonic_rms > np.percentile(harmonic_rms, 30))

        return {
            "chroma": chroma.tolist(),
            "spectral_contrast": spectral_contrast.tolist(),
            "bass_energy": bass_energy.tolist(),
            "mid_energy": mid_energy.tolist(),
            "harmonic_content_ratio": float(harmonic_content_ratio),
            "harmonic_rms": harmonic_rms.tolist(),
        }

    def analyze_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio features for dance generation with instrument separation.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio features including instrument separation
        """
        print(f"Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path)

        # Separate instruments first
        separated_tracks = self.separate_instruments(y, sr)

        # Analyze drum patterns
        drum_analysis = self.analyze_drum_patterns(separated_tracks, sr)

        # Analyze vocal features
        vocal_analysis = self.analyze_vocal_features(separated_tracks["vocals"], sr)

        # Basic audio features (using full track)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Extend beats to cover full song duration if detection stopped early
        duration = len(y) / sr
        if (
            len(beat_times) > 1 and beat_times[-1] < duration - 1.0
        ):  # If more than 1 second missing
            print(
                f"Beat detection stopped at {beat_times[-1]:.1f}s, extending to {duration:.1f}s"
            )

            # Calculate average beat interval from detected beats
            beat_intervals = np.diff(beat_times[-10:])  # Use last 10 intervals
            avg_interval = np.mean(beat_intervals)

            # Extend beats to cover remaining duration
            extended_beats = []
            current_time = beat_times[-1]
            while current_time + avg_interval < duration:
                current_time += avg_interval
                extended_beats.append(current_time)

            beat_times = np.concatenate([beat_times, extended_beats])
            print(f"Extended {len(extended_beats)} beats to cover full song")

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # RMS energy for intensity
        rms = librosa.feature.rms(y=y)[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # Analyze harmonic instruments (bass, guitar, piano, etc.)
        harmonic_analysis = self.analyze_harmonic_instruments(
            separated_tracks["other_instruments"], sr
        )

        return {
            "audio_path": audio_path,
            "sample_rate": sr,
            "duration": len(y) / sr,
            "tempo": float(tempo),
            "beats": beat_times.tolist(),
            "onsets": onset_times.tolist(),
            "spectral_centroids": spectral_centroids.tolist(),
            "spectral_rolloff": spectral_rolloff.tolist(),
            "mfcc": mfcc.tolist(),
            "chroma": chroma.tolist(),
            "rms_energy": rms.tolist(),
            "zero_crossing_rate": zcr.tolist(),
            # Add audio data for chord analysis
            "audio_data": y,
            # New instrument-specific features
            "drums": drum_analysis,
            "vocals": vocal_analysis,
            "harmonic_instruments": harmonic_analysis,
            "separated_tracks_info": {
                "has_kick": "kick" in drum_analysis
                and drum_analysis["kick"]["hit_count"] > 0,
                "has_snare": "snare" in drum_analysis
                and drum_analysis["snare"]["hit_count"] > 0,
                "has_hihat": "hihat" in drum_analysis
                and drum_analysis["hihat"]["hit_count"] > 0,
                "has_vocals": vocal_analysis["vocal_presence_ratio"] > 0.1,
                "has_harmonic": harmonic_analysis["harmonic_content_ratio"] > 0.1,
            },
        }

    def calculate_multi_factor_energy(
        self,
        rms_energy: np.ndarray,
        spectral_centroids: np.ndarray,
        beats: np.ndarray,
        audio_features: Dict[str, Any],
    ) -> np.ndarray:
        """Calculate energy levels using multiple factors beyond just RMS amplitude.

        This method addresses music compression issues by considering:
        1. Dynamic range analysis
        2. Spectral content (brightness/darkness)
        3. Rhythmic complexity
        4. Instrument separation
        5. Adaptive percentile thresholds instead of fixed values

        Args:
            rms_energy: Raw RMS energy values
            spectral_centroids: Spectral centroid values
            beats: Beat timestamps
            audio_features: Full audio analysis data

        Returns:
            Energy scores from 0-1 with better distribution across energy levels
        """
        # Factor 1: RMS Energy with better normalization
        # Use percentiles instead of min/max to handle outliers better
        rms_p10 = np.percentile(rms_energy, 10)
        rms_p90 = np.percentile(rms_energy, 90)
        rms_normalized = np.clip((rms_energy - rms_p10) / (rms_p90 - rms_p10), 0, 1)

        # Factor 2: Spectral brightness (high freq content indicates energy)
        spectral_p10 = np.percentile(spectral_centroids, 10)
        spectral_p90 = np.percentile(spectral_centroids, 90)
        spectral_normalized = np.clip(
            (spectral_centroids - spectral_p10) / (spectral_p90 - spectral_p10), 0, 1
        )

        # Factor 3: Beat density/rhythm complexity
        # Calculate local beat density - more beats in short time = higher energy
        beat_density = np.zeros(len(rms_energy))
        window_size = 4.0  # 4 second window

        for i, frame_time in enumerate(
            np.linspace(0, audio_features["duration"], len(rms_energy))
        ):
            # Count beats within window around this time
            beats_in_window = np.sum(
                (beats >= frame_time - window_size / 2)
                & (beats <= frame_time + window_size / 2)
            )
            beat_density[i] = beats_in_window / window_size  # beats per second

        # Normalize beat density
        if np.max(beat_density) > np.min(beat_density):
            beat_density_normalized = (beat_density - np.min(beat_density)) / (
                np.max(beat_density) - np.min(beat_density)
            )
        else:
            beat_density_normalized = np.ones_like(beat_density) * 0.5

        # Factor 4: Onset strength - rapid changes indicate energy
        onsets = np.array(audio_features["onsets"])
        onset_density = np.zeros(len(rms_energy))

        for i, frame_time in enumerate(
            np.linspace(0, audio_features["duration"], len(rms_energy))
        ):
            # Count onsets within small window
            onsets_in_window = np.sum(
                (onsets >= frame_time - 1.0) & (onsets <= frame_time + 1.0)
            )
            onset_density[i] = onsets_in_window / 2.0  # onsets per second in 2s window

        # Normalize onset density
        if np.max(onset_density) > np.min(onset_density):
            onset_density_normalized = (onset_density - np.min(onset_density)) / (
                np.max(onset_density) - np.min(onset_density)
            )
        else:
            onset_density_normalized = np.ones_like(onset_density) * 0.5

        # Factor 5: Instrument complexity (if available)
        instrument_factor = np.ones(len(rms_energy)) * 0.5  # default neutral

        # Use drum intensity if available
        if "drums" in audio_features and "intensity" in audio_features["drums"]:
            drum_intensity = np.array(audio_features["drums"]["intensity"])
            if len(drum_intensity) > 0:
                # Resample to match energy length
                drum_resampled = np.interp(
                    np.linspace(0, 1, len(rms_energy)),
                    np.linspace(0, 1, len(drum_intensity)),
                    drum_intensity,
                )
                # Normalize drum intensity
                if np.max(drum_resampled) > np.min(drum_resampled):
                    instrument_factor = (drum_resampled - np.min(drum_resampled)) / (
                        np.max(drum_resampled) - np.min(drum_resampled)
                    )

        # Combine all factors with weights (optimized for better peak detection)
        # Weights sum to 1.0 for proper scaling
        energy_combined = (
            0.30 * rms_normalized  # RMS energy (reduced slightly)
            + 0.25 * spectral_normalized  # Spectral brightness (increased)
            + 0.25 * beat_density_normalized  # Rhythmic complexity (increased)
            + 0.15 * onset_density_normalized  # Event density (increased)
            + 0.05 * instrument_factor  # Instrument contribution (reduced)
        )

        # Apply smoothing to reduce jitter
        energy_smoothed = uniform_filter1d(energy_combined, size=5)

        # Final adaptive thresholding using percentiles of the song itself
        # This ensures each song uses its full dynamic range appropriately
        p25 = np.percentile(energy_smoothed, 25)  # Low energy threshold
        p75 = np.percentile(energy_smoothed, 75)  # High energy threshold

        # Create final energy levels with better distribution
        final_energy = np.zeros_like(energy_smoothed)

        # Map to 0-1 range with better distribution
        # Use more dynamic thresholds based on song's energy distribution
        p10 = np.percentile(energy_smoothed, 10)
        p30 = np.percentile(energy_smoothed, 30)
        p70 = np.percentile(energy_smoothed, 70)
        p90 = np.percentile(energy_smoothed, 90)

        # Create more balanced distribution: 30% low, 40% medium, 30% high
        for i, energy in enumerate(energy_smoothed):
            if energy <= p30:
                # Map lowest 30% to 0-0.33 range (low energy)
                final_energy[i] = (
                    0.33 * ((energy - p10) / (p30 - p10)) if (p30 - p10) > 0 else 0.16
                )
                final_energy[i] = max(0.0, min(0.33, final_energy[i]))
            elif energy <= p70:
                # Map middle 40% to 0.33-0.67 range (medium energy)
                final_energy[i] = (
                    0.33 + 0.34 * ((energy - p30) / (p70 - p30))
                    if (p70 - p30) > 0
                    else 0.5
                )
            else:
                # Map top 30% to 0.67-1.0 range (high energy)
                final_energy[i] = (
                    0.67 + 0.33 * ((energy - p70) / (p90 - p70))
                    if (p90 - p70) > 0
                    else 0.83
                )
                final_energy[i] = min(1.0, final_energy[i])

        return final_energy

    def detect_musical_structure(
        self, audio_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect musical structure with improved song section identification.

        This method identifies:
        1. Musical sections (intro, verse, chorus, bridge, outro)
        2. Repetitive patterns to distinguish verses from choruses
        3. Energy transitions based on musical context
        4. Structural changes using multiple audio features

        Args:
            audio_features: Full audio analysis data

        Returns:
            List of musical sections with assigned energy levels and section types
        """

        duration = audio_features["duration"]
        beats = np.array(audio_features["beats"])
        rms_energy = np.array(audio_features["rms_energy"])
        spectral_centroids = np.array(audio_features["spectral_centroids"])
        onsets = np.array(audio_features["onsets"])

        # Extract chord/harmonic information for pattern detection
        y = audio_features.get("audio_data")
        sr = audio_features.get("sample_rate", 22050)

        # Use larger segments for better structural analysis (32 beats ≈ 8 bars)
        segment_length = 32  # beats per segment for structural analysis
        overlap = 0.25  # 25% overlap for smoother transitions

        segments = []
        segment_features = []

        # Create segments with minimal overlap for clearer section boundaries
        step_size = int(segment_length * (1 - overlap))
        for i in range(0, len(beats), step_size):
            segment_start = i
            segment_end = min(i + segment_length, len(beats))

            if (
                segment_end - segment_start < segment_length // 3
            ):  # Skip very short segments
                break

            segment_beats = beats[segment_start:segment_end]
            if len(segment_beats) < 8:  # Need minimum beats for analysis
                continue

            segment_duration = segment_beats[-1] - segment_beats[0]
            segment_center_time = (segment_beats[0] + segment_beats[-1]) / 2

            # Enhanced feature extraction for better section identification
            time_ratio = segment_center_time / duration

            # RMS features (energy)
            rms_start_idx = max(0, int((segment_beats[0] / duration) * len(rms_energy)))
            rms_end_idx = min(
                len(rms_energy), int((segment_beats[-1] / duration) * len(rms_energy))
            )
            segment_rms = (
                rms_energy[rms_start_idx:rms_end_idx]
                if rms_end_idx > rms_start_idx
                else [0]
            )

            avg_rms = np.mean(segment_rms)
            rms_variance = np.var(segment_rms)
            rms_peak = np.max(segment_rms) if len(segment_rms) > 0 else 0

            # Spectral features (brightness/timbre)
            spectral_start_idx = max(
                0, int((segment_beats[0] / duration) * len(spectral_centroids))
            )
            spectral_end_idx = min(
                len(spectral_centroids),
                int((segment_beats[-1] / duration) * len(spectral_centroids)),
            )
            segment_spectral = (
                spectral_centroids[spectral_start_idx:spectral_end_idx]
                if spectral_end_idx > spectral_start_idx
                else [0]
            )

            avg_spectral = np.mean(segment_spectral)
            spectral_variance = np.var(segment_spectral)

            # Rhythmic features
            beat_intervals = np.diff(segment_beats) if len(segment_beats) > 1 else [1.0]
            tempo_stability = 1.0 / (np.std(beat_intervals) + 0.01)
            avg_tempo = (
                60.0 / np.mean(beat_intervals) if len(beat_intervals) > 0 else 120.0
            )

            # Onset density and patterns
            onsets_in_segment = np.sum(
                (onsets >= segment_beats[0]) & (onsets <= segment_beats[-1])
            )
            onset_density = (
                onsets_in_segment / segment_duration if segment_duration > 0 else 0
            )

            # Harmonic/Chord analysis for pattern detection
            chromagram = None
            chord_pattern = np.zeros(12)  # Default empty chromagram

            if y is not None and sr is not None:
                try:
                    # Extract audio segment for chord analysis
                    start_sample = int(segment_beats[0] * sr)
                    end_sample = int(segment_beats[-1] * sr)

                    if start_sample < len(y) and end_sample <= len(y):
                        audio_segment = y[start_sample:end_sample]

                        # Compute chromagram (pitch class distribution)
                        chromagram = librosa.feature.chroma_stft(
                            y=audio_segment, sr=sr, hop_length=512, n_fft=2048
                        )

                        # Average chord pattern over time
                        chord_pattern = np.mean(chromagram, axis=1)

                        # Normalize chord pattern
                        if np.sum(chord_pattern) > 0:
                            chord_pattern = chord_pattern / np.sum(chord_pattern)

                except Exception:
                    # Fallback if chord analysis fails
                    chord_pattern = np.zeros(12)

            # Harmonic stability (using spectral features as proxy)
            harmonic_stability = 1.0 / (spectral_variance + 0.01)

            # Chord complexity (entropy of pitch distribution)
            chord_complexity = -np.sum(chord_pattern * np.log(chord_pattern + 1e-10))

            # Create comprehensive feature vector for clustering including chord features
            feature_vector = (
                [
                    avg_rms,  # Energy level
                    rms_variance,  # Energy variation (chorus typically more consistent)
                    rms_peak,  # Peak energy moments
                    avg_spectral,  # Brightness
                    spectral_variance,  # Timbre variation
                    tempo_stability,  # Rhythmic consistency
                    avg_tempo,  # Tempo
                    onset_density,  # Event density
                    harmonic_stability,  # Harmonic consistency
                    chord_complexity,  # Chord complexity
                ]
                + chord_pattern.tolist()
            )  # Add 12 chord features (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)

            segment_info = {
                "start_time": segment_beats[0],
                "end_time": segment_beats[-1],
                "start_beat": segment_start,
                "end_beat": segment_end - 1,
                "features": feature_vector,
                "beat_count": len(segment_beats),
                "avg_rms": avg_rms,
                "avg_spectral": avg_spectral,
                "tempo_stability": tempo_stability,
                "onset_density": onset_density,
                "chord_pattern": chord_pattern,
                "chord_complexity": chord_complexity,
            }

            segments.append(segment_info)
            segment_features.append(feature_vector)

        # Advanced clustering with chord-based pattern detection
        if len(segment_features) > 3:
            # Calculate chord similarity matrix for pattern detection
            chord_patterns = [seg["chord_pattern"] for seg in segments]
            chord_similarity_matrix = np.zeros((len(segments), len(segments)))

            for i in range(len(segments)):
                for j in range(len(segments)):
                    # Calculate cosine similarity between chord patterns
                    chord_i = chord_patterns[i]
                    chord_j = chord_patterns[j]

                    # Cosine similarity
                    dot_product = np.dot(chord_i, chord_j)
                    norm_i = np.linalg.norm(chord_i)
                    norm_j = np.linalg.norm(chord_j)

                    if norm_i > 0 and norm_j > 0:
                        chord_similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                    else:
                        chord_similarity_matrix[i, j] = 0.0

            # Find repetitive patterns (chord similarity) - lowered threshold
            similarity_threshold = (
                0.5  # Lower threshold to catch more repetitive patterns
            )
            repetitive_groups = []
            used_segments = set()

            for i in range(len(segments)):
                if i in used_segments:
                    continue

                # Find all segments similar to this one
                similar_segments = []
                for j in range(len(segments)):
                    if chord_similarity_matrix[i, j] > similarity_threshold:
                        similar_segments.append(j)

                if (
                    len(similar_segments) > 1
                ):  # Only consider groups with 2+ similar segments
                    repetitive_groups.append(similar_segments)
                    used_segments.update(similar_segments)

            # Normalize features for clustering (giving more weight to chord features)
            scaler = StandardScaler()

            # Weight chord features more heavily for structure detection
            weighted_features = []
            for feature_vector in segment_features:
                # First 10 are audio features, last 12 are chord features
                audio_features = feature_vector[:10]
                chord_features = feature_vector[10:]

                # Give chord features 2x weight
                weighted_feature = audio_features + [x * 2.0 for x in chord_features]
                weighted_features.append(weighted_feature)

            normalized_features = scaler.fit_transform(weighted_features)

            # Use more clusters to capture repetitive patterns
            n_clusters = min(max(5, len(segments) // 2), 8)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_features)

            # Analyze cluster characteristics and assign section types
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_indices = [
                    i for i, label in enumerate(cluster_labels) if label == cluster_id
                ]

                if not cluster_indices:
                    continue

                cluster_segments = [segments[i] for i in cluster_indices]
                cluster_features = [segment_features[i] for i in cluster_indices]

                avg_features = np.mean(cluster_features, axis=0)
                (
                    avg_rms,
                    rms_variance,
                    rms_peak,
                    avg_spectral,
                    spectral_variance,
                    tempo_stability,
                    avg_tempo,
                    onset_density,
                    harmonic_stability,
                    chord_complexity,
                ) = avg_features[:10]

                # Calculate position in song (for intro/outro detection)
                first_segment_time = min(seg["start_time"] for seg in cluster_segments)
                last_segment_time = max(seg["end_time"] for seg in cluster_segments)
                relative_position = first_segment_time / duration
                section_spread = (last_segment_time - first_segment_time) / duration

                # Repetition frequency (how often this cluster appears)
                repetition_frequency = len(cluster_indices) / len(segments)

                # Check if this cluster is part of a repetitive group (chord-based pattern)
                is_repetitive_pattern = any(
                    len(set(cluster_indices) & set(group)) >= 2
                    for group in repetitive_groups
                )

                # Get chord similarity within this cluster
                cluster_chord_similarity = 0.0
                if len(cluster_indices) > 1:
                    similarities = []
                    for i in range(len(cluster_indices)):
                        for j in range(i + 1, len(cluster_indices)):
                            idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                            similarities.append(chord_similarity_matrix[idx_i, idx_j])
                    cluster_chord_similarity = (
                        np.mean(similarities) if similarities else 0.0
                    )

                # Enhanced section type classification using chord patterns
                section_type = "section"
                energy_level = "medium_energy"

                # Intro detection: low repetition + early position (remove energy constraint)
                if relative_position < 0.15 and repetition_frequency < 0.2:
                    section_type = "intro"
                    # Use multi-factor energy analysis for more accurate classification
                    # Consider RMS, spectral brightness, and peak energy
                    rms_percentile = np.percentile([f[0] for f in segment_features], 50)
                    spectral_percentile = np.percentile(
                        [f[3] for f in segment_features], 60
                    )
                    peak_percentile = np.percentile(
                        [f[2] for f in segment_features], 60
                    )

                    # Multi-factor energy score
                    energy_factors = 0
                    if avg_rms > rms_percentile * 0.8:  # Lower threshold for intro RMS
                        energy_factors += 1
                    if (
                        avg_spectral > spectral_percentile
                    ):  # Bright sound indicates energy
                        energy_factors += 1
                    if rms_peak > peak_percentile:  # High peaks indicate energy bursts
                        energy_factors += 1

                    # Classify based on multiple energy indicators
                    if energy_factors >= 2:
                        energy_level = "high_energy"
                    elif energy_factors >= 1:
                        energy_level = "medium_energy"
                    else:
                        energy_level = "low_energy"

                # Outro detection: low repetition + late position + varying energy
                elif relative_position > 0.8 and repetition_frequency < 0.25:
                    section_type = "outro"
                    energy_level = (
                        "medium_energy"
                        if avg_rms > np.percentile([f[0] for f in segment_features], 50)
                        else "low_energy"
                    )

                # Chorus detection (enhanced with chord patterns, more sensitive):
                # Lower thresholds to catch more choruses
                elif (
                    avg_rms > np.percentile([f[0] for f in segment_features], 60)
                    and (is_repetitive_pattern or repetition_frequency > 0.15)
                    and (cluster_chord_similarity > 0.4 or repetition_frequency > 0.2)
                    and tempo_stability
                    > np.percentile([f[5] for f in segment_features], 30)
                ):
                    section_type = "chorus"
                    energy_level = "high_energy"

                # Verse detection (enhanced with chord patterns):
                # Repetitive chord pattern + lower energy than chorus + good repetition
                elif (
                    is_repetitive_pattern
                    and repetition_frequency > 0.2
                    and cluster_chord_similarity > 0.4
                    and avg_rms < np.percentile([f[0] for f in segment_features], 65)
                ):  # Lower energy than chorus
                    section_type = "verse"
                    energy_level = "medium_energy"

                # Bridge detection: unique features + middle position + low chord similarity (different harmony)
                elif (
                    repetition_frequency < 0.2
                    and 0.3 < relative_position < 0.8
                    and (
                        cluster_chord_similarity < 0.5  # Different chord progression
                        or spectral_variance
                        > np.percentile([f[4] for f in segment_features], 70)
                        or avg_spectral
                        > np.percentile([f[3] for f in segment_features], 75)
                    )
                ):
                    section_type = "bridge"
                    energy_level = (
                        "high_energy"
                        if avg_rms > np.percentile([f[0] for f in segment_features], 65)
                        else "medium_energy"
                    )

                # Pre-chorus detection: moderate energy + chord progression similarity + before choruses
                elif (
                    np.percentile([f[0] for f in segment_features], 40)
                    < avg_rms
                    < np.percentile([f[0] for f in segment_features], 70)
                    and repetition_frequency > 0.15
                    and cluster_chord_similarity > 0.5
                    and tempo_stability
                    > np.percentile([f[5] for f in segment_features], 50)
                ):
                    section_type = "pre_chorus"
                    energy_level = "medium_energy"

                # Default fallback based on energy and repetition patterns
                else:
                    if is_repetitive_pattern and repetition_frequency > 0.2:
                        if avg_rms > np.percentile(
                            [f[0] for f in segment_features], 65
                        ):  # Lower threshold
                            section_type = "chorus"
                            energy_level = "high_energy"
                        else:
                            section_type = "verse"
                            energy_level = "medium_energy"
                    else:
                        # For non-repetitive sections, still check for high energy choruses
                        if (
                            avg_rms
                            > np.percentile([f[0] for f in segment_features], 70)
                            and relative_position > 0.2
                        ):
                            section_type = "chorus"
                            energy_level = "high_energy"
                        elif avg_rms > np.percentile(
                            [f[0] for f in segment_features], 35
                        ):
                            section_type = "section"
                            energy_level = "medium_energy"
                        else:
                            section_type = "section"
                            energy_level = "low_energy"

                cluster_analysis[cluster_id] = {
                    "section_type": section_type,
                    "energy_level": energy_level,
                    "avg_rms": avg_rms,
                    "repetition_frequency": repetition_frequency,
                    "relative_position": relative_position,
                }

            # Create musical sections with enhanced information
            musical_sections = []

            # First, calculate RMS for all segments to identify top energy sections
            segment_rms_values = []
            for i, segment in enumerate(segments):
                segment_rms_values.append((i, segment["avg_rms"]))

            # Sort by RMS energy (highest first)
            segment_rms_values.sort(key=lambda x: x[1], reverse=True)

            # Mark top 30% of high-energy segments as potential choruses
            top_energy_count = max(
                2, int(len(segments) * 0.30)
            )  # At least 2, up to 30%
            top_energy_indices = set(
                [idx for idx, _ in segment_rms_values[:top_energy_count]]
            )

            for i, segment in enumerate(segments):
                cluster_id = cluster_labels[i]
                analysis = cluster_analysis.get(
                    cluster_id,
                    {"section_type": "section", "energy_level": "medium_energy"},
                )

                # Override section type for high energy segments not in intro/outro
                segment_start_time = segment["start_time"]
                relative_pos = segment_start_time / duration

                if (
                    i in top_energy_indices
                    and relative_pos > 0.15
                    and relative_pos < 0.9  # Not intro or outro
                    and analysis["section_type"] not in ["intro", "outro"]
                ):
                    analysis["section_type"] = "chorus"
                    analysis["energy_level"] = "high_energy"

                section = {
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "start_beat": segment["start_beat"],
                    "end_beat": segment["end_beat"],
                    "energy_level": analysis["energy_level"],
                    "cluster_id": cluster_id,
                    "section_type": analysis["section_type"],
                    "beat_count": segment["beat_count"],
                    "avg_rms": segment["avg_rms"],
                    "onset_density": segment["onset_density"],
                }
                musical_sections.append(section)

        else:
            # Fallback for very short songs - simple energy assignment
            musical_sections = []
            for segment in segments:
                avg_rms = segment["features"][0]
                # Simple threshold-based assignment for short songs
                if avg_rms > 0.6:
                    energy_level = "high_energy"
                elif avg_rms > 0.3:
                    energy_level = "medium_energy"
                else:
                    energy_level = "low_energy"

                section = {
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "start_beat": segment["start_beat"],
                    "end_beat": segment["end_beat"],
                    "energy_level": energy_level,
                    "cluster_id": 0,
                    "section_type": "section_0",
                    "beat_count": segment["beat_count"],
                }
                musical_sections.append(section)

        return musical_sections

    def detect_key_moments(
        self, audio_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect key moments in the audio for dance generation.

        Args:
            audio_features: Audio features dictionary

        Returns:
            List of key moments with timestamps and characteristics
        """
        key_moments = []

        beats = np.array(audio_features["beats"])
        onsets = np.array(audio_features["onsets"])
        rms_energy = np.array(audio_features["rms_energy"])
        spectral_centroids = np.array(audio_features["spectral_centroids"])

        # Use musical structure detection instead of beat-by-beat energy analysis
        musical_sections = self.detect_musical_structure(audio_features)

        # Create a mapping of beat numbers to energy levels and section types based on musical sections
        beat_to_energy = {}
        beat_to_section = {}
        for section in musical_sections:
            start_beat = section["start_beat"]
            end_beat = section["end_beat"]
            energy_level = section["energy_level"]
            section_type = section["section_type"]

            # Assign energy level and section type to all beats in this section
            for beat_idx in range(start_beat, end_beat + 1):
                if beat_idx < len(beats):
                    beat_to_energy[beat_idx] = energy_level
                    beat_to_section[beat_idx] = section_type

        # Detect tempo changes
        tempo_changes = self.detect_tempo_changes(beats)

        # Use beat timestamps as the primary time grid instead of fixed intervals
        duration = audio_features["duration"]

        # Extract instrument-specific features for moments
        drum_features = audio_features.get("drums", {})
        vocal_features = audio_features.get("vocals", {})
        harmonic_features = audio_features.get("harmonic_instruments", {})

        # Process each beat timestamp
        for i, beat_time in enumerate(beats):
            # Beat strength is always 1.0 since we're processing actual beats
            beat_strength = 1.0

            # Calculate beat intensity based on surrounding beats
            if i > 0 and i < len(beats) - 1:
                # Check if this beat is stronger than neighbors (shorter intervals = stronger beat)
                prev_interval = beat_time - beats[i - 1]
                next_interval = beats[i + 1] - beat_time
                avg_interval = (prev_interval + next_interval) / 2
                # Stronger beats have shorter intervals around them
                beat_intensity = 1.0 / (
                    avg_interval + 0.1
                )  # Add small value to avoid division by zero
                beat_strength = min(beat_intensity / 2.0, 1.0)  # Normalize to 0-1 range

            # Find nearest onset
            if len(onsets) > 0:
                onset_distances = np.abs(onsets - beat_time)
                nearest_onset_idx = np.argmin(onset_distances)
                onset_strength = 1.0 / (1.0 + onset_distances[nearest_onset_idx])
            else:
                onset_strength = 0.0

            # Get energy level and section type from musical structure analysis
            energy_category = beat_to_energy.get(
                i, "medium_energy"
            )  # Default to medium_energy if not found
            section_type = beat_to_section.get(
                i, "section"
            )  # Default to 'section' if not found

            # Convert energy category to numeric value for compatibility with existing code
            if energy_category == "high_energy":
                energy_level = 0.8
            elif energy_category == "medium_energy":
                energy_level = 0.5
            else:  # low_energy
                energy_level = 0.2

            # Check for tempo change
            tempo_change = any(abs(tc - beat_time) < 1.0 for tc in tempo_changes)

            # Calculate instrument-specific features at this beat
            instrument_features = self.get_instrument_features_at_time(
                beat_time, duration, drum_features, vocal_features, harmonic_features
            )

            # Add every beat as a key moment (since we want moves on every beat)
            moment_data = {
                "timestamp": float(beat_time),
                "energy_level": float(energy_level),
                "energy_category": energy_category,  # Add categorical energy for dance move selection
                "section_type": section_type,  # Add section type for dance move selection
                "beat_strength": float(beat_strength),
                "onset_strength": float(onset_strength),
                "tempo_change": tempo_change,
                "spectral_brightness": float(
                    spectral_centroids[
                        min(
                            int((beat_time / duration) * len(spectral_centroids)),
                            len(spectral_centroids) - 1,
                        )
                    ]
                ),
                "beat_number": i + 1,  # Add beat number for reference
            }

            # Add instrument-specific features
            moment_data.update(instrument_features)
            key_moments.append(moment_data)

        return key_moments

    def get_instrument_features_at_time(
        self,
        timestamp: float,
        duration: float,
        drum_features: Dict,
        vocal_features: Dict,
        harmonic_features: Dict,
    ) -> Dict[str, Any]:
        """Get instrument-specific features at a given timestamp.

        Args:
            timestamp: Time in seconds
            duration: Total duration of audio
            drum_features: Drum analysis results
            vocal_features: Vocal analysis results
            harmonic_features: Harmonic instrument analysis results

        Returns:
            Dictionary with instrument features at this time
        """
        features = {}

        # Calculate frame index for time-series data
        def get_frame_idx(time_series_length: int) -> int:
            return min(
                int((timestamp / duration) * time_series_length), time_series_length - 1
            )

        # Drum features at this time
        for drum_type in ["kick", "snare", "hihat"]:
            if drum_type in drum_features:
                drum_data = drum_features[drum_type]

                # Check if there's a drum hit near this timestamp (within 0.1 seconds)
                onsets = np.array(drum_data["onsets"])
                if len(onsets) > 0:
                    nearest_onset_distance = np.min(np.abs(onsets - timestamp))
                    drum_hit = nearest_onset_distance < 0.1  # Within 100ms
                    drum_strength = (
                        1.0 / (1.0 + nearest_onset_distance) if drum_hit else 0.0
                    )
                else:
                    drum_hit = False
                    drum_strength = 0.0

                # Get RMS energy at this time
                rms_energy = drum_data["rms_energy"]
                rms_idx = get_frame_idx(len(rms_energy))
                drum_energy = rms_energy[rms_idx] if rms_idx < len(rms_energy) else 0.0

                features[f"{drum_type}_hit"] = drum_hit
                features[f"{drum_type}_strength"] = float(drum_strength)
                features[f"{drum_type}_energy"] = float(drum_energy)

        # Vocal features at this time
        if vocal_features:
            vocal_activity = vocal_features.get("vocal_activity", [])
            vocal_rms = vocal_features.get("vocal_rms", [])
            pitch_track = vocal_features.get("pitch_track", [])

            vocal_idx = get_frame_idx(len(vocal_activity)) if vocal_activity else 0
            features["vocal_active"] = (
                bool(vocal_activity[vocal_idx])
                if vocal_idx < len(vocal_activity)
                else False
            )
            features["vocal_energy"] = (
                float(vocal_rms[vocal_idx])
                if vocal_rms and vocal_idx < len(vocal_rms)
                else 0.0
            )
            features["vocal_pitch"] = (
                float(pitch_track[vocal_idx])
                if pitch_track and vocal_idx < len(pitch_track)
                else 0.0
            )

        # Harmonic instrument features at this time
        if harmonic_features:
            bass_energy = harmonic_features.get("bass_energy", [])
            mid_energy = harmonic_features.get("mid_energy", [])
            harmonic_rms = harmonic_features.get("harmonic_rms", [])

            harmonic_idx = get_frame_idx(len(bass_energy)) if bass_energy else 0
            features["bass_energy"] = (
                float(bass_energy[harmonic_idx])
                if harmonic_idx < len(bass_energy)
                else 0.0
            )
            features["mid_energy"] = (
                float(mid_energy[harmonic_idx])
                if mid_energy and harmonic_idx < len(mid_energy)
                else 0.0
            )
            features["harmonic_energy"] = (
                float(harmonic_rms[harmonic_idx])
                if harmonic_rms and harmonic_idx < len(harmonic_rms)
                else 0.0
            )

        return features

    def detect_tempo_changes(
        self, beats: np.ndarray, threshold: float = 0.2
    ) -> List[float]:
        """Detect tempo changes in the beat sequence.

        Args:
            beats: Array of beat timestamps
            threshold: Threshold for detecting tempo change

        Returns:
            List of timestamps where tempo changes occur
        """
        if len(beats) < 10:
            return []

        # Calculate inter-beat intervals
        intervals = np.diff(beats)

        # Smooth intervals to reduce noise
        from scipy import ndimage

        smoothed_intervals = ndimage.gaussian_filter1d(intervals, sigma=2)

        # Find significant changes
        tempo_changes = []
        for i in range(5, len(smoothed_intervals) - 5):
            before = np.mean(smoothed_intervals[i - 5 : i])
            after = np.mean(smoothed_intervals[i : i + 5])

            if abs(before - after) / before > threshold:
                tempo_changes.append(beats[i])

        return tempo_changes

    def generate_dance_sequence(
        self, key_moments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate dance sequence based on key moments.

        Args:
            key_moments: List of key moments with audio features

        Returns:
            List of dance moves with timestamps and interpolated frames
        """
        # Generate beat and half-beat dance moves
        beat_moves = []

        for i, moment in enumerate(key_moments):
            # Generate main beat movement with instrument features
            main_beat_move = self.dance_generator.generate_move(
                energy_level=moment["energy_level"],
                beat_strength=moment["beat_strength"],
                onset_strength=moment["onset_strength"],
                tempo_change=moment["tempo_change"],
                instrument_features=moment,
                energy_category=moment.get("energy_category"),
            )

            main_beat_move["timestamp"] = moment["timestamp"]
            main_beat_move["audio_features"] = moment
            main_beat_move["move_type"] = "main_beat"
            beat_moves.append(main_beat_move)

            # Generate half-beat movement if there's space before the next beat
            if i < len(key_moments) - 1:
                next_timestamp = key_moments[i + 1]["timestamp"]
                half_beat_time = (
                    moment["timestamp"] + (next_timestamp - moment["timestamp"]) / 2
                )

                # Create half-beat movement with same energy category but softer intensity
                # Keep the same energy level for consistency in CSV output
                half_beat_energy = moment[
                    "energy_level"
                ]  # Keep same energy level as main beat

                # Create modified instrument features for half-beat (reduced intensity)
                half_beat_instrument_features = {}
                for key, value in moment.items():
                    if key.endswith("_hit"):
                        half_beat_instrument_features[key] = (
                            False  # No hits on half-beats
                        )
                    elif key.endswith("_strength") or key.endswith("_energy"):
                        half_beat_instrument_features[key] = (
                            value * 0.3 if isinstance(value, (int, float)) else value
                        )
                    else:
                        half_beat_instrument_features[key] = value

                half_beat_move = self.dance_generator.generate_move(
                    energy_level=half_beat_energy,
                    beat_strength=moment["beat_strength"] * 0.4,  # Softer beat strength
                    onset_strength=moment["onset_strength"] * 0.3,
                    tempo_change=False,  # Half-beats don't usually have tempo changes
                    instrument_features=half_beat_instrument_features,
                    is_main_beat=False,  # This is a half-beat
                    energy_category=moment.get(
                        "energy_category"
                    ),  # Use same energy category as main beat
                )

                # Create audio features for half-beat (interpolated)
                half_beat_features = moment.copy()
                half_beat_features["energy_level"] = half_beat_energy
                half_beat_features["beat_strength"] = moment["beat_strength"] * 0.4
                half_beat_features["onset_strength"] = moment["onset_strength"] * 0.3
                half_beat_features["beat_number"] = f"{moment.get('beat_number', '')}.5"

                half_beat_move["timestamp"] = half_beat_time
                half_beat_move["audio_features"] = half_beat_features
                half_beat_move["move_type"] = "half_beat"
                beat_moves.append(half_beat_move)

        # Sort by timestamp to ensure proper order
        beat_moves.sort(key=lambda x: x["timestamp"])

        # Return direct beat moves without interpolation
        return beat_moves

    def analyze_video(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio file for features and generate dance moves.

        Args:
            audio_path: Path to audio file (can be video or audio file)

        Returns:
            Complete analysis results with dance moves
        """
        print(f"Analyzing audio for dance generation: {audio_path}")

        # If it's a video file, extract audio first
        if Path(audio_path).suffix.lower() in [".mp4", ".webm", ".mkv", ".avi"]:
            extracted_audio_path = self.extract_audio_from_video(audio_path)
            if not extracted_audio_path:
                return {"error": "Failed to extract audio from video"}
            audio_path = extracted_audio_path

        # Check for cached dance analysis first
        cached_analysis = self.load_cached_analysis(audio_path)
        if cached_analysis:
            print("Found cached dance analysis, skipping computation")
            return cached_analysis

        # Analyze audio features
        print("Extracting audio features...")
        audio_features = self.analyze_audio_features(audio_path)

        # Detect key moments
        print("Detecting key moments...")
        key_moments = self.detect_key_moments(audio_features)

        # Generate dance sequence
        print("Generating dance moves...")
        dance_sequence = self.generate_dance_sequence(key_moments)

        analysis_results = {
            "audio_path": audio_path,
            "audio_features": audio_features,
            "key_moments_count": len(key_moments),
            "dance_moves_count": len(dance_sequence),
            "dance_sequence": dance_sequence,
            "summary": {
                "duration": audio_features["duration"],
                "tempo": audio_features["tempo"],
                "total_beats": len(audio_features["beats"]),
                "total_onsets": len(audio_features["onsets"]),
                "key_moments": len(key_moments),
                "dance_moves_generated": len(dance_sequence),
            },
        }

        # Save analysis results
        self.save_analysis_results(analysis_results, audio_path)

        return analysis_results

    def save_analysis_results(self, results: Dict[str, Any], audio_path: str):
        """Save analysis results to JSON file.

        Args:
            results: Analysis results dictionary
            audio_path: Original audio path for naming
        """
        audio_name = Path(audio_path).stem
        output_path = self.output_dir / f"{audio_name}_dance_analysis.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print(f"Dance analysis results saved to: {output_path}")

    def load_cached_analysis(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Load cached dance analysis if it exists.

        Args:
            audio_path: Path to audio file

        Returns:
            Cached analysis results if found, None otherwise
        """
        audio_name = Path(audio_path).stem
        cache_path = self.output_dir / f"{audio_name}_dance_analysis.json"

        if cache_path.exists():
            try:
                print(f"Loading cached analysis from: {cache_path}")
                with open(cache_path, "r") as f:
                    cached_results = json.load(f)

                # Verify the cached analysis has the required fields
                required_fields = ["audio_path", "dance_sequence", "summary"]
                if all(field in cached_results for field in required_fields):
                    # Update audio_path in case the file was moved
                    cached_results["audio_path"] = audio_path
                    return cached_results
                else:
                    print("Cached analysis missing required fields, will recompute")
                    return None

            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading cached analysis: {e}, will recompute")
                return None

        return None

    def export_dance_csv(
        self, analysis_results: Dict[str, Any], audio_path: str
    ) -> str:
        """Export dance moves to CSV format.

        Args:
            analysis_results: Analysis results dictionary
            audio_path: Original audio path for naming

        Returns:
            Path to saved CSV file
        """
        import csv

        audio_name = Path(audio_path).stem
        csv_path = self.output_dir / f"{audio_name}_dance_moves.csv"

        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame_number",
                "timestamp",
                "move_type",
                "beat_number",
                "energy_level",
                "beat_strength",
                "sequence_type",
                "sequence_variation",
                "sequence_position",
                "sequence_repetition",
                "head_movement_name",
                "x_cm",
                "y_cm",
                "z_cm",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "body_yaw_deg",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for frame_number, frame in enumerate(analysis_results["dance_sequence"]):
                # Handle the new direct beat format (no interpolation)
                if frame.get("head_movements"):
                    coordinates = frame["head_movements"][0]["coords"]
                    movement_name = frame["head_movements"][0]["name"]
                else:
                    coordinates = [0, 0, 0, 0, 0, 0]
                    movement_name = "no movement"

                row = {
                    "frame_number": frame_number + 1,
                    "timestamp": frame["timestamp"],
                    "move_type": frame.get("move_type", "beat"),
                    "beat_number": frame.get("audio_features", {}).get(
                        "beat_number", ""
                    ),
                    "energy_level": frame.get("energy_level", 0),
                    "beat_strength": frame.get("beat_strength", 0),
                    "sequence_type": frame.get("sequence_type", ""),
                    "sequence_variation": frame.get("sequence_variation", ""),
                    "sequence_position": frame.get("sequence_position", ""),
                    "sequence_repetition": frame.get("sequence_repetition", ""),
                    "head_movement_name": movement_name,
                    "x_cm": coordinates[0],
                    "y_cm": coordinates[1],
                    "z_cm": coordinates[2],
                    "roll_deg": coordinates[3],
                    "pitch_deg": coordinates[4],
                    "yaw_deg": coordinates[5],
                    "body_yaw_deg": frame.get("body_yaw", 0),
                }
                writer.writerow(row)

        print(f"Dance moves CSV saved to: {csv_path}")
        return str(csv_path)
