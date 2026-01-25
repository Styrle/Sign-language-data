"""BSL Data Extraction - Hand landmark constants.

This module defines constants for MediaPipe Hand Landmarker,
including landmark names, connections, and finger groupings.

MediaPipe detects 21 landmarks per hand:
- 0: Wrist
- 1-4: Thumb (CMC, MCP, IP, TIP)
- 5-8: Index finger (MCP, PIP, DIP, TIP)
- 9-12: Middle finger (MCP, PIP, DIP, TIP)
- 13-16: Ring finger (MCP, PIP, DIP, TIP)
- 17-20: Pinky finger (MCP, PIP, DIP, TIP)
"""

# =============================================================================
# Landmark Names
# =============================================================================

LANDMARK_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)
"""Names of all 21 hand landmarks in index order."""

# Reverse mapping: name -> index
LANDMARK_INDICES: dict[str, int] = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}
"""Mapping from landmark name to index."""


# =============================================================================
# Landmark Connections
# =============================================================================

LANDMARK_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Thumb
    (0, 1),   # WRIST -> THUMB_CMC
    (1, 2),   # THUMB_CMC -> THUMB_MCP
    (2, 3),   # THUMB_MCP -> THUMB_IP
    (3, 4),   # THUMB_IP -> THUMB_TIP
    # Index finger
    (0, 5),   # WRIST -> INDEX_FINGER_MCP
    (5, 6),   # INDEX_FINGER_MCP -> INDEX_FINGER_PIP
    (6, 7),   # INDEX_FINGER_PIP -> INDEX_FINGER_DIP
    (7, 8),   # INDEX_FINGER_DIP -> INDEX_FINGER_TIP
    # Middle finger
    (0, 9),   # WRIST -> MIDDLE_FINGER_MCP
    (9, 10),  # MIDDLE_FINGER_MCP -> MIDDLE_FINGER_PIP
    (10, 11), # MIDDLE_FINGER_PIP -> MIDDLE_FINGER_DIP
    (11, 12), # MIDDLE_FINGER_DIP -> MIDDLE_FINGER_TIP
    # Ring finger
    (0, 13),  # WRIST -> RING_FINGER_MCP
    (13, 14), # RING_FINGER_MCP -> RING_FINGER_PIP
    (14, 15), # RING_FINGER_PIP -> RING_FINGER_DIP
    (15, 16), # RING_FINGER_DIP -> RING_FINGER_TIP
    # Pinky
    (0, 17),  # WRIST -> PINKY_MCP
    (17, 18), # PINKY_MCP -> PINKY_PIP
    (18, 19), # PINKY_PIP -> PINKY_DIP
    (19, 20), # PINKY_DIP -> PINKY_TIP
    # Palm connections (for visualization)
    (5, 9),   # INDEX_FINGER_MCP -> MIDDLE_FINGER_MCP
    (9, 13),  # MIDDLE_FINGER_MCP -> RING_FINGER_MCP
    (13, 17), # RING_FINGER_MCP -> PINKY_MCP
)
"""Connections between landmarks as (from_index, to_index) pairs."""


# =============================================================================
# Finger Indices
# =============================================================================

FINGER_INDICES: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
"""Mapping from finger name to landmark indices (excluding wrist)."""

FINGERTIP_INDICES: dict[str, int] = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
"""Mapping from finger name to fingertip landmark index."""

FINGER_BASE_INDICES: dict[str, int] = {
    "thumb": 1,   # CMC for thumb
    "index": 5,   # MCP for fingers
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}
"""Mapping from finger name to base joint landmark index."""


# =============================================================================
# Joint Groups (for angle calculation)
# =============================================================================

# Each tuple is (proximal, middle, distal) for angle calculation
THUMB_JOINTS: dict[str, tuple[int, int, int]] = {
    "cmc": (0, 1, 2),     # Wrist -> CMC -> MCP angle
    "mcp": (1, 2, 3),     # CMC -> MCP -> IP angle
    "ip": (2, 3, 4),      # MCP -> IP -> TIP angle
}
"""Thumb joint triplets for angle calculation."""

FINGER_JOINTS: dict[str, dict[str, tuple[int, int, int]]] = {
    "index": {
        "mcp": (0, 5, 6),    # Wrist -> MCP -> PIP angle
        "pip": (5, 6, 7),    # MCP -> PIP -> DIP angle
        "dip": (6, 7, 8),    # PIP -> DIP -> TIP angle
    },
    "middle": {
        "mcp": (0, 9, 10),
        "pip": (9, 10, 11),
        "dip": (10, 11, 12),
    },
    "ring": {
        "mcp": (0, 13, 14),
        "pip": (13, 14, 15),
        "dip": (14, 15, 16),
    },
    "pinky": {
        "mcp": (0, 17, 18),
        "pip": (17, 18, 19),
        "dip": (18, 19, 20),
    },
}
"""Finger joint triplets for angle calculation (excluding thumb)."""


# =============================================================================
# Utility Constants
# =============================================================================

NUM_LANDMARKS: int = 21
"""Total number of landmarks per hand."""

NUM_FINGERS: int = 5
"""Number of fingers (including thumb)."""

WRIST_INDEX: int = 0
"""Index of the wrist landmark."""

# Palm landmarks used for hand orientation
PALM_INDICES: tuple[int, ...] = (0, 5, 9, 13, 17)
"""Indices forming the palm base (wrist + finger MCPs)."""


# =============================================================================
# Individual Landmark Indices
# =============================================================================

# Wrist
WRIST: int = 0

# Thumb
THUMB_CMC: int = 1
THUMB_MCP: int = 2
THUMB_IP: int = 3
THUMB_TIP: int = 4

# Index finger
INDEX_FINGER_MCP: int = 5
INDEX_FINGER_PIP: int = 6
INDEX_FINGER_DIP: int = 7
INDEX_FINGER_TIP: int = 8

# Middle finger
MIDDLE_FINGER_MCP: int = 9
MIDDLE_FINGER_PIP: int = 10
MIDDLE_FINGER_DIP: int = 11
MIDDLE_FINGER_TIP: int = 12

# Ring finger
RING_FINGER_MCP: int = 13
RING_FINGER_PIP: int = 14
RING_FINGER_DIP: int = 15
RING_FINGER_TIP: int = 16

# Pinky
PINKY_MCP: int = 17
PINKY_PIP: int = 18
PINKY_DIP: int = 19
PINKY_TIP: int = 20
