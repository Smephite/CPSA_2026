"""Shared data types. Coordinates are frame pixels unless named *_m (metres)."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

# COCO-17 keypoint order (MoveNet)
KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
ARM_IDS = [5, 6, 7, 8, 9, 10]


class Level(IntEnum):
    NONE = 0
    WARN = 1
    STOP = 2


class NodeState(IntEnum):
    IDLE = 0      # no beacon: camera off, no inference
    DETECT = 1    # beacon: person detection only
    TRACK = 2     # robot and human closing: pose + rules


class Band(IntEnum):
    FAR = 0
    APPROACH = 1
    CLOSE = 2


@dataclass
class Detection:
    box: np.ndarray               # (4,) x1, y1, x2, y2
    score: float
    label: str = "person"


@dataclass
class TruthPerson:
    """Ground truth attached to synthetic frames (scenario) so the replay backend can 'detect' it."""
    name: str
    box: np.ndarray               # (4,)
    kp: np.ndarray                # (17, 3)


@dataclass
class Frame:
    image: np.ndarray             # BGR
    t: float                      # seconds (node clock)
    truth: Optional[List[TruthPerson]] = None


@dataclass
class BeaconState:
    present: bool
    robot_id: str = ""
    robot_type: str = ""
    reach_m: float = 0.8


@dataclass
class Danger:
    rule: str
    level_now: Level
    predicted: Level              # worst level reached within the look-ahead
    ttc_s: Optional[float]        # seconds until the predicted STOP (None if not predicted)
    human_id: int = -1


@dataclass
class Decision:
    level: Level
    rules: Tuple[str, ...] = ()
    since: float = 0.0
    dangers: List[Danger] = field(default_factory=list)
