"""Shared data types: the vocabulary the module boxes use to talk to each other.

    sensors         -> Frame, BeaconState, watts
    VIDEO_pipeline  -> Detection, (17, 3) keypoints, Track (VIDEO_pipeline/tracking.py)
    core            -> Danger, Decision (to actuators), Snapshot (to the dashboard)

Coordinates are frame pixels unless a name ends in _m (metres). Keypoints are (17, 3) arrays of (x, y, score)
in COCO-17 order (KEYPOINTS below).
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

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
    """What core decided this frame; the input of every actuator."""
    level: Level
    rules: Tuple[str, ...] = ()           # rules behind the current level
    since: float = 0.0                    # node time the level last changed
    changed: bool = False                 # level differs from the previous frame
    robot_id: str = ""                    # from the beacon ('' when no robot is announced)
    dangers: List[Danger] = field(default_factory=list)


@dataclass
class Snapshot:
    """Everything the dashboard needs to draw one frame. Produced by core.GuardianNode.step()."""
    t: float
    state: NodeState
    level: Level
    frame: Optional[Frame] = None         # None while IDLE (camera off)
    beacon: Optional[BeaconState] = None
    schedule: Any = None                  # core.scheduler.Schedule: band, models, rates
    tracks: List[Any] = field(default_factory=list)      # VIDEO_pipeline.tracking.Track
    rules: Tuple[str, ...] = ()
    dangers: List[Danger] = field(default_factory=list)
    future: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)   # track id -> (kp, box) at the horizon
    pair_human: Any = None                # the human nearest to the robot (Track)
    gap_m: Optional[float] = None         # robot <-> nearest human, floor distance
    body_gap_m: Optional[float] = None    # robot body hull <-> nearest human joint
    closing_mps: Optional[float] = None   # positive = the gap is shrinking
    m_per_px: Optional[float] = None      # at the robot's depth
    cascade: Dict[str, Any] = field(default_factory=dict)  # "det" / "pose" -> cascade (statistics)
    power_w: Optional[float] = None
    times: Dict[str, float] = field(default_factory=dict)  # stage -> ms this frame
    fps: float = 0.0
    caption: str = ""
