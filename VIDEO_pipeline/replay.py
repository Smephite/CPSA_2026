"""Replay backend: 'detects' and 'estimates poses' from the ground truth of synthetic frames.

Lets the whole node (tracker, scheduler, predictor, rules, views) run on a laptop without the DPU.
Optional seeded noise makes it a bit less perfect than the truth.
"""
import numpy as np

from utils.geometry import box_center
from utils.types import Detection


class ReplayDetector:
    name = "replay"

    def __init__(self, box_noise_px=0.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.box_noise_px = box_noise_px
        self.times = {}

    def detect(self, frame):
        dets = []
        for p in frame.truth or []:
            box = p.box + (self.rng.normal(0, self.box_noise_px, 4) if self.box_noise_px else 0)
            dets.append(Detection(box=np.asarray(box, np.float64), score=0.9))
        return dets


class ReplayPose:
    def __init__(self, kp_noise_px=0.0, seed=1, margin=(0.25, 0.12), min_cover=0.6):
        self.rng = np.random.default_rng(seed)
        self.kp_noise_px = kp_noise_px
        self.margin, self.min_cover = margin, min_cover
        self.times = {}

    def estimate(self, frame, box):
        """Truth keypoints of the person whose box centre is closest to the crop's centre.

        Like a real crop model, it only sees the person if the crop (box + margin) covers most of them: a stale
        upright box over someone lying down gives low-confidence keypoints.
        """
        if not frame.truth:
            return np.zeros((17, 3))
        c = box_center(box)
        p = min(frame.truth, key=lambda q: np.linalg.norm(box_center(q.box) - c))
        kp = p.kp.copy()
        w, h = box[2] - box[0], box[3] - box[1]
        crop = np.array([box[0] - self.margin[0] * w, box[1] - self.margin[1] * h,
                         box[2] + self.margin[0] * w, box[3] + self.margin[1] * h])
        tb = p.box
        iw = max(0.0, min(crop[2], tb[2]) - max(crop[0], tb[0]))
        ih = max(0.0, min(crop[3], tb[3]) - max(crop[1], tb[1]))
        if iw * ih < self.min_cover * max((tb[2] - tb[0]) * (tb[3] - tb[1]), 1.0):
            kp[:, 2] *= 0.2
        if self.kp_noise_px:
            kp[:, :2] += self.rng.normal(0, self.kp_noise_px, (17, 2))
        return kp


class CheapReplayDetector(ReplayDetector):
    """Stand-in for a pruned pedestrian detector: misses lying people, low score for boxes cut by the frame edge."""
    name = "replay_cheap"

    def __init__(self, box_noise_px=3.0, seed=2, lying_aspect=1.0, frame_w=640):
        super().__init__(box_noise_px, seed)
        self.lying_aspect, self.frame_w = lying_aspect, frame_w

    def detect(self, frame):
        dets = []
        for d in super().detect(frame):
            b = d.box
            if (b[2] - b[0]) > self.lying_aspect * (b[3] - b[1]):
                continue
            d.score = 0.45 if b[0] < 0 or b[2] > self.frame_w else 0.85
            dets.append(d)
        return dets


class CheapReplayPose(ReplayPose):
    """Stand-in for SPnet: no face points, noisier, and unreliable on people who are not upright."""

    FACE = [0, 1, 2, 3, 4]

    def __init__(self, kp_noise_px=3.0, seed=3):
        super().__init__(kp_noise_px, seed)

    def estimate(self, frame, box):
        kp = super().estimate(frame, box)
        kp[self.FACE, 2] = 0.0
        if (box[2] - box[0]) > 0.8 * (box[3] - box[1]):
            kp[:, 2] *= 0.3
        return kp


class ReplayHourglass(ReplayPose):
    """Stand-in for Hourglass (MPII): no face points, but fine on lying and bent people."""

    FACE = [0, 1, 2, 3, 4]

    def estimate(self, frame, box):
        kp = super().estimate(frame, box)
        kp[self.FACE, 2] = 0.0
        return kp


class ReplayOrientation:
    """Stand-in for the orientation classifier: reads the facing direction off the scenario's true pose.

    Labels as the real model: left / right (facing image left / right), front, back (relative to the camera).
    """

    def __init__(self, min_score=0.3):
        self.min_score = min_score
        self.times = {}

    def classify(self, frame, box):
        if not frame.truth:
            return "front", 0.25
        c = box_center(box)
        p = min(frame.truth, key=lambda q: np.linalg.norm(box_center(q.box) - c))
        kp, s = p.kp, p.kp[:, 2] >= self.min_score
        if not (s[0] or s[1] or s[2]):
            return "back", 0.9
        shoulders_w = abs(kp[5, 0] - kp[6, 0])
        body_h = max(p.box[3] - p.box[1], 1.0)
        if shoulders_w < 0.15 * body_h and s[0]:                 # seen edge-on: profile
            return ("right" if kp[0, 0] > (kp[5, 0] + kp[6, 0]) / 2 else "left"), 0.9
        return "front", 0.9


# Traits of the real models, for their laptop stand-ins (see VIDEO_pipeline/catalog.py)
CHEAP_DETECTORS = {"yolov2_voc_pruned", "refinedet_096", "refinedet_092"}   # pruned: miss lying people


def replay_models(frame_w):
    """Stand-ins for every catalog model -> (detectors, poses, orientation), same names as on the board."""
    from VIDEO_pipeline.catalog import CATALOG, names       # noqa: PLC0415
    full, cheap = ReplayDetector(box_noise_px=1.5), CheapReplayDetector(frame_w=frame_w)
    detectors = {n: (cheap if n in CHEAP_DETECTORS else full) for n in names("detector")}
    poses = {"movenet": ReplayPose(kp_noise_px=1.5), "hourglass": ReplayHourglass(kp_noise_px=1.5)}
    assert set(poses) == set(names("pose")), "every pose model in the catalog needs a stand-in"
    return detectors, poses, ReplayOrientation() if "orientation" in CATALOG else None
