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
