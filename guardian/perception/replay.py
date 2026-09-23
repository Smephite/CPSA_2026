"""Replay backend: 'detects' and 'estimates poses' from the ground truth of synthetic frames.

Lets the whole node (tracker, scheduler, predictor, rules, views) run on a laptop without the DPU.
Optional seeded noise makes it a bit less perfect than the truth.
"""
import numpy as np

from guardian.geometry import box_center
from guardian.types import Detection


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
    def __init__(self, kp_noise_px=0.0, seed=1):
        self.rng = np.random.default_rng(seed)
        self.kp_noise_px = kp_noise_px
        self.times = {}

    def estimate(self, frame, box):
        """Truth keypoints of the person whose box centre is closest to the crop's centre."""
        if not frame.truth:
            return np.zeros((17, 3))
        c = box_center(box)
        p = min(frame.truth, key=lambda q: np.linalg.norm(box_center(q.box) - c))
        kp = p.kp.copy()
        if self.kp_noise_px:
            kp[:, :2] += self.rng.normal(0, self.kp_noise_px, (17, 2))
        return kp
