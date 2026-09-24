"""Monocular 'fake 3D': floor position of a person from its box, assuming a known body size.

depth Z = focal_px * person_height_m / size_px, with size_px = max(box height, box width) so a lying person
(wide box) keeps roughly the right scale. X = (foot_u - W/2) * Z / focal_px. Every joint of a person is placed
at that person's depth (a billboard): good enough for distances between people and for the virtual scene,
not a metric reconstruction.
"""
import numpy as np


class WorldModel:
    def __init__(self, cfg, frame_w, frame_h):
        self.w, self.h = frame_w, frame_h
        self.configure(cfg)

    def configure(self, cfg):
        self.f = cfg["camera"]["focal_px"]
        self.person_h = cfg["person_height_m"]

    def size_px(self, box):
        return max(box[3] - box[1], box[2] - box[0], 1.0)

    def depth(self, box):
        return self.f * self.person_h / self.size_px(box)

    def m_per_px(self, box):
        return self.depth(box) / self.f

    def floor(self, box):
        """(X, Z) metres of the box's foot point."""
        z = self.depth(box)
        u = (box[0] + box[2]) / 2.0
        return np.array([(u - self.w / 2.0) * z / self.f, z])

    def gap_m(self, box_a, box_b):
        return float(np.linalg.norm(self.floor(box_a) - self.floor(box_b)))

    def joints_3d(self, kp, box):
        """(17, 3) keypoints (u, v, score) -> (17, 3) world (X, Y up, Z); all joints at the person's depth."""
        z = self.depth(box)
        mpp = z / self.f
        foot_v = box[3]
        X = (kp[:, 0] - self.w / 2.0) * mpp
        Y = (foot_v - kp[:, 1]) * mpp
        return np.stack([X, Y, np.full(17, z)], axis=1)
