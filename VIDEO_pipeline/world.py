"""Monocular 'fake 3D': floor position of a person from its box, assuming a known body size.

depth Z = focal_px * person_height_m / size_px, with size_px = max(box height, box width) so a lying person
(wide box) keeps roughly the right scale. X = (foot_u - W/2) * Z / focal_px. Every joint of a person is placed
at that person's depth (a billboard): good enough for distances between people and for the virtual scene,
not a metric reconstruction.

Boxes cut off by the frame (someone close to the camera) are handled explicitly:
    cut at top or bottom only   depth from the width instead (person_width_m)
    cut on both axes            depth is a lower bound only: `depth_reliable()` is False and the rules do not
                                use it to rule out contact
"""
import numpy as np


class WorldModel:
    def __init__(self, cfg, frame_w, frame_h):
        self.w, self.h = frame_w, frame_h
        self.configure(cfg)

    def configure(self, cfg):
        self.f = cfg["camera"]["focal_px"]
        self.edge = cfg["camera"]["edge_px"]
        self.person_h = cfg["person_height_m"]
        self.person_w = cfg["person_width_m"]

    def size_px(self, box):
        return max(box[3] - box[1], box[2] - box[0], 1.0)

    def cut(self, box):
        """-> (cut at top/bottom, cut at left/right): which box sides touch the frame border."""
        e = self.edge
        vertical = box[1] <= e or box[3] >= self.h - e
        horizontal = box[0] <= e or box[2] >= self.w - e
        return bool(vertical), bool(horizontal)

    def depth_reliable(self, box):
        return not all(self.cut(box))

    def depth(self, box):
        vertical, horizontal = self.cut(box)
        if vertical and not horizontal:
            return self.f * self.person_w / max(box[2] - box[0], 1.0)
        return self.f * self.person_h / self.size_px(box)

    def depth_gap_m(self, box_a, box_b):
        """|depth difference| in metres, or 0.0 when either depth is unreliable (cannot rule out contact)."""
        if not (self.depth_reliable(box_a) and self.depth_reliable(box_b)):
            return 0.0
        return abs(self.depth(box_a) - self.depth(box_b))

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
