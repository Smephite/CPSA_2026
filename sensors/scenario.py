"""Synthetic demo scenario: the brief's 30-second choreography with two people, one playing the robot.

Renders simple camera frames (stick figures on a wall/floor background) and attaches the ground truth
(boxes + 17 keypoints) so the replay backend can stand in for YOLO and MoveNet. All people stand at the same
depth; x positions are metres from the image centre.

    0-3 s    no beacon                                    -> IDLE
    3-9 s    beat 1: robot passes far away (gap > 3 m)    -> DETECT, nothing else
    9-16 s   beat 2: robot approaches from the front      -> TRACK, WARN (reach predicted), robot pauses
    16-19 s  beat 3: human turns away, robot keeps closing -> STOP (from_behind)
    19-23 s  robot backs off, human falls (21.4-22 s, accelerating)
    23-30 s  beat 4: robot approaches the fallen human    -> WARN, then STOP (down)
    30 s     beacon gone                                   -> IDLE after the timeout
"""
from dataclasses import dataclass

import cv2
import numpy as np

from utils.types import SKELETON, TruthPerson

# Standing, facing the camera; metres, x to the image right, y up from the floor. Person's left = image right.
FRONT = np.array([
    [0.00, 1.60], [0.035, 1.64], [-0.035, 1.64], [0.08, 1.61], [-0.08, 1.61],
    [0.20, 1.42], [-0.20, 1.42], [0.24, 1.12], [-0.24, 1.12], [0.25, 0.85], [-0.25, 0.85],
    [0.12, 0.95], [-0.12, 0.95], [0.12, 0.50], [-0.12, 0.50], [0.12, 0.06], [-0.12, 0.06]])
VISIBLE, HIDDEN = 0.85, 0.05


def pose_front():
    return FRONT.copy(), np.full(17, VISIBLE)


def pose_profile(facing):
    """Side view, facing +1 (image right) or -1. The far eye and ear are hidden."""
    p = FRONT.copy()
    p[:, 0] *= 0.2                                          # body seen edge-on
    p[0] = [0.10 * facing, 1.58]
    p[1] = [0.07 * facing, 1.63]
    p[2] = [0.07 * facing, 1.63]
    p[3] = [-0.01 * facing, 1.61]
    p[4] = [-0.01 * facing, 1.61]
    s = np.full(17, VISIBLE)
    far_eye, far_ear = (2, 4) if facing > 0 else (1, 3)     # facing right: the person's right side is far
    s[far_eye] = s[far_ear] = HIDDEN
    return p, s


def pose_fallen(head_dir=1):
    """Lying on the floor along x, head towards head_dir."""
    p = FRONT.copy()
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0] = (y - 0.85) * head_dir
    p[:, 1] = 0.12 + (x + 0.25) * 0.3
    return p, np.full(17, 0.75)


def pose_overhead():
    p, s = pose_front()
    p[7], p[8] = [0.25, 1.72], [-0.25, 1.72]
    p[9], p[10] = [0.15, 1.95], [-0.15, 1.95]
    return p, s


def _lerp(keys, t):
    """Piecewise-linear value from [(t, v), ...]."""
    ts = [k[0] for k in keys]
    vs = [k[1] for k in keys]
    return float(np.interp(t, ts, vs))


@dataclass
class Beat:
    name: str
    t0: float
    t1: float


class DemoScenario:
    size = (640, 480)
    px_per_m = 141.0
    floor_y = 440
    beacon_intervals = [(3.0, 30.0)]
    duration = 34.0
    beats = [Beat("idle", 0, 3), Beat("beat1_far", 3, 9), Beat("beat2_front", 9, 16),
             Beat("beat3_behind", 16, 19.5), Beat("reset", 19.5, 23), Beat("beat4_fallen", 23, 30),
             Beat("idle_end", 32.5, 34)]

    HUMAN_X = 1.4
    FALLEN_SHIFT = 0.15                                 # lying, the head would leave the frame at HUMAN_X
    FALL_T, FALL_S = 22.0, 0.6                          # the human is on the floor at FALL_T after FALL_S falling
    ROBOT_X = [(0, -1.9), (3, -1.9), (9, -1.6), (15, 0.4), (17, 0.4), (19, 0.75), (22, -1.4), (23, -1.4),
               (28, 0.3), (40, 0.3)]

    def robot_x(self, t):
        return _lerp(self.ROBOT_X, t)

    def people(self, t):
        """-> [(name, x_m, pose (17, 2) metres, scores (17,))]"""
        robot = ("robot", self.robot_x(t)) + pose_front()
        t0 = self.FALL_T - self.FALL_S
        if t < 16.5:
            human = pose_front()
        elif t < t0:
            human = pose_profile(+1)                    # facing image right, away from the robot on the left
        elif t < self.FALL_T:
            u = ((t - t0) / self.FALL_S) ** 2           # falling: accelerating, like a body under gravity
            (a, sa), (b, sb) = pose_profile(+1), pose_fallen(+1)
            human = ((1 - u) * a + u * b, np.minimum(sa, sb) if u > 0.5 else sa)
        else:
            human = pose_fallen(+1)
        u = 0.0 if t < t0 else min(1.0, ((t - t0) / self.FALL_S) ** 2)
        x = self.HUMAN_X - self.FALLEN_SHIFT * u
        out = [robot] if t >= 3.4 else []               # the robot walks in just after its beacon
        return out + [("human", x) + human]

    def beat(self, t):
        return next((b.name for b in self.beats if b.t0 <= t < b.t1), "")

    # ---------------------------------------------------------------- rendering

    def to_px(self, x_m, pose):
        W = self.size[0]
        u = W / 2 + (x_m + pose[:, 0]) * self.px_per_m
        v = self.floor_y - pose[:, 1] * self.px_per_m
        return np.stack([u, v], axis=1)

    def render(self, t):
        W, H = self.size
        img = np.empty((H, W, 3), np.uint8)
        img[:300] = (196, 200, 204)
        img[300:] = (150, 156, 160)
        for i in range(0, W, 80):
            cv2.line(img, (i, 300), (int(W / 2 + (i - W / 2) * 1.6), H), (140, 146, 150), 1)
        cv2.line(img, (0, 300), (W, 300), (120, 124, 128), 2)
        truth = []
        for name, x_m, pose, scores in self.people(t):
            uv = self.to_px(x_m, pose)
            col = (40, 120, 230) if name == "robot" else (190, 110, 50)
            for a, b in SKELETON:
                cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), col, 7, cv2.LINE_AA)
            head = uv[[0, 3, 4]].mean(axis=0)
            cv2.circle(img, tuple(head.astype(int)), int(0.11 * self.px_per_m), col, -1, cv2.LINE_AA)
            pad = np.array([0.06, 0.12, 0.06, 0.03]) * self.px_per_m
            box = np.array([uv[:, 0].min() - pad[0], uv[:, 1].min() - pad[1],
                            uv[:, 0].max() + pad[2], uv[:, 1].max() + pad[3]])
            kp = np.concatenate([uv, scores[:, None]], axis=1)
            truth.append(TruthPerson(name=name, box=box, kp=kp))
        return img, truth
