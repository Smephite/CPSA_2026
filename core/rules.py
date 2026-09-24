"""Deterministic rule engine: five geometric predicates on the robot's and a human's skeletons.

Each predicate looks at one snapshot (now, or a predicted future) and returns NONE / WARN / STOP:

    reach        a human keypoint inside the robot's arm hull (shoulders, elbows, wrists) dilated by a margin
    from_behind  robot closing on a human who faces away from it
    down         human hip at or below knee height, or lying (wide box); WARN within down_warn_m, STOP within reach_m,
                 both measured from any human joint to the hull of all robot joints
    pinned       human between the robot and a configured static line (wall, bed, rack), robot closing
    overhead     robot wrists above its shoulders with a human inside the drop radius

A predicate that is STOP now -> STOP. A predicate that becomes STOP within the look-ahead -> WARN, with the
time to it (TTC). Only this module decides; no model output reaches the outputs directly.

Distances: image-plane geometry in pixels, converted with metres-per-pixel from the robot's apparent size;
the gap between the two people is the floor distance from `WorldModel` (includes depth).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.geometry import box_center, convex_hull, distance_to_hull, point_segment_distance
from utils.types import ARM_IDS, Danger, KP, Level


@dataclass
class Body:
    kp: np.ndarray            # (17, 3)
    box: np.ndarray           # (4,)
    vel: np.ndarray           # (2,) px/s, body


@dataclass
class Pair:
    robot: Body
    human: Body
    m_per_px: float
    gap_m: float
    lines_px: List[np.ndarray]


class RuleEngine:
    def __init__(self, cfg, frame_w, frame_h):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.configure(cfg)

    def configure(self, cfg):
        frame_w, frame_h = self.frame_w, self.frame_h
        self.c = cfg["rules"]
        self.min_score = cfg["pose"]["min_score"]
        self.enabled = list(self.c["enabled"])
        self.lines_px = [np.array([l[0] * frame_w, l[1] * frame_h, l[2] * frame_w, l[3] * frame_h], float)
                         for l in self.c["static_lines"]]
        self.rules = {name: getattr(self, "rule_" + name) for name in self.enabled}

    # ---------------------------------------------------------------- helpers

    def _vis(self, kp, idx):
        return [i for i in idx if kp[i, 2] >= self.min_score]

    def arm_hull(self, kp):
        idx = self._vis(kp, ARM_IDS)
        return convex_hull(kp[idx, :2]) if len(idx) >= 2 else None

    def body_gap_m(self, p: Pair):
        """Closest visible human joint to the hull of all visible robot joints, in metres (gap_m if none)."""
        r = p.robot.kp[p.robot.kp[:, 2] >= self.min_score, :2]
        h = p.human.kp[p.human.kp[:, 2] >= self.min_score, :2]
        if not len(r) or not len(h):
            return p.gap_m
        hull = convex_hull(r)
        return min(distance_to_hull(q, hull) for q in h) * p.m_per_px

    def _closing_mps(self, p: Pair):
        d = box_center(p.human.box) - box_center(p.robot.box)
        n = np.linalg.norm(d)
        return 0.0 if n < 1e-6 else float(p.robot.vel @ (d / n)) * p.m_per_px

    def facing_away(self, human: Body, robot_x):
        kp = human.kp
        ls, rs = kp[KP["left_shoulder"]], kp[KP["right_shoulder"]]
        if ls[2] < self.min_score or rs[2] < self.min_score:
            return False
        face = [kp[KP[n], 2] >= self.min_score for n in ("nose", "left_eye", "right_eye")]
        if not any(face):
            return True                               # back of the head towards the camera
        nose = kp[KP["nose"]]
        if nose[2] < self.min_score:
            return False
        box_h = max(human.box[3] - human.box[1], 1.0)
        mid = (ls[0] + rs[0]) / 2.0
        off = (nose[0] - mid) / max(abs(ls[0] - rs[0]), 0.1 * box_h)
        if abs(off) < 0.3:
            return False                              # facing the camera: not away from a robot at its side
        robot_side = np.sign(robot_x - box_center(human.box)[0])
        return np.sign(off) == -robot_side

    # ---------------------------------------------------------------- predicates

    def rule_reach(self, p: Pair) -> Level:
        hull = self.arm_hull(p.robot.kp)
        if hull is None:
            return Level.NONE
        margin = self.c["reach_margin_m"] / p.m_per_px
        pts = p.human.kp[p.human.kp[:, 2] >= self.min_score, :2]
        return Level.STOP if any(distance_to_hull(q, hull) <= margin for q in pts) else Level.NONE

    def rule_from_behind(self, p: Pair) -> Level:
        if p.gap_m > self.c["behind_m"] or self._closing_mps(p) < self.c["closing_min_mps"]:
            return Level.NONE
        if self.is_down(p.human, p.m_per_px):
            return Level.NONE                         # no facing direction when lying: `down` covers it
        return Level.STOP if self.facing_away(p.human, box_center(p.robot.box)[0]) else Level.NONE

    def is_down(self, human: Body, m_per_px):
        kp, box = human.kp, human.box
        if (box[2] - box[0]) > self.c["lying_aspect"] * (box[3] - box[1]):
            return True
        hips = self._vis(kp, [KP["left_hip"], KP["right_hip"]])
        knees = self._vis(kp, [KP["left_knee"], KP["right_knee"]])
        if not hips or not knees:
            return False
        tol = self.c["down_tol_m"] / m_per_px
        return kp[hips, 1].mean() >= kp[knees, 1].mean() - tol          # image y grows downwards

    def rule_down(self, p: Pair) -> Level:
        if not self.is_down(p.human, p.m_per_px):
            return Level.NONE
        gap = self.body_gap_m(p)
        if gap <= self.c["reach_m"] or self.rule_reach(p) == Level.STOP:
            return Level.STOP
        return Level.WARN if gap <= self.c["down_warn_m"] else Level.NONE

    def rule_pinned(self, p: Pair) -> Level:
        if not p.lines_px or p.gap_m > self.c["pin_gap_m"] or self._closing_mps(p) < self.c["closing_min_mps"]:
            return Level.NONE
        h, r = box_center(p.human.box), box_center(p.robot.box)
        for line in p.lines_px:
            a, b = line[:2], line[2:]
            if point_segment_distance(h, a, b) * p.m_per_px > self.c["pin_line_m"]:
                continue
            ab = b - a
            t = float(np.clip((r - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
            q = a + t * ab                                              # robot's nearest point on the line
            if point_segment_distance(h, r, q) * p.m_per_px <= self.c["pin_corridor_m"]:
                return Level.STOP
        return Level.NONE

    def load_overhead(self, kp, m_per_px):
        """Robot wrists above its shoulders (carrying something overhead)."""
        sh = self._vis(kp, [KP["left_shoulder"], KP["right_shoulder"]])
        wr = self._vis(kp, [KP["left_wrist"], KP["right_wrist"]])
        if not sh or not wr:
            return False
        return kp[wr, 1].min() < kp[sh, 1].min() - self.c["overhead_tol_m"] / m_per_px

    def rule_overhead(self, p: Pair) -> Level:
        overhead = self.load_overhead(p.robot.kp, p.m_per_px)
        return Level.STOP if overhead and p.gap_m <= self.c["drop_radius_m"] else Level.NONE

    # ---------------------------------------------------------------- decision sensitivity

    def near_threshold(self, p: Pair, band_m) -> bool:
        """True if a distance an enabled rule compares against a threshold is within band_m of it."""
        margins = []
        if "reach" in self.enabled:
            hull = self.arm_hull(p.robot.kp)
            pts = p.human.kp[p.human.kp[:, 2] >= self.min_score, :2]
            if hull is not None and len(pts):
                d = min(distance_to_hull(q, hull) for q in pts) * p.m_per_px
                margins.append(abs(d - self.c["reach_margin_m"]))
        if "down" in self.enabled and self.is_down(p.human, p.m_per_px):
            gap = self.body_gap_m(p)
            margins += [abs(gap - self.c["reach_m"]), abs(gap - self.c["down_warn_m"])]
        if "from_behind" in self.enabled and self._closing_mps(p) >= self.c["closing_min_mps"]:
            margins.append(abs(p.gap_m - self.c["behind_m"]))
        if "overhead" in self.enabled and self.load_overhead(p.robot.kp, p.m_per_px):
            margins.append(abs(p.gap_m - self.c["drop_radius_m"]))
        return bool(margins) and min(margins) < band_m

    # ---------------------------------------------------------------- evaluation

    def evaluate(self, now: Pair, future: List[tuple]) -> List[Danger]:
        """now: Pair; future: [(dt, Pair), ...] ascending. -> one Danger per rule that is not NONE."""
        dangers = []
        for name, rule in self.rules.items():
            level_now = rule(now)
            predicted, ttc = level_now, None
            for dt, pf in future:
                lv = rule(pf)
                if lv == Level.STOP and ttc is None:
                    ttc = dt
                predicted = max(predicted, lv)
            if level_now == Level.STOP:
                ttc = 0.0
            if predicted > Level.NONE:
                dangers.append(Danger(rule=name, level_now=level_now, predicted=predicted, ttc_s=ttc))
        return dangers


def combine(dangers: List[Danger]) -> Level:
    """STOP if any rule is STOP now; WARN if any rule is WARN now or predicts STOP; else NONE."""
    level = Level.NONE
    for d in dangers:
        level = max(level, d.level_now)
        if d.predicted == Level.STOP:
            level = max(level, Level.WARN)
    return level


class DecisionLatch:
    """Holds STOP for stop_hold_s after the last STOP condition and WARN for warn_hold_s."""

    def __init__(self, cfg):
        self.configure(cfg)
        self.reset()

    def configure(self, cfg):
        self.stop_hold, self.warn_hold = cfg["decision"]["stop_hold_s"], cfg["decision"]["warn_hold_s"]

    def reset(self):
        self.stop_until = self.warn_until = -1e9
        self.level = Level.NONE
        self.since = 0.0
        self.rules = {Level.WARN: (), Level.STOP: ()}

    def update(self, raw: Level, dangers: List[Danger], t: float):
        if raw == Level.STOP:
            self.stop_until = t + self.stop_hold
            self.rules[Level.STOP] = tuple(dict.fromkeys(d.rule for d in dangers if d.level_now == Level.STOP))
        if raw >= Level.WARN:
            self.warn_until = t + self.warn_hold
            if raw == Level.WARN:
                self.rules[Level.WARN] = tuple(dict.fromkeys(d.rule for d in dangers))
        level = Level.STOP if t < self.stop_until else Level.WARN if t < self.warn_until else Level.NONE
        if level != self.level:
            self.level, self.since = level, t
        return level, self.rules.get(level, ())
