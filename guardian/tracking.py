"""Multi-person tracking with persistent ids, and the robot/human role lock.

Nearest-centroid matching, gated by the track's box height. Between detector frames (the detector may run at
1-3 Hz) boxes are predicted at constant velocity; pose observations (up to 15 Hz) keep the velocity current.

Roles: when the second confirmed track appears, roles are locked once (leftmost = robot by default, the rest
humans) and kept by the tracker. `assign_robot()` is the hook for the production rule (the robot is the track
matching the beacon's reported position).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from guardian.geometry import box_center


def _size(box):
    return max(box[3] - box[1], box[2] - box[0], 1.0)


def _aspect_flip(old, new):
    """True when a box changes between upright and lying (aspect ratio crosses 1 by a clear margin)."""
    a_old = (old[2] - old[0]) / max(old[3] - old[1], 1.0)
    a_new = (new[2] - new[0]) / max(new[3] - new[1], 1.0)
    return (a_old < 0.8 and a_new > 1.25) or (a_old > 1.25 and a_new < 0.8)


def _pose_anchor(kp, min_score):
    """Torso centre (shoulders + hips) of a pose, or the mean of all visible joints; None if too few."""
    vis = kp[:, 2] >= min_score
    torso = [i for i in (5, 6, 11, 12) if vis[i]]
    idx = torso if len(torso) >= 2 else list(np.nonzero(vis)[0])
    if len(idx) < 2:
        return None
    return kp[idx, :2].mean(axis=0)


@dataclass
class Track:
    id: int
    box: np.ndarray                      # (4,) at time t_box
    t_box: float
    t_obs: float                         # last observation (detector or pose)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))   # px/s of the box centre
    hits: int = 1
    role: Optional[str] = None           # "robot" | "human" | None
    kp: Optional[np.ndarray] = None      # (17, 3) latest pose, frame px
    kp_t: float = -1e9
    _anchor: Optional[np.ndarray] = None
    _anchor_t: float = -1e9
    _c_obs: Optional[np.ndarray] = None

    def box_at(self, t):
        d = self.vel * (t - self.t_box)
        return self.box + np.array([d[0], d[1], d[0], d[1]])

    @property
    def center(self):
        return box_center(self.box)


class Tracker:
    def __init__(self, cfg):
        self.configure(cfg)
        self.tracks: Dict[int, Track] = {}
        self.robot_id: Optional[int] = None
        self._next_id = 1

    def configure(self, cfg):
        c = cfg["tracker"]
        self.gate = c["gate"]
        self.max_age_s = c["max_age_s"]
        self.min_hits = c["min_hits"]
        self.alpha = c["vel_alpha"]
        self.pose_min_kps = c["pose_min_kps"]
        self.min_score = cfg["pose"]["min_score"]
        self.robot_is = cfg["roles"]["robot_is"]

    # ---------------------------------------------------------------- observations

    def predict(self, t):
        for tr in self.tracks.values():
            tr.box = tr.box_at(t)
            tr.t_box = t

    def update(self, detections, t):
        """Match detections (persons) to tracks; new tracks for the rest; drop stale tracks."""
        self.predict(t)
        pairs = []
        for tid, tr in self.tracks.items():
            h = _size(tr.box)                    # person size: a lying person has a wide, flat box
            for di, d in enumerate(detections):
                cost = np.linalg.norm(box_center(d.box) - tr.center) / h
                if cost <= self.gate:
                    pairs.append((cost, tid, di))
        used_t, used_d = set(), set()
        for cost, tid, di in sorted(pairs):
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            self._observe_box(self.tracks[tid], detections[di].box, t)
        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks[self._next_id] = Track(id=self._next_id, box=np.asarray(d.box, float).copy(),
                                                   t_box=t, t_obs=t, _c_obs=box_center(d.box))
                self._next_id += 1
        self.prune(t)

    def _observe_box(self, tr, box, t):
        c = box_center(box)
        if _aspect_flip(tr.box, box):
            tr.vel = np.zeros(2)                 # stood up / fell: the centre jump is not motion
        elif tr._c_obs is not None and t - tr.t_obs > 1e-3 and tr.kp_t < tr.t_obs - 1e-6:
            # velocity from detector boxes only when no fresher pose drives it
            tr.vel = (1 - self.alpha) * tr.vel + self.alpha * (c - tr._c_obs) / (t - tr.t_obs)
        tr.box = np.asarray(box, float).copy()
        tr.t_box = tr.t_obs = t
        tr._c_obs = c
        tr.hits += 1

    def observe_pose(self, track_id, kp, t):
        """Store a pose; use its torso anchor to keep the velocity (and so the predicted box) current."""
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        tr.kp, tr.kp_t = kp, t
        n_vis = int((kp[:, 2] >= self.min_score).sum())
        anchor = _pose_anchor(kp, self.min_score) if n_vis >= self.pose_min_kps else None
        if anchor is None:
            return
        if tr._anchor is not None and t - tr._anchor_t > 1e-3:
            tr.box = tr.box_at(t)
            tr.t_box = t
            sample = (anchor - tr._anchor) / (t - tr._anchor_t)
            tr.vel = (1 - self.alpha) * tr.vel + self.alpha * sample
            tr.t_obs = t
        tr._anchor, tr._anchor_t = anchor, t

    def prune(self, t):
        for tid in [tid for tid, tr in self.tracks.items() if t - tr.t_obs > self.max_age_s]:
            del self.tracks[tid]
            if tid == self.robot_id:
                self.robot_id = None          # robot lost: roles are locked again when two people are seen

    def reset(self):
        self.tracks.clear()
        self.robot_id = None

    # ---------------------------------------------------------------- roles

    def confirmed(self) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.hits >= self.min_hits]

    def assign_roles(self):
        conf = self.confirmed()
        if self.robot_id is None and len(conf) >= 2:
            pick = min if self.robot_is == "leftmost" else max
            self.robot_id = pick(conf, key=lambda tr: tr.center[0]).id
        for tr in self.tracks.values():
            if self.robot_id is None:
                tr.role = None
            else:
                tr.role = "robot" if tr.id == self.robot_id else ("human" if tr.hits >= self.min_hits else None)

    def assign_robot(self, track_id):
        """Production hook: the robot is the track matching the beacon's reported position."""
        if track_id in self.tracks:
            self.robot_id = track_id
            self.assign_roles()

    @property
    def robot(self) -> Optional[Track]:
        return self.tracks.get(self.robot_id) if self.robot_id is not None else None

    @property
    def humans(self) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.role == "human"]
