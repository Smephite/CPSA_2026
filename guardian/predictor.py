"""Time-series predictor: joint histories -> per-joint velocities -> future poses within the look-ahead.

Each track keeps a short history of poses. Joint velocities are least-squares slopes over the last `window_s`
(joints below the visibility threshold are skipped; joints without enough samples fall back to the track's
body velocity). A future pose at t + dt is the current pose moved by velocity * dt (constant velocity).
The rule engine evaluates the same predicates on these future poses to predict dangers before they happen.
"""
from collections import deque

import numpy as np

TORSO = (5, 6, 11, 12)                          # shoulders and hips: whole-body motion, not arm swings


class PoseHistory:
    def __init__(self, window_s, min_score, maxlen=64):
        self.window_s = window_s
        self.min_score = min_score
        self.samples = deque(maxlen=maxlen)      # (t, kp (17, 3))

    def add(self, t, kp):
        if self.samples and t <= self.samples[-1][0]:
            return
        self.samples.append((t, kp.copy()))

    def joint_velocities(self, fallback_vel):
        """(17, 2) px/s. Least squares over the window per joint; body velocity where data is missing."""
        vel = np.tile(np.asarray(fallback_vel, float), (17, 1))
        if len(self.samples) < 2:
            return vel
        t_end = self.samples[-1][0]
        recent = [(t, kp) for t, kp in self.samples if t >= t_end - self.window_s]
        if len(recent) < 2:
            return vel
        ts = np.array([t for t, _ in recent])
        kps = np.stack([kp for _, kp in recent])            # (n, 17, 3)
        for j in range(17):
            ok = kps[:, j, 2] >= self.min_score
            if ok.sum() >= 2 and np.ptp(ts[ok]) > 1e-3:
                tt = ts[ok] - ts[ok].mean()
                for d in range(2):
                    yy = kps[ok, j, d]
                    vel[j, d] = float((tt * (yy - yy.mean())).sum() / (tt * tt).sum())
        return vel


    def torso_acceleration(self, window_s, min_samples=5, min_span_s=0.15):
        """(2,) px/s^2: mean over confident torso joints of a quadratic least-squares fit, or None.

        Uses the last `window_s`, stretched back to `min_samples` poses (up to 2.5 x window_s) when poses are
        slow: a short window keeps a jolt sharp at 15 Hz, the stretch keeps 5 Hz usable."""
        if len(self.samples) < min_samples:
            return None
        t_end = self.samples[-1][0]
        recent = [(t, kp) for t, kp in self.samples if t >= t_end - window_s]
        if len(recent) < min_samples:
            recent = list(self.samples)[-min_samples:]
            if t_end - recent[0][0] > 2.5 * window_s:
                return None
        ts = np.array([t for t, _ in recent])
        kps = np.stack([kp for _, kp in recent])
        acc = []
        for j in TORSO:
            ok = kps[:, j, 2] >= self.min_score
            if ok.sum() >= 4 and np.ptp(ts[ok]) >= min_span_s:
                tt = ts[ok] - ts[ok].mean()
                acc.append([2.0 * np.polyfit(tt, kps[ok, j, d], 2)[0] for d in range(2)])
        return np.mean(acc, axis=0) if len(acc) >= 2 else None


class Predictor:
    def __init__(self, cfg):
        self.histories = {}
        self.configure(cfg)

    def configure(self, cfg):
        p = cfg["predictor"]
        self.window_s, self.horizon_s, self.steps = p["window_s"], p["horizon_s"], int(p["steps"])
        self.accel_window_s = p["accel_window_s"]
        self.min_score = cfg["pose"]["min_score"]
        for h in self.histories.values():
            h.window_s, h.min_score = self.window_s, self.min_score

    def observe(self, track_id, t, kp):
        h = self.histories.get(track_id)
        if h is None:
            h = self.histories[track_id] = PoseHistory(self.window_s, self.min_score)
        h.add(t, kp)

    def forget_missing(self, live_ids):
        for tid in list(self.histories):
            if tid not in live_ids:
                del self.histories[tid]

    def reset(self):
        self.histories.clear()

    def horizon(self):
        """Look-ahead sample times, e.g. [0.25, 0.5, 0.75, 1.0]."""
        return [self.horizon_s * (i + 1) / self.steps for i in range(self.steps)]

    def acceleration(self, track):
        """(2,) px/s^2 torso acceleration of a track, or None."""
        h = self.histories.get(track.id)
        return h.torso_acceleration(self.accel_window_s) if h else None

    def velocities(self, track):
        h = self.histories.get(track.id)
        return h.joint_velocities(track.vel) if h else np.tile(track.vel, (17, 1))

    @staticmethod
    def future(kp, box, joint_vel, body_vel, dt):
        """Pose and box `dt` seconds ahead (constant velocity)."""
        kp_f = kp.copy()
        kp_f[:, :2] += joint_vel * dt
        d = np.asarray(body_vel) * dt
        return kp_f, box + np.array([d[0], d[1], d[0], d[1]])
