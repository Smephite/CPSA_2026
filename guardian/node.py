"""The node: one loop that runs the cascade and produces a snapshot per frame for the views.

    IDLE    beacon absent (for absent_timeout_s): camera released, no inference, trackers reset
    DETECT  beacon present: detector at the band's rate; poses only once a robot/human pair is APPROACH or CLOSE
    TRACK   pair in APPROACH/CLOSE: poses at the band's rate, predictor + rules every frame

All timing uses the node clock (real or simulated), so a scenario runs identically fast or in real time.
"""
import time

import numpy as np

from guardian.predictor import Predictor
from guardian.rules import Body, DecisionLatch, Pair, RuleEngine, combine
from guardian.scheduler import Scheduler
from guardian.tracking import Tracker
from guardian.types import Band, Level, NodeState
from guardian.world import WorldModel


class GuardianNode:
    def __init__(self, cfg, clock, camera, beacon, detectors, pose, audio, robot_link, power, events,
                 caption=None):
        self.cfg, self.clock, self.camera, self.beacon = cfg, clock, camera, beacon
        self.detectors, self.pose = detectors, pose
        self.audio, self.robot_link, self.power, self.events = audio, robot_link, power, events
        self.caption = caption or (lambda t: "")
        w, h = camera.frame_size
        self.world = WorldModel(cfg, w, h)
        self.tracker = Tracker(cfg)
        self.scheduler = Scheduler(cfg)
        self.predictor = Predictor(cfg)
        self.engine = RuleEngine(cfg, w, h)
        self.latch = DecisionLatch(cfg)
        self.min_score = cfg["pose"]["min_score"]
        self.pose_max_age = cfg["pose"]["max_age_s"]
        self.absent_timeout = cfg["beacon"]["absent_timeout_s"]

        self.state = NodeState.IDLE
        self.schedule = self.scheduler.update(None)
        self.last_beacon_t = -1e9
        self.last_det_t = -1e9
        self.gap_prev = None                     # (t, gap) for the closing speed
        self.closing = 0.0
        self.body_gap = None
        self.fps, self._t_prev = 0.0, None

    # ---------------------------------------------------------------- helpers

    def _set_state(self, s, t):
        if s != self.state:
            self.events.system(f"[node] {self.state.name} -> {s.name} at {t:.1f}s")
            self.state = s

    def _pair(self):
        """Robot and its nearest human with a fresh-enough track -> (robot, human, gap_m) or None."""
        robot = self.tracker.robot
        humans = self.tracker.humans
        if robot is None or not humans:
            return None
        gaps = [(self.world.gap_m(robot.box, h.box), h) for h in humans]
        gap, human = min(gaps, key=lambda g: g[0])
        return robot, human, gap

    def _fresh(self, tr, t):
        return tr.kp is not None and t - tr.kp_t <= self.pose_max_age

    def _body(self, tr, kp=None, box=None):
        return Body(kp=tr.kp if kp is None else kp, box=tr.box if box is None else box, vel=tr.vel)

    # ---------------------------------------------------------------- one iteration

    def step(self):
        t = self.clock.now()
        beacon = self.beacon.poll(t)
        if beacon.present:
            self.last_beacon_t = t

        if t - self.last_beacon_t > self.absent_timeout:
            return self._idle(t, beacon)

        if not self.camera.is_open:
            self.camera.open()
        frame = self.camera.read()
        t = frame.t
        times = {}

        # --- detector (rate from the band schedule)
        sched = self.schedule
        if sched.detector_hz > 0 and t - self.last_det_t >= 1.0 / sched.detector_hz - 1e-6:
            det = self.detectors[sched.detector] if sched.detector in self.detectors else next(iter(self.detectors.values()))
            t0 = time.perf_counter()
            dets = det.detect(frame)
            times["det"] = (time.perf_counter() - t0) * 1e3
            self.tracker.update(dets, t)
            self.last_det_t = t
        else:
            self.tracker.predict(t)
            self.tracker.prune(t)
        self.tracker.assign_roles()
        self.predictor.forget_missing(set(self.tracker.tracks))

        # --- distance band -> schedule, state
        pair = self._pair()
        gap = pair[2] if pair else None
        if gap is not None and self.gap_prev is not None and t - self.gap_prev[0] > 1e-3:
            sample = -(gap - self.gap_prev[1]) / (t - self.gap_prev[0])      # positive = closing
            self.closing = 0.7 * self.closing + 0.3 * sample
        self.gap_prev = (t, gap) if gap is not None else None
        if gap is None:
            self.closing = 0.0
        self.schedule = sched = self.scheduler.update(gap, self.closing)
        tracking = sched.band in (Band.APPROACH, Band.CLOSE)
        self._set_state(NodeState.TRACK if tracking else NodeState.DETECT, t)

        # --- poses on person crops (robot and every human), at the band's pose rate
        if tracking and sched.pose_hz > 0:
            t0 = time.perf_counter()
            for tr in [self.tracker.robot] + self.tracker.humans:
                if tr is not None and t - tr.kp_t >= 1.0 / sched.pose_hz - 1e-6:
                    kp = self.pose.estimate(frame, tr.box_at(t))
                    self.tracker.observe_pose(tr.id, kp, t)
                    self.predictor.observe(tr.id, t, kp)
            times["pose"] = (time.perf_counter() - t0) * 1e3

        # --- predictor + rules (every frame while tracking)
        t0 = time.perf_counter()
        dangers, future = [], {}
        self.body_gap = None
        robot = self.tracker.robot
        if tracking and robot is not None and self._fresh(robot, t):
            m_per_px = self.world.m_per_px(robot.box)
            r_vel = self.predictor.velocities(robot)
            for tr in self.tracker.tracks.values():
                if self._fresh(tr, t):
                    jv = self.predictor.velocities(tr)
                    future[tr.id] = Predictor.future(tr.kp, tr.box, jv, tr.vel, self.predictor.horizon_s)
            for human in self.tracker.humans:
                if not self._fresh(human, t):
                    continue
                h_vel = self.predictor.velocities(human)
                now = Pair(self._body(robot), self._body(human), m_per_px,
                           self.world.gap_m(robot.box, human.box), self.engine.lines_px)
                if pair and human is pair[1]:
                    self.body_gap = self.engine.body_gap_m(now)
                fut = []
                for dt in self.predictor.horizon():
                    rk, rb = Predictor.future(robot.kp, robot.box, r_vel, robot.vel, dt)
                    hk, hb = Predictor.future(human.kp, human.box, h_vel, human.vel, dt)
                    fut.append((dt, Pair(self._body(robot, rk, rb), self._body(human, hk, hb), m_per_px,
                                         self.world.gap_m(rb, hb), self.engine.lines_px)))
                for d in self.engine.evaluate(now, fut):
                    d.human_id = human.id
                    dangers.append(d)
        times["rules"] = (time.perf_counter() - t0) * 1e3

        raw = combine(dangers)
        level = self._decide(raw, dangers, t, beacon)
        return self._snapshot(t, frame, beacon, level, dangers, future, pair, times)

    def _decide(self, raw, dangers, t, beacon):
        prev = self.latch.level
        level, rules = self.latch.update(raw, dangers, t)
        changed = level != prev
        if changed:
            self.events.system(f"[decision] {prev.name} -> {level.name} {', '.join(rules)} at {t:.1f}s")
            if level > Level.NONE:
                self.events.event(level.name, rules)
        self.audio.update(level, t, changed)
        self.robot_link.update(level, beacon.robot_id)
        return level

    def _idle(self, t, beacon):
        if self.state != NodeState.IDLE:
            self.camera.release()
            self.tracker.reset()
            self.predictor.reset()
            self.scheduler.reset()
            self.latch.reset()
            self.gap_prev, self.closing, self.body_gap = None, 0.0, None
            self.schedule = self.scheduler.update(None)
            self._set_state(NodeState.IDLE, t)
            self.robot_link.update(Level.NONE, "")
        self.clock.sleep(0.1)
        t = self.clock.now()
        return self._snapshot(t, None, beacon, Level.NONE, [], {}, None, {})

    def _snapshot(self, t, frame, beacon, level, dangers, future, pair, times):
        if self._t_prev is not None and t > self._t_prev:
            inst = 1.0 / (t - self._t_prev)
            self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
        self._t_prev = t
        robot = self.tracker.robot
        return {
            "t": t, "frame": frame, "state": self.state, "beacon": beacon, "schedule": self.schedule,
            "tracks": list(self.tracker.tracks.values()), "level": level, "rules": self.latch.rules.get(level, ()),
            "dangers": dangers, "future": future, "gap_m": pair[2] if pair else None,
            "body_gap_m": self.body_gap,
            "pair_human": pair[1] if pair else None, "closing_mps": self.closing if pair else None,
            "m_per_px": self.world.m_per_px(robot.box) if robot is not None else None,
            "power_w": self.power.read(), "times": times, "fps": self.fps, "caption": self.caption(t),
        }
