"""GuardianNode: the coordinator (the role of upstream CPSA_2026's EventDispatcher). One call to step() = one frame.

    sensors ──Frame, BeaconState──► VIDEO_pipeline ──tracks, poses──► core ──Decision──► actuators
                                                                          └──Snapshot──► dashboard

    IDLE    beacon absent (for absent_timeout_s): camera released, no inference, trackers reset
    DETECT  beacon present: detector at the band's rate; poses only once a robot/human pair is APPROACH or CLOSE
    TRACK   pair in APPROACH/CLOSE: poses at the band's rate, predictor + rules every frame

All timing uses the node clock (real or simulated), so a scenario runs identically fast or in real time.
"""
import time

import numpy as np

from VIDEO_pipeline.cascade import DetectorCascade, PoseCascade
from core.predictor import Predictor
from core.rules import Body, DecisionLatch, Pair, RuleEngine, combine
from core.scheduler import Scheduler
from VIDEO_pipeline.tracking import Tracker
from utils.types import Band, Decision, Level, NodeState, Snapshot
from VIDEO_pipeline.world import WorldModel


class GuardianNode:
    def __init__(self, cfg, clock, camera, beacon, power, detectors, poses, actuators, events, caption=None):
        """
        cfg                 settings dict (utils/settings.py)
        clock               utils.clock.RealClock or SimClock
        camera, beacon, power   sensors (sensors/__init__.py for their interfaces)
        detectors, poses    model name -> Detector / PoseEstimator, for every name the band table and
                            pose.models use (VIDEO_pipeline/__init__.py)
        actuators           actuators.ActuatorManager
        events              utils.event_log.EventLog (system log for state changes)
        caption             t -> text shown on the dashboard (scenario beat), optional
        """
        self.cfg, self.clock, self.camera, self.beacon = cfg, clock, camera, beacon
        self.det_cascade = DetectorCascade(cfg, detectors)
        self.pose_cascade = PoseCascade(cfg, poses)

        self.power, self.actuators, self.events = power, actuators, events
        self.caption = caption or (lambda t: "")
        w, h = camera.frame_size
        self.world = WorldModel(cfg, w, h)
        self.tracker = Tracker(cfg)
        self.scheduler = Scheduler(cfg)
        self.predictor = Predictor(cfg)
        self.engine = RuleEngine(cfg, w, h)
        self.latch = DecisionLatch(cfg)
        self.configure_self(cfg)

        self.state = NodeState.IDLE
        self.schedule = self.scheduler.update(None)
        self.last_beacon_t = -1e9
        self.last_det_t = -1e9
        self.gap_prev = None                     # (t, gap) for the closing speed
        self.closing = 0.0
        self.body_gap = None
        self.sensitive = False                   # last frame's rules were near a threshold or predicting a STOP
        self.sudden = set()                      # track ids whose torso accelerated suddenly (last frame)
        self.fps, self._t_prev = 0.0, None

    def configure_self(self, cfg):
        self.min_score = cfg["pose"]["min_score"]
        self.pose_max_age = cfg["pose"]["max_age_s"]
        self.absent_timeout = cfg["beacon"]["absent_timeout_s"]
        self.pose_models = cfg["pose"]["models"]

    def reconfigure(self):
        """Re-read every tunable value from self.cfg (after it was changed in place). Keeps tracks and history."""
        self.configure_self(self.cfg)
        for part in (self.world, self.tracker, self.scheduler, self.predictor, self.engine, self.latch,
                     self.det_cascade, self.pose_cascade, self.actuators):
            part.configure(self.cfg)

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

    def _facing_matters(self, tr):
        """Could `from_behind` fire for this track within the look-ahead? (needs a pose model with face points)"""
        robot = self.tracker.robot
        if tr.role != "human" or robot is None or "from_behind" not in self.engine.enabled:
            return False
        if (tr.box[2] - tr.box[0]) > self.cfg["cascade"]["lying_aspect"] * (tr.box[3] - tr.box[1]):
            return False                         # lying: no facing direction, `from_behind` ignores it
        gap = self.world.gap_m(robot.box, tr.box) - max(self.closing, 0.0) * self.predictor.horizon_s
        return gap <= self.engine.c["behind_m"] + self.cfg["cascade"]["sensitivity_m"]

    def _accel_mps2(self, tr):
        a = self.predictor.acceleration(tr)
        return 0.0 if a is None else float(np.linalg.norm(a)) * self.world.m_per_px(tr.box)

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
            expected = [tr.box_at(t) for tr in self.tracker.confirmed()]
            t0 = time.perf_counter()
            dets = self.det_cascade.detect(frame, sched.detector, expected, self.sensitive, t, bool(self.sudden))
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
                    kp = self.pose_cascade.estimate(frame, tr.box_at(t), self.pose_models,
                                                    self._facing_matters(tr), self.sensitive, tr.id in self.sudden)
                    self.tracker.observe_pose(tr.id, kp, t)
                    self.predictor.observe(tr.id, t, kp)
            times["pose"] = (time.perf_counter() - t0) * 1e3
        self.sudden = {tr.id for tr in self.tracker.tracks.values() if self._accel_mps2(tr) > self.cfg["cascade"]["sudden_accel_mps2"]}

        # --- predictor + rules (every frame while tracking)
        t0 = time.perf_counter()
        dangers, future = [], {}
        self.body_gap = None
        sensitive = False
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
                sensitive |= self.engine.near_threshold(now, self.cfg["cascade"]["sensitivity_m"])
                for d in self.engine.evaluate(now, fut):
                    d.human_id = human.id
                    dangers.append(d)
        self.sensitive = sensitive or any(d.predicted > d.level_now for d in dangers)
        times["rules"] = (time.perf_counter() - t0) * 1e3

        raw = combine(dangers)
        level = self._decide(raw, dangers, t, beacon)
        return self._snapshot(t, frame, beacon, level, dangers, future, pair, times)

    def _decide(self, raw, dangers, t, beacon):
        """Latch the combined rule level and hand the decision to the actuators."""
        prev = self.latch.level
        level, rules = self.latch.update(raw, dangers, t)
        decision = Decision(level=level, rules=tuple(rules), since=self.latch.since, changed=level != prev,
                            robot_id=beacon.robot_id, dangers=dangers)
        self.actuators.update(decision, t)
        return level

    def _idle(self, t, beacon):
        if self.state != NodeState.IDLE:
            self.camera.release()
            self.tracker.reset()
            self.predictor.reset()
            self.scheduler.reset()
            self.latch.reset()
            self.gap_prev, self.closing, self.body_gap, self.sensitive = None, 0.0, None, False
            self.sudden = set()
            self.schedule = self.scheduler.update(None)
            self._set_state(NodeState.IDLE, t)
            self.actuators.update(Decision(Level.NONE, changed=True), t)
        self.clock.sleep(0.1)
        t = self.clock.now()
        return self._snapshot(t, None, beacon, Level.NONE, [], {}, None, {})

    def _snapshot(self, t, frame, beacon, level, dangers, future, pair, times):
        if self._t_prev is not None and t > self._t_prev:
            inst = 1.0 / (t - self._t_prev)
            self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
        self._t_prev = t
        robot = self.tracker.robot
        return Snapshot(
            t=t, state=self.state, level=level, frame=frame, beacon=beacon, schedule=self.schedule,
            tracks=list(self.tracker.tracks.values()), rules=self.latch.rules.get(level, ()),
            dangers=dangers, future=future, pair_human=pair[1] if pair else None,
            gap_m=pair[2] if pair else None, body_gap_m=self.body_gap,
            closing_mps=self.closing if pair else None,
            m_per_px=self.world.m_per_px(robot.box) if robot is not None else None,
            cascade={"det": self.det_cascade, "pose": self.pose_cascade},
            power_w=self.power.read(), times=times, fps=self.fps, caption=self.caption(t),
        )
