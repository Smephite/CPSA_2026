"""Low -> high fidelity model cascades for the detector and the pose stage.

A band names its detector (and the node its pose model) either as one model or as a list, cheapest first.
Each stage but the last must earn acceptance; otherwise the next stage runs and its result replaces the cheap one.

Escalation reasons are of two kinds:
    before a cheap stage runs (from context; the cheap stage is then skipped):
        watchdog        the last detector stage has not run for watchdog_s (catches what cheap models miss)
        near threshold  the rules were within sensitivity_m of flipping a decision on the last frame
        sudden motion   a torso accelerated faster than sudden_accel_mps2 (lunge, fall, abrupt start or stop)
        lying box       a tracked person has a lying-shaped box (pedestrian-trained models miss these)
        face needed     `from_behind` could fire and the cheap pose model has no face points
        not upright     the person's box is not upright (cheap pose models are trained on standing people)
    after it ran (from its output):
        missing track   a confirmed track has no matching detection: "nobody there" is never taken on trust
        low score       a detection below accept_score
        low keypoints   the rule-relevant joints are not confident

Uncertain *absence* always escalates; a confident detection that matches what the tracker expects does not.
"""
from collections import Counter

import numpy as np

from utils.geometry import box_center

BODY_IDS = list(range(5, 17))              # shoulders .. ankles: the joints the rules use


def _aspect(box):
    return (box[2] - box[0]) / max(box[3] - box[1], 1.0)


def _names(spec):
    return [spec] if isinstance(spec, str) else list(spec)


class _Cascade:
    def __init__(self, cfg, models):
        self.models = models
        self.configure(cfg)
        self.calls = Counter()                   # stage name -> calls
        self.reasons = Counter()                 # escalation reason -> count
        self.last = ("", ())                     # (stage that produced the result, reasons that led there)
        self.runs = self.cheap_runs = 0          # cascade invocations / those answered by the first stage
        self.stage_times = Counter()             # model stage -> ms since the last take_times() (profiling)

    def configure(self, cfg):
        self.c = cfg["cascade"]

    def _run(self, names, pre, post, call):
        """pre: reasons known before running; post(result) -> reasons from a cheap result."""
        names = _names(names)
        missing = [n for n in names if n not in self.models]
        if missing:
            raise KeyError(f"no model loaded for {missing} (have {sorted(self.models)})")
        led = ()
        for i, name in enumerate(names):
            last = i == len(names) - 1
            if not last and pre:
                led = tuple(pre)
                self.reasons.update(pre)
                continue
            out = call(self.models[name])
            self.calls[name] += 1
            self.stage_times.update(getattr(self.models[name], "times", {}) or {})
            if last:
                break
            why = post(out)
            if not why:
                break
            led = tuple(why)
            self.reasons.update(why)
        self.last = (name, led)
        self.runs += 1
        self.cheap_runs += name == names[0]
        return out, name

    def take_times(self):
        """Per-stage ms of the model calls since the last call (pre / dpu / post), then reset."""
        out, self.stage_times = dict(self.stage_times), Counter()
        return out

    def cheap_share(self):
        """Fraction of invocations answered by the first (cheapest) stage."""
        return self.cheap_runs / self.runs if self.runs else 1.0


class DetectorCascade(_Cascade):
    def __init__(self, cfg, detectors):
        super().__init__(cfg, detectors)
        self.last_full_t = -1e9

    def configure(self, cfg):
        super().configure(cfg)
        self.gate = cfg["tracker"]["gate"]

    def detect(self, frame, names, expected, sensitive, t, sudden=False):
        """expected: predicted boxes of confirmed tracks at t. -> detections of the stage that was accepted."""
        names = _names(names)
        pre = []
        if len(names) > 1:
            if t - self.last_full_t >= self.c["watchdog_s"]:
                pre.append("watchdog")
            if sensitive:
                pre.append("near threshold")
            if sudden:
                pre.append("sudden motion")
            if any(_aspect(b) > self.c["lying_aspect"] for b in expected):
                pre.append("lying box")
        dets, name = self._run(names, pre, lambda d: self._post(d, expected), lambda m: m.detect(frame))
        if name == names[-1]:
            self.last_full_t = t
        return dets

    def _post(self, dets, expected):
        why = []
        if any(d.score < self.c["accept_score"] for d in dets):
            why.append("low score")
        if any(_aspect(d.box) > self.c["lying_aspect"] for d in dets):
            why.append("lying box")
        for b in expected:
            size = max(b[3] - b[1], b[2] - b[0], 1.0)
            if not any(np.linalg.norm(box_center(d.box) - box_center(b)) / size <= self.gate for d in dets):
                why.append("missing track")
                break
        return why


class PoseCascade(_Cascade):
    def estimate(self, frame, box, names, need_face, sensitive, sudden=False):
        names = _names(names)
        pre = []
        if len(names) > 1:
            if need_face:
                pre.append("face needed")
            if sensitive:
                pre.append("near threshold")
            if sudden:
                pre.append("sudden motion")
            if _aspect(box) >= self.c["upright_aspect"]:
                pre.append("not upright")
        post = lambda kp: [] if kp[BODY_IDS, 2].mean() >= self.c["accept_kp"] else ["low keypoints"]
        return self._run(names, pre, post, lambda m: m.estimate(frame, box))[0]
