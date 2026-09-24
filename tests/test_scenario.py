"""End-to-end: the synthetic 4-beat demo on the simulated clock, replay perception, no outputs."""
import pytest

import main
from utils import settings as gcfg
from utils.types import Level, NodeState


# YOLOv3 alone in every band (the default is the YOLOv2 -> YOLOv3 cascade)
YOLOV3_ONLY = {"bands": {"rates": {b: {"detector": "yolov3_voc"} for b in ("detect", "far", "approach", "close")}}}


def simulate(overrides=None, models=None):
    args = main.parse_args(["--fast", "--no-log", "--no-audio", "--port", "0"])
    node, _, scenario, _, _ = main.build(args, gcfg._merge(gcfg.DEFAULTS, overrides))
    if models:
        node.det_cascade.models, node.pose_cascade.models = models
    node.events.echo = False
    snaps = []
    while True:
        s = node.step()
        snaps.append((s.t, s.state, s.level, tuple(s.rules)))
        if s.t >= scenario.duration:
            return snaps, node


@pytest.fixture(scope="module", params=[YOLOV3_ONLY, None, main.DEMO_CASCADE], ids=["yolov3", "default", "cascade"])
def run(request):
    return simulate(request.param)[0]


def first(snaps, level, rule, t0, t1):
    return next((t for t, _, lv, rules in snaps if t0 <= t < t1 and lv == level and rule in rules), None)


def test_idle_without_beacon(run):
    assert all(st == NodeState.IDLE for t, st, _, _ in run if t < 3.0)
    assert run[-1][1] == NodeState.IDLE


def test_beat1_far_pass_is_quiet(run):
    assert all(lv == Level.NONE for t, _, lv, _ in run if 3.0 <= t < 9.0)


def test_beat2_reach_warns_before_contact(run):
    assert first(run, Level.WARN, "reach", 9.0, 16.0) is not None
    assert first(run, Level.STOP, "reach", 9.0, 16.0) is None


def test_beat3_stop_from_behind(run):
    assert first(run, Level.STOP, "from_behind", 16.0, 19.5) is not None


def test_beat4_fallen_warns_then_stops(run):
    warn = first(run, Level.WARN, "down", 23.0, 30.0)
    stop = first(run, Level.STOP, "down", 23.0, 30.0)
    assert warn is not None and stop is not None and warn < stop <= 28.0


def test_idle_after_beacon_timeout(run):
    t_idle = next(t for t, st, _, _ in run if t > 30.0 and st == NodeState.IDLE)
    assert 31.5 <= t_idle <= 33.0


def test_cheap_models_alone_are_unsafe():
    """Control for the cascade: the cheap stand-ins alone false-STOP in beat 2 and lose the fallen human."""
    from VIDEO_pipeline.replay import CheapReplayDetector, CheapReplayPose
    snaps, _ = simulate(YOLOV3_ONLY, models=({"yolov3_voc": CheapReplayDetector()}, {"movenet": CheapReplayPose()}))
    assert first(snaps, Level.STOP, "from_behind", 9.0, 16.0) is not None
    assert first(snaps, Level.STOP, "down", 23.0, 30.0) is None


def test_cascade_uses_both_stages_for_the_right_reasons():
    _, node = simulate(main.DEMO_CASCADE)
    assert node.det_cascade.calls["refinedet_096"] > 0 and node.pose_cascade.calls["spnet"] > 0
    assert node.pose_cascade.reasons["face needed"] > 0 and node.det_cascade.reasons["lying box"] > 0


@pytest.mark.parametrize("overrides", [YOLOV3_ONLY, None, main.DEMO_CASCADE], ids=["yolov3", "default", "cascade"])
def test_sudden_motion_only_on_the_fall_and_the_robot_reversal(overrides, monkeypatch):
    """Torso acceleration crosses the threshold during the fall (21.4-22 s) and when the scripted robot reverses
    in one frame (19 s), nowhere else: walking and pose noise stay below it."""
    import core.guardian_node as gn
    hits, orig = [], gn.GuardianNode._accel_mps2

    def record(self, tr):
        a = orig(self, tr)
        if a > self.cfg["cascade"]["sudden_accel_mps2"]:
            hits.append((self.clock.now(), tr.role))
        return a
    monkeypatch.setattr(gn.GuardianNode, "_accel_mps2", record)
    simulate(overrides)
    assert any(r == "human" and 21.4 <= t <= 22.7 for t, r in hits)
    assert all((r == "human" and 21.4 <= t <= 22.7) or (r == "robot" and 19.0 <= t <= 19.5) for t, r in hits), hits


def test_default_cascade_runs_yolov2_first():
    _, node = simulate()
    assert node.det_cascade.calls["yolov2_voc_pruned"] > 0 and node.det_cascade.calls["yolov3_voc"] > 0
