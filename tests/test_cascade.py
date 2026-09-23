import numpy as np
import pytest

from guardian.cascade import DetectorCascade, PoseCascade
from guardian.rules import Body, Pair, RuleEngine
from guardian.scenario import pose_fallen, pose_front, pose_overhead
from guardian.types import Detection
from tests.conftest import M_PER_PX, person

UPRIGHT = np.array([300.0, 100.0, 380.0, 440.0])
LYING = np.array([200.0, 380.0, 460.0, 440.0])


class Fake:
    """Returns a fixed result and counts calls."""

    def __init__(self, result):
        self.result, self.n = result, 0

    def detect(self, frame):
        self.n += 1
        return self.result

    def estimate(self, frame, box):
        self.n += 1
        return self.result.copy()


def det(box, score=0.9):
    return Detection(box=np.asarray(box, float), score=score)


@pytest.fixture
def dets(cfg):
    def make(cheap_result):
        cheap, full = Fake(cheap_result), Fake([det(UPRIGHT)])
        casc = DetectorCascade(cfg, {"cheap": cheap, "full": full})
        casc.last_full_t = 0.0                    # watchdog satisfied at t < watchdog_s
        return casc, cheap, full
    return make


def run(casc, expected=(UPRIGHT,), sensitive=False, t=0.5):
    return casc.detect(None, ["cheap", "full"], list(expected), sensitive, t)


def test_confident_matching_cheap_result_is_accepted(dets):
    casc, cheap, full = dets([det(UPRIGHT + 3)])
    run(casc)
    assert (cheap.n, full.n) == (1, 0) and casc.last == ("cheap", ())


@pytest.mark.parametrize("cheap_result, reason", [
    ([], "missing track"),                        # "nobody there" is never taken on trust
    ([det(UPRIGHT, 0.4)], "low score"),
    ([det(UPRIGHT), det(LYING)], "lying box"),
])
def test_uncertain_cheap_result_escalates(dets, cheap_result, reason):
    casc, cheap, full = dets(cheap_result)
    out = run(casc)
    assert (cheap.n, full.n) == (1, 1) and reason in casc.last[1] and out is full.result


@pytest.mark.parametrize("kwargs, reason", [
    ({"t": 5.0}, "watchdog"),
    ({"sensitive": True}, "near threshold"),
    ({"expected": (LYING,)}, "lying box"),
])
def test_context_skips_the_cheap_stage(dets, kwargs, reason):
    casc, cheap, full = dets([det(UPRIGHT)])
    run(casc, **kwargs)
    assert (cheap.n, full.n) == (0, 1) and reason in casc.last[1]


def test_single_stage_and_unknown_names(cfg):
    only = Fake([])
    casc = DetectorCascade(cfg, {"yolov3_voc": only})
    casc.detect(None, "yolov3_voc", [UPRIGHT], True, 99.0)
    assert only.n == 1 and casc.cheap_share() == 1.0
    with pytest.raises(KeyError):
        casc.detect(None, ["nope", "yolov3_voc"], [], False, 0.0)


def test_cheap_share_counts_invocations(dets):
    casc, _, _ = dets([det(UPRIGHT)])
    run(casc)
    run(casc, sensitive=True)
    assert casc.cheap_share() == 0.5


@pytest.fixture
def poses(cfg):
    good = np.column_stack([np.zeros((17, 2)), np.full(17, 0.8)])
    def make(cheap_scores=0.8):
        cheap = Fake(np.column_stack([np.zeros((17, 2)), np.full(17, cheap_scores)]))
        full = Fake(good)
        return PoseCascade(cfg, {"spnet": cheap, "movenet": full}), cheap, full
    return make


def test_pose_cheap_accepted_for_upright_confident_person(poses):
    casc, cheap, full = poses()
    casc.estimate(None, UPRIGHT, ["spnet", "movenet"], need_face=False, sensitive=False)
    assert (cheap.n, full.n) == (1, 0)


@pytest.mark.parametrize("box, need_face, sensitive, reason", [
    (UPRIGHT, True, False, "face needed"),
    (LYING, False, False, "not upright"),
    (UPRIGHT, False, True, "near threshold"),
])
def test_pose_context_skips_cheap(poses, box, need_face, sensitive, reason):
    casc, cheap, full = poses()
    casc.estimate(None, box, ["spnet", "movenet"], need_face, sensitive)
    assert (cheap.n, full.n) == (0, 1) and reason in casc.last[1]


def test_pose_low_keypoints_escalate(poses):
    casc, cheap, full = poses(cheap_scores=0.2)
    casc.estimate(None, UPRIGHT, ["spnet", "movenet"], False, False)
    assert (cheap.n, full.n) == (1, 1) and casc.last == ("movenet", ("low keypoints",))


def body(x_m, pose):
    kp, box = person(x_m, pose)
    return Body(kp=kp, box=box, vel=np.zeros(2))


@pytest.mark.parametrize("robot_pose, human_x, human_pose, gap, near", [
    (pose_front(), 1.5, pose_fallen(+1), 3.0, False),   # body gap 0.55 m: 0.25 from reach_m 0.8, clear of 0.2
    (pose_front(), 1.75, pose_fallen(+1), 3.0, True),   # body gap ~0.8 m: on reach_m
    (pose_front(), 3.0, pose_front(), 1.0, False),      # overhead distance only counts with a raised load
    (pose_overhead(), 3.0, pose_front(), 1.0, True),
])
def test_near_threshold(cfg, robot_pose, human_x, human_pose, gap, near):
    engine = RuleEngine(cfg, 640, 480)
    p = Pair(body(0.0, robot_pose), body(human_x, human_pose), M_PER_PX, gap, [])
    assert engine.near_threshold(p, cfg["cascade"]["sensitivity_m"]) == near
