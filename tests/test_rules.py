import numpy as np
import pytest

from core.rules import Body, DecisionLatch, Pair, RuleEngine, combine
from sensors.scenario import pose_fallen, pose_front, pose_overhead, pose_profile
from utils.types import Danger, Level
from tests.conftest import M_PER_PX, person

W, H = 640, 480


def body(x_m, pose, vel=(0.0, 0.0)):
    kp, box = person(x_m, pose)
    return Body(kp=kp, box=box, vel=np.array(vel, float))


def pair(robot, human, gap_m, lines=()):
    return Pair(robot=robot, human=human, m_per_px=M_PER_PX, gap_m=gap_m, lines_px=list(lines))


@pytest.fixture
def engine(cfg):
    return RuleEngine(cfg, W, H)


def test_reach(engine):
    robot = body(0.0, pose_front())
    assert engine.rule_reach(pair(robot, body(0.6, pose_front()), 0.6)) == Level.STOP
    assert engine.rule_reach(pair(robot, body(1.5, pose_front()), 1.5)) == Level.NONE


@pytest.mark.parametrize("facing, robot_vel, level", [
    (+1, 100.0, Level.STOP),     # facing away from the robot on its left, robot closing
    (-1, 100.0, Level.NONE),     # facing the robot
    (+1, 0.0, Level.NONE),       # robot not moving
    (+1, -100.0, Level.NONE),    # robot backing off
])
def test_from_behind(engine, facing, robot_vel, level):
    p = pair(body(0.0, pose_front(), (robot_vel, 0.0)), body(1.0, pose_profile(facing)), 1.0)
    assert engine.rule_from_behind(p) == level


def test_from_behind_ignores_a_lying_human(engine):
    p = pair(body(0.0, pose_front(), (100.0, 0.0)), body(1.0, pose_fallen(+1)), 1.0)
    assert engine.rule_from_behind(p) == Level.NONE


@pytest.mark.parametrize("human_x, level", [(1.5, Level.STOP), (2.2, Level.WARN), (3.2, Level.NONE)])
def test_down_uses_body_gap(engine, human_x, level):
    # floor gap deliberately large: only the joint-to-robot-hull distance should decide
    p = pair(body(0.0, pose_front()), body(human_x, pose_fallen(+1)), gap_m=3.0)
    assert engine.rule_down(p) == level


def test_down_needs_a_down_posture(engine):
    p = pair(body(0.0, pose_front()), body(1.0, pose_front()), 1.0)
    assert engine.rule_down(p) == Level.NONE


def test_body_gap_between_hulls(engine):
    # nearest: the fallen human's feet (0.71 m, 0.16-0.23 m up) to the robot's ankle-wrist edge (x 0.12-0.25 m)
    p = pair(body(0.0, pose_front()), body(1.5, pose_fallen(+1)), 3.0)
    assert 0.5 < engine.body_gap_m(p) < 0.6


def test_body_gap_falls_back_to_floor_gap_without_joints(engine):
    robot = body(0.0, pose_front())
    human = body(1.0, pose_front())
    human.kp[:, 2] = 0.0
    assert engine.body_gap_m(pair(robot, human, 1.23)) == 1.23


def test_pinned(cfg):
    line_u = (W / 2 + 1.5 * 141 + 30) / W                       # just right of a human at x = 1.5 m
    cfg["rules"]["static_lines"] = [[line_u, 0.0, line_u, 1.0]]
    engine = RuleEngine(cfg, W, H)
    human = body(1.5, pose_front())
    closing = pair(body(0.5, pose_front(), (100.0, 0.0)), human, 1.0, engine.lines_px)
    assert engine.rule_pinned(closing) == Level.STOP
    backing = pair(body(0.5, pose_front(), (-100.0, 0.0)), human, 1.0, engine.lines_px)
    assert engine.rule_pinned(backing) == Level.NONE
    assert RuleEngine(cfg | {"rules": {**cfg["rules"], "static_lines": []}}, W, H).rule_pinned(
        pair(body(0.5, pose_front(), (100.0, 0.0)), human, 1.0)) == Level.NONE


def test_overhead(engine):
    human = body(1.0, pose_front())
    assert engine.rule_overhead(pair(body(0.0, pose_overhead()), human, 0.8)) == Level.STOP
    assert engine.rule_overhead(pair(body(0.0, pose_overhead()), human, 1.5)) == Level.NONE
    assert engine.rule_overhead(pair(body(0.0, pose_front()), human, 0.8)) == Level.NONE


def test_predicted_stop_warns_with_time_to_contact(engine):
    robot, human = body(0.0, pose_front()), body(1.5, pose_front())
    now = pair(robot, human, 1.5)
    future = [(0.5, pair(robot, body(1.2, pose_front()), 1.2)), (1.0, pair(robot, body(0.6, pose_front()), 0.6))]
    reach = next(d for d in engine.evaluate(now, future) if d.rule == "reach")
    assert (reach.level_now, reach.predicted, reach.ttc_s) == (Level.NONE, Level.STOP, 1.0)
    assert combine([reach]) == Level.WARN


def test_combine():
    assert combine([]) == Level.NONE
    assert combine([Danger("down", Level.WARN, Level.WARN, None)]) == Level.WARN
    assert combine([Danger("down", Level.WARN, Level.WARN, None),
                    Danger("reach", Level.STOP, Level.STOP, 0.0)]) == Level.STOP


def test_latch_holds_stop_then_releases(cfg):
    latch = DecisionLatch(cfg)
    stop = [Danger("reach", Level.STOP, Level.STOP, 0.0)]
    assert latch.update(Level.STOP, stop, 0.0) == (Level.STOP, ("reach",))
    assert latch.update(Level.NONE, [], 1.9) == (Level.STOP, ("reach",))
    assert latch.update(Level.NONE, [], 2.1)[0] == Level.NONE
    latch.update(Level.WARN, [Danger("down", Level.WARN, Level.WARN, None)], 3.0)
    assert latch.update(Level.NONE, [], 3.9) == (Level.WARN, ("down",))
    assert latch.update(Level.NONE, [], 4.1)[0] == Level.NONE


# ---------------------------------------------------------------- depth gate (live test: false STOPs across depth)


def test_contact_rules_ignore_pairs_apart_in_depth(engine):
    """Overlapping in the image but metres apart in depth: no contact rule fires."""
    robot, touching = body(0.0, pose_front()), body(0.6, pose_front())
    fallen = body(0.9, pose_fallen(+1))
    near = Pair(robot, touching, M_PER_PX, 0.6, [], depth_gap_m=0.3)
    apart = Pair(robot, touching, M_PER_PX, 0.6, [], depth_gap_m=3.0)
    assert engine.rule_reach(near) == Level.STOP
    assert engine.rule_reach(apart) == Level.NONE
    assert engine.rule_down(Pair(robot, fallen, M_PER_PX, 0.9, [], depth_gap_m=3.0)) == Level.NONE
    assert not engine.near_threshold(apart, 10.0)


def test_unknown_depth_never_rules_contact_out(engine):
    """depth_gap_m = 0 (the default, and what an unreliable depth gives) keeps the rules active."""
    robot, touching = body(0.0, pose_front()), body(0.6, pose_front())
    assert engine.rule_reach(pair(robot, touching, 0.6)) == Level.STOP
