"""End-to-end: the synthetic 4-beat demo on the simulated clock, replay perception, no outputs."""
import pytest

import guardian_main
from guardian import config as gcfg
from guardian.types import Level, NodeState


@pytest.fixture(scope="module")
def run():
    args = guardian_main.parse_args(["--fast", "--no-log", "--no-audio", "--port", "0"])
    node, _, scenario, _, _ = guardian_main.build(args, gcfg._merge(gcfg.DEFAULTS, {}))
    node.events.echo = False
    snaps = []
    while True:
        s = node.step()
        snaps.append((s["t"], s["state"], s["level"], tuple(s["rules"])))
        if s["t"] >= scenario.duration:
            return snaps


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
