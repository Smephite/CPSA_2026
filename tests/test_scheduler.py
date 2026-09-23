import pytest

from guardian.scheduler import Scheduler
from guardian.types import Band


@pytest.fixture
def sched(cfg):
    return Scheduler(cfg)


def bands(s, gaps, closing=0.0):
    return [s.update(g, closing).band for g in gaps]


def test_no_pair_uses_detect_rates(sched, cfg):
    s = sched.update(None)
    assert s.band is None and s.pose_hz == cfg["bands"]["rates"]["detect"]["pose_hz"]


def test_hysteresis_at_the_far_edge(sched):
    # far_m 3.0, hysteresis 0.2
    assert bands(sched, [4.0, 2.9, 2.7, 3.1, 3.3]) == [Band.FAR, Band.FAR, Band.APPROACH, Band.APPROACH, Band.FAR]


def test_hysteresis_at_the_close_edge(sched):
    # close_m 1.5
    assert bands(sched, [2.0, 1.4, 1.2, 1.6, 1.8]) == [Band.APPROACH, Band.APPROACH, Band.CLOSE, Band.CLOSE,
                                                       Band.APPROACH]


def test_fast_closing_moves_one_band_closer(sched, cfg):
    s = sched.update(4.0, closing_mps=cfg["bands"]["fast_closing_mps"] + 0.1)
    assert s.band == Band.APPROACH and sched.band == Band.FAR
    assert s.pose_hz == cfg["bands"]["rates"]["approach"]["pose_hz"]
