"""Shared fixtures: default config and people placed with the demo scenario's camera model."""
import numpy as np
import pytest

from utils import settings as gcfg
from sensors.scenario import DemoScenario

SCENE = DemoScenario()
M_PER_PX = 1.0 / SCENE.px_per_m


@pytest.fixture
def cfg():
    return gcfg._merge(gcfg.DEFAULTS, {})


def person(x_m, pose):
    """(pose (17, 2) metres, scores (17,)) at x_m -> (kp (17, 3) frame px, box (4,))."""
    p, s = pose
    uv = SCENE.to_px(x_m, p)
    kp = np.column_stack([uv, s])
    return kp, np.array([uv[:, 0].min() - 8, uv[:, 1].min() - 16, uv[:, 0].max() + 8, uv[:, 1].max() + 4])
