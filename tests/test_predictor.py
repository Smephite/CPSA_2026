import numpy as np
import pytest

from guardian.predictor import TORSO, PoseHistory

PX_PER_M = 141.0


def history(rate_hz, duration_s, accel_mps2=0.0, vel_mps=0.5, noise_px=0.0, seed=0):
    """Torso moving right with constant acceleration, sampled at rate_hz."""
    h, rng = PoseHistory(window_s=0.6, min_score=0.3), np.random.default_rng(seed)
    for t in np.arange(0, duration_s, 1.0 / rate_hz):
        kp = np.zeros((17, 3))
        x = (vel_mps * t + 0.5 * accel_mps2 * t * t) * PX_PER_M
        kp[:, 0] = 300 + x + rng.normal(0, noise_px, 17)
        kp[:, 1] = 200 + np.arange(17) * 10 + rng.normal(0, noise_px, 17)
        kp[list(TORSO), 2] = 0.9
        h.add(t, kp)
    return h


@pytest.mark.parametrize("rate_hz", [5, 15])
def test_recovers_constant_acceleration(rate_hz):
    a = history(rate_hz, 2.0, accel_mps2=3.0).torso_acceleration(0.5)
    assert a[0] / PX_PER_M == pytest.approx(3.0, rel=0.05) and abs(a[1]) < 1e-6


@pytest.mark.parametrize("rate_hz", [5, 15])
def test_noise_alone_stays_below_the_sudden_threshold(rate_hz):
    """1.5 px pose noise (the replay stand-in's) at constant velocity: well under 2 m/s^2."""
    worst = 0.0
    for seed in range(50):
        a = history(rate_hz, 2.0, noise_px=1.5, seed=seed).torso_acceleration(0.5)
        worst = max(worst, np.linalg.norm(a) / PX_PER_M)
    assert worst < 1.0


def test_too_few_or_too_old_samples():
    assert history(15, 0.2).torso_acceleration(0.5) is None              # 3 samples
    h = history(1, 5.0)                                                  # 1 Hz: 5 samples span 4 s > 2.5 x 0.5 s
    assert h.torso_acceleration(0.5) is None
