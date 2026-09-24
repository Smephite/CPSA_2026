import numpy as np
import pytest

from VIDEO_pipeline.world import WorldModel

W, H = 640, 480


@pytest.fixture
def world(cfg):
    return WorldModel(cfg, W, H)


def test_full_box_depth_from_height(world, cfg):
    box = np.array([300, 100, 360, 350])                          # 250 px tall, inside the frame
    assert world.cut(box) == (False, False)
    assert world.depth(box) == pytest.approx(cfg["camera"]["focal_px"] * cfg["person_height_m"] / 250)
    assert world.depth_reliable(box)


def test_box_cut_at_the_bottom_uses_width(world, cfg):
    """Someone seated near the camera: box runs off the bottom edge, so its height says nothing about depth."""
    box = np.array([40, 20, 490, H])                              # 450 px wide, cut at the bottom
    assert world.cut(box) == (True, False)
    assert world.depth(box) == pytest.approx(cfg["camera"]["focal_px"] * cfg["person_width_m"] / 450)
    assert world.depth(box) < 1.0                                 # close to the camera, as seen live
    assert world.depth_reliable(box)


def test_box_cut_on_both_axes_is_unreliable(world):
    box = np.array([0, 0, W, H])
    assert world.cut(box) == (True, True)
    assert not world.depth_reliable(box)
    assert world.depth_gap_m(box, np.array([300, 100, 360, 350])) == 0.0


def test_live_case_near_and_far_people_are_apart_in_depth(world, cfg):
    """The live office run: a seated person filling the frame bottom vs. people standing ~5 m behind."""
    near = np.array([42, 20, 490, H])
    far = np.array([395, 45, 460, 190])                           # 145 px tall
    assert world.depth_gap_m(near, far) > cfg["rules"]["depth_gate_m"]
