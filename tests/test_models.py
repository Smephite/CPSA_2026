"""The model catalog and the new decoders (RefineDet, Hourglass, orientation, YOLOv5-style boxes)."""
import os

import numpy as np
import pytest

from VIDEO_pipeline import catalog
from VIDEO_pipeline.HOURGLASS import hourglass
from VIDEO_pipeline.ORIENTATION import orientation
from VIDEO_pipeline.REFINEDET import refinedet
from VIDEO_pipeline.YOLO import yolo


def test_every_catalog_model_file_exists():
    for m in catalog.CATALOG.values():
        assert os.path.isfile(os.path.join("VIDEO_pipeline", m.file)), m.file


def test_every_catalog_model_has_a_laptop_stand_in():
    from VIDEO_pipeline.replay import replay_models
    detectors, poses, orient = replay_models(640)
    assert set(detectors) == set(catalog.names("detector"))
    assert set(poses) == set(catalog.names("pose"))
    assert orient is not None


def test_choices_pair_cheap_with_full_models():
    choices = catalog.detector_choices()
    assert "yolov2_voc_pruned > yolov3_voc" in choices and "refinedet_096 > ofa_yolo_05" in choices
    assert "yolov3_voc > yolov2_voc_pruned" not in choices            # never the expensive one first
    assert set(catalog.names("detector")) <= set(choices)
    assert "movenet > hourglass" in catalog.pose_choices()


def test_refinedet_decodes_a_single_prior():
    n = len(refinedet.PRIORS)
    assert n == 12240                                                  # 4 layers x 3 aspect ratios
    i = 1000
    arm_conf = np.tile([4.0, -4.0], (n, 1)); arm_conf[i] = [-4.0, 4.0]
    odm_conf = np.tile([4.0, -4.0], (n, 1)); odm_conf[i] = [-4.0, 4.0]
    heads = {"arm_loc": np.zeros((1, n * 4)), "arm_conf": arm_conf[None], "odm_loc": np.zeros((1, n * 4)),
             "odm_conf": odm_conf[None]}
    dets = refinedet.decode(heads, (480, 640))
    assert len(dets) == 1 and dets[0].score > 0.99
    cx, cy, w, h = refinedet.PRIORS[i]
    assert np.allclose(dets[0].box, [(cx - w / 2) * 640, (cy - h / 2) * 480, (cx + w / 2) * 640, (cy + h / 2) * 480],
                       atol=1e-3)


def test_refinedet_rejects_arm_negatives():
    n = len(refinedet.PRIORS)
    arm_conf = np.tile([9.0, -9.0], (n, 1))                           # every anchor surely background
    odm_conf = np.tile([-4.0, 4.0], (n, 1))                           # even if the ODM says person
    heads = {"arm_loc": np.zeros((1, n * 4)), "arm_conf": arm_conf[None], "odm_loc": np.zeros((1, n * 4)),
             "odm_conf": odm_conf[None]}
    assert refinedet.decode(heads, (480, 640)) == []


def test_hourglass_peak_maps_to_coco_joint():
    hm = np.zeros((64, 64, 16), np.float32)
    hm[40, 20, 15] = 0.4                                               # MPII 15 = left wrist
    kp = hourglass.decode(hm, score_gain=2.0)
    assert kp[9, 2] == pytest.approx(0.8)                              # COCO 9 = left wrist
    assert kp[9, 0] == pytest.approx(20.5 * 4) and kp[9, 1] == pytest.approx(40.5 * 4)
    assert (kp[:5, 2] == 0).all()                                      # no face points in MPII


def test_orientation_labels():
    assert orientation.decode([0, 0, 5, 0])[0] == "front"
    assert orientation.decode([0, 0, 0, 5])[0] == "back"
    assert orientation.decode([5, 0, 0, 0]) == ("left", pytest.approx(np.exp(5) / (np.exp(5) + 3)))


def test_yolov5_style_box():
    """One OFA-YOLO cell: centre = (2*sigmoid - 0.5 + cell) / g, size = (2*sigmoid)^2 * anchor."""
    heads = [np.full((1, g, g, 255), -12.0, np.float32) for g in (20, 40, 80)]
    v = heads[0][0, 10, 10].reshape(3, 85)                             # 20x20 head, cell (10, 10), anchor 2
    v[2, :4] = 0.0                                                     # sigmoid 0.5: centre at cell + 0.5, size = anchor
    v[2, 4] = 8.0
    v[2, 5] = 8.0                                                      # COCO person
    heads[0][0, 10, 10] = v.reshape(-1)
    dets = yolo.decode(heads, (640, 640), 640, 0.5, 0.45, (yolo.COCO_PERSON,), yolo.ANCHORS, yolo.COCO_CLASSES, "v5")
    assert len(dets) == 1
    aw, ah = yolo.ANCHORS[8]                                           # coarsest grid uses anchors 6..8
    cx, cy = 10.5 / 20 * 640, 10.5 / 20 * 640
    assert np.allclose(dets[0].box, [cx - aw / 2, cy - ah / 2, cx + aw / 2, cy + ah / 2], atol=0.5)
