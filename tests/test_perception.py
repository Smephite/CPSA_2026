import numpy as np
import pytest

from VIDEO_pipeline.MOVENET import movenet
from VIDEO_pipeline.YOLO import yolo


def test_yolo_encode_decode_round_trip():
    boxes = [[40, 120, 140, 440], [420, 100, 560, 460]]
    dets = yolo.decode(yolo.encode_for_test(boxes, (480, 640)), (480, 640))
    got = sorted(d.box.tolist() for d in dets)
    assert len(got) == 2 and all(d.label == "person" for d in dets)
    np.testing.assert_allclose(got, boxes, atol=1.0)


def test_yolo_keeps_only_requested_classes():
    heads = yolo.encode_for_test([[40, 120, 140, 440]], (480, 640), cls=yolo.VOC_CLASSES.index("chair"))
    assert yolo.decode(heads, (480, 640)) == []


def test_yolo_letterbox_shape_and_range():
    x = yolo.letterbox(np.full((480, 640, 3), 255, np.uint8))
    assert x.shape == (1, 416, 416, 3) and x.dtype == np.float32
    assert x.min() == pytest.approx(128 / 255) and x.max() == 1.0


def _movenet_heads(joints_cell, center_cell, g=48, off=0.25):
    """Logit heads with one person: centre peak, a heatmap peak per joint, regs pointing at it, offsets."""
    heads = {"centers": np.full((g, g, 1), -8.0), "heatmaps": np.full((g, g, 17), -8.0),
             "regs": np.zeros((g, g, 34)), "offsets": np.full((g, g, 34), off)}
    cy, cx = center_cell
    heads["centers"][cy, cx, 0] = 8.0
    for k, (y, x) in enumerate(joints_cell):
        heads["heatmaps"][y, x, k] = 4.0
        heads["regs"][cy, cx, 2 * k], heads["regs"][cy, cx, 2 * k + 1] = x - cx, y - cy
    return heads


def test_movenet_decode_finds_joints():
    g, size = 48, 192
    joints = [(10 + k, 20 + (k % 5)) for k in range(17)]       # (y, x) cells
    kp = movenet.decode(_movenet_heads(joints, (18, 22)), np.ones((g, g)), size)
    expect = np.array([[(x + 0.25) / g * size, (y + 0.25) / g * size] for y, x in joints])
    np.testing.assert_allclose(kp[:, :2], expect, atol=1e-6)
    assert np.all(kp[:, 2] > 0.9)


def test_movenet_center_weight_picks_the_person():
    g, size = 48, 192
    a = _movenet_heads([(5, 5)] * 17, (5, 5))
    b = _movenet_heads([(40, 40)] * 17, (40, 40))
    heads = {k: np.maximum(a[k], b[k]) if k != "regs" else a[k] + b[k] for k in a}
    weight = np.zeros((g, g))
    weight[40, 40] = 1.0                                        # prior favours the second person
    kp = movenet.decode(heads, weight, size)
    np.testing.assert_allclose(kp[:, :2], (40.25 / g * size), atol=1e-6)


def test_movenet_to_frame_undoes_letterbox_and_crop():
    crop = np.zeros((300, 150, 3), np.uint8)
    _, k, ox, oy = movenet.preprocess(crop, 192, np.array([127.5] * 3, np.float32), np.array([1 / 127.5] * 3, np.float32))
    kp_model = np.array([[ox + 0.0, oy + 0.0, 1.0], [ox + 150 * k, oy + 300 * k, 1.0]] + [[0, 0, 0]] * 15)
    kp = movenet.to_frame(kp_model, k, ox, oy, 100, 50)
    np.testing.assert_allclose(kp[:2, :2], [[100, 50], [250, 350]], atol=1e-6)



def test_dpu_model_files_exist():
    """The board backend's model paths point at files in the repo (otherwise this only shows up on the board)."""
    import os

    from VIDEO_pipeline import dpu                    # pynq_dpu is imported lazily: safe on a laptop
    for path in list(dpu.MODELS.values()) + [dpu.MOVENET_PROTOTXT]:
        assert os.path.isfile(path), path


def test_yolov2_round_trip():
    from VIDEO_pipeline.YOLO import yolov2
    boxes = [(100, 40, 260, 420), (400, 120, 520, 400)]
    dets = yolov2.decode(yolov2.encode_for_test(boxes, (480, 640)), (480, 640))
    assert len(dets) == 2
    got = sorted(d.box.tolist() for d in dets)
    for g, b in zip(got, sorted(boxes)):
        assert np.allclose(g, b, atol=2.0), (g, b)
    assert all(d.label == "person" and d.score > 0.9 for d in dets)


def test_yolov2_ignores_other_classes():
    from VIDEO_pipeline.YOLO import yolov2
    head = yolov2.encode_for_test([(100, 40, 260, 420)], (480, 640), cls=11)      # dog
    assert yolov2.decode(head, (480, 640)) == []
