"""YOLOv2 (Pascal VOC, 20 classes) pre- and post-processing, pure NumPy/OpenCV.

Model: Vitis AI 2.5 zoo `dk_yolov2_voc_448_448_0.77_7.82G` (xmodel `yolov2_voc_pruned_0_77`, KV260 build,
DPUCZDX8G_ISA1_B4096). Settings from its prototxt (beside this file): input 448x448, pixel / 256
(mean 0, scale 0.00390625), 5 anchors in grid units, 20 classes, conf 0.3, NMS 0.45.

One head: (1, 14, 14, 125) = 14 x 14 cells x 5 anchors x (tx, ty, tw, th, objectness, 20 class logits).
Box centre = (cell + sigmoid(t)) / 14, size = anchor * exp(t) / 14; score = sigmoid(obj) * softmax(class).
Darknet YOLOv2 was trained on plainly resized images (no letterbox), so boxes map back by scaling.

Measured on the board: 15.7 ms per call vs. 75.6 ms for YOLOv3-VOC (2026-09-24).
"""
import cv2
import numpy as np

from utils.types import Detection
from VIDEO_pipeline.YOLO.yolo import PERSON, VOC_CLASSES

ANCHORS = np.array([1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
                   np.float32).reshape(5, 2)          # (w, h) in grid cells
NUM_CLASSES = 20


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


SCALE = 1.0 / 256.0          # prototxt: mean 0, scale 0.00390625


def preprocess_u8(image_bgr, size=448):
    """BGR image -> (size, size, 3) uint8 RGB (plain resize, no letterbox)."""
    return cv2.cvtColor(cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def preprocess(image_bgr, size=448):
    """BGR image -> (1, size, size, 3) float32 RGB, pixel / 256 (plain resize, no letterbox)."""
    return (preprocess_u8(image_bgr, size).astype(np.float32) * SCALE)[None]


def decode(output, image_hw, score_thresh=0.3, nms_iou=0.45, classes=(PERSON,)):
    """(1, g, g, 125) head -> list of Detection for `classes`, image pixels."""
    ih, iw = image_hw
    g = output.shape[1]
    p = output.reshape(g, g, len(ANCHORS), 5 + NUM_CLASSES)
    obj = _sigmoid(p[..., 4])
    gy, gx, a = np.nonzero(obj >= score_thresh)          # score = obj * p(class) <= obj
    if len(a) == 0:
        return []
    c = p[gy, gx, a]
    logits = c[:, 5:] - c[:, 5:].max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    scores = obj[gy, gx, a][:, None] * probs
    bx = (_sigmoid(c[:, 0]) + gx) / g
    by = (_sigmoid(c[:, 1]) + gy) / g
    bw = np.exp(c[:, 2]) * ANCHORS[a, 0] / g
    bh = np.exp(c[:, 3]) * ANCHORS[a, 1] / g
    boxes = np.stack([(bx - bw / 2) * iw, (by - bh / 2) * ih, (bx + bw / 2) * iw, (by + bh / 2) * ih], 1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, iw)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ih)

    dets = []
    for cls in classes:
        s = scores[:, cls]
        m = s >= score_thresh
        if not m.any():
            continue
        b, s = boxes[m], s[m]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).tolist()
        for i in np.asarray(cv2.dnn.NMSBoxes(xywh, s.tolist(), score_thresh, nms_iou)).flatten():
            dets.append(Detection(box=b[i].astype(np.float64), score=float(s[i]), label=VOC_CLASSES[cls]))
    return dets


def encode_for_test(boxes_xyxy, image_hw, grid=14, cls=PERSON, logit=6.0):
    """Inverse of `decode` for unit tests: a head tensor that decodes to the given boxes."""
    ih, iw = image_hw
    head = np.full((1, grid, grid, len(ANCHORS) * (5 + NUM_CLASSES)), -12.0, np.float32)
    for (x1, y1, x2, y2) in boxes_xyxy:
        bx, by = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
        bw, bh = (x2 - x1) / iw, (y2 - y1) / ih
        gx, gy, a = int(bx * grid), int(by * grid), 4
        fx, fy = bx * grid - gx, by * grid - gy
        v = head[0, gy, gx].reshape(len(ANCHORS), 5 + NUM_CLASSES)
        v[a, 0], v[a, 1] = np.log(fx / (1 - fx)), np.log(fy / (1 - fy))
        v[a, 2] = np.log(bw * grid / ANCHORS[a, 0])
        v[a, 3] = np.log(bh * grid / ANCHORS[a, 1])
        v[a, 4] = logit
        v[a, 5:] = -logit
        v[a, 5 + cls] = logit
        head[0, gy, gx] = v.reshape(-1)
    return head
