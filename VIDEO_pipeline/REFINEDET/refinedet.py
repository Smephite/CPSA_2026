"""RefineDet pedestrian detector (persons only) pre- and post-processing, pure NumPy/OpenCV.

Models: Vitis AI 2.5 zoo `cf_refinedet_coco_360_480_{0.8,0.92,0.96}` (Caffe, trained on COCO persons), KV260 builds.
Settings from the zoo prototxt: input 360x480 (HxW, same 4:3 aspect as the camera: plain resize), BGR minus
(104, 117, 123), 2 classes (background, person), conf 0.5, NMS 0.4.

Two-step SSD decoding over 12,240 priors (4 layers x 3 aspect ratios):
    ARM (anchor refinement): arm_loc refines each prior; arm_conf rejects anchors that are surely background
    ODM (object detection):  odm_loc refines the refined anchor again; odm_conf is the person score
The DPU returns logits: softmax is done here (it was a CPU subgraph in the xmodel).
"""
import cv2
import numpy as np

from utils.types import Detection

INPUT_HW = (360, 480)
MEAN_BGR = np.array([104.0, 117.0, 123.0], np.float32)
VARIANCES = np.array([0.1, 0.1, 0.2, 0.2], np.float32)
# (layer_w, layer_h, min_size, step) from the prototxt; aspect ratios [1, 2, 1/2] per cell (flip)
PRIOR_LAYERS = [(64, 48, 32.0, 8.0), (32, 24, 64.0, 16.0), (16, 12, 128.0, 32.0), (8, 6, 256.0, 64.0)]
ARM_NEGATIVE = 0.99                 # anchors with background probability above this are dropped (RefineDet paper)


def priors(input_hw=INPUT_HW):
    """(N, 4) priors as normalised (cx, cy, w, h), in the Caffe PriorBox order: per cell [ar 1, ar 2, ar 1/2]."""
    ih, iw = input_hw
    out = []
    for lw, lh, s, step in PRIOR_LAYERS:
        ys, xs = np.mgrid[0:lh, 0:lw]
        cx = ((xs + 0.5) * step / iw).reshape(-1, 1)
        cy = ((ys + 0.5) * step / ih).reshape(-1, 1)
        r = np.sqrt(2.0)
        sizes = [(s, s), (s * r, s / r), (s / r, s * r)]
        cell = np.concatenate([np.hstack([cx, cy, np.full_like(cx, w / iw), np.full_like(cx, h / ih)])[:, None, :]
                               for w, h in sizes], axis=1)
        out.append(cell.reshape(-1, 4))
    return np.concatenate(out).astype(np.float32)


PRIORS = priors()


def preprocess_u8(image_bgr):
    """BGR image -> (360, 480, 3) uint8 BGR (plain resize: same 4:3 aspect as the camera)."""
    return cv2.resize(image_bgr, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)


def preprocess(image_bgr):
    """BGR image -> (1, 360, 480, 3) float32 BGR minus the mean."""
    return (preprocess_u8(image_bgr).astype(np.float32) - MEAN_BGR)[None]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _apply(loc, anchors):
    """SSD centre-size decoding of (N, 4) offsets against (N, 4) anchors (cx, cy, w, h)."""
    cx = anchors[:, 0] + loc[:, 0] * VARIANCES[0] * anchors[:, 2]
    cy = anchors[:, 1] + loc[:, 1] * VARIANCES[1] * anchors[:, 3]
    w = anchors[:, 2] * np.exp(loc[:, 2] * VARIANCES[2])
    h = anchors[:, 3] * np.exp(loc[:, 3] * VARIANCES[3])
    return np.stack([cx, cy, w, h], axis=1)


def decode(heads, image_hw, score_thresh=0.5, nms_iou=0.4):
    """heads: name -> array for arm_loc, arm_conf, odm_loc, odm_conf (batch dim allowed) -> [Detection] in image px."""
    ih, iw = image_hw
    arm_loc = heads["arm_loc"].reshape(-1, 4)
    arm_conf = _softmax(heads["arm_conf"].reshape(-1, 2))
    odm_loc = heads["odm_loc"].reshape(-1, 4)
    odm_conf = _softmax(heads["odm_conf"].reshape(-1, 2))
    keep = (arm_conf[:, 0] <= ARM_NEGATIVE) & (odm_conf[:, 1] >= score_thresh)
    if not keep.any():
        return []
    refined = _apply(arm_loc[keep], PRIORS[keep])
    boxes = _apply(odm_loc[keep], refined)
    scores = odm_conf[keep, 1]
    x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2) * iw, 0, iw)
    y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2) * ih, 0, ih)
    x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2) * iw, 0, iw)
    y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2) * ih, 0, ih)
    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1).tolist()
    dets = []
    for i in np.asarray(cv2.dnn.NMSBoxes(xywh, scores.tolist(), score_thresh, nms_iou)).flatten():
        dets.append(Detection(box=np.array([x1[i], y1[i], x2[i], y2[i]], np.float64), score=float(scores[i])))
    return dets
