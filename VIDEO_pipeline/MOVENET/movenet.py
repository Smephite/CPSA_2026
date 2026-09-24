"""MoveNet Lightning (17 COCO keypoints) pre-processing and single-pose decoding, pure numpy/OpenCV.

As specified by the Vitis AI 2.5 model zoo (VIDEO_pipeline/MOVENET/movenet_ntd_pt.prototxt): letterbox
192x192 padded with the mean colour, RGB, (pixel - 127.5) / 127.5, then MoveNet's single-pose decoder:
person centre from the `centers` head weighted by the 48x48 `center_weight` prior -> regressed joint positions
(`regs`) -> best heatmap cell near each -> sub-cell `offsets`.

Upstream CPSA_2026 feeds [0, 1] pixels and takes a per-joint argmax; that is what made its poses unusable.
"""
import re

import cv2
import numpy as np

HEADS = ("centers", "heatmaps", "regs", "offsets")


def load_prototxt(path):
    """-> (mean (3,), scale (3,), center_weight (48, 48), conf_threshold)."""
    txt = open(path).read()
    mean = np.array([float(v) for v in re.findall(r"mean:\s*([-\d.eE]+)", txt)], np.float32)
    scale = np.array([float(v) for v in re.findall(r"scale:\s*([-\d.eE]+)", txt)], np.float32)
    cw = np.array([float(v) for v in re.findall(r"center_weight:\s*([-\d.eE]+)", txt)], np.float32)
    thr = re.search(r"conf_threshold:\s*([-\d.eE]+)", txt)
    side = int(round(np.sqrt(cw.size)))
    return mean, scale, cw.reshape(side, side), float(thr.group(1)) if thr else 0.1


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(image_bgr, size, mean, scale):
    """Letterbox to size x size -> (normalised float32 HWC RGB, k, ox, oy) to map keypoints back."""
    canvas, k, ox, oy = preprocess_u8(image_bgr, size, mean)
    return (canvas.astype(np.float32) - mean) * scale, k, ox, oy


def preprocess_u8(image_bgr, size, mean):
    """Letterbox to size x size, padded with the mean colour -> (uint8 HWC RGB, k, ox, oy) (geometry only)."""
    ih, iw = image_bgr.shape[:2]
    k = min(size / iw, size / ih)
    nw, nh = max(1, int(round(iw * k))), max(1, int(round(ih * k)))
    canvas = np.full((size, size, 3), mean.astype(np.uint8), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), k, ox, oy


def decode(heads, center_weight, size):
    """heads: name -> (48, 48, C) float logits/values -> (17, 3) (x, y, score) in model-input pixels."""
    hm = _sigmoid(heads["heatmaps"])                       # (g, g, 17)
    ct = _sigmoid(heads["centers"][..., 0])                # (g, g)
    regs, offs = heads["regs"], heads["offsets"]           # (g, g, 34): [2k] = x, [2k+1] = y
    g = hm.shape[0]
    cy, cx = np.unravel_index(np.argmax(ct * center_weight), ct.shape)
    reg_x = np.clip((regs[cy, cx, 0::2] + cx + 0.5).astype(np.int32), 0, g - 1)
    reg_y = np.clip((regs[cy, cx, 1::2] + cy + 0.5).astype(np.int32), 0, g - 1)
    ys, xs = np.mgrid[0:g, 0:g]
    dist = np.sqrt((xs[..., None] - reg_x) ** 2 + (ys[..., None] - reg_y) ** 2) + 1.8
    flat = (hm / dist).reshape(-1, hm.shape[2]).argmax(0)  # best cell per joint near its regressed spot
    py, px = np.unravel_index(flat, (g, g))
    k = np.arange(hm.shape[2])
    score = hm[py, px, k]
    x = (px + offs[py, px, 2 * k]) / g * size
    y = (py + offs[py, px, 2 * k + 1]) / g * size
    return np.stack([x, y, score], axis=1)


def to_frame(kp_model, k, ox, oy, crop_x0, crop_y0):
    """Keypoints in model-input pixels -> frame pixels."""
    kp = kp_model.copy()
    kp[:, 0] = (kp[:, 0] - ox) / k + crop_x0
    kp[:, 1] = (kp[:, 1] - oy) / k + crop_y0
    return kp
