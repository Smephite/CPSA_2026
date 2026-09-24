"""Hourglass (MPII, 16 joints) pose pre- and post-processing, pure NumPy/OpenCV.

Model: Vitis AI 2.5 zoo `cf_hourglass_mpii_256_256_10.2G` (xmodel `hourglass-pe_mpii`, KV260 build).
Settings from the zoo prototxt: input 256x256, (pixel - (112.3, 113.2, 110.3)) / 255, RGB (checked on the board:
RGB gives higher joint responses than BGR). Output (1, 64, 64, 16) heatmaps.

Trained on MPII, which includes people lying, sitting and bending: the escalation stage for poses MoveNet
handles badly. MPII has no eye/ear/nose joints, so the COCO face points come back with score 0 (the facing
rule then needs the orientation model or MoveNet). Heatmap maxima are not probabilities (0.1 - 0.6 on a clear
person): `score_gain` scales them into the 0..1 range the rules use; calibrate it on real footage.
"""
import cv2
import numpy as np

SIZE = 256
MEAN_RGB = np.array([112.302, 113.22, 110.3385], np.float32)
# MPII joint order -> COCO-17 index (-1 = no COCO equivalent: pelvis, thorax, upper neck, head top)
MPII_TO_COCO = {0: 16, 1: 14, 2: 12, 3: 11, 4: 13, 5: 15, 10: 10, 11: 8, 12: 6, 13: 5, 14: 7, 15: 9}


def preprocess(crop_bgr):
    """Crop -> (x (256, 256, 3) float32, k, ox, oy): letterboxed with black, RGB, normalised."""
    ih, iw = crop_bgr.shape[:2]
    k = min(SIZE / iw, SIZE / ih)
    nw, nh = max(1, int(round(iw * k))), max(1, int(round(ih * k)))
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    ox, oy = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (rgb - MEAN_RGB) / 255.0, k, ox, oy


def decode(heatmaps, score_gain=2.0):
    """(64, 64, 16) heatmaps -> (17, 3) COCO keypoints (x, y, score) in model-input pixels (256 x 256).

    Peak per joint, moved a quarter cell towards the higher neighbour (standard hourglass refinement).
    """
    hm = heatmaps.reshape(heatmaps.shape[-3], heatmaps.shape[-2], heatmaps.shape[-1])
    g_h, g_w, n = hm.shape
    kp = np.zeros((17, 3), np.float32)
    for j, c in MPII_TO_COCO.items():
        m = hm[:, :, j]
        y, x = np.unravel_index(np.argmax(m), m.shape)
        px, py = float(x), float(y)
        if 0 < x < g_w - 1:
            px += 0.25 * np.sign(m[y, x + 1] - m[y, x - 1])
        if 0 < y < g_h - 1:
            py += 0.25 * np.sign(m[y + 1, x] - m[y - 1, x])
        kp[c] = [(px + 0.5) * SIZE / g_w, (py + 0.5) * SIZE / g_h, min(1.0, float(m[y, x]) * score_gain)]
    return kp


def to_frame(kp_model, k, ox, oy, crop_x0, crop_y0):
    kp = kp_model.copy()
    kp[:, 0] = (kp[:, 0] - ox) / k + crop_x0
    kp[:, 1] = (kp[:, 1] - oy) / k + crop_y0
    return kp
