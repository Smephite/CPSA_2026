"""int8 DPU I/O: quantise uint8 images straight to the model's int8 input, dequantise int8 outputs.

Handing the runner float32 buffers makes VART convert every input and output on the ARM. Measured on the board
(2026-09-24): YOLOv3 75.5 -> 67.9 ms, YOLOv2 15.8 -> 9.4, RefineDet-0.96 17.9 -> 9.1, MoveNet 5.8 -> 2.8,
Hourglass 17.8 -> 15.1 ms per call with int8 buffers. Normalisation + quantisation is one per-channel lookup
table (cv2.LUT, exact): q = clip(round((pixel - mean) * scale * 2^fix_point), -128, 127).
"""
import cv2
import numpy as np


class Quantizer:
    """uint8 HWC image (in the model's channel order) -> int8 input for a tensor with `fix_point`."""

    def __init__(self, mean, scale, fix_point):
        mean = np.broadcast_to(np.asarray(mean, np.float64), (3,))
        scale = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        v = np.arange(256, dtype=np.float64)[:, None]
        q = np.clip(np.round((v - mean[None]) * scale[None] * 2.0 ** fix_point), -128, 127)
        self.lut = q.astype(np.int8).reshape(1, 256, 3)

    def __call__(self, img_u8):
        return cv2.LUT(img_u8, self.lut)


def dequantize(out_i8, fix_point):
    return out_i8.astype(np.float32) * np.float32(2.0 ** -fix_point)


def fix_point(tensor):
    return tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else 0
