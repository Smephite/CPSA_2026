"""Person-orientation classifier pre- and post-processing, pure NumPy/OpenCV.

Model: Vitis AI 2.5 zoo `pt_person-orientation_224_112_558M` (xmodel `person-orientation_pruned_558m_pt`, KV260
build). Input (1, 176, 80, 3) person crop, RGB, (pixel - (103.5, 116.3, 123.6)) * (0.017124, 0.017507, 0.017429)
(zoo prototxt). Output: 4 logits in the Vitis AI Library's label order (orien_label.txt): Left, Right, Front, Back.

Front / Back are relative to the camera. Whether "Left" means facing image-left or the person's own left is not
documented: `swap_left_right` in the settings flips it, check it once on the board with someone in profile.
"""
import cv2
import numpy as np

LABELS = ("left", "right", "front", "back")
INPUT_HW = (176, 80)
MEAN = np.array([103.5, 116.3, 123.6], np.float32)
SCALE = np.array([0.017124, 0.017507, 0.017429], np.float32)


def preprocess_u8(crop_bgr):
    """Crop -> (176, 80, 3) uint8 RGB (geometry only)."""
    return cv2.cvtColor(cv2.resize(crop_bgr, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_BGR2RGB)


def preprocess(crop_bgr):
    return (preprocess_u8(crop_bgr).astype(np.float32) - MEAN) * SCALE


def decode(logits):
    """4 logits -> (label, probability)."""
    z = np.asarray(logits, np.float64).reshape(-1)
    p = np.exp(z - z.max())
    p /= p.sum()
    i = int(np.argmax(p))
    return LABELS[i], float(p[i])
