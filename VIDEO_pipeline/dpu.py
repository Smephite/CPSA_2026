"""DPU backends for the KV260 (PYNQ-DPU 2.5). Board only: imports pynq_dpu lazily.

`DpuModels` programs the FPGA once with PYNQ-DPU's dpu.bit (the DPUCZDX8G_ISA1_B4096 all zoo models here were
compiled for) and creates one vart runner per xmodel. Upstream CPSA_2026 used `xmutil loadapp` + an xclbin instead;
that firmware package is not in the repo and the board is set up for PYNQ.

Gotcha (found on the board): a runner points into its xir graph and `load_model()` replaces `overlay.graph`,
so every graph is kept referenced here; otherwise the earlier runners segfault on execute.
"""
import os
import time

import numpy as np

from utils.geometry import expand_box
from VIDEO_pipeline.MOVENET import movenet as mn
from VIDEO_pipeline.YOLO import yolo

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {
    "yolov3_voc": os.path.join(REPO, "VIDEO_pipeline", "YOLO",
                               "pynqdpu.tf_yolov3_voc.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel"),
    "movenet": os.path.join(REPO, "VIDEO_pipeline", "MOVENET", "kv260_MoveNet_int.xmodel"),
}
MOVENET_PROTOTXT = os.path.join(REPO, "VIDEO_pipeline", "MOVENET", "movenet_ntd_pt.prototxt")


class DpuModels:
    def __init__(self, bitfile="dpu.bit"):
        from pynq_dpu import DpuOverlay        # noqa: PLC0415 (board only)
        t0 = time.perf_counter()
        self.overlay = DpuOverlay(bitfile)
        self.load_s = time.perf_counter() - t0
        self._graphs = []
        self.runners = {}

    def runner(self, name):
        if name not in self.runners:
            self.overlay.load_model(MODELS[name])
            self._graphs.append(self.overlay.graph)
            self.runners[name] = self.overlay.runner
        return self.runners[name]

    def close(self):
        self.runners.clear()
        self._graphs.clear()
        self.overlay = None


class YoloV3VocDetector:
    name = "yolov3_voc"

    def __init__(self, runner, score_thresh=0.5):
        self.runner = runner
        self.score_thresh = score_thresh
        self.size = tuple(runner.get_input_tensors()[0].dims)[1]
        self.out = [np.empty(tuple(t.dims), np.float32, order="C") for t in runner.get_output_tensors()]
        self.times = {}

    def detect(self, frame):
        t0 = time.perf_counter()
        x = np.ascontiguousarray(yolo.letterbox(frame.image, self.size))
        t1 = time.perf_counter()
        self.runner.wait(self.runner.execute_async([x], self.out))
        t2 = time.perf_counter()
        dets = yolo.decode(self.out, frame.image.shape[:2], self.size, self.score_thresh)
        self.times = {"det_pre": (t1 - t0) * 1e3, "det_dpu": (t2 - t1) * 1e3,
                      "det_post": (time.perf_counter() - t2) * 1e3}
        return dets


class MoveNetPose:
    def __init__(self, runner, prototxt=MOVENET_PROTOTXT, margin=(0.25, 0.12)):
        self.runner = runner
        self.mean, self.scale, self.center_weight, _ = mn.load_prototxt(prototxt)
        self.margin = margin
        it, ot = runner.get_input_tensors(), runner.get_output_tensors()
        self.shape_in = tuple(it[0].dims)
        self.size = self.shape_in[1]
        self.inp = [np.empty(self.shape_in, np.float32, order="C")]   # float in: vart applies fix_point
        self.out = [np.empty(tuple(t.dims), np.float32, order="C") for t in ot]
        names = [t.name for t in ot]
        self.idx = {h: next(i for i, n in enumerate(names) if f"header_{h}" in n) for h in mn.HEADS}
        self.times = {}

    def estimate(self, frame, box):
        H, W = frame.image.shape[:2]
        x0, y0, x1, y1 = expand_box(box, self.margin[0], self.margin[1], W, H).astype(int)
        crop = frame.image[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        t0 = time.perf_counter()
        x, k, ox, oy = mn.preprocess(crop, self.size, self.mean, self.scale)
        self.inp[0][0, ...] = x
        t1 = time.perf_counter()
        self.runner.wait(self.runner.execute_async(self.inp, self.out))
        t2 = time.perf_counter()
        heads = {h: self.out[i][0] for h, i in self.idx.items()}
        kp = mn.to_frame(mn.decode(heads, self.center_weight, self.size), k, ox, oy, x0, y0)
        self.times = {"pose_pre": (t1 - t0) * 1e3, "pose_dpu": (t2 - t1) * 1e3,
                      "pose_post": (time.perf_counter() - t2) * 1e3}
        return kp


DETECTORS = {"yolov3_voc": YoloV3VocDetector}


POSES = {"movenet": MoveNetPose}


def detector_names(cfg):
    """Every detector the band table uses (single names and cascade lists)."""
    names = set()
    for r in cfg["bands"]["rates"].values():
        names.update([r["detector"]] if isinstance(r["detector"], str) else r["detector"])
    return names


def build(cfg):
    """-> (models, {detector_name: Detector}, {pose_name: PoseEstimator}) for every model the config names."""
    names, pose_names = detector_names(cfg), set(cfg["pose"]["models"])
    unknown = sorted(names - set(DETECTORS)) + sorted(pose_names - set(POSES))
    if unknown:
        raise ValueError(f"no DPU backend implemented for {unknown} (have {sorted(DETECTORS) + sorted(POSES)})")
    models = DpuModels()
    detectors = {n: DETECTORS[n](models.runner(n)) for n in names}
    margin = (cfg["pose"]["crop_margin_x"], cfg["pose"]["crop_margin_y"])
    poses = {n: POSES[n](models.runner(n), margin=margin) for n in pose_names}
    return models, detectors, poses
