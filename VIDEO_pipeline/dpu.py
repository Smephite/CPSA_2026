"""DPU backends for the KV260 (PYNQ-DPU 2.5). Board only: imports pynq_dpu lazily.

`DpuModels` programs the FPGA once with PYNQ-DPU's dpu.bit (the DPUCZDX8G_ISA1_B4096 every catalog model was
compiled for) and creates one vart runner per xmodel. Every model in `catalog.CATALOG` is loaded at startup
(≈ 400 MB), so the web UI can switch detectors, pose models and cascades while the node runs.

One class per model family, all with the same interface as their kind (VIDEO_pipeline/__init__.py):
    detectors     detect(frame) -> [Detection]
    pose models   estimate(frame, box) -> (17, 3) keypoints in frame pixels
    orientation   classify(frame, box) -> (label, probability)
Each keeps `times` (pre / dpu / post ms of its last call) for the dashboard.

Gotcha (found on the board): a runner points into its xir graph and `load_model()` replaces `overlay.graph`, so
every graph is kept referenced here; otherwise the earlier runners segfault on execute. Runners are created
directly from each graph's DPU subgraph with vart, which also works for models PYNQ's load_model rejects.
"""
import os
import time

import numpy as np

from utils.geometry import expand_box
from VIDEO_pipeline.catalog import CATALOG
from VIDEO_pipeline.quant import Quantizer, dequantize, fix_point
from VIDEO_pipeline.HOURGLASS import hourglass
from VIDEO_pipeline.MOVENET import movenet as mn
from VIDEO_pipeline.ORIENTATION import orientation
from VIDEO_pipeline.REFINEDET import refinedet
from VIDEO_pipeline.YOLO import yolo, yolov2

HERE = os.path.dirname(os.path.abspath(__file__))          # VIDEO_pipeline/
MODELS = {name: os.path.join(HERE, m.file) for name, m in CATALOG.items()}
MOVENET_PROTOTXT = os.path.join(HERE, "MOVENET", "movenet_ntd_pt.prototxt")


class DpuModels:
    """The overlay plus one runner per model name."""

    def __init__(self, bitfile="dpu.bit"):
        from pynq_dpu import DpuOverlay        # noqa: PLC0415 (board only)
        t0 = time.perf_counter()
        self.overlay = DpuOverlay(bitfile)
        self.load_s = time.perf_counter() - t0
        self._graphs = []
        self.runners = {}

    def runner(self, name):
        if name not in self.runners:
            import vart                         # noqa: PLC0415 (board only)
            import xir                          # noqa: PLC0415
            graph = xir.Graph.deserialize(MODELS[name])
            subs = [s for s in graph.get_root_subgraph().toposort_child_subgraph()
                    if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
            if len(subs) != 1:
                raise ValueError(f"{name}: expected one DPU subgraph, found {len(subs)}")
            self._graphs.append(graph)
            self.runners[name] = vart.Runner.create_runner(subs[0], "run")
        return self.runners[name]

    def close(self):
        self.runners.clear()
        self._graphs.clear()
        self.overlay = None


class _DpuModel:
    """Runner with its I/O buffers and per-call timing.

    int8=True: int8 buffers; `prepare(u8)` quantises with a lookup table and outputs are dequantised here.
    int8=False: float32 buffers; VART converts on the ARM (slower, kept as a fallback: dpu.int8_io).
    `mean` / `scale` are the model's normalisation, applied to the uint8 image in its channel order.
    """

    def __init__(self, runner, mean=0.0, scale=1.0, int8=True):
        self.runner, self.int8 = runner, int8
        it = runner.get_input_tensors()[0]
        self.shape_in = tuple(it.dims)
        tensors = runner.get_output_tensors()
        self.out_names = [t.name for t in tensors]
        self.out_fix = [fix_point(t) for t in tensors]
        dtype = np.int8 if int8 else np.float32
        self.inp = [np.empty(self.shape_in, dtype, order="C")]
        self.out = [np.empty(tuple(t.dims), dtype, order="C") for t in tensors]
        self.mean = np.broadcast_to(np.asarray(mean, np.float32), (3,)).copy()
        self.scale = np.broadcast_to(np.asarray(scale, np.float32), (3,)).copy()
        self.quant = Quantizer(self.mean, self.scale, fix_point(it)) if int8 else None
        self.times = {}

    def prepare(self, img_u8):
        """uint8 HWC image -> the input tensor's content (int8 via lookup, or normalised float32)."""
        if self.int8:
            return self.quant(img_u8)
        return (img_u8.astype(np.float32) - self.mean) * self.scale

    def run(self, x, prefix):
        t1 = time.perf_counter()
        self.inp[0][0, ...] = x
        self.runner.wait(self.runner.execute_async(self.inp, self.out))
        self.times[prefix + "_dpu"] = (time.perf_counter() - t1) * 1e3
        if self.int8:
            return [dequantize(o, f) for o, f in zip(self.out, self.out_fix)]
        return self.out


MIN_CROP_PX = 8                          # smaller crops (e.g. a box extrapolated off the frame) are not run


def _crop(frame, box, margin):
    """-> (crop, x0, y0), or (None, 0, 0) when the expanded box has (almost) no area inside the frame."""
    H, W = frame.image.shape[:2]
    x0, y0, x1, y1 = expand_box(box, margin[0], margin[1], W, H).astype(int)
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None, 0, 0
    return frame.image[y0:y1, x0:x1], x0, y0


# ---------------------------------------------------------------- detectors


class YoloDetector(_DpuModel):
    """YOLOv3-family detectors: YOLOv3-VOC, YOLOv4-COCO, OFA-YOLO (v5 style). Letterboxed RGB input."""

    VARIANTS = {   # name: (fill, scale, anchors, class names, person index, style)
        "yolov3_voc": (128, 1 / 255.0, yolo.ANCHORS, yolo.VOC_CLASSES, yolo.PERSON, "v3"),
        "yolov4_pruned": (128, 1 / 256.0, yolo.YOLOV4_ANCHORS, yolo.COCO_CLASSES, yolo.COCO_PERSON, "v3"),
        "ofa_yolo_05": (114, 0.00392156, yolo.ANCHORS, yolo.COCO_CLASSES, yolo.COCO_PERSON, "v5"),
    }

    def __init__(self, runner, name, score_thresh=0.5, int8=True):
        fill, scale, self.anchors, self.classes, self.person, self.style = self.VARIANTS[name]
        super().__init__(runner, 0.0, scale, int8)
        self.name, self.fill = name, fill
        self.size, self.score_thresh = self.shape_in[1], score_thresh

    def detect(self, frame):
        t0 = time.perf_counter()
        x = self.prepare(yolo.letterbox_u8(frame.image, self.size, self.fill))
        self.times = {"det_pre": (time.perf_counter() - t0) * 1e3}
        out = self.run(x, "det")
        t2 = time.perf_counter()
        dets = yolo.decode(out, frame.image.shape[:2], self.size, self.score_thresh, 0.45, (self.person,),
                           self.anchors, self.classes, self.style)
        for d in dets:
            d.label = "person"
        self.times["det_post"] = (time.perf_counter() - t2) * 1e3
        return dets


class YoloV2Detector(_DpuModel):
    name = "yolov2_voc_pruned"

    def __init__(self, runner, score_thresh=0.3, int8=True):
        super().__init__(runner, 0.0, yolov2.SCALE, int8)
        self.size, self.score_thresh = self.shape_in[1], score_thresh

    def detect(self, frame):
        t0 = time.perf_counter()
        x = self.prepare(yolov2.preprocess_u8(frame.image, self.size))
        self.times = {"det_pre": (time.perf_counter() - t0) * 1e3}
        out = self.run(x, "det")
        t2 = time.perf_counter()
        dets = yolov2.decode(out[0], frame.image.shape[:2], self.score_thresh)
        self.times["det_post"] = (time.perf_counter() - t2) * 1e3
        return dets


class RefineDetDetector(_DpuModel):
    """RefineDet pedestrian (persons only). Heads looked up by name: the runner's output order is not fixed."""

    def __init__(self, runner, name, score_thresh=0.5, int8=True):
        super().__init__(runner, refinedet.MEAN_BGR, 1.0, int8)
        self.name, self.score_thresh = name, score_thresh
        self.idx = {h: next(i for i, n in enumerate(self.out_names) if n.startswith(h))
                    for h in ("arm_loc", "arm_conf", "odm_loc", "odm_conf")}

    def detect(self, frame):
        t0 = time.perf_counter()
        x = self.prepare(refinedet.preprocess_u8(frame.image))
        self.times = {"det_pre": (time.perf_counter() - t0) * 1e3}
        out = self.run(x, "det")
        t2 = time.perf_counter()
        dets = refinedet.decode({h: out[i] for h, i in self.idx.items()}, frame.image.shape[:2], self.score_thresh)
        self.times["det_post"] = (time.perf_counter() - t2) * 1e3
        return dets


# ---------------------------------------------------------------- pose


class MoveNetPose(_DpuModel):
    name = "movenet"

    def __init__(self, runner, margin=(0.25, 0.12), prototxt=MOVENET_PROTOTXT, int8=True):
        mean, scale, self.center_weight, _ = mn.load_prototxt(prototxt)
        super().__init__(runner, mean, scale, int8)
        self.margin = margin
        self.size = self.shape_in[1]
        self.idx = {h: next(i for i, n in enumerate(self.out_names) if f"header_{h}" in n) for h in mn.HEADS}

    def estimate(self, frame, box):
        crop, x0, y0 = _crop(frame, box, self.margin)
        if crop is None:
            self.times = {}
            return np.zeros((17, 3), np.float32)             # nothing to see: every joint invisible
        t0 = time.perf_counter()
        u8, k, ox, oy = mn.preprocess_u8(crop, self.size, self.mean)
        x = self.prepare(u8)
        self.times = {"pose_pre": (time.perf_counter() - t0) * 1e3}
        out = self.run(x, "pose")
        t2 = time.perf_counter()
        heads = {h: out[i][0] for h, i in self.idx.items()}
        kp = mn.to_frame(mn.decode(heads, self.center_weight, self.size), k, ox, oy, x0, y0)
        self.times["pose_post"] = (time.perf_counter() - t2) * 1e3
        return kp


class HourglassPose(_DpuModel):
    name = "hourglass"

    def __init__(self, runner, margin=(0.25, 0.12), score_gain=2.0, int8=True):
        super().__init__(runner, hourglass.MEAN_RGB, hourglass.SCALE, int8)
        self.margin, self.score_gain = margin, score_gain

    def configure(self, cfg):
        self.score_gain = cfg["pose"]["hourglass_gain"]

    def estimate(self, frame, box):
        crop, x0, y0 = _crop(frame, box, self.margin)
        if crop is None:
            self.times = {}
            return np.zeros((17, 3), np.float32)
        t0 = time.perf_counter()
        u8, k, ox, oy = hourglass.preprocess_u8(crop)
        x = self.prepare(u8)
        self.times = {"pose_pre": (time.perf_counter() - t0) * 1e3}
        out = self.run(x, "pose")
        t2 = time.perf_counter()
        kp = hourglass.to_frame(hourglass.decode(out[0][0], self.score_gain), k, ox, oy, x0, y0)
        self.times["pose_post"] = (time.perf_counter() - t2) * 1e3
        return kp


# ---------------------------------------------------------------- orientation


class OrientationModel(_DpuModel):
    name = "orientation"

    def __init__(self, runner, margin=(0.05, 0.02), int8=True):
        super().__init__(runner, orientation.MEAN, orientation.SCALE, int8)
        self.margin = margin

    def classify(self, frame, box):
        crop, _, _ = _crop(frame, box, self.margin)
        if crop is None:
            self.times = {}
            return None, 0.0                                    # unknown
        out = self.run(self.prepare(orientation.preprocess_u8(crop)), "orient")
        return orientation.decode(out[0][0])


# ---------------------------------------------------------------- construction


def _make(name, runner, cfg):
    margin = (cfg["pose"]["crop_margin_x"], cfg["pose"]["crop_margin_y"])
    q = cfg["dpu"]["int8_io"]
    if name in YoloDetector.VARIANTS:
        return YoloDetector(runner, name, int8=q)
    if name == "yolov2_voc_pruned":
        return YoloV2Detector(runner, int8=q)
    if name.startswith("refinedet"):
        return RefineDetDetector(runner, name, int8=q)
    if name == "movenet":
        return MoveNetPose(runner, margin, int8=q)
    if name == "hourglass":
        return HourglassPose(runner, margin, cfg["pose"]["hourglass_gain"], int8=q)
    if name == "orientation":
        return OrientationModel(runner, int8=q)
    raise ValueError(f"no DPU backend for {name}")


def build(cfg):
    """-> (models, detectors {name: Detector}, poses {name: PoseEstimator}, orientation model or None).

    Loads every catalog model (they fit together), so any of them can be selected at runtime.
    """
    models = DpuModels()
    made = {name: _make(name, models.runner(name), cfg) for name in CATALOG}
    detectors = {n: m for n, m in made.items() if CATALOG[n].kind == "detector"}
    poses = {n: m for n, m in made.items() if CATALOG[n].kind == "pose"}
    return models, detectors, poses, made.get("orientation")
