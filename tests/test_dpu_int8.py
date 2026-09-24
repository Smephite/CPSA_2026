"""Board backends with a fake vart runner: int8 I/O gives the same results as float I/O.

Tensor shapes, names and fix points are the ones the board reported for each model (probe, 2026-09-24).
"""
import numpy as np
import pytest

from utils import settings as gcfg
from utils.types import Frame
from VIDEO_pipeline import dpu


class Tensor:
    def __init__(self, name, dims, fix):
        self.name, self.dims, self._fix = name, dims, fix

    def has_attr(self, key):
        return key == "fix_point"

    def get_attr(self, key):
        return self._fix


class FakeRunner:
    """Returns fixed int8 outputs (dequantised when the caller uses float buffers), records the input."""

    def __init__(self, inp, outs, seed=0):
        rng = np.random.default_rng(seed)
        self.inputs = [Tensor("in", *inp)]
        self.outputs = [Tensor(n, d, f) for n, d, f in outs]
        self.q = [rng.integers(-128, 24, size=d, dtype=np.int8) for _, d, _ in outs]
        self.seen = None

    def get_input_tensors(self):
        return self.inputs

    def get_output_tensors(self):
        return self.outputs

    def execute_async(self, inp, out):
        self.seen = inp[0].copy()
        for buf, q, t in zip(out, self.q, self.outputs):
            buf[...] = q if buf.dtype == np.int8 else q.astype(np.float32) * 2.0 ** -t._fix
        return 1

    def wait(self, job):
        pass


YOLO3 = ((1, 416, 416, 3), 6), [("conv2d_59", (1, 13, 13, 75), 2), ("conv2d_67", (1, 26, 26, 75), 2),
                                  ("conv2d_75", (1, 52, 52, 75), 2)]
YOLO4 = ((1, 416, 416, 3), 7), [("layer138", (1, 52, 52, 255), 4), ("layer149", (1, 26, 26, 255), 4),
                                  ("layer160", (1, 13, 13, 255), 4)]
OFA = ((1, 640, 640, 3), 6), [("m240", (1, 80, 80, 255), 3), ("m241", (1, 40, 40, 255), 4), ("m242", (1, 20, 20, 255), 4)]
YOLO2 = ((1, 448, 448, 3), 7), [("layer30-conv_fixed", (1, 14, 14, 125), 4)]
REFINEDET = ((1, 360, 480, 3), -1), [("arm_conf_reshape_fix", (1, 12240, 2), 6), ("arm_loc_fixed", (1, 48960), 3),
                                      ("odm_conf_reshape_fix", (1, 12240, 2), 5), ("odm_loc_fixed", (1, 48960), 4)]
MOVENET = ((1, 192, 192, 3), 6), [("header_centers", (1, 48, 48, 1), 3), ("header_heatmaps", (1, 48, 48, 17), 3),
                                   ("header_offsets", (1, 48, 48, 34), 7), ("header_regs", (1, 48, 48, 34), 2)]
HOURGLASS = ((1, 256, 256, 3), 7), [("ConvNd_56_fixed", (1, 64, 64, 16), 7)]
ORIENT = ((1, 176, 80, 3), 5), [("classifier", (1, 4), 3)]

CASES = {
    "yolov3_voc": YOLO3, "yolov4_pruned": YOLO4, "ofa_yolo_05": OFA, "yolov2_voc_pruned": YOLO2,
    "refinedet_096": REFINEDET, "movenet": MOVENET, "hourglass": HOURGLASS, "orientation": ORIENT,
}


def run_model(name, int8, frame, box):
    (inp, outs) = CASES[name]
    runner = FakeRunner(inp, outs)
    cfg = gcfg._merge(gcfg.DEFAULTS, {"dpu": {"int8_io": int8}})
    model = dpu._make(name, runner, cfg)
    if hasattr(model, "detect"):
        out = [(d.box.round(3).tolist(), round(d.score, 5)) for d in model.detect(frame)]
    elif hasattr(model, "estimate"):
        out = model.estimate(frame, box).round(3).tolist()
    else:
        out = model.classify(frame, box)
    return out, runner.seen, model


@pytest.fixture(scope="module")
def frame():
    img = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return Frame(image=img, t=0.0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_int8_matches_float(name, frame):
    box = np.array([200.0, 60.0, 360.0, 420.0])
    out_q, seen_q, model_q = run_model(name, True, frame, box)
    out_f, seen_f, _ = run_model(name, False, frame, box)
    assert out_q == out_f                                          # same outputs -> identical results
    assert seen_q.dtype == np.int8 and seen_f.dtype == np.float32
    fix = model_q.runner.get_input_tensors()[0].get_attr("fix_point")
    ref = np.clip(np.round(seen_f.astype(np.float64) * 2.0 ** fix), -128, 127)
    assert np.abs(seen_q.astype(np.int16) - ref).max() <= 1        # int8 input = what VART would have made


@pytest.mark.parametrize("name", ["movenet", "hourglass", "orientation"])
@pytest.mark.parametrize("box", [[700.0, 50.0, 800.0, 400.0],      # right of the frame (live crash, 2026-09-24)
                                 [-300.0, 50.0, -100.0, 400.0],    # left of it
                                 [100.0, 500.0, 200.0, 600.0]])    # below it
def test_crop_models_survive_boxes_outside_the_frame(name, box, frame):
    out, seen, _ = run_model(name, True, frame, np.array(box))
    assert seen is None                                           # no DPU call for an empty crop
    if name == "orientation":
        assert out == (None, 0.0)
    else:
        assert np.asarray(out).shape == (17, 3) and not np.asarray(out)[:, 2].any()


def test_expand_box_clamps_both_edges():
    from utils.geometry import expand_box, visible_fraction
    b = expand_box(np.array([700.0, 50.0, 800.0, 400.0]), 0.25, 0.12, 640, 480)
    assert b[0] == b[2] == 640                                    # zero width, not negative
    assert visible_fraction([600, 0, 700, 100], 640, 480) == pytest.approx(0.4)
    assert visible_fraction([700, 0, 800, 100], 640, 480) == 0.0
