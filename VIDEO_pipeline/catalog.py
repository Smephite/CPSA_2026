"""Model catalog: every DPU model Guardian can use, in one place.

The board backends (dpu.py), the laptop stand-ins (replay.py) and the web UI's model choices all read this table.
Adding a model = one entry here + its decoder in its folder + a class in dpu.py.

All models are Vitis AI 2.5 zoo KV260 builds for DPUCZDX8G_ISA1_B4096 (fingerprint 0x101000016010407), checked on the
board on 2026-09-24: all of them load together on one overlay (≈ 400 MB). `ms` = DPU time per call measured there.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str          # config / UI name
    kind: str          # detector | pose | orientation
    file: str          # xmodel, relative to VIDEO_pipeline/
    ms: float          # DPU time per call on the KV260 (measured)
    zoo: str           # Vitis AI 2.5 zoo id
    note: str


CATALOG = {m.name: m for m in [
    # ---- person detectors (frame -> person boxes)
    ModelInfo("yolov2_voc_pruned", "detector", "YOLO/yolov2_voc_pruned_0_77.xmodel", 15.7,
              "dk_yolov2_voc_448_448_0.77_7.82G", "YOLOv2 VOC, 7.8 GOPs: cheapest YOLO"),
    ModelInfo("refinedet_096", "detector", "REFINEDET/refinedet_pruned_0_96.xmodel", 18.0,
              "cf_refinedet_coco_360_480_0.96_5.08G", "RefineDet persons-only, 5.1 GOPs"),
    ModelInfo("refinedet_092", "detector", "REFINEDET/refinedet_pruned_0_92.xmodel", 22.7,
              "cf_refinedet_coco_360_480_0.92_10.10G", "RefineDet persons-only, 10.1 GOPs"),
    ModelInfo("refinedet_08", "detector", "REFINEDET/refinedet_pruned_0_8.xmodel", 40.1,
              "cf_refinedet_coco_360_480_0.8_25G", "RefineDet persons-only, 25 GOPs: most accurate RefineDet"),
    ModelInfo("yolov4_pruned", "detector", "YOLO/yolov4_leaky_spp_m_pruned_0_36.xmodel", 59.8,
              "dk_yolov4_coco_416_416_0.36_38.2G", "YOLOv4 COCO, 38 GOPs"),
    ModelInfo("ofa_yolo_05", "detector", "YOLO/ofa_yolo_pruned_0_50_pt.xmodel", 61.6,
              "pt_OFA-yolo_coco_640_640_0.5_24.62G", "OFA-YOLO (YOLOv5-style) COCO, 640 input"),
    ModelInfo("yolov3_voc", "detector", "YOLO/pynqdpu.tf_yolov3_voc.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel", 75.6,
              "tf_yolov3_voc_416_416_65.63G", "YOLOv3 VOC, 65.6 GOPs: the original"),
    # ---- pose estimators (crop -> 17 COCO keypoints)
    ModelInfo("movenet", "pose", "MOVENET/kv260_MoveNet_int.xmodel", 5.8,
              "pt_movenet_coco_192_192_0.5G", "MoveNet Lightning: 17 joints incl. face"),
    ModelInfo("hourglass", "pose", "HOURGLASS/hourglass-pe_mpii.xmodel", 17.7,
              "cf_hourglass_mpii_256_256_10.2G", "Hourglass MPII: 16 joints, no face; trained incl. lying people"),
    # ---- auxiliary
    ModelInfo("orientation", "orientation", "ORIENTATION/person-orientation_pruned_558m_pt.xmodel", 1.8,
              "pt_person-orientation_224_112_558M", "Which way a person faces: left / right / front / back"),
]}


def names(kind):
    """Model names of one kind, cheapest first."""
    return [m.name for m in sorted(CATALOG.values(), key=lambda m: m.ms) if m.kind == kind]


def detector_choices():
    """UI choices: every detector alone, plus cheap -> full pairs where the first is at least 2x faster."""
    dets = names("detector")
    pairs = [f"{a} > {b}" for a in dets for b in dets if CATALOG[a].ms * 2 <= CATALOG[b].ms]
    return dets + pairs


def pose_choices():
    poses = names("pose")
    return poses + [f"{a} > {b}" for a in poses for b in poses if a != b]
