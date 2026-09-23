"""Perception backends behind two small interfaces:

    Detector.detect(frame: Frame) -> list[Detection]          persons only, frame pixels
    PoseEstimator.estimate(frame: Frame, box) -> (17, 3)      keypoints (x, y, score) in frame pixels

Implementations:
    dpu.py     YOLOv3-VOC and MoveNet on the KV260 DPU (PYNQ-DPU 2.5 runners); board only
    replay.py  reads the ground truth of synthetic frames (guardian.scenario); runs anywhere
The pure decode math (yolo.py, movenet.py) has no DPU dependency and is unit-tested locally.
"""
