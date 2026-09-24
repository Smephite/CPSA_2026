"""VIDEO_pipeline: who is in the frame, where, and in what pose. Perception only: no rules, no decisions.

Interfaces (models; one instance per model name):
    Detector       detect(frame: Frame) -> list[Detection]            persons, frame pixels
    PoseEstimator  estimate(frame: Frame, box) -> (17, 3) keypoints   single person inside box (+ margin)

    YOLO/yolo.py        YOLOv3-VOC pre/post-processing (pure NumPy/OpenCV) + the xmodel
    MOVENET/movenet.py  MoveNet pre/post-processing (pure) + the xmodel and the model-zoo prototxt
    dpu.py              board backends: DpuModels (PYNQ overlay + vart runners), YoloV3VocDetector, MoveNetPose
    replay.py           laptop stand-ins that read the scenario's ground truth (and worse "cheap" variants)
    cascade.py          DetectorCascade / PoseCascade: cheapest model first, escalate when unsure
    tracking.py         Tracker: persistent ids, velocities, robot/human roles  -> Track
    world.py            WorldModel: monocular 'fake 3D' (depth from box size, floor positions, gaps)

Add a model: implement Detector or PoseEstimator in dpu.py (board) and a stand-in in replay.py (laptop),
register it in dpu.DETECTORS / dpu.POSES, then name it in the band table (utils/settings.py).
"""
