"""Guardian Node: infrastructure-side safety supervisor for robots working among people.

Pipeline (one process, one camera):

    beacon gate -> camera -> person detector (YOLO, DPU) -> tracker + roles
                -> pose on person crops (MoveNet, DPU, rate scaled by distance band)
                -> predictor (joint time series, look-ahead) -> rule engine -> decision latch
                -> outputs: audio + robot link + event log
                -> views: camera overlay | virtual scene (avatars on the joints, predicted dangers)

Everything except `guardian.perception.dpu` is plain numpy/OpenCV and runs on a laptop, driven by the
synthetic demo scenario (`guardian.scenario`) or a webcam with the replay/CPU-free backends.
"""
