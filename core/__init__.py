"""core: decides NONE / WARN / STOP. The only place a decision is made.

Interface:
    GuardianNode(cfg, clock, camera, beacon, power, detectors, poses, actuators, events)
    GuardianNode.step() -> Snapshot       one frame: sense -> perceive -> predict -> decide -> act
    GuardianNode.reconfigure()            re-read cfg after live tuning

    guardian_node.py  the per-frame loop and IDLE / DETECT / TRACK states (upstream: EventDispatcher)
    scheduler.py      distance band -> which detector / pose model runs, and how often
    predictor.py      pose time series -> joint velocities, future poses, torso acceleration
    rules.py          the five geometric rules (reach, from_behind, down, pinned, overhead), combine, DecisionLatch

Inputs come from sensors and VIDEO_pipeline; outputs go to actuators (Decision) and dashboard (Snapshot).
Rules are deterministic geometry on keypoints: no model output reaches an actuator without passing rules.py.
"""
