#!/usr/bin/env python3
"""Guardian Node: entry point. Builds the module boxes, wires them together, runs the loop.

    sensors/  ─► VIDEO_pipeline/ ─► core/ ─► actuators/     (README.md: "How it works")
                                      └────► dashboard/

Laptop (no FPGA):
    uv run python main.py                                # synthetic demo, real time, MJPEG on :8080
    uv run python main.py --fast --record out/demo.mp4   # render the demo to a video as fast as possible
Board (KV260, root, PYNQ environment: see run.sh):
    ./run.sh --source webcam --backend dpu --hdmi                 # live: webcam -> DPU -> HDMI (+ MJPEG)

Beacon: --beacon scenario (demo timeline) | always | manual (type 'b' + Enter to toggle).
"""
import argparse
import os
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from actuators.actuator_manager import ActuatorManager  # noqa: E402
from actuators.audio import AudioOut, NullAudio  # noqa: E402
from actuators.event_diary import EventDiary  # noqa: E402
from actuators.robot_link import LogRobotLink  # noqa: E402
from core.guardian_node import GuardianNode  # noqa: E402
from dashboard import tuning as gtune  # noqa: E402
from dashboard.sinks import DisplayPortSink, MjpegSink, VideoFileSink  # noqa: E402
from dashboard.views import Dashboard  # noqa: E402
from sensors.beacon import AlwaysBeacon, ManualBeacon, ScheduledBeacon  # noqa: E402
from sensors.camera import ScenarioCamera, WebcamCamera  # noqa: E402
from sensors.power import NullPower, board_power  # noqa: E402
from sensors.scenario import DemoScenario  # noqa: E402
from utils import settings as gcfg  # noqa: E402
from utils.clock import RealClock, SimClock  # noqa: E402
from utils.event_log import EventLog  # noqa: E402
from utils.types import NodeState  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["scenario", "webcam"], default="scenario")
    p.add_argument("--backend", choices=["replay", "dpu"], default=None,
                   help="perception backend (default: replay for scenario, dpu for webcam)")
    p.add_argument("--beacon", choices=["scenario", "always", "manual"], default=None,
                   help="default: scenario timeline for the scenario source, always for the webcam")
    p.add_argument("--fast", action="store_true", help="simulated clock: run the scenario as fast as possible")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds (scenario default: its length)")
    p.add_argument("--port", type=int, default=8080, help="MJPEG stream port, 0 = off (default 8080)")
    p.add_argument("--hdmi", action="store_true", help="dashboard full screen on the HDMI monitor (board)")
    p.add_argument("--record", help="write the dashboard to this .mp4")
    p.add_argument("--snapshots", help="save a dashboard PNG every N seconds into this folder (with --snapshot-every)")
    p.add_argument("--snapshot-every", type=float, default=2.0)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--power", action="store_true", help="read the board's power rails (pynq)")
    p.add_argument("--no-log", action="store_true", help="do not write the upstream system log / event diary")
    p.add_argument("--tuning", default=os.path.join(HERE, "guardian_tuning.json"),
                   help="file that keeps settings changed in the web UI (loaded at start); '' = do not load or save")
    p.add_argument("--cascade", action="store_true",
                   help="low -> high fidelity cascades (survey models; replay stand-ins on the laptop, see DEMO_CASCADE)")
    return p.parse_args(argv)


# Survey recommendation (docs/model_survey.md) as cascades. On the board these need their DPU backends first.
DEMO_CASCADE = {
    "bands": {"rates": {band: {"detector": ["refinedet_096", "ofa_yolo_05"]}
                        for band in ("detect", "far", "approach", "close")}},
    "pose": {"models": ["spnet", "movenet"]},
}


def replay_models(cfg, frame_w):
    """Replay stand-ins for every configured model: the cheap variant for all but the last stage of a cascade."""
    from VIDEO_pipeline.replay import CheapReplayDetector, CheapReplayPose, ReplayDetector, ReplayPose  # noqa
    full, cheap = ReplayDetector(box_noise_px=1.5), CheapReplayDetector(frame_w=frame_w)
    detectors = {}
    for r in cfg["bands"]["rates"].values():
        names = [r["detector"]] if isinstance(r["detector"], str) else r["detector"]
        for n in names[:-1]:
            detectors.setdefault(n, cheap)
        detectors[names[-1]] = full
    names = cfg["pose"]["models"]
    poses = {n: CheapReplayPose() for n in names[:-1]}
    poses[names[-1]] = ReplayPose(kp_noise_px=1.5)
    return detectors, poses


def build(args, cfg):
    """Create every box and wire them into a GuardianNode. -> (node, beacon, scenario, models, events)"""
    # --- sensors
    fast = args.fast and args.source == "scenario"
    clock = SimClock() if fast else RealClock()
    scenario = DemoScenario() if args.source == "scenario" else None
    if scenario:
        camera = ScenarioCamera(clock, scenario)
    else:
        c = cfg["camera"]
        camera = WebcamCamera(clock, c["index"], c["width"], c["height"])
    beacon_mode = args.beacon or ("scenario" if scenario else "always")
    if beacon_mode == "scenario":
        beacon = ScheduledBeacon(scenario.beacon_intervals if scenario else [(0, 1e9)])
    elif beacon_mode == "manual":
        beacon = ManualBeacon()
    else:
        beacon = AlwaysBeacon()
    power = board_power(cfg["power"]["rails"]) if args.power else NullPower()

    # --- VIDEO_pipeline: one model object per configured model name
    backend = args.backend or ("replay" if scenario else "dpu")
    models = None
    if backend == "dpu":
        from VIDEO_pipeline import dpu   # noqa: PLC0415 (board only)
        models, detectors, poses = dpu.build(cfg)
    else:
        detectors, poses = replay_models(cfg, camera.frame_size[0])

    # --- actuators
    events = EventLog(enabled=not args.no_log)
    audio = NullAudio() if args.no_audio or fast else AudioOut(cfg, log=events.system)
    actuators = ActuatorManager([audio, LogRobotLink(events.system), EventDiary(events)], log=events.system)

    # --- core
    caption = (lambda t: f"scenario t={t:5.1f}s  {scenario.beat(t)}") if scenario else None
    node = GuardianNode(cfg, clock, camera, beacon, power, detectors, poses, actuators, events, caption=caption)
    return node, beacon, scenario, models, events


def configs(args, log=print):
    """-> (live cfg, base). Base = defaults <- config.yaml <- command line; live = base with the tuning file
    applied underneath the command line (an explicit flag still wins for this run)."""
    cli = gcfg._merge(DEMO_CASCADE if args.cascade else {}, {"audio": {"enabled": False}} if args.no_audio else {})
    base = gcfg.load(cli)
    cfg = gcfg.load()
    gtune.load_file(args.tuning, cfg, log)
    return gcfg._merge(cfg, cli), base


def make_tuning(args, node, base, log):
    """Tuning over the node's live config; model choices = each loaded model alone plus the configured specs."""
    cfg = node.cfg
    specs = [gtune.spec_str(r["detector"]) for r in cfg["bands"]["rates"].values()]
    det = list(dict.fromkeys(specs + sorted(node.det_cascade.models)))
    pose = list(dict.fromkeys([gtune.spec_str(cfg["pose"]["models"])] + sorted(node.pose_cascade.models)))
    return gtune.Tuning(cfg, base, args.tuning or None, log, det, pose)


def main(argv=None):
    args = parse_args(argv)
    cfg, base = configs(args)
    node, beacon, scenario, models, events = build(args, cfg)
    tuning = make_tuning(args, node, base, events.system)
    dash = Dashboard(cfg, node.world, node.engine, cfg["display"]["width"], cfg["display"]["height"])

    sinks = []
    if args.hdmi:
        import subprocess   # noqa: PLC0415
        subprocess.run(["systemctl", "stop", "gdm"], check=False)
        sinks.append(DisplayPortSink(cfg["display"]["width"], cfg["display"]["height"], cfg["display"]["dp_pixel_format"]))
    if args.port:
        sinks.append(MjpegSink(args.port, tuning=tuning))
        print(f"stream + tuning: http://localhost:{args.port}/" + (f"  (saved to {args.tuning})" if args.tuning else ""))
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        sinks.append(VideoFileSink(args.record, fps=15.0))
    if args.snapshots:
        os.makedirs(args.snapshots, exist_ok=True)

    if isinstance(beacon, ManualBeacon):
        def keys():
            for line in sys.stdin:
                if line.strip().lower() == "b":
                    beacon.toggle()
                    print(f"beacon {'ON' if beacon.present else 'OFF'}")
        threading.Thread(target=keys, daemon=True).start()
        print("manual beacon: type b + Enter to toggle")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    duration = args.duration if args.duration is not None else (scenario.duration if scenario else 0)
    next_snap = 0.0
    try:
        while not stop.is_set():
            if tuning.apply_pending():
                node.reconfigure()
            snap = node.step()
            view = dash.render(snap)
            for s in list(sinks):
                try:
                    s.show(view)
                except Exception as e:           # e.g. HDMI without a monitor: keep the other sinks
                    print(f"{type(s).__name__} disabled: {e}")
                    sinks.remove(s)
            if args.snapshots and snap.t >= next_snap:
                import cv2   # noqa: PLC0415
                cv2.imwrite(os.path.join(args.snapshots, f"t{snap.t:06.2f}.png"), view)
                next_snap = snap.t + args.snapshot_every
            if snap.state == NodeState.IDLE and isinstance(node.clock, RealClock):
                time.sleep(0.05)
            if duration and snap.t >= duration:
                break
    finally:
        node.camera.release()
        for s in sinks:
            s.close()
        if models is not None:
            models.close()
    return node


if __name__ == "__main__":
    main()
