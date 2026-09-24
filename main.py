#!/usr/bin/env python3
"""Guardian Node: entry point. Builds the module boxes, wires them together, runs the loop.

    sensors/  ─► VIDEO_pipeline/ ─► core/ ─► actuators/     (README.md: "How it works")
                                      └────► dashboard/

Laptop (no FPGA):
    uv run python main.py                                # synthetic demo, real time, MJPEG on :8080
    uv run python main.py --fast --record out/demo.mp4   # render the demo to a video as fast as possible
Board (KV260, root, PYNQ environment: see run.sh):
    ./run.sh --source webcam --backend dpu --hdmi                 # live: webcam -> DPU -> HDMI (+ MJPEG)
    ./record.sh out/clips/reach.avi                               # record raw webcam clips (live view on HDMI)
    ./run.sh --source video --video out/clips/reach.avi --loop    # replay a clip live, tune in the web UI
    ./run.sh --source video --video out/clips/reach.avi --fast    # every frame as fast as possible (repeatable)

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
from dashboard.output import OutputWorker  # noqa: E402
from dashboard.views import Dashboard  # noqa: E402
from sensors.beacon import AlwaysBeacon, ManualBeacon, ScheduledBeacon  # noqa: E402
from sensors.camera import ScenarioCamera, VideoFileCamera, WebcamCamera  # noqa: E402
from sensors.power import NullPower, board_power  # noqa: E402
from sensors.system import SystemMonitor  # noqa: E402
from sensors.scenario import DemoScenario  # noqa: E402
from utils import settings as gcfg  # noqa: E402
from utils.clock import RealClock, SimClock  # noqa: E402
from utils.profile import Profiler  # noqa: E402
from utils.event_log import EventLog  # noqa: E402
from utils.types import NodeState  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["scenario", "webcam", "video"], default="scenario")
    p.add_argument("--video", metavar="CLIP", help="clip for --source video (recorded with tools/record.py)")
    p.add_argument("--loop", action="store_true", help="--source video: play the clip over and over")
    p.add_argument("--backend", choices=["replay", "dpu"], default=None,
                   help="perception backend (default: replay for scenario, dpu for webcam and video)")
    p.add_argument("--beacon", choices=["scenario", "always", "manual"], default=None,
                   help="default: scenario timeline for the scenario source, always for webcam and video")
    p.add_argument("--fast", action="store_true",
                   help="simulated clock: run the scenario or video as fast as possible, every frame (repeatable)")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds (scenario default: its length)")
    p.add_argument("--port", type=int, default=8080, help="MJPEG stream port, 0 = off (default 8080)")
    p.add_argument("--hdmi", action="store_true", help="dashboard full screen on the HDMI monitor (board)")
    p.add_argument("--record", help="write the dashboard to this .mp4")
    p.add_argument("--snapshots", help="save a dashboard PNG every N seconds into this folder (with --snapshot-every)")
    p.add_argument("--snapshot-every", type=float, default=2.0)
    p.add_argument("--silent", "--no-audio", dest="no_audio", action="store_true",
                   help="silent mode: no sound output (visual warnings only)")
    p.add_argument("--power", action="store_true", help="read the board's power rails (pynq)")
    p.add_argument("--no-log", action="store_true", help="do not write the upstream system log / event diary")
    p.add_argument("--tuning", default=os.path.join(HERE, "guardian_tuning.json"),
                   help="file that keeps settings changed in the web UI (loaded at start); '' = do not load or save")
    p.add_argument("--profile", metavar="CSV", help="record per-frame stage timings to CSV; summary printed at exit")
    p.add_argument("--cascade", action="store_true",
                   help="low -> high fidelity cascades (survey models; replay stand-ins on the laptop, see DEMO_CASCADE)")
    return p.parse_args(argv)


# Survey-style cascades: person-only RefineDet first, OFA-YOLO when unsure; MoveNet first, Hourglass for lying /
# unusual poses. All run on the board; any other combination can be picked live in the web UI.
DEMO_CASCADE = {
    "bands": {"rates": {band: {"detector": ["refinedet_096", "ofa_yolo_05"]}
                        for band in ("detect", "far", "approach", "close")}},
    "pose": {"models": ["movenet", "hourglass"]},
}


def build(args, cfg):
    """Create every box and wire them into a GuardianNode. -> (node, beacon, scenario, models, events)"""
    # --- sensors
    fast = args.fast and args.source in ("scenario", "video")
    clock = SimClock() if fast else RealClock()
    scenario = DemoScenario() if args.source == "scenario" else None
    if scenario:
        camera = ScenarioCamera(clock, scenario)
    elif args.source == "video":
        if not args.video:
            raise SystemExit("--source video needs --video CLIP")
        camera = VideoFileCamera(clock, args.video, loop=args.loop)
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

    # --- VIDEO_pipeline: every catalog model, so any of them can be chosen at runtime
    backend = args.backend or ("replay" if scenario else "dpu")
    models = None
    if backend == "dpu":
        from VIDEO_pipeline import dpu   # noqa: PLC0415 (board only)
        models, detectors, poses, orientation = dpu.build(cfg)
    else:
        from VIDEO_pipeline.replay import replay_models   # noqa: PLC0415
        detectors, poses, orientation = replay_models(camera.frame_size[0])

    # --- actuators
    events = EventLog(enabled=not args.no_log)
    audio = NullAudio() if args.no_audio or fast else AudioOut(cfg, log=events.system)
    actuators = ActuatorManager([audio, LogRobotLink(events.system), EventDiary(events)], log=events.system)

    # --- core
    caption = (lambda t: f"scenario t={t:5.1f}s  {scenario.beat(t)}") if scenario else None
    node = GuardianNode(cfg, clock, camera, beacon, power, detectors, poses, actuators, events, caption=caption,
                        orientation=orientation, async_detect=not fast, system=SystemMonitor())
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
    """Tuning over the node's live config. Model choices: the configured specs, every catalog model alone, and
    cheap -> full cascades (VIDEO_pipeline/catalog.py); all of them are loaded, so any choice works live."""
    from VIDEO_pipeline import catalog   # noqa: PLC0415
    cfg = node.cfg
    specs = [gtune.spec_str(r["detector"]) for r in cfg["bands"]["rates"].values()]
    det = [c for c in dict.fromkeys(specs + catalog.detector_choices())
           if all(n in node.det_cascade.models for n in gtune.spec_value(c) if isinstance(gtune.spec_value(c), list))
           and (not isinstance(gtune.spec_value(c), str) or gtune.spec_value(c) in node.det_cascade.models)]
    pose = [c for c in dict.fromkeys([gtune.spec_str(cfg["pose"]["models"])] + catalog.pose_choices())
            if all(n in node.pose_cascade.models for n in ([gtune.spec_value(c)] if isinstance(gtune.spec_value(c), str)
                                                          else gtune.spec_value(c)))]
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
    profiler = Profiler(args.profile) if args.profile else None
    snaps = {"next": 0.0}

    def save_snapshot(view, snap):
        if args.snapshots and snap.t >= snaps["next"]:
            import cv2   # noqa: PLC0415
            cv2.imwrite(os.path.join(args.snapshots, f"t{snap.t:06.2f}.png"), view)
            snaps["next"] = snap.t + args.snapshot_every

    # drawing + sinks in their own thread on a live node; inline (every frame) for --fast and --record
    output = OutputWorker(dash, sinks, threaded=not (args.fast or args.record), on_frame=save_snapshot,
                          profiler=profiler, log=events.system)
    try:
        while not stop.is_set():
            t_loop = time.perf_counter()
            if tuning.apply_pending():
                node.reconfigure()
            snap = node.step()
            output.submit(snap)
            if profiler is not None and snap.state != NodeState.IDLE:
                profiler.record(snap.t, snap.state.name, dict(snap.times, loop=(time.perf_counter() - t_loop) * 1e3))
            if snap.state == NodeState.IDLE and isinstance(node.clock, RealClock):
                time.sleep(0.05)
            if duration and snap.t >= duration or getattr(node.camera, "finished", False):
                break
    finally:
        output.close()
        if profiler is not None:
            print(profiler.close())
            if output.threaded:
                print(f"output: {output.frames} frames drawn, {output.skipped} skipped (newer snapshot arrived first)")
        node.camera.release()
        if models is not None:
            models.close()
    return node


if __name__ == "__main__":
    main()
