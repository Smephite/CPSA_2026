#!/usr/bin/env python3
"""Guardian Node entry point.

Laptop (no FPGA):
    uv run python guardian_main.py                                # synthetic demo, real time, MJPEG on :8080
    uv run python guardian_main.py --fast --record out/demo.mp4   # render the demo to a video as fast as possible
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

from guardian import config as gcfg  # noqa: E402
from guardian import io  # noqa: E402
from guardian.node import GuardianNode  # noqa: E402
from guardian.scenario import DemoScenario  # noqa: E402
from guardian.sinks import DisplayPortSink, MjpegSink, VideoFileSink  # noqa: E402
from guardian.types import NodeState  # noqa: E402
from guardian.views import Dashboard  # noqa: E402


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
    return p.parse_args(argv)


def build(args, cfg):
    fast = args.fast and args.source == "scenario"
    clock = io.SimClock() if fast else io.RealClock()
    scenario = DemoScenario() if args.source == "scenario" else None
    if scenario:
        camera = io.ScenarioCamera(clock, scenario)
    else:
        c = cfg["camera"]
        camera = io.WebcamCamera(clock, c["index"], c["width"], c["height"])

    beacon_mode = args.beacon or ("scenario" if scenario else "always")
    if beacon_mode == "scenario":
        beacon = io.ScheduledBeacon(scenario.beacon_intervals if scenario else [(0, 1e9)])
    elif beacon_mode == "manual":
        beacon = io.ManualBeacon()
    else:
        beacon = io.AlwaysBeacon()

    backend = args.backend or ("replay" if scenario else "dpu")
    models = None
    if backend == "dpu":
        from guardian.perception import dpu   # noqa: PLC0415 (board only)
        models, detectors, pose = dpu.build(cfg)
    else:
        from guardian.perception.replay import ReplayDetector, ReplayPose   # noqa: PLC0415
        det = ReplayDetector(box_noise_px=1.5)
        detectors = {r["detector"]: det for r in cfg["bands"]["rates"].values()}
        pose = ReplayPose(kp_noise_px=1.5)

    events = io.EventLog(enabled=not args.no_log)
    audio = io.NullAudio() if args.no_audio or fast else io.AudioOut(cfg, log=events.system)
    power = io.PynqRailsPower(cfg["power"]["rails"]) if args.power else io.NullPower()
    caption = (lambda t: f"scenario t={t:5.1f}s  {scenario.beat(t)}") if scenario else None
    node = GuardianNode(cfg, clock, camera, beacon, detectors, pose, audio, io.LogRobotLink(events.system),
                        power, events, caption=caption)
    return node, beacon, scenario, models, events


def main(argv=None):
    args = parse_args(argv)
    cfg = gcfg.load()
    if args.no_audio:
        cfg["audio"]["enabled"] = False
    node, beacon, scenario, models, events = build(args, cfg)
    dash = Dashboard(cfg, node.world, node.engine, cfg["display"]["width"], cfg["display"]["height"])

    sinks = []
    if args.hdmi:
        import subprocess   # noqa: PLC0415
        subprocess.run(["systemctl", "stop", "gdm"], check=False)
        sinks.append(DisplayPortSink(cfg["display"]["width"], cfg["display"]["height"], cfg["display"]["dp_pixel_format"]))
    if args.port:
        sinks.append(MjpegSink(args.port))
        print(f"stream: http://localhost:{args.port}/")
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        sinks.append(VideoFileSink(args.record, fps=15.0))
    if args.snapshots:
        os.makedirs(args.snapshots, exist_ok=True)

    if isinstance(beacon, io.ManualBeacon):
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
            snap = node.step()
            view = dash.render(snap)
            for s in list(sinks):
                try:
                    s.show(view)
                except Exception as e:           # e.g. HDMI without a monitor: keep the other sinks
                    print(f"{type(s).__name__} disabled: {e}")
                    sinks.remove(s)
            if args.snapshots and snap["t"] >= next_snap:
                import cv2   # noqa: PLC0415
                cv2.imwrite(os.path.join(args.snapshots, f"t{snap['t']:06.2f}.png"), view)
                next_snap = snap["t"] + args.snapshot_every
            if snap["state"] == NodeState.IDLE and isinstance(node.clock, io.RealClock):
                time.sleep(0.05)
            if duration and snap["t"] >= duration:
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
