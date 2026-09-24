#!/usr/bin/env python3
"""Record raw webcam clips for offline evaluation, with a live view on the HDMI monitor and the web stream.

Board (root, PYNQ environment for HDMI): use ../record.sh, which sets that up:
    ./record.sh out/clips/reach_from_left.avi                  # Ctrl+C to stop
    ./record.sh out/clips/walk_past.avi --duration 30 --no-hdmi
    ./record.sh out/clips/stereo_reach.avi --index 0 --right 2     # stereo: two webcams, 15 fps requested
    ./record.sh out/clips/close.avi --depth                         # RealSense F200: colour + depth (video2)

Writes clip.avi (MJPEG, high quality) and clip.csv (frame, t: capture time in s from the start); with --right also
clip_right.avi + clip_right.csv, with --depth clip_depth.u16 (raw uint16, lossless) + clip_depth.csv, all on the
same time base. Two separate webcams are not synchronised: pair their frames
by time. Replay (left / single camera) with
    ./run.sh --source video --video clip.avi [--loop | --fast]
No DPU and no Guardian logic here: stop run.sh first (only one process can own the webcam and the monitor).
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.output import OutputWorker  # noqa: E402
from dashboard.sinks import DisplayPortSink, MjpegSink  # noqa: E402
from dashboard.views import BG, MUTED, TEXT, colorize_depth, put  # noqa: E402
from sensors.camera import DepthCamera, WebcamCamera  # noqa: E402
from utils import settings  # noqa: E402
from utils.clock import RealClock  # noqa: E402


class CameraRecorder:
    """One camera in its own thread (each delivers frames at its own pace), saving to `path`:
        .avi    colour, MJPEG (high quality)
        .u16    depth, raw uint16 frames back to back (lossless; load with sensors.camera.load_depth_clip)
        None    nothing saved (live view only)
    plus the .csv next to it: frame, t (s on the shared clock since `t0`, to pair cameras by time), width, height.
    `preview(image) -> BGR` turns the latest frame into something to look at (depth: colour map).
    """

    def __init__(self, camera, path, t0, preview=None):
        self.camera, self.path, self.t0 = camera, path, t0
        self.preview = preview or (lambda img: img)
        self.frames, self.fps, self.latest, self.error = 0, 0.0, None, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rec:{path}", daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        video = raw = f = rows = None
        t_prev = None
        try:
            if self.path is not None:
                f = open(os.path.splitext(self.path)[0] + ".csv", "w", newline="")
                rows = csv.writer(f)
                rows.writerow(["frame", "t", "width", "height"])
            while not self._stop.is_set():
                frame = self.camera.read()
                img = frame.image
                h, w = img.shape[:2]
                t = frame.t - self.t0
                if self.path is None:
                    pass
                elif self.path.endswith(".u16"):
                    raw = raw or open(self.path, "wb")
                    raw.write(img.astype("<u2").tobytes())
                else:
                    if video is None:
                        video = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
                        video.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
                        if not video.isOpened():
                            raise RuntimeError(f"cannot write {self.path}")
                    video.write(img)
                if rows is not None:
                    rows.writerow([self.frames, f"{t:.4f}", w, h])
                self.frames += 1
                if t_prev is not None and t > t_prev:
                    self.fps = 1 / (t - t_prev) if not self.fps else 0.9 * self.fps + 0.1 / (t - t_prev)
                t_prev, self.latest = t, img
        except Exception as e:                      # noqa: BLE001: reported by main, which stops the recording
            self.error = e
        finally:
            if video is not None:
                video.release()
            for fh in (raw, f):
                if fh is not None:
                    fh.close()
            self.camera.release()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


class RecordView:
    """Live view: each camera's image (uncropped, side by side for stereo) and a REC panel on the right.

    render(status) -> image; status = {"t": s, "cams": [(name, image or None, frames, fps), ...]}.
    """

    def __init__(self, path, width=1280, height=720):
        self.name, self.w, self.h = os.path.basename(path), width, height

    def render(self, s):
        out = np.full((self.h, self.w, 3), BG, np.uint8)
        cams = s["cams"]
        area_w = self.w - 260
        slot_w = area_w // len(cams)
        y = 60
        for i, (name, img, frames, fps) in enumerate(cams):
            x0 = i * slot_w
            if img is not None:
                cw = min(slot_w - 8, int(img.shape[1] * (self.h - 140) / img.shape[0]))
                ch = int(img.shape[0] * cw / img.shape[1])
                y = (self.h - ch) // 2
                out[y:y + ch, x0 + 4:x0 + 4 + cw] = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
            put(out, f"{name}   {frames} frames   {fps:.1f} fps", (x0 + 8, max(30, y - 14)), 0.55, TEXT)
        x = area_w + 20
        if int(s["t"] * 2) % 2 == 0:
            cv2.circle(out, (x + 12, 52), 12, (40, 40, 230), -1, cv2.LINE_AA)
        put(out, "REC", (x + 34, 62), 1.1, (40, 40, 230), 2)
        m, sec = divmod(int(s["t"]), 60)
        put(out, f"{m:02d}:{sec:02d}", (x, 120), 1.2, TEXT, 2)
        put(out, self.name[:20], (x, 170), 0.55, MUTED)
        put(out, "Ctrl+C to stop", (x, self.h - 30), 0.55, MUTED)
        return out


def main(argv=None):
    cfg = settings.load()
    cam_cfg, disp = cfg["camera"], cfg["display"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="clip path (.avi); clip.csv is written next to it")
    p.add_argument("--index", type=int, default=cam_cfg["index"], help="webcam /dev/videoN (the left one for stereo)")
    p.add_argument("--right", type=int, metavar="INDEX",
                   help="stereo: second webcam /dev/videoN, recorded to clip_right.avi + clip_right.csv")
    p.add_argument("--depth", type=int, nargs="?", const=cfg["depth"]["index"], metavar="INDEX",
                   help=f"also record depth (default /dev/video{cfg['depth']['index']}: RealSense F200) to "
                        "clip_depth.u16 + clip_depth.csv")
    p.add_argument("--depth-fps", type=int, default=cfg["depth"]["fps"])
    p.add_argument("--width", type=int, default=cam_cfg["width"])
    p.add_argument("--height", type=int, default=cam_cfg["height"])
    p.add_argument("--fps", type=int, default=None,
                   help="requested fps (default 30, 15 for stereo: two cameras at 30 do not fit on the board's USB 2.0)")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (default: Ctrl+C)")
    p.add_argument("--port", type=int, default=8080, help="live view on http://<board>:PORT/stream, 0 = off")
    p.add_argument("--no-hdmi", action="store_true", help="no live view on the HDMI monitor")
    p.add_argument("--force", action="store_true", help="overwrite an existing clip")
    args = p.parse_args(argv)

    if not args.out.endswith(".avi"):
        raise SystemExit("the clip must be an .avi (MJPEG)")
    paths = [args.out] + ([args.out[:-4] + "_right.avi"] if args.right is not None else []) \
        + ([args.out[:-4] + "_depth.u16"] if args.depth is not None else [])
    for path in paths:
        if os.path.exists(path) and not args.force:
            raise SystemExit(f"{path} exists (--force to overwrite)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fps = args.fps or (15 if args.right is not None else 30)

    clock = RealClock()
    cameras = [WebcamCamera(clock, i, args.width, args.height, fps)
               for i in [args.index] + ([args.right] if args.right is not None else [])]
    names = [f"video{cam.index}" + ("" if len(cameras) == 1 else (" left", " right")[i])
             for i, cam in enumerate(cameras)]
    previews = [None] * len(cameras)
    if args.depth is not None:
        d = cfg["depth"]
        cameras.append(DepthCamera(clock, args.depth, args.width, args.height, args.depth_fps))
        names.append(f"video{args.depth} depth")
        previews.append(lambda z: colorize_depth(z, d["unit_mm"], d["near_m"], d["far_m"]))
    for cam in cameras:                             # open all before recording starts: fail early
        cam.open()
    t0 = clock.now()
    recorders = [CameraRecorder(cam, path, t0, pv) for cam, path, pv in zip(cameras, paths, previews)]

    sinks = []
    if not args.no_hdmi:
        subprocess.run(["systemctl", "stop", "gdm"], check=False)
        sinks.append(DisplayPortSink(disp["width"], disp["height"], disp["dp_pixel_format"]))
    if args.port:
        sinks.append(MjpegSink(args.port))
        print(f"live view: http://localhost:{args.port}/stream")
    output = OutputWorker(RecordView(args.out, disp["width"], disp["height"]), sinks, threaded=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"recording to {', '.join(paths)} (Ctrl+C to stop)")
    for r in recorders:
        r.start()
    try:
        while not stop.is_set():
            t = clock.now() - t0
            errors = [r.error for r in recorders if r.error is not None]
            if errors:
                print(f"camera error, stopping: {errors[0]}")
                break
            output.submit({"t": t, "cams": [(n, None if r.latest is None else r.preview(r.latest), r.frames, r.fps)
                                            for n, r in zip(names, recorders)]})
            if args.duration and t >= args.duration:
                break
            time.sleep(1 / 15)
    finally:
        for r in recorders:
            r.stop()
        output.close()
    for n, r in zip(names, recorders):
        print(f"{n}: {r.frames} frames ({r.fps:.1f} fps) -> {r.path} + .csv")


if __name__ == "__main__":
    main()
