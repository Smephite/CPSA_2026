#!/usr/bin/env python3
"""Depth demo: the RealSense F200's colour and depth streams side by side, live on HDMI and the web stream.

Board: ./tools/board.sh tools/depth_demo.py [--no-hdmi] [--port 8080]      (Ctrl+C to stop)

Left: colour (/dev/video0). Right: depth (/dev/video2), near = red ... far = blue, black = no depth (out of the
F200's ~0.2-1.2 m range, or washed out by sunlight). The crosshair shows the depth at the image centre: hold
something at a measured distance there to check settings depth.unit_mm. The two cameras sit a few cm apart on
the F200 and are not aligned to each other: the images match only roughly. Nothing is saved (tools/record.py does).
"""
import argparse
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
from tools.record import CameraRecorder  # noqa: E402
from utils import settings  # noqa: E402
from utils.clock import RealClock  # noqa: E402


class DepthView:
    """render({"color", "depth", "fps"}) -> 1280x720: colour | depth, crosshair distance, colour scale, coverage."""

    def __init__(self, depth_cfg, width=1280, height=720):
        self.d, self.w, self.h = depth_cfg, width, height

    def render(self, s):
        out = np.full((self.h, self.w, 3), BG, np.uint8)
        pw = self.w // 2 - 12
        ph = pw * 3 // 4
        y0 = 70
        d = self.d
        z = s["depth"]
        panels = [("colour", s["color"]),
                  ("depth", None if z is None else colorize_depth(z, d["unit_mm"], d["near_m"], d["far_m"]))]
        for i, (name, img) in enumerate(panels):
            x0 = 8 + i * (pw + 8)
            put(out, f"{name}   {s['fps'][i]:.1f} fps", (x0, y0 - 14), 0.6, TEXT)
            if img is None:
                put(out, "waiting...", (x0 + pw // 2 - 50, y0 + ph // 2), 0.7, MUTED)
                continue
            out[y0:y0 + ph, x0:x0 + pw] = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_NEAREST)
            cx, cy = x0 + pw // 2, y0 + ph // 2
            cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        if z is not None:
            h, w = z.shape
            c = z[h // 2 - 4:h // 2 + 5, w // 2 - 4:w // 2 + 5]
            c = c[c > 0]
            centre = f"{np.median(c) * d['unit_mm'] / 1e3:.2f} m" if c.size else "no depth"
            put(out, f"centre: {centre}", (8, y0 + ph + 44), 0.9, TEXT, 2)
            put(out, f"depth coverage {np.count_nonzero(z) / z.size:.0%}", (8, y0 + ph + 80), 0.6, MUTED)
            self._scale(out, pw + 16, y0 + ph + 26, pw, 18)
        put(out, "RealSense F200: coded-light depth, ~0.2-1.2 m, indoors", (8, 30), 0.6, MUTED)
        return out

    def _scale(self, out, x, y, w, h):
        d = self.d
        raw = (np.linspace(d["near_m"], d["far_m"], w) / (d["unit_mm"] / 1e3)).astype(np.uint16)[None].repeat(h, 0)
        out[y:y + h, x:x + w] = colorize_depth(raw, d["unit_mm"], d["near_m"], d["far_m"])
        put(out, f"{d['near_m']:.1f} m", (x, y + h + 20), 0.5, MUTED)
        put(out, f"{d['far_m']:.1f} m", (x + w - 50, y + h + 20), 0.5, MUTED)


def main(argv=None):
    cfg = settings.load()
    cam, dep, disp = cfg["camera"], cfg["depth"], cfg["display"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--color", type=int, default=0, help="colour /dev/videoN (default 0)")
    p.add_argument("--depth", type=int, default=dep["index"], help=f"depth /dev/videoN (default {dep['index']})")
    p.add_argument("--port", type=int, default=8080, help="live view on http://<board>:PORT/stream, 0 = off")
    p.add_argument("--no-hdmi", action="store_true")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (default: Ctrl+C)")
    args = p.parse_args(argv)

    clock = RealClock()
    cams = [WebcamCamera(clock, args.color, cam["width"], cam["height"]),
            DepthCamera(clock, args.depth, cam["width"], cam["height"], 30)]
    for c in cams:
        c.open()
    readers = [CameraRecorder(c, None, 0.0) for c in cams]
    for r in readers:
        r.start()

    sinks = []
    if not args.no_hdmi:
        subprocess.run(["systemctl", "stop", "gdm"], check=False)
        sinks.append(DisplayPortSink(disp["width"], disp["height"], disp["dp_pixel_format"]))
    if args.port:
        sinks.append(MjpegSink(args.port))
        print(f"live view: http://localhost:{args.port}/stream")
    output = OutputWorker(DepthView(dep, disp["width"], disp["height"]), sinks, threaded=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    t0 = time.monotonic()
    try:
        while not stop.is_set():
            errors = [r.error for r in readers if r.error is not None]
            if errors:
                print(f"camera error, stopping: {errors[0]}")
                break
            output.submit({"color": readers[0].latest, "depth": readers[1].latest, "fps": [r.fps for r in readers]})
            if args.duration and time.monotonic() - t0 >= args.duration:
                break
            time.sleep(1 / 20)
    finally:
        for r in readers:
            r.stop()
        output.close()


if __name__ == "__main__":
    main()
