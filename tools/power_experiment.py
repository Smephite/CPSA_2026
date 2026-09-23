#!/usr/bin/env python3
"""Board power experiment (docs/model_survey.md section 4.4, steps 1-4 and 6). KV260 only; run as root in the PYNQ env:

    sudo -i; source /etc/profile.d/pynq_venv.sh; cd <repo>
    python3 tools/power_experiment.py --out out/power            # ~15 min with the defaults

Steps, in this order (the overlay cannot be measured unloaded once it is loaded, so step 1 comes first):
    idle            Linux, no overlay, no camera                              x repeats
    overlay_load    DpuOverlay(...) + one runner per model: time and energy   once
    overlay_idle    overlay and runners loaded, no inference                  x repeats
    camera_<fps>    webcam capture at each --camera-fps, no inference         x repeats
    model_<name>    tight loop over a fixed input, full pre/DPU/post call     x repeats

Rail power is polled from pynq.get_rails() at --hz. Writes <out>_samples.csv (one row per sample, one column per
rail plus the total) and <out>_summary.csv (per step and repeat: mean/std power, calls/s, mean DPU time, energy
per call above overlay_idle, memory available). Not covered: duty-cycled band schedules (step 5), PL unload
(PYNQ has no clean unload; reboot for step 1 instead) and the thermal run (step 7).
"""
import argparse
import csv
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="out/power", help="output prefix (default out/power)")
    p.add_argument("--duration", type=float, default=60.0, help="seconds per step (default 60)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--hz", type=float, default=10.0, help="rail sampling rate (default 10)")
    p.add_argument("--rails", nargs="*", default=[], help="rail names to record (default: all with a power sensor)")
    p.add_argument("--camera", type=int, default=0, help="webcam index, -1 = skip the camera steps")
    p.add_argument("--camera-fps", type=int, nargs="*", default=[15, 30])
    p.add_argument("--models", nargs="*", default=["yolov3_voc", "movenet"])
    return p.parse_args(argv)


class RailSampler:
    """Background thread: every 1/hz s, read each rail's power (W) and tag the sample with the current step."""

    def __init__(self, rails, hz):
        import pynq                                   # noqa: PLC0415 (board only)
        all_rails = pynq.get_rails()
        names = rails or sorted(n for n, r in all_rails.items() if getattr(r, "power", None) is not None)
        missing = [n for n in names if n not in all_rails]
        if missing or not names:
            raise SystemExit(f"rails {missing or names} not found; have {sorted(all_rails)}")
        self.names, self.rails, self.period = names, [all_rails[n] for n in names], 1.0 / hz
        self.rows, self.tag = [], None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.t0 = time.perf_counter()
        self._thread.start()

    def _run(self):
        nxt = time.perf_counter()
        while not self._stop.is_set():
            if self.tag is not None:
                w = [float(r.power.value) for r in self.rails]
                self.rows.append((*self.tag, time.perf_counter() - self.t0, sum(w), *w))
            nxt += self.period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def measure(self, step, repeat, duration, work=None):
        """Sample while `work()` runs in a loop (or while idling) for `duration` s -> (power (N,), calls)."""
        start = len(self.rows)
        self.tag = (step, repeat)
        calls, t_end = 0, time.perf_counter() + duration
        while time.perf_counter() < t_end:
            if work is None:
                time.sleep(0.05)
            else:
                work()
                calls += 1
        self.tag = None
        return np.array([r[3] for r in self.rows[start:]]), calls

    def close(self):
        self._stop.set()
        self._thread.join()


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sampler = RailSampler(args.rails, args.hz)
    summary = []

    def record(step, repeat, power, calls=0, duration=None, dpu_ms=float("nan"), baseline=None):
        dur = duration or args.duration
        rate = calls / dur if calls else 0.0
        e_call = (power.mean() - baseline) / rate * 1e3 if rate and baseline is not None else float("nan")
        row = {"step": step, "repeat": repeat, "samples": len(power), "mean_w": power.mean() if len(power) else float("nan"),
               "std_w": power.std() if len(power) else float("nan"), "duration_s": dur, "calls_per_s": rate,
               "dpu_ms": dpu_ms, "e_call_mj": e_call, "mem_available_mb": mem_available_mb()}
        summary.append(row)
        print(f"{step:<16s} #{repeat}  {row['mean_w']:6.2f} +- {row['std_w']:4.2f} W  {rate:6.1f} calls/s  "
              f"dpu {dpu_ms:5.1f} ms  E_call {e_call:6.1f} mJ", flush=True)

    try:
        for r in range(args.repeats):
            record("idle", r, sampler.measure("idle", r, args.duration)[0])

        from guardian.perception import dpu    # noqa: PLC0415 (board only: imports pynq_dpu)
        sampler.tag = ("overlay_load", 0)
        start, t0 = len(sampler.rows), time.perf_counter()
        models = dpu.DpuModels()
        runners = {name: models.runner(name) for name in args.models}
        load_s = time.perf_counter() - t0
        sampler.tag = None
        power = np.array([row[3] for row in sampler.rows[start:]])
        if not len(power):                       # faster than one sample period: read the rails once
            power = np.array([sum(float(rail.power.value) for rail in sampler.rails)])
        record("overlay_load", 0, power, duration=load_s)
        print(f"overlay + runners loaded in {load_s:.2f} s (DpuOverlay alone {models.load_s:.2f} s), "
              f"energy {power.mean() * load_s if len(power) else float('nan'):.1f} J", flush=True)

        idle_w = []
        for r in range(args.repeats):
            p = sampler.measure("overlay_idle", r, args.duration)[0]
            idle_w.append(p.mean())
            record("overlay_idle", r, p)
        baseline = float(np.mean(idle_w))

        if args.camera >= 0:
            import cv2                          # noqa: PLC0415
            for fps in args.camera_fps:
                cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, fps)
                if not cap.isOpened():
                    print(f"camera {args.camera} did not open; skipping camera steps", flush=True)
                    break
                for r in range(args.repeats):
                    p, calls = sampler.measure(f"camera_{fps}", r, args.duration, lambda: cap.read())
                    record(f"camera_{fps}", r, p, calls, baseline=baseline)
                cap.release()

        from guardian.types import Frame        # noqa: PLC0415
        rng = np.random.default_rng(0)
        frame = Frame(image=rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), t=0.0)
        box = np.array([220.0, 60.0, 420.0, 460.0])
        for name in args.models:
            if name == "movenet":
                model = dpu.MoveNetPose(runners[name])
                work, key = (lambda m=model: m.estimate(frame, box)), "pose_dpu"
            else:
                model = dpu.DETECTORS[name](runners[name])
                work, key = (lambda m=model: m.detect(frame)), "det_dpu"
            for r in range(args.repeats):
                dpu_ms = []
                p, calls = sampler.measure(f"model_{name}", r, args.duration,
                                           lambda: (work(), dpu_ms.append(model.times.get(key, float("nan")))))
                record(f"model_{name}", r, p, calls, dpu_ms=float(np.mean(dpu_ms)), baseline=baseline)
        models.close()
    finally:
        sampler.close()
        with open(args.out + "_samples.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "repeat", "t_s", "total_w"] + [f"{n}_w" for n in sampler.names])
            w.writerows(sampler.rows)
        if summary:
            with open(args.out + "_summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0]))
                w.writeheader()
                w.writerows(summary)
        print(f"wrote {args.out}_samples.csv ({len(sampler.rows)} samples) and {args.out}_summary.csv")


if __name__ == "__main__":
    main()
