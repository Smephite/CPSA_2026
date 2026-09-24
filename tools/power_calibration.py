#!/usr/bin/env python3
"""Fit the power split model (settings `power.model`) on the KV260. Run as root in the PYNQ env, Guardian stopped:

    sudo -i; source /etc/profile.d/pynq_venv.sh; cd <repo>
    python3 tools/power_calibration.py                       # ~2.5 min; prints the fitted model as JSON

The board measures total SOM power only (INA260), so the split comes from a controlled experiment: with the DPU
overlay loaded, hold known loads for --duration s each and read power, CPU and DPU busy at the same time:
    idle        nothing running
    cpu_1..4    1..4 processes spinning on the ARM cores
    dpu         one DPU runner in a tight loop on a fixed buffer (the ARM only waits), for each --models
Then least squares:  P = static_w + cpu_core_w * busy_cores + dpu_w * dpu_busy.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.power import HwmonPower  # noqa: E402
from sensors.system import SystemMonitor  # noqa: E402


def _spin(stop):
    while not stop.is_set():
        pass


def measure(meter, duration, work=None):
    """-> (mean W, busy cores, dpu busy) while `work()` (returns DPU ms) loops, or while idling."""
    mon = SystemMonitor(window_s=0.0)
    mon.read(time.perf_counter())                                    # CPU baseline
    watts, t_end, dpu_ms = [], time.perf_counter() + duration, 0.0
    t0 = time.perf_counter()
    next_sample = t0
    while time.perf_counter() < t_end:
        if work is None:
            time.sleep(0.05)
        else:
            dpu_ms += work()
        if time.perf_counter() >= next_sample:
            watts.append(meter.read())
            next_sample += 0.1
    load = mon.read(time.perf_counter())
    return float(np.mean(watts)), load.cpu_cores, min(1.0, dpu_ms / 1e3 / (time.perf_counter() - t0))


def dpu_loop(runner):
    it = runner.get_input_tensors()[0]
    inp = [np.zeros(tuple(it.dims), np.int8, order="C")]
    out = [np.empty(tuple(t.dims), np.int8, order="C") for t in runner.get_output_tensors()]

    def work():
        t = time.perf_counter()
        runner.wait(runner.execute_async(inp, out))
        return (time.perf_counter() - t) * 1e3
    return work


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=15.0, help="seconds per step (default 15)")
    p.add_argument("--models", nargs="*", default=["ofa_yolo_05", "refinedet_096"])
    args = p.parse_args(argv)

    from VIDEO_pipeline.dpu import DpuModels   # noqa: PLC0415 (board only)
    meter = HwmonPower()
    models = DpuModels()                                             # overlay loaded: its static power is in static_w
    time.sleep(2.0)
    rows = []

    def step(name, work=None):
        w, cores, dpu = measure(meter, args.duration, work)
        rows.append((name, w, cores, dpu))
        print(f"{name:<22s} {w:6.2f} W   cpu {cores:4.2f} cores   dpu {dpu:4.0%}", flush=True)

    step("idle")
    for n in range(1, os.cpu_count() + 1):
        stop = mp.Event()
        procs = [mp.Process(target=_spin, args=(stop,)) for _ in range(n)]
        for pr in procs:
            pr.start()
        time.sleep(1.0)
        step(f"cpu_{n}")
        stop.set()
        for pr in procs:
            pr.join()
    for name in args.models:
        step(f"dpu_{name}", dpu_loop(models.runner(name)))
    models.close()

    a = np.array([[1.0, c, d] for _, _, c, d in rows])
    y = np.array([w for _, w, _, _ in rows])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    model = {"static_w": round(float(coef[0]), 3), "cpu_core_w": round(float(coef[1]), 3),
             "dpu_w": round(float(coef[2]), 3)}
    print(f"fit residuals (W): {' '.join(f'{r:+.2f}' for r in resid)}")
    print(json.dumps({"power": {"model": model}}, indent=2))


if __name__ == "__main__":
    main()
