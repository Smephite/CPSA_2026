"""Per-frame timing profile: where each frame's time goes (ARM pre/post-processing, DPU, drawing, sinks).

    main.py --profile out/profile.csv    one CSV row per frame, summary table printed at exit

Keys (ms): cap (camera read), det / pose (whole stage) with det_pre, det_dpu, det_post, pose_pre, pose_dpu,
pose_post, orient_dpu (per model call, summed over the frame), rules, render (dashboard drawing), sink:<Name>
(one per output), loop (whole iteration). `*_dpu` includes the runtime's float <-> int8 conversion.
"""
import csv
from collections import defaultdict

import numpy as np


class Profiler:
    def __init__(self, path=None):
        self.path = path
        self.rows = []

    def record(self, t, state, times):
        self.rows.append({"t": round(t, 3), "state": state, **{k: round(v, 3) for k, v in times.items()}})

    def summary(self):
        """-> text table: per state, per key: median, p90, mean and share of the mean loop time."""
        by_state = defaultdict(list)
        for r in self.rows:
            by_state[r["state"]].append(r)
        lines = []
        for state, rows in sorted(by_state.items()):
            keys = sorted({k for r in rows for k in r} - {"t", "state"})
            total = "loop" if any("loop" in r for r in rows) else "output"
            loop = np.mean([r.get(total, 0.0) for r in rows]) or 1.0
            rate = f"{len(rows) / max(rows[-1]['t'] - rows[0]['t'], 1e-6):.1f} per s" if len(rows) > 1 else ""
            lines.append(f"\n{state}: {len(rows)} frames ({rate}), {total} {loop:.1f} ms mean")
            lines.append(f"  {'stage':16s} {'median':>8s} {'p90':>8s} {'mean':>8s} {'share':>7s}  frames")
            for k in sorted(keys, key=lambda k: -np.mean([r.get(k, 0.0) for r in rows])):
                vals = np.array([r[k] for r in rows if k in r])
                mean_all = np.mean([r.get(k, 0.0) for r in rows])
                lines.append(f"  {k:16s} {np.median(vals):8.1f} {np.percentile(vals, 90):8.1f} "
                             f"{mean_all:8.1f} {mean_all / loop:6.0%}  {len(vals)}")
        return "\n".join(lines)

    def close(self):
        if self.path and self.rows:
            keys = ["t", "state"] + sorted({k for r in self.rows for k in r} - {"t", "state"})
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
        return self.summary()
