"""System load: how busy the CPU cores, the DPU (PL) and the GPU are, plus PS/PL temperatures. Linux sysfs only.

    SystemMonitor.read(t, dpu_ms) -> SystemLoad

The KV260 has one power sensor: the INA260 on the SOM supply (sensors/power.py), i.e. total SOM power. There is no
per-rail current telemetry (AMS gives voltages and temperatures only; the DA9130/DA9131 PMICs report nothing), so
the CPU / DPU / GPU split is an *estimate*: split_power() applies a linear model fitted on the board with
tools/power_calibration.py (settings `power.model`).
"""
import glob
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

IIO_ROOT = "/sys/bus/iio/devices"
MALI_STATUS = "/sys/devices/platform/axi/fd4b0000.gpu/power/runtime_status"


@dataclass
class SystemLoad:
    cpu: Tuple[float, ...] = ()           # per core, 0..1, since the last read
    dpu: float = 0.0                      # fraction of wall time the DPU ran (from the models' *_dpu timings)
    gpu: str = "n/a"                      # Mali-400 runtime PM status: "suspended" = off, "active" = in use
    temps: Dict[str, float] = field(default_factory=dict)   # "ps" / "pl" -> deg C

    @property
    def cpu_cores(self):
        """Busy cores (0..n): what the power model uses."""
        return sum(self.cpu)


class SystemMonitor:
    def __init__(self, proc_stat="/proc/stat", iio_root=IIO_ROOT, mali_status=MALI_STATUS, window_s=1.0):
        self.proc_stat, self.mali_status, self.window_s = proc_stat, mali_status, window_s
        self.temp_files = _ams_temps(iio_root)
        self._cpu_prev = None
        self._dpu_ms, self._dpu_t0, self._dpu = 0.0, None, 0.0
        self._t_read = None
        self._last = SystemLoad()

    def read(self, t, dpu_ms=0.0):
        """Called every frame with that frame's DPU milliseconds; sysfs is re-read at most once per window."""
        self._dpu_ms += dpu_ms
        if self._dpu_t0 is None:
            self._dpu_t0 = t
        if self._t_read is not None and t - self._t_read < self.window_s:
            return self._last
        if t > self._dpu_t0:
            self._dpu = min(1.0, self._dpu_ms / 1e3 / (t - self._dpu_t0))
        self._dpu_ms, self._dpu_t0, self._t_read = 0.0, t, t
        self._last = SystemLoad(cpu=self._read_cpu(), dpu=self._dpu, gpu=_read(self.mali_status) or "n/a",
                                temps={k: v for k, v in ((k, _temp(f)) for k, f in self.temp_files.items()) if v is not None})
        return self._last

    def _read_cpu(self):
        try:
            lines = [ln.split() for ln in open(self.proc_stat) if ln.startswith("cpu") and ln[3].isdigit()]
        except OSError:
            return ()
        now = [(sum(map(int, f[1:9])), int(f[4]) + int(f[5])) for f in lines]    # (total, idle + iowait) jiffies
        prev, self._cpu_prev = self._cpu_prev, now
        if prev is None or len(prev) != len(now):
            return ()
        return tuple(max(0.0, min(1.0, 1 - (i1 - i0) / (t1 - t0))) if t1 > t0 else 0.0
                     for (t0, i0), (t1, i1) in zip(prev, now))


def split_power(total_w, load, model):
    """Estimated watts per consumer: {"static", "cpu", "dpu", "gpu", "rest"}, summing to total_w.

    model = settings `power.model`: static_w (idle board: PS + PL + DDR + carrier), cpu_core_w (per fully busy core),
    dpu_w (DPU running 100 %).
    "rest" = what the model does not explain (DDR traffic, USB camera, HDMI, error), can be negative.
    """
    if total_w is None or load is None:
        return None
    parts = {
        "static": model["static_w"],
        "cpu": model["cpu_core_w"] * load.cpu_cores,
        "dpu": model["dpu_w"] * load.dpu,
        "gpu": 0.0,                                  # runtime-suspended: nothing here uses the Mali-400
    }
    parts["rest"] = total_w - sum(parts.values())
    return parts


def _ams_temps(root):
    """AMS (Zynq MPSoC system monitor) temperature channels: {"ps": path, "pl": path}, empty off the board."""
    for dev in sorted(glob.glob(os.path.join(root, "iio:device*"))):
        if _read(os.path.join(dev, "name")) != "ams":
            continue
        out = {}
        for key, pattern in (("ps", "in_temp*_ps_temp_raw"), ("pl", "in_temp*_pl_temp_raw")):
            hits = glob.glob(os.path.join(dev, pattern))
            if hits:
                out[key] = hits[0][:-len("raw")]
        return out
    return {}


def _temp(prefix):
    try:
        raw, offset, scale = (float(_read(prefix + s)) for s in ("raw", "offset", "scale"))
    except (TypeError, ValueError):
        return None
    return (raw + offset) * scale / 1e3                          # IIO: millidegrees C


def _read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None
