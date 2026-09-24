"""Edges of the node: beacon, clocks, cameras, audio, robot link, power, event log.

Beacon: only simulated sources for now. `BeaconSource.poll(t) -> BeaconState` is the interface a real BLE
listener (USB dongle + BlueZ, robot ID / type / reach in the advertisement) implements later; running it as a
separate process that writes to a socket would keep PYNQ's pinned venv untouched.
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import cv2
import numpy as np

from guardian.types import BeaconState, Frame, Level

# ---------------------------------------------------------------- beacon


class BeaconSource:
    def poll(self, t) -> BeaconState:
        raise NotImplementedError


class AlwaysBeacon(BeaconSource):
    def __init__(self, robot_id="sim-robot"):
        self.state = BeaconState(True, robot_id, "humanoid")

    def poll(self, t):
        return self.state


class ScheduledBeacon(BeaconSource):
    """Present during the given [(t_on, t_off), ...] intervals (scenario time)."""

    def __init__(self, intervals, robot_id="sim-robot"):
        self.intervals = intervals
        self.robot_id = robot_id

    def poll(self, t):
        on = any(a <= t < b for a, b in self.intervals)
        return BeaconState(on, self.robot_id if on else "", "humanoid" if on else "")


class ManualBeacon(BeaconSource):
    """Toggled from the keyboard (see guardian_main.py): 'b' + Enter."""

    def __init__(self, present=False, robot_id="sim-robot"):
        self.present = present
        self.robot_id = robot_id

    def toggle(self):
        self.present = not self.present

    def poll(self, t):
        return BeaconState(self.present, self.robot_id if self.present else "", "humanoid" if self.present else "")


# ---------------------------------------------------------------- clocks and cameras


class RealClock:
    def __init__(self):
        self.t0 = time.monotonic()

    def now(self):
        return time.monotonic() - self.t0

    def sleep(self, s):
        time.sleep(max(0.0, s))


class SimClock:
    """Scenario time that advances only when the node sleeps (runs as fast as the CPU allows)."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class WebcamCamera:
    """USB webcam via V4L2; opened only while the node is not IDLE."""

    def __init__(self, clock, index=0, width=640, height=480, fps=30):
        self.clock, self.index, self.size, self.fps = clock, index, (width, height), fps
        self.cap = None

    @property
    def is_open(self):
        return self.cap is not None

    def open(self):
        if self.cap is None:
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open /dev/video{self.index}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap

    def read(self):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return Frame(image=img, t=self.clock.now())

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def frame_size(self):
        return self.size


class ScenarioCamera:
    """Renders the synthetic scenario at `fps` (scenario time = clock time)."""

    def __init__(self, clock, scenario, fps=15):
        self.clock, self.scenario, self.fps = clock, scenario, fps
        self.is_open = False

    def open(self):
        self.is_open = True

    def read(self):
        self.clock.sleep(1.0 / self.fps)
        t = self.clock.now()
        image, truth = self.scenario.render(t)
        return Frame(image=image, t=t, truth=truth)

    def release(self):
        self.is_open = False

    @property
    def frame_size(self):
        return self.scenario.size


# ---------------------------------------------------------------- audio (HDMI screen speakers on the board)


def _tone_wav(path, parts, rate=22050):
    """parts: [(freq_hz or 0 for silence, seconds), ...] -> 16-bit mono wav."""
    chunks = []
    for f, s in parts:
        n = int(rate * s)
        t = np.arange(n) / rate
        x = np.sin(2 * np.pi * f * t) if f else np.zeros(n)
        env = np.minimum(1, np.minimum(t, s - t) / 0.01) if n else x   # 10 ms ramps, no clicks
        chunks.append(0.6 * x * env)
    data = (np.concatenate(chunks) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


class AudioOut:
    """WARN: two short beeps. STOP: fast two-tone alarm. Played with `aplay` (non-blocking, no overlap)."""

    def __init__(self, cfg, log=print):
        a = cfg["audio"]
        self.log = log
        self.configure(cfg)
        self.device = a["device"]
        self.proc = None
        self.last = {Level.WARN: -1e9, Level.STOP: -1e9}
        self.dir = tempfile.mkdtemp(prefix="guardian_audio_")
        self.files = {Level.WARN: os.path.join(self.dir, "warn.wav"), Level.STOP: os.path.join(self.dir, "stop.wav")}
        _tone_wav(self.files[Level.WARN], [(880, 0.12), (0, 0.08), (880, 0.12)])
        _tone_wav(self.files[Level.STOP], [(1200, 0.15), (800, 0.15)] * 3)
        if a["enabled"] and not self.enabled:
            log("[audio] aplay not found: audio disabled")

    def configure(self, cfg):
        a = cfg["audio"]
        self.enabled = a["enabled"] and shutil.which("aplay") is not None
        self.repeat_s = a["repeat_s"]

    def update(self, level, t, changed):
        if not self.enabled or level == Level.NONE:
            return
        if not changed and t - self.last[level] < self.repeat_s:
            return
        if self.proc is not None and self.proc.poll() is None:
            if not (changed and level == Level.STOP):
                return
            self.proc.terminate()                     # STOP pre-empts a WARN tone
        cmd = ["aplay", "-q"] + (["-D", self.device] if self.device else []) + [self.files[level]]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.last[level] = t
        except OSError as e:
            self.log(f"[audio] {e}")
            self.enabled = False


class NullAudio:
    def configure(self, cfg):
        pass

    def update(self, level, t, changed):
        pass


# ---------------------------------------------------------------- robot link


class LogRobotLink:
    """'slow' on WARN, 'stop' on STOP, 'resume' when clear. Future: BLE/Wi-Fi message to the robot."""

    def __init__(self, log=print):
        self.log = log
        self.last = None

    def update(self, level, robot_id):
        msg = {Level.NONE: "resume", Level.WARN: "slow", Level.STOP: "stop"}[level]
        if msg != self.last:
            self.log(f"[robot-link] -> {robot_id or '?'}: {msg}")
            self.last = msg


# ---------------------------------------------------------------- power


class NullPower:
    label = "n/a"

    def read(self):
        return None


class PynqRailsPower:
    """Sum of on-board rail power sensors via pynq.get_rails() (board only; names vary, see README)."""

    label = "W"

    def __init__(self, rails=()):
        import pynq                                   # noqa: PLC0415 (board only)
        all_rails = pynq.get_rails()
        names = rails or [n for n, r in all_rails.items() if getattr(r, "power", None) is not None]
        self.rails = [all_rails[n] for n in names if n in all_rails]
        if not self.rails:
            raise RuntimeError(f"no power rails found (have {sorted(all_rails)})")

    def read(self):
        return float(sum(r.power.value for r in self.rails))


class HwmonPower:
    """Power straight from Linux hwmon sysfs (`power1_input`, microwatts), no libsensors needed.

    On the KV260 this is the carrier's INA260 (`ina260_u14`) on the SOM supply: total SOM power.
    PYNQ's libsensors fails to initialise on the board image, so pynq.get_rails() returns nothing there.
    """

    label = "W"

    def __init__(self, names=("ina260",), root="/sys/class/hwmon"):
        self.files = []
        for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            path = os.path.join(root, d)
            try:
                name = open(os.path.join(path, "name")).read().strip()
            except OSError:
                continue
            f = os.path.join(path, "power1_input")
            if any(name.startswith(n) for n in names) and os.path.exists(f):
                self.files.append(f)
        if not self.files:
            raise RuntimeError(f"no hwmon power sensor matching {list(names)} under {root}")

    def read(self):
        return sum(int(open(f).read()) for f in self.files) / 1e6


def board_power(rails=()):
    """pynq.get_rails() if it finds power sensors, else hwmon sysfs (INA260)."""
    try:
        return PynqRailsPower(rails)
    except Exception:
        return HwmonPower()


# ---------------------------------------------------------------- event log (upstream CPSA_2026 logger)


class EventLog:
    """System log + event diary via upstream's utils.logger (log files under log_base_path)."""

    def __init__(self, enabled=True, echo=True):
        self.enabled, self.echo = enabled, echo
        self._lock = threading.Lock()
        if enabled:
            from utils.logger import log_event, log_system   # noqa: PLC0415
            self._log_system, self._log_event = log_system, log_event

    def system(self, msg, level="INFO"):
        if self.enabled:
            self._log_system(msg, level=level)
        elif self.echo:
            print(msg)

    def event(self, name, rules, source="guardian"):
        if self.enabled:
            from datetime import datetime               # noqa: PLC0415
            with self._lock:
                self._log_event(timestamp=datetime.now().isoformat(), feature_type="rules", event=name,
                                actuations=[{"target": r, "params": {}} for r in rules], source=source)
        elif self.echo:
            print(f"[event] {name} {', '.join(rules)}")
