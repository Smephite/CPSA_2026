"""Audio actuator: warning tones through the HDMI screen's speakers (via `aplay`).

    WARN  two short beeps, repeated every repeat_s while WARN holds
    STOP  fast two-tone alarm; pre-empts a playing WARN tone
"""
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

from utils.types import Decision, Level


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

    def update(self, decision: Decision, t):
        level, changed = decision.level, decision.changed
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
    """Silent stand-in (tests, --fast runs)."""

    def configure(self, cfg):
        pass

    def update(self, decision: Decision, t):
        pass
