"""Clocks. Everything in the node takes time from one of these, so the same code runs live or simulated."""
import time


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
