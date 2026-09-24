"""Beacon: 'is a robot near?' (ID, type, reach). Drives IDLE vs. active.

Only simulated sources exist. A real BLE listener (USB dongle + BlueZ; robot ID, type and reach in the
advertisement) implements the same `poll(t) -> BeaconState`; run it as a separate process writing to a socket so
PYNQ's pinned venv stays untouched.
"""
from utils.types import BeaconState


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
