"""sensors: what the node measures. Pure inputs: no decisions, no drawing.

Interfaces:
    Camera        open(); read() -> Frame; release(); is_open; frame_size -> (w, h)
    BeaconSource  poll(t) -> BeaconState            is a robot announced? (drives IDLE vs. active)
    PowerMeter    read() -> watts or None           for the power trace on the dashboard
    SystemMonitor read(t, dpu_ms) -> SystemLoad     CPU per core, DPU and GPU busy, PS/PL temperatures

    camera.py     WebcamCamera (USB, V4L2), ScenarioCamera (synthetic frames from scenario.py)
    beacon.py     AlwaysBeacon, ScheduledBeacon, ManualBeacon (all simulated; a BLE listener fits the same interface)
    power.py      HwmonPower (KV260: INA260 via sysfs), PynqRailsPower, NullPower, board_power()
    system.py     SystemMonitor (/proc/stat, AMS temperatures, Mali runtime status), split_power() (estimated)
    scenario.py   DemoScenario: the four-beat demo world, with ground truth for the replay models

Upstream CPSA_2026 read BlueCoin IMUs here (sensor_manager.py); that pipeline was removed (see legacy/).
"""
