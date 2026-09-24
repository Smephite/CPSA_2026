"""Robot link actuator: tells the robot to slow / stop / resume. Only logs today (future: BLE / Wi-Fi message)."""
from utils.types import Decision, Level


class LogRobotLink:
    """'slow' on WARN, 'stop' on STOP, 'resume' when clear. Future: BLE/Wi-Fi message to the robot."""

    def __init__(self, log=print):
        self.log = log
        self.last = None

    def configure(self, cfg):
        pass

    def update(self, decision: Decision, t):
        msg = {Level.NONE: "resume", Level.WARN: "slow", Level.STOP: "stop"}[decision.level]
        if msg != self.last:
            self.log(f"[robot-link] -> {decision.robot_id or '?'}: {msg}")
            self.last = msg
