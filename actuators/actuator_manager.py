"""ActuatorManager: hands every decision to every actuator (same role as upstream CPSA_2026's manager).

core calls `update(decision, t)` once per frame; the manager never decides anything itself. An actuator that
raises is logged and switched off so one broken output cannot stop the others.
"""
from utils.types import Decision


class ActuatorManager:
    def __init__(self, actuators, log=print):
        self.actuators = list(actuators)
        self.log = log

    def configure(self, cfg):
        for a in self.actuators:
            a.configure(cfg)

    def update(self, decision: Decision, t):
        for a in list(self.actuators):
            try:
                a.update(decision, t)
            except Exception as e:                   # noqa: BLE001: keep the other outputs alive
                self.log(f"[actuators] {type(a).__name__} disabled: {e!r}")
                self.actuators.remove(a)

    def names(self):
        return [type(a).__name__ for a in self.actuators]
