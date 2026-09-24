"""Event diary actuator: one diary line per WARN / STOP (upstream CPSA_2026's event diary format).

Level changes also go to the system log. Uses utils.event_log.EventLog.
"""
from utils.types import Decision, Level


class EventDiary:
    def __init__(self, events):
        self.events = events

    def configure(self, cfg):
        pass

    def update(self, decision: Decision, t):
        if not decision.changed:
            return
        self.events.system(f"[decision] -> {decision.level.name} {', '.join(decision.rules)} at {t:.1f}s")
        if decision.level > Level.NONE:
            self.events.event(decision.level.name, decision.rules)
