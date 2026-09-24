"""Event log: system messages and the event diary, through upstream CPSA_2026's utils.logger."""
import threading


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
