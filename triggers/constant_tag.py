# constant_tag.py
# Stand-in for the removed IMU pipeline (BlueCoin sensors -> synchronizer -> buffer -> stereotypy classifier).
# Publishes a fixed stereotypy tag into the shared event queue at a fixed rate, like the classifier
# published one event per sliding window. Tag 2 (DANGEROUS) runs the YOLO -> MoveNet video branch.

import threading
import uuid
from datetime import datetime

from utils.event_queue import get_event_queue, enqueue_drop_oldest
from utils.logger import log_system


class ConstantTagSource(threading.Thread):
    def __init__(self, tag=2, period_sec=0.5, source="constant_tag"):
        super().__init__(daemon=True)
        self.tag = int(tag)
        self.period_sec = period_sec
        self.source = source
        self.stop_event = threading.Event()

    def make_event(self):
        # Same fields as the former StereotipyClassifier events.
        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "source": self.source,
            "feature_vector": None,
            "stereotipy_tag": self.tag,
        }

    def run(self):
        log_system(f"[ConstantTagSource] Publishing tag={self.tag} every {self.period_sec}s")
        q = get_event_queue()
        while not self.stop_event.is_set():
            enqueue_drop_oldest(q, self.make_event(), kind="event")
            self.stop_event.wait(self.period_sec)

    def stop(self):
        self.stop_event.set()
        if threading.current_thread() is not self and self.is_alive():
            self.join(timeout=2.0)
