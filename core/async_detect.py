"""AsyncDetector: runs the detector cascade in its own thread so poses and rules do not wait for it.

YOLOv3 takes ~76 ms on the DPU plus ~15 ms of ARM work, longer than a 15 Hz pose period. The node submits a
frame when a detection is due and the worker is idle, keeps tracking with predicted boxes meanwhile, and
applies the result (with the frame's own time stamp) when it is ready. Synchronous mode (threaded=False) runs
the same call inline: used on the simulated clock so tests and --fast runs stay deterministic.
"""
import threading
import time


class AsyncDetector:
    def __init__(self, cascade, threaded=True):
        self.cascade, self.threaded = cascade, threaded
        self._result = None                      # (detections, frame time, ms, stage times)
        self._busy = False
        self._lock = threading.Lock()

    @property
    def busy(self):
        return self._busy

    def submit(self, frame, names, expected, sensitive, t, sudden):
        """Start a detection on `frame` (taken at t). Ignored if one is still running."""
        if self._busy:
            return False
        self._busy = True

        def work():
            t0 = time.perf_counter()
            try:
                dets = self.cascade.detect(frame, names, expected, sensitive, t, sudden)
                times = self.cascade.take_times()
            except Exception as e:               # noqa: BLE001: surface on the main thread
                dets, times = e, {}
            with self._lock:
                self._result = (dets, t, (time.perf_counter() - t0) * 1e3, times)
            self._busy = False

        if self.threaded:
            threading.Thread(target=work, name="detector", daemon=True).start()
        else:
            work()
        return True

    def poll(self):
        """-> (detections, frame time, ms, stage times) of a finished detection, or None."""
        with self._lock:
            res, self._result = self._result, None
        if res is not None and isinstance(res[0], Exception):
            raise res[0]
        return res
