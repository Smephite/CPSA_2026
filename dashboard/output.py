"""OutputWorker: draws the dashboard and feeds the sinks off the pipeline's critical path.

On the board, drawing the dashboard (~75-115 ms) plus the HDMI write cost more than the whole perception loop.
In threaded mode a worker thread takes the latest snapshot, renders it and hands the image to every sink; a newer
snapshot replaces one that was not drawn yet (frames are skipped, never queued). OpenCV drawing releases the GIL,
so this runs on another core. threaded=False draws inline: every frame, deterministic (--fast, --record).
"""
import threading
import time


class OutputWorker:
    def __init__(self, dashboard, sinks, threaded=True, on_frame=None, profiler=None, log=print):
        self.dashboard, self.sinks = dashboard, list(sinks)
        self.on_frame, self.profiler, self.log = on_frame, profiler, log
        self.threaded = threaded
        self.frames = self.skipped = 0
        self._pending = None
        self._cond = threading.Condition()
        self._stop = False
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._loop, name="output", daemon=True)
            self._thread.start()

    def submit(self, snap):
        if not self.threaded:
            self._draw(snap)
            return
        with self._cond:
            if self._pending is not None:
                self.skipped += 1
            self._pending = snap
            self._cond.notify()

    def _loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None or self._stop)
                if self._stop and self._pending is None:
                    return
                snap, self._pending = self._pending, None
            try:
                self._draw(snap)
            except Exception as e:                   # noqa: BLE001: never take the pipeline down
                self.log(f"[output] {e!r}")

    def _draw(self, snap):
        t0 = time.perf_counter()
        view = self.dashboard.render(snap)
        times = {"render": (time.perf_counter() - t0) * 1e3}
        for s in list(self.sinks):
            t0 = time.perf_counter()
            try:
                s.show(view)
            except Exception as e:                   # e.g. HDMI without a monitor: keep the other sinks
                self.log(f"{type(s).__name__} disabled: {e}")
                self.sinks.remove(s)
            times[f"sink:{type(s).__name__}"] = (time.perf_counter() - t0) * 1e3
        if self.on_frame is not None:
            self.on_frame(view, snap)
        self.frames += 1
        if self.profiler is not None:
            times["output"] = sum(times.values())
            self.profiler.record(snap.t, "OUTPUT", times)

    def close(self):
        if self._thread is not None:
            with self._cond:
                self._stop = True
                self._cond.notify()
            self._thread.join(timeout=5)
        for s in self.sinks:
            try:
                s.close()
            except Exception:                        # noqa: BLE001
                pass
