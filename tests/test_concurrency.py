"""Threaded parts of the live node: async detector, output worker, lazy MJPEG encoding."""
import time
import urllib.request

import numpy as np
import pytest

from core.async_detect import AsyncDetector
from dashboard.output import OutputWorker
from dashboard.sinks import MjpegSink


class SlowCascade:
    def __init__(self, delay=0.05, fail=False):
        self.delay, self.fail, self.calls = delay, fail, 0

    def detect(self, frame, names, expected, sensitive, t, sudden):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("dpu gone")
        return [f"det@{t}"]

    def take_times(self):
        return {"det_dpu": 1.0}


def wait_result(det, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        r = det.poll()
        if r is not None:
            return r
        time.sleep(0.005)
    raise AssertionError("no result")


def test_async_detector_returns_later_with_the_frame_time():
    det = AsyncDetector(SlowCascade(), threaded=True)
    assert det.submit("frame", "m", [], False, 3.25, False)
    assert det.busy and det.poll() is None                       # not ready yet: the node keeps tracking
    assert not det.submit("frame2", "m", [], False, 3.30, False)  # one at a time
    dets, t, ms, times = wait_result(det)
    assert dets == ["det@3.25"] and t == 3.25 and ms >= 40 and times == {"det_dpu": 1.0}
    assert not det.busy


def test_async_detector_surfaces_errors_on_poll():
    det = AsyncDetector(SlowCascade(fail=True), threaded=True)
    det.submit("frame", "m", [], False, 0.0, False)
    with pytest.raises(RuntimeError):
        wait_result(det)


def test_sync_detector_is_inline():
    det = AsyncDetector(SlowCascade(delay=0.0), threaded=False)
    det.submit("frame", "m", [], False, 1.0, False)
    assert det.poll()[0] == ["det@1.0"]


class Recorder:
    def __init__(self):
        self.shown = []

    def show(self, img):
        self.shown.append(img)

    def close(self):
        pass


class SlowDashboard:
    def render(self, snap):
        time.sleep(0.02)
        return np.full((4, 4, 3), snap, np.uint8)


def test_output_worker_skips_to_the_newest_snapshot():
    rec = Recorder()
    out = OutputWorker(SlowDashboard(), [rec], threaded=True)
    for i in range(20):
        out.submit(i)                                            # faster than it can draw
    time.sleep(0.2)
    out.close()
    assert rec.shown and rec.shown[-1][0, 0, 0] == 19            # the last snapshot is always drawn
    assert out.skipped > 0 and out.frames + out.skipped == 20


def test_mjpeg_encodes_only_on_demand():
    sink = MjpegSink(0)
    try:
        sink.show(np.zeros((32, 32, 3), np.uint8))
        assert sink._jpeg is None                                # nobody watching: no encoding
        data = urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/snapshot.jpg", timeout=2).read()
        assert data[:2] == b"\xff\xd8"                           # JPEG
        assert sink.jpeg() is data or sink.jpeg() == data        # cached for the same frame
    finally:
        sink.close()
