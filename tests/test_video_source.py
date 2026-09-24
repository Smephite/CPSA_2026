import time

import cv2
import numpy as np
import pytest

from sensors.camera import VideoFileCamera
from tools.record import CameraRecorder, RecordView
from utils.clock import SimClock
from utils.types import Frame

TIMES = [0.0, 0.05, 0.13, 0.20, 0.31]           # uneven, as a real webcam delivers them


@pytest.fixture
def clip(tmp_path):
    path = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(len(TIMES)):
        w.write(np.full((48, 64, 3), 40 * i, np.uint8))    # frame i has brightness 40 i
    w.release()
    (tmp_path / "clip.csv").write_text("frame,t\n" + "".join(f"{i},{t}\n" for i, t in enumerate(TIMES)))
    return path


def _brightness(frame):
    return round(frame.image.mean() / 40)


def test_simulated_clock_plays_every_frame_at_its_recorded_time(clip):
    cam = VideoFileCamera(SimClock(), clip)
    cam.open()
    got = [cam.read() for _ in TIMES]
    assert [_brightness(f) for f in got] == [0, 1, 2, 3, 4]
    assert np.allclose([f.t for f in got], TIMES) and cam.finished and cam.frame_size == (64, 48)


class SlowClock(SimClock):
    """A live node that is busy for 0.1 s per step: the camera must drop the frames it missed."""

    def advance(self):
        self.t += 0.1


def test_live_pacing_drops_missed_frames(clip):
    clock = SlowClock()
    cam = VideoFileCamera(clock, clip)
    cam.open()
    seen = []
    while not cam.finished:
        seen.append(_brightness(cam.read()))
        clock.advance()
    assert seen == [0, 1, 3, 4]                  # at t=0.2 frame 2 (0.13) is stale: frame 3 (0.20) is due


def test_loop_continues_the_time(clip):
    cam = VideoFileCamera(SimClock(), clip, loop=True)
    cam.open()
    got = [cam.read() for _ in range(len(TIMES) + 2)]
    assert [_brightness(f) for f in got][-2:] == [0, 1] and not cam.finished
    ts = [f.t for f in got]
    assert all(b > a for a, b in zip(ts, ts[1:]))


@pytest.mark.parametrize("n", [1, 2])
def test_record_view_draws_every_camera(n):
    img = np.full((480, 640, 3), 200, np.uint8)
    cams = [(f"video{i}", img, 50, 15.0) for i in range(n)]
    out = RecordView("out/clip.avi").render({"t": 3.2, "cams": cams})
    slot = (1280 - 260) // n
    assert out.shape == (720, 1280, 3) and all(out[360, i * slot + slot // 2].mean() == 200 for i in range(n))


class FakeWebcam:
    """Frames i = 0, 1, 2 ... at t = 0.1 i; then blocks like a camera with nothing new."""

    def __init__(self, clock, n):
        self.clock, self.n, self.i, self.released = clock, n, 0, False

    def read(self):
        if self.i >= self.n:
            time.sleep(0.01)
            raise RuntimeError("webcam read failed")
        self.i += 1
        return Frame(image=np.full((48, 64, 3), 40 * (self.i - 1), np.uint8), t=0.1 * (self.i - 1) + 5.0)

    def release(self):
        self.released = True


def test_recorded_clip_replays_with_its_times(tmp_path):
    cam = FakeWebcam(None, 4)
    rec = CameraRecorder(cam, str(tmp_path / "c.avi"), t0=5.0)
    rec.start()
    rec._thread.join(timeout=5)
    assert rec.frames == 4 and str(rec.error) == "webcam read failed" and cam.released
    replay = VideoFileCamera(SimClock(), str(tmp_path / "c.avi"))
    replay.open()
    got = [replay.read() for _ in range(4)]
    assert [_brightness(f) for f in got] == [0, 1, 2, 3] and np.allclose([f.t for f in got], [0, 0.1, 0.2, 0.3])
