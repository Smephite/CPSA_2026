"""Cameras: `open()`, `read() -> Frame`, `release()`, `is_open`, `frame_size`. Opened only while not IDLE."""
import csv
import os

import cv2

from utils.types import Frame


class WebcamCamera:
    """USB webcam via V4L2; opened only while the node is not IDLE."""

    def __init__(self, clock, index=0, width=640, height=480, fps=30):
        self.clock, self.index, self.size, self.fps = clock, index, (width, height), fps
        self.cap = None

    @property
    def is_open(self):
        return self.cap is not None

    def open(self):
        if self.cap is None:
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open /dev/video{self.index}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap

    def read(self):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return Frame(image=img, t=self.clock.now())

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def frame_size(self):
        return self.size


class ScenarioCamera:
    """Renders the synthetic scenario at `fps` (scenario time = clock time)."""

    def __init__(self, clock, scenario, fps=15):
        self.clock, self.scenario, self.fps = clock, scenario, fps
        self.is_open = False

    def open(self):
        self.is_open = True

    def read(self):
        self.clock.sleep(1.0 / self.fps)
        t = self.clock.now()
        image, truth = self.scenario.render(t)
        return Frame(image=image, t=t, truth=truth)

    def release(self):
        self.is_open = False

    @property
    def frame_size(self):
        return self.scenario.size


class VideoFileCamera:
    """Plays a clip recorded by tools/record.py (clip.avi + clip.csv with each frame's capture time) as a camera.

    Live clock: frames come at their recorded times; a frame whose time has passed is dropped, as a live webcam
    would drop it while the node is busy. Simulated clock (--fast): every frame, in order, as fast as the node runs.
    Without the .csv, frames are spaced by the file's nominal fps. loop=True starts over at the end (time keeps
    counting up); otherwise `finished` turns True after the last frame and main.py stops.
    """

    def __init__(self, clock, path, loop=False):
        self.clock, self.path, self.loop = clock, path, loop
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.times = self._load_times(os.path.splitext(path)[0] + ".csv") or [i / fps for i in range(n)]
        if not self.times:
            raise RuntimeError(f"no frames in {path}")
        self.duration = self.times[-1] - self.times[0] + 1.0 / fps
        self.cap = None

    @staticmethod
    def _load_times(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return [float(row["t"]) for row in csv.DictReader(f)]

    @property
    def is_open(self):
        return self.cap is not None

    @property
    def finished(self):
        return self.cap is not None and not self.loop and self.i >= len(self.times)

    def open(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)
            self.i, self.offset, self.img = 0, self.clock.now() - self.times[0], None

    def _next(self):
        """Advance one frame -> its clock time, or None at the end of a clip that does not loop."""
        if self.i >= len(self.times):
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.i, self.offset = 0, self.offset + self.duration - self.times[0]
        ok, img = self.cap.read()
        if not ok:                               # fewer frames than timestamps: treat as the end
            self.i = len(self.times)
            return self._next() if self.loop else None
        self.img, t = img, self.offset + self.times[self.i]
        self.i += 1
        return t

    def read(self):
        t = self._next()
        if t is None:
            return Frame(image=self.img, t=self.clock.now())
        self.clock.sleep(t - self.clock.now())               # wait for the frame's time (simulated: jump there)
        while self.i < len(self.times) and self.offset + self.times[self.i] <= self.clock.now():
            t_next = self._next()                            # live and behind: drop frames, keep the newest
            if t_next is None:
                break
            t = t_next
        return Frame(image=self.img, t=t)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def frame_size(self):
        return self.size
