"""Cameras: `open()`, `read() -> Frame`, `release()`, `is_open`, `frame_size`. Opened only while not IDLE."""
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
