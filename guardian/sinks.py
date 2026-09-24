"""Where the dashboard goes: HDMI monitor (board), MJPEG web stream, video file, snapshot."""
import http.server
import json
import threading

import cv2

from guardian.webui import PAGE


class MjpegSink:
    """http://<host>:<port>/ page, /stream MJPEG, /snapshot.jpg last frame. Demo only: frames leave the node.

    With a `tuning.Tuning`: GET /api/params lists the tunable settings, POST /api/params {path: value} and
    POST /api/reset {"paths": [...]} (empty = all) stage changes; the main loop applies them at the next frame.
    No authentication: anyone who can reach the port can change thresholds (lab use).
    """

    def __init__(self, port=8080, quality=80, tuning=None):
        self.jpeg = None
        self.cond = threading.Condition()
        self.quality = quality
        sink = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, ctype, data):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if tuning is None or self.path not in ("/api/params", "/api/reset"):
                    return self.send_error(404)
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("expected a JSON object")
                except ValueError as e:
                    return self._json({"ok": {}, "errors": {"": f"bad request: {e}"}}, 400)
                if self.path == "/api/params":
                    ok, errors = tuning.stage(body)
                else:
                    ok, errors = tuning.reset(body.get("paths"))
                self._json({"ok": {k: str(v) for k, v in ok.items()}, "errors": errors}, 200 if not errors else 422)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send("text/html; charset=utf-8", PAGE.encode())
                elif self.path == "/api/params":
                    if tuning is None:
                        return self.send_error(404)
                    self._json(tuning.describe())
                elif self.path == "/snapshot.jpg":
                    self._send("image/jpeg", sink.jpeg or b"")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            with sink.cond:
                                sink.cond.wait_for(lambda: sink.jpeg is not None and sink.jpeg is not last, timeout=1)
                                data = sink.jpeg
                            if data is None or data is last:
                                continue
                            last = data
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

        self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.port = self.httpd.server_address[1]
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def show(self, img):
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
        with self.cond:
            self.jpeg = data
            self.cond.notify_all()

    def close(self):
        self.httpd.shutdown()


class VideoFileSink:
    def __init__(self, path, fps=15.0):
        self.path, self.fps, self.writer = path, fps, None

    def show(self, img):
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(img)

    def close(self):
        if self.writer is not None:
            self.writer.release()


class DisplayPortSink:
    """Full-screen HDMI/DP output via PYNQ's DisplayPort (board only; the desktop must be stopped:
    `systemctl stop gdm`). Opened lazily in the calling thread with its own asyncio loop.

    pixel_format: 'rgb' or 'bgr'. With PIXEL_BGR the colours looked swapped on the board, so 'rgb' is the default;
    not yet confirmed on the monitor.
    """

    def __init__(self, width=1280, height=720, pixel_format="rgb"):
        self.want = (width, height)
        self.pixel_format = pixel_format
        self.dp = self.mode = None

    def _open(self):
        import asyncio                                            # noqa: PLC0415
        from pynq.lib.video import DisplayPort, PIXEL_BGR, PIXEL_RGB   # noqa: PLC0415
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.dp = DisplayPort(event_loop=loop)
        modes = self.dp.modes
        if not modes:
            raise RuntimeError("no display modes: is a monitor connected?")
        exact = [m for m in modes if (m.width, m.height) == self.want]
        self.mode = (exact or sorted(modes, key=lambda m: m.width * m.height, reverse=True))[0]
        self.dp.configure(self.mode, PIXEL_RGB if self.pixel_format == "rgb" else PIXEL_BGR)

    def show(self, img):
        if self.dp is None:
            self._open()
        h, w = self.mode.height, self.mode.width
        out = self.dp.newframe()
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        out[:] = img
        self.dp.writeframe(out)

    def close(self):
        if self.dp is not None:
            self.dp.close()
            self.dp = None
