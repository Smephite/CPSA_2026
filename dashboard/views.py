"""Views: camera overlay, virtual scene (robot / nonna sprites on the floor), and the composed dashboard.

People are drawn as sprites (dashboard/assets: robot.png for the robot, nonna.png for every human), not as
keypoints; the poses are still estimated and used by the rules. The virtual scene puts each person at its floor
position from `WorldModel` in an oblique projection: a floor grid, shadows, the sprites (turned on their side
when the box says lying), predicted ghosts at the end of the look-ahead, the robot's reach zone tinted by the
decision, and a label on each endangered nonna with the rule and time to contact.
"""
import functools
import os
from collections import deque

import cv2
import numpy as np

from sensors.system import split_power
from utils.geometry import dilate_polygon
from utils.types import Level, NodeState

TITLE = "NonnaGuardian"
ROBOT = (40, 120, 235)         # BGR
HUMAN = (200, 130, 40)
NAMES = {"robot": "robot", "human": "nonna", None: "nonna"}      # role -> label on screen
LEVEL_COL = {Level.NONE: (90, 170, 60), Level.WARN: (0, 170, 255), Level.STOP: (40, 40, 230)}
BG, PANEL, TEXT, MUTED = (24, 24, 26), (34, 35, 38), (225, 225, 225), (140, 140, 145)
FONT = cv2.FONT_HERSHEY_SIMPLEX

SPLIT_COL = {"static": (110, 110, 118), "cpu": (230, 160, 60), "dpu": (60, 200, 250), "gpu": (200, 90, 200),
             "rest": (80, 80, 86)}                                    # BGR, power split bar


def put(img, text, org, scale=0.5, col=TEXT, th=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, col, th, cv2.LINE_AA)


def colorize_depth(z, unit_mm, near_m, far_m):
    """uint16 depth (raw units) -> BGR: near = red ... far = blue (turbo), no depth = black."""
    m = z.astype(np.float32) * (unit_mm / 1e3)
    u8 = np.clip((far_m - m) / (far_m - near_m) * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    out[z == 0] = 0
    return out


def _fit(img, w, h):
    box = np.full((h, w, 3), BG, np.uint8)
    k = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    oy, ox = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    box[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
    return box


# ---------------------------------------------------------------- sprites

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _load_sprite(name):
    img = cv2.imread(os.path.join(ASSETS, name), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    if img[:, :, 3].min() == 255:           # no transparency in the file: clear the black background around the figure
        _, lab = cv2.connectedComponents((img[:, :, :3].max(axis=2) < 40).astype(np.uint8), connectivity=4)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        alpha = np.where(np.isin(lab, border[border > 0]), 0, 255).astype(np.uint8)
        img[:, :, 3] = cv2.GaussianBlur(alpha, (3, 3), 0)
    return img


SPRITES = {"robot": _load_sprite("robot.png"), "human": _load_sprite("nonna.png")}


@functools.lru_cache(maxsize=512)
def sprite(role, height, lying=False):
    """BGRA sprite for a role (robot, else nonna), `height` px tall before turning; None if it cannot be drawn.

    Cached per size: callers round `height` (to 4 px) so a moving person does not resize every frame.
    """
    src = SPRITES["robot" if role == "robot" else "human"]
    if src is None or height < 4:
        return None
    w = max(1, round(src.shape[1] * height / src.shape[0]))
    img = cv2.resize(src, (w, height), interpolation=cv2.INTER_AREA)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if lying else img


def paste(img, spr, x, y, alpha=1.0):
    """Alpha-blend a BGRA sprite onto a BGR image with its top-left corner at (x, y), clipped to the image."""
    h, w = spr.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    roi = img[y0:y1, x0:x1]
    roi[:] = (s[:, :, :3] * a + roi * (1.0 - a)).astype(np.uint8)


def _lying(box, aspect):
    return (box[2] - box[0]) / max(box[3] - box[1], 1.0) > aspect


# ---------------------------------------------------------------- camera overlay



def _when(d, snap, digits):
    """'now' for a STOP, 'in 0.8 s' for a predicted one, else how close the human is ('at 1.4 m')."""
    if d.level_now == Level.STOP:
        return "now"
    if d.ttc_s is not None:
        return f"in {d.ttc_s:.{digits}f} s"
    h, body = snap.pair_human, snap.body_gap_m
    return f"at {body:.{digits}f} m" if body is not None and h is not None and h.id == d.human_id else ""


def camera_view(snap, engine, lying_aspect):
    img = snap.frame.image.copy()
    for line in engine.lines_px:
        cv2.line(img, tuple(line[:2].astype(int)), tuple(line[2:].astype(int)), (60, 60, 200), 3)
    for tr in snap.tracks:
        col = ROBOT if tr.role == "robot" else HUMAN if tr.role == "human" else MUTED
        x1, y1, x2, y2 = tr.box.astype(int)
        lying = _lying(tr.box, lying_aspect)
        spr = sprite(tr.role, int(0.55 * ((x2 - x1) if lying else (y2 - y1))) // 4 * 4, lying)
        if spr is not None:                       # the sprite over the person, in place of the keypoints
            paste(img, spr, (x1 + x2 - spr.shape[1]) // 2, (y1 + y2 - spr.shape[0]) // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        facing = f"  faces {tr.facing}" if getattr(tr, "facing", None) and snap.t - tr.facing_t <= 1.0 else ""
        put(img, f"{NAMES.get(tr.role, 'nonna')} #{tr.id}{facing}", (x1 + 3, max(14, y1 - 5)), 0.45, col, 1)
        fresh = tr.kp is not None and snap.t - tr.kp_t <= 0.5
        if fresh:
            if tr.role == "robot":
                hull = engine.arm_hull(tr.kp)
                if hull is not None and snap.m_per_px:
                    ring = dilate_polygon(hull, engine.c["reach_margin_m"] / snap.m_per_px)
                    cv2.polylines(img, [ring.astype(np.int32)], True, LEVEL_COL[snap.level], 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- virtual scene


class SceneRenderer:
    """Oblique projection of the floor: X right, Z away from the camera, Y up."""

    def __init__(self, w, h, scale=78.0):
        self.w, self.h, self.s = w, h, scale
        self.zc = 4.0

    def project(self, X, Y, Z):
        s = self.s
        dz = Z - self.zc
        return np.stack([self.w / 2 + s * (X + 0.55 * dz), self.h * 0.78 - s * (Y + 0.42 * dz)], -1)

    def _floor(self, img):
        for X in np.arange(-4, 4.01, 1.0):
            a, b = self.project(np.array([X, X]), np.zeros(2), np.array([self.zc - 2.5, self.zc + 2.5]))
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (58, 60, 64), 1, cv2.LINE_AA)
        for Z in np.arange(self.zc - 2.5, self.zc + 2.51, 1.0):
            a, b = self.project(np.array([-4.0, 4.0]), np.zeros(2), np.array([Z, Z]))
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (58, 60, 64), 1, cv2.LINE_AA)
        cam = self.project(np.array([0.0]), np.array([0.0]), np.array([self.zc - 2.5]))[0]
        cv2.drawMarker(img, tuple(cam.astype(int)), (120, 120, 120), cv2.MARKER_TRIANGLE_UP, 14, 2)
        put(img, "node", (cam[0] + 10, cam[1] + 4), 0.4, MUTED)

    def _floor_circle(self, img, X, Z, r, col, alpha):
        ang = np.linspace(0, 2 * np.pi, 48)
        pts = self.project(X + r * np.cos(ang), np.zeros(48), Z + r * np.sin(ang)).astype(np.int32)
        over = img.copy()
        cv2.fillPoly(over, [pts], col, cv2.LINE_AA)
        cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)

    def _figure(self, img, role, X, Z, person_h, lying, alpha=1.0):
        """Sprite standing on the floor at (X, Z), person_h metres tall. -> top-centre point (for labels)."""
        foot = self.project(np.array([X]), np.zeros(1), np.array([Z]))[0]
        head = self.project(np.array([X]), np.array([person_h]), np.array([Z]))[0]
        if alpha == 1.0:
            cv2.ellipse(img, tuple(foot.astype(int)), (int(0.35 * self.s), int(0.12 * self.s)), 0, 0, 360,
                        (18, 18, 20), -1, cv2.LINE_AA)
        spr = sprite(role, int(max(8.0, foot[1] - head[1])) // 4 * 4, lying)
        if spr is None:                            # no image file: a plain disc
            cv2.circle(img, (int(foot[0]), int(foot[1]) - 20), 18, ROBOT if role == "robot" else HUMAN, -1)
            return foot - np.array([0.0, 40.0])
        h, w = spr.shape[:2]
        paste(img, spr, int(foot[0] - w / 2), int(foot[1] - h), alpha)
        return np.array([foot[0], foot[1] - h])

    def render(self, snap, world, reach_m, person_h, lying_aspect):
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        people = [tr for tr in snap.tracks if tr.role]
        if people:
            self.zc = 0.8 * self.zc + 0.2 * float(np.mean([world.floor(tr.box)[1] for tr in people]))
        self._floor(img)
        level = snap.level
        by_id = {d.human_id: d for d in snap.dangers} if snap.dangers else {}
        robot = next((tr for tr in people if tr.role == "robot"), None)
        if robot is not None:
            X, Z = world.floor(robot.box)
            self._floor_circle(img, X, Z, reach_m, LEVEL_COL[level], 0.28)
        for tr in sorted(people, key=lambda q: -world.floor(q.box)[1]):       # far first
            X, Z = world.floor(tr.box)
            fut = snap.future.get(tr.id)
            if fut is not None:                    # ghost at the end of the look-ahead
                _, box_f = fut
                Xf, Zf = world.floor(box_f)
                self._figure(img, tr.role, Xf, Zf, person_h, _lying(box_f, lying_aspect), alpha=0.3)
                a, b = self.project(np.array([X, Xf]), np.zeros(2), np.array([Z, Zf]))
                if np.linalg.norm(b - a) > 4:
                    cv2.arrowedLine(img, tuple(a.astype(int)), tuple(b.astype(int)), (200, 200, 200), 2,
                                    cv2.LINE_AA, tipLength=0.25)
            top = self._figure(img, tr.role, X, Z, person_h, _lying(tr.box, lying_aspect))
            foot = self.project(np.array([X]), np.zeros(1), np.array([Z]))[0]
            put(img, f"{NAMES[tr.role]} #{tr.id}", foot + np.array([-30, 24]), 0.45,
                ROBOT if tr.role == "robot" else HUMAN)
            if tr.role == "human" and tr.id in by_id:
                self._danger_label(img, top, by_id[tr.id], snap)
        if robot is not None and snap.gap_m is not None and snap.pair_human is not None:
            h = snap.pair_human
            fa, fb = world.floor(robot.box), world.floor(h.box)
            pa = self.project(np.array([fa[0]]), np.zeros(1), np.array([fa[1]]))[0]
            pb = self.project(np.array([fb[0]]), np.zeros(1), np.array([fb[1]]))[0]
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), (170, 170, 170), 1, cv2.LINE_AA)
            put(img, f"{snap.gap_m:.1f} m", (pa + pb) / 2 + np.array([-18, 18]), 0.5, (200, 200, 200))
        put(img, "virtual scene", (12, 22), 0.55, MUTED)
        return img

    def _danger_label(self, img, top, d, snap):
        lvl = snap.level if snap.level > Level.NONE else d.predicted
        col = LEVEL_COL[max(Level.WARN, lvl)]
        txt = d.rule.replace("_", " ") + " " + _when(d, snap, 1)
        tw = cv2.getTextSize(txt, FONT, 0.55, 2)[0][0]
        x, y = min(int(top[0]) + 20, self.w - tw - 44), int(top[1]) - 40       # keep the label on screen
        tri = np.array([[x, y - 18], [x - 14, y + 6], [x + 14, y + 6]], np.int32)
        cv2.fillPoly(img, [tri], col, cv2.LINE_AA)
        put(img, "!", (x - 3, y + 3), 0.5, (20, 20, 20), 2)
        put(img, txt, (x + 20, y), 0.55, col, 2)


# ---------------------------------------------------------------- dashboard


class Dashboard:
    """1280x720: title bar | camera overlay (left) | virtual scene (right) | status strip with dangers, rates, power."""

    def __init__(self, cfg, world, engine, width=1280, height=720):
        self.cfg, self.world, self.engine = cfg, world, engine
        self.w, self.h = width, height
        self.hh = max(32, int(height * 0.06))                  # title bar
        self.pw, self.ph = width // 2, int(height * 0.72) - self.hh
        self.scene = SceneRenderer(self.pw, self.ph)
        self.power = deque(maxlen=240)

    def render(self, snap):
        out = np.full((self.h, self.w, 3), BG, np.uint8)
        state = snap.state
        if state == NodeState.IDLE or snap.frame is None:
            cam = np.full((self.ph, self.pw, 3), (12, 12, 14), np.uint8)
            put(cam, "camera off", (self.pw // 2 - 70, self.ph // 2 - 10), 0.8, MUTED, 2)
            put(cam, "waiting for a robot beacon", (self.pw // 2 - 125, self.ph // 2 + 22), 0.55, MUTED)
        else:
            cam = _fit(camera_view(snap, self.engine, self.cfg["rules"]["lying_aspect"]), self.pw, self.ph)
        y0, y1 = self.hh, self.hh + self.ph
        out[y0:y1, :self.pw] = cam
        out[y0:y1, self.pw:] = self.scene.render(snap, self.world, self.cfg["rules"]["reach_m"],
                                                 self.cfg["person_height_m"], self.cfg["rules"]["lying_aspect"])
        cv2.line(out, (self.pw, y0), (self.pw, y1), (70, 70, 75), 1)
        self._title(out)
        self._strip(out, snap)
        level = snap.level
        if level > Level.NONE:
            col = LEVEL_COL[level]
            th = 14 if level == Level.STOP and int(snap.t * 4) % 2 == 0 else 8
            cv2.rectangle(out, (0, y0), (self.w - 1, y1 - 1), col, th)
            label = "STOP" if level == Level.STOP else "WARNING"
            cx = self.pw // 2                                   # over the camera panel, clear of the scene title
            cv2.rectangle(out, (cx - 150, y0 + 10), (cx + 150, y0 + 64), col, -1)
            put(out, label, (cx - (70 if level == Level.STOP else 105), y0 + 52), 1.4, (20, 20, 20), 3)
        return out

    def _title(self, out):
        cv2.rectangle(out, (0, 0), (self.w, self.hh - 1), PANEL, -1)
        scale = 0.9 * self.hh / 43
        (tw, th), _ = cv2.getTextSize(TITLE, FONT, scale, 2)
        put(out, TITLE, ((self.w - tw) // 2, (self.hh + th) // 2), scale, TEXT, 2)
        cv2.line(out, (0, self.hh - 1), (self.w, self.hh - 1), (70, 70, 75), 1)

    def _strip(self, out, snap):
        y0 = self.hh + self.ph
        cv2.rectangle(out, (0, y0), (self.w, self.h), PANEL, -1)
        state = snap.state
        sched = snap.schedule
        band = sched.band.name if sched is not None and sched.band is not None else "-"
        put(out, f"{state.name}", (20, y0 + 40), 1.0, (0, 220, 255) if state == NodeState.TRACK else TEXT, 2)
        put(out, f"band {band}", (20, y0 + 72), 0.6, TEXT)
        gap = snap.gap_m
        body = snap.body_gap_m
        put(out, (f"gap {gap:.2f} m" if gap is not None else "gap -") + (f"  body {body:.2f}" if body is not None and gap is not None else ""),
            (20, y0 + 98), 0.55, TEXT)
        cl = snap.closing_mps
        put(out, f"closing {cl:+.2f} m/s" if cl is not None else "", (20, y0 + 122), 0.55, TEXT)
        b = snap.beacon
        put(out, f"beacon {'on: ' + b.robot_id if b and b.present else 'off'}", (20, y0 + 150), 0.5, MUTED)
        if snap.caption:
            put(out, snap.caption, (20, y0 + 176), 0.5, MUTED)

        x = 300
        put(out, "dangers", (x, y0 + 30), 0.55, MUTED)
        yy = y0 + 58
        for d in (snap.dangers or [])[:5]:
            lvl = max(d.level_now, Level.WARN if d.predicted == Level.STOP else d.predicted)
            put(out, f"{d.rule:<12s} {lvl.name:<5s} {_when(d, snap, 2)}", (x, yy), 0.55, LEVEL_COL[lvl], 1)
            yy += 26
        if not snap.dangers:
            level = snap.level
            for rule in snap.rules[:5] if level > Level.NONE else ():
                put(out, f"{rule:<12s} {level.name:<5s} held", (x, yy), 0.55, LEVEL_COL[level], 1)
                yy += 26
            if level == Level.NONE or not snap.rules:
                put(out, "none", (x, yy), 0.55, LEVEL_COL[Level.NONE])

        x = 620
        put(out, "rates", (x, y0 + 30), 0.55, MUTED)
        casc = snap.cascade or {}
        if sched is not None and state != NodeState.IDLE:
            for i, (key, hz) in enumerate((("det", sched.detector_hz), ("pose", sched.pose_hz))):
                c, y = casc.get(key), y0 + 56 + 42 * i
                stage, why = c.last if c is not None and c.last[0] else ("-", ())
                put(out, f"{key} {stage} @ {hz:g} Hz", (x, y), 0.5, TEXT)
                if c is not None and len(c.calls) > 1:
                    put(out, f"cheap {c.cheap_share():.0%}" + (f"  ({why[0]})" if why else ""), (x + 12, y + 18),
                        0.42, MUTED)
        else:
            put(out, "no inference", (x, y0 + 58), 0.5, TEXT)
        tm = snap.times or {}
        put(out, f"{snap.fps:.1f} fps  " + "  ".join(f"{k} {v:.0f}" for k, v in tm.items()),
            (x, y0 + 150), 0.42, MUTED)

        self._power(out, snap, 900, y0 + 16, self.w - 920, self.h - y0 - 32)

    def _power(self, out, snap, x, y, w, h):
        """Measured total power + its trace; CPU / DPU / GPU load and the estimated split (sensors/system.py)."""
        p, load = snap.power_w, snap.system
        if p is not None:
            self.power.append(p)
        put(out, "power", (x, y + 14), 0.55, MUTED)
        if load is not None and load.temps:
            put(out, "  ".join(f"{k.upper()} {v:.0f}C" for k, v in load.temps.items()), (x + w - 110, y + 14), 0.42, MUTED)
        if load is not None:
            put(out, "CPU", (x, y + 38), 0.42, MUTED)
            for i, c in enumerate(load.cpu):                         # one bar per core
                bx = x + 32 + i * 16
                cv2.rectangle(out, (bx, y + 26), (bx + 11, y + 40), (60, 60, 66), -1)
                cv2.rectangle(out, (bx, y + 40 - int(14 * c)), (bx + 11, y + 40), SPLIT_COL["cpu"], -1)
            cpu = sum(load.cpu) / len(load.cpu) if load.cpu else 0.0
            put(out, f"{cpu:.0%}   DPU {load.dpu:.0%}   GPU {'off' if load.gpu == 'suspended' else load.gpu}",
                (x + 36 + 16 * len(load.cpu), y + 38), 0.42, TEXT)
        if not self.power:
            put(out, "n/a (no power sensor: --power)", (x, y + 64), 0.5, MUTED)
            return
        put(out, f"{self.power[-1]:.2f} W", (x + 70, y + 14), 0.55, TEXT)
        parts = split_power(self.power[-1], load, self.cfg["power"]["model"]) if load is not None else None
        top = y + 50
        if parts is not None:                                    # stacked bar of the estimated split, then labels
            total = max(self.power[-1], sum(max(v, 0.0) for v in parts.values()), 1e-6)
            bx = x
            for k, v in parts.items():
                bw = int(w * max(v, 0.0) / total)
                cv2.rectangle(out, (bx, top), (bx + bw, top + 10), SPLIT_COL[k], -1)
                bx += bw
            put(out, "est. " + "  ".join(f"{k} {v:.1f}" for k, v in parts.items() if k != "gpu" or v),
                (x, top + 26), 0.4, MUTED)
            top += 34
        vals = np.array(self.power)
        lo, hi = vals.min() - 0.1, vals.max() + 0.1
        xs = x + np.linspace(0, w, len(vals))
        ys = y + h - (vals - lo) / (hi - lo) * (y + h - top - 6)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(out, [pts], False, (0, 200, 160), 2, cv2.LINE_AA)
