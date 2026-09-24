"""Views: camera overlay, virtual scene ('fake 3D' avatars on the joints), and the composed dashboard.

The virtual scene lifts each person's joints to the floor plane with `WorldModel` (all joints of a person at
its estimated depth) and draws them in an oblique projection: a floor grid, shadows, a robot avatar (boxy,
orange) and human avatars (rounded, blue), predicted ghosts at the end of the look-ahead, the robot's reach
zone tinted by the decision, and a label on each endangered human with the rule and time to contact.
"""
from collections import deque

import cv2
import numpy as np

from sensors.system import split_power
from utils.geometry import dilate_polygon
from utils.types import SKELETON, Level, NodeState

ROBOT = (40, 120, 235)         # BGR
HUMAN = (200, 130, 40)
LEVEL_COL = {Level.NONE: (90, 170, 60), Level.WARN: (0, 170, 255), Level.STOP: (40, 40, 230)}
BG, PANEL, TEXT, MUTED = (24, 24, 26), (34, 35, 38), (225, 225, 225), (140, 140, 145)
FONT = cv2.FONT_HERSHEY_SIMPLEX

SPLIT_COL = {"static": (110, 110, 118), "cpu": (230, 160, 60), "dpu": (60, 200, 250), "gpu": (200, 90, 200),
             "rest": (80, 80, 86)}                                    # BGR, power split bar


def put(img, text, org, scale=0.5, col=TEXT, th=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, col, th, cv2.LINE_AA)


def _fit(img, w, h):
    box = np.full((h, w, 3), BG, np.uint8)
    k = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    oy, ox = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    box[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
    return box


# ---------------------------------------------------------------- camera overlay


def draw_skeleton(img, kp, col, min_score, thick=2):
    for a, b in SKELETON:
        if kp[a, 2] >= min_score and kp[b, 2] >= min_score:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), col, thick, cv2.LINE_AA)
    for x, y, s in kp:
        if s >= min_score:
            cv2.circle(img, (int(x), int(y)), thick + 2, col, -1, cv2.LINE_AA)


def _when(d, snap, digits):
    """'now' for a STOP, 'in 0.8 s' for a predicted one, else how close the human is ('at 1.4 m')."""
    if d.level_now == Level.STOP:
        return "now"
    if d.ttc_s is not None:
        return f"in {d.ttc_s:.{digits}f} s"
    h, body = snap.pair_human, snap.body_gap_m
    return f"at {body:.{digits}f} m" if body is not None and h is not None and h.id == d.human_id else ""


def camera_view(snap, min_score, engine):
    img = snap.frame.image.copy()
    for line in engine.lines_px:
        cv2.line(img, tuple(line[:2].astype(int)), tuple(line[2:].astype(int)), (60, 60, 200), 3)
    for tr in snap.tracks:
        col = ROBOT if tr.role == "robot" else HUMAN if tr.role == "human" else MUTED
        x1, y1, x2, y2 = tr.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        facing = f"  faces {tr.facing}" if getattr(tr, "facing", None) and snap.t - tr.facing_t <= 1.0 else ""
        put(img, f"{tr.role or 'person'} #{tr.id}{facing}", (x1 + 3, max(14, y1 - 5)), 0.45, col, 1)
        fresh = tr.kp is not None and snap.t - tr.kp_t <= 0.5
        if fresh:
            draw_skeleton(img, tr.kp, col, min_score)
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

    def _avatar(self, img, J, vis, role, alpha=1.0):
        """J: (17, 3) world joints. Robot: thick square limbs; human: rounded limbs with a torso."""
        P = self.project(J[:, 0], J[:, 1], J[:, 2])
        col = ROBOT if role == "robot" else HUMAN
        dark = tuple(int(c * 0.55) for c in col)
        layer = img.copy() if alpha < 1 else img
        foot = (J[15, [0, 2]] + J[16, [0, 2]]) / 2 if vis[15] or vis[16] else J[[11, 12]][:, [0, 2]].mean(0)
        sh = self.project(np.array([foot[0]]), np.array([0.0]), np.array([foot[1]]))[0]
        cv2.ellipse(layer, tuple(sh.astype(int)), (int(0.35 * self.s), int(0.12 * self.s)), 0, 0, 360,
                    (18, 18, 20), -1, cv2.LINE_AA)
        torso = [5, 6, 12, 11]
        if all(vis[i] for i in torso):
            cv2.fillPoly(layer, [P[torso].astype(np.int32)], dark, cv2.LINE_AA)
        thick = 11 if role == "robot" else 8
        for a, b in SKELETON:
            if vis[a] and vis[b] and a > 4 and b > 4:
                cv2.line(layer, tuple(P[a].astype(int)), tuple(P[b].astype(int)), dark, thick + 3, cv2.LINE_AA)
                cv2.line(layer, tuple(P[a].astype(int)), tuple(P[b].astype(int)), col, thick, cv2.LINE_AA)
        for i in range(5, 17):
            if vis[i]:
                c = tuple(P[i].astype(int))
                if role == "robot":
                    cv2.rectangle(layer, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (30, 30, 30), -1)
                else:
                    cv2.circle(layer, c, 4, (240, 240, 240), -1, cv2.LINE_AA)
        face = [i for i in range(5) if vis[i]]
        if face:
            hc = P[face].mean(0).astype(int)
            r = int(0.13 * self.s)
            if role == "robot":
                cv2.rectangle(layer, (hc[0] - r, hc[1] - r), (hc[0] + r, hc[1] + r), col, -1)
                cv2.rectangle(layer, (hc[0] - r + 3, hc[1] - 4), (hc[0] + r - 3, hc[1] + 3), (40, 40, 40), -1)
            else:
                cv2.circle(layer, tuple(hc), r, col, -1, cv2.LINE_AA)
                cv2.circle(layer, tuple(hc), r, dark, 2, cv2.LINE_AA)
        if alpha < 1:
            cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)
        return P

    def render(self, snap, world, min_score, reach_m):
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        people = [tr for tr in snap.tracks if tr.role and tr.kp is not None and snap.t - tr.kp_t <= 1.0]
        if people:
            self.zc = 0.8 * self.zc + 0.2 * float(np.mean([world.floor(tr.box)[1] for tr in people]))
        self._floor(img)
        level = snap.level
        by_id = {d.human_id: d for d in snap.dangers} if snap.dangers else {}
        robot = next((tr for tr in people if tr.role == "robot"), None)
        if robot is not None:
            X, Z = world.floor(robot.box)
            self._floor_circle(img, X, Z, reach_m, LEVEL_COL[level], 0.28)
        for tr in snap.tracks:                   # detected but no pose yet (FAR / DETECT): floor marker only
            if tr.role and tr not in people:
                X, Z = world.floor(tr.box)
                col = ROBOT if tr.role == "robot" else HUMAN
                self._floor_circle(img, X, Z, 0.25, col, 0.6)
                p = self.project(np.array([X]), np.zeros(1), np.array([Z]))[0]
                put(img, f"{tr.role} (no pose)", p + np.array([-40, 30]), 0.45, col)
        for tr in sorted(people, key=lambda q: -world.floor(q.box)[1]):       # far first
            J = world.joints_3d(tr.kp, tr.box)
            vis = tr.kp[:, 2] >= min_score
            fut = snap.future.get(tr.id)
            if fut is not None:
                kp_f, box_f = fut
                Jf = world.joints_3d(kp_f, box_f)
                self._avatar(img, Jf, vis, tr.role, alpha=0.28)
                a = self.project(*[np.array([v]) for v in (J[[11, 12], 0].mean(), 0.0, J[0, 2])])[0]
                b = self.project(*[np.array([v]) for v in (Jf[[11, 12], 0].mean(), 0.0, Jf[0, 2])])[0]
                if np.linalg.norm(b - a) > 4:
                    cv2.arrowedLine(img, tuple(a.astype(int)), tuple(b.astype(int)), (200, 200, 200), 2,
                                    cv2.LINE_AA, tipLength=0.25)
            P = self._avatar(img, J, vis, tr.role)
            if tr.role == "human" and tr.id in by_id:
                self._danger_label(img, P, by_id[tr.id], snap)
        if robot is not None and snap.gap_m is not None and snap.pair_human is not None:
            h = snap.pair_human
            fa, fb = world.floor(robot.box), world.floor(h.box)
            pa = self.project(np.array([fa[0]]), np.zeros(1), np.array([fa[1]]))[0]
            pb = self.project(np.array([fb[0]]), np.zeros(1), np.array([fb[1]]))[0]
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), (170, 170, 170), 1, cv2.LINE_AA)
            put(img, f"{snap.gap_m:.1f} m", (pa + pb) / 2 + np.array([-18, 18]), 0.5, (200, 200, 200))
        put(img, "virtual scene", (12, 22), 0.55, MUTED)
        return img

    def _danger_label(self, img, P, d, snap):
        lvl = snap.level if snap.level > Level.NONE else d.predicted
        col = LEVEL_COL[max(Level.WARN, lvl)]
        top = P[:5].mean(0) if P.shape[0] else np.array([0, 0])
        txt = d.rule.replace("_", " ") + " " + _when(d, snap, 1)
        tw = cv2.getTextSize(txt, FONT, 0.55, 2)[0][0]
        x, y = min(int(top[0]) + 20, self.w - tw - 44), int(top[1]) - 40       # keep the label on screen
        tri = np.array([[x, y - 18], [x - 14, y + 6], [x + 14, y + 6]], np.int32)
        cv2.fillPoly(img, [tri], col, cv2.LINE_AA)
        put(img, "!", (x - 3, y + 3), 0.5, (20, 20, 20), 2)
        put(img, txt, (x + 20, y), 0.55, col, 2)


# ---------------------------------------------------------------- dashboard


class Dashboard:
    """1280x720: camera overlay (left) | virtual scene (right) | status strip with dangers, rates, power."""

    def __init__(self, cfg, world, engine, width=1280, height=720):
        self.cfg, self.world, self.engine = cfg, world, engine
        self.w, self.h = width, height
        self.pw, self.ph = width // 2, int(height * 0.72)
        self.scene = SceneRenderer(self.pw, self.ph)
        self.power = deque(maxlen=240)

    @property
    def min_score(self):
        return self.cfg["pose"]["min_score"]

    def render(self, snap):
        out = np.full((self.h, self.w, 3), BG, np.uint8)
        state = snap.state
        if state == NodeState.IDLE or snap.frame is None:
            cam = np.full((self.ph, self.pw, 3), (12, 12, 14), np.uint8)
            put(cam, "camera off", (self.pw // 2 - 70, self.ph // 2 - 10), 0.8, MUTED, 2)
            put(cam, "waiting for a robot beacon", (self.pw // 2 - 125, self.ph // 2 + 22), 0.55, MUTED)
        else:
            cam = _fit(camera_view(snap, self.min_score, self.engine), self.pw, self.ph)
        out[:self.ph, :self.pw] = cam
        out[:self.ph, self.pw:] = self.scene.render(snap, self.world, self.min_score, self.cfg["rules"]["reach_m"])
        cv2.line(out, (self.pw, 0), (self.pw, self.ph), (70, 70, 75), 1)
        self._strip(out, snap)
        level = snap.level
        if level > Level.NONE:
            col = LEVEL_COL[level]
            th = 14 if level == Level.STOP and int(snap.t * 4) % 2 == 0 else 8
            cv2.rectangle(out, (0, 0), (self.w - 1, self.ph - 1), col, th)
            label = "STOP" if level == Level.STOP else "WARNING"
            cx = self.pw // 2                                   # over the camera panel, clear of the scene title
            cv2.rectangle(out, (cx - 150, 10), (cx + 150, 64), col, -1)
            put(out, label, (cx - (70 if level == Level.STOP else 105), 52), 1.4, (20, 20, 20), 3)
        return out

    def _strip(self, out, snap):
        y0 = self.ph
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
