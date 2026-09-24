"""Runtime tuning: which settings can change while the node runs, their ranges, and where changes are kept.

    web thread   Tuning.stage(changes)       validate and queue (never touches the live config)
    main loop    Tuning.apply_pending()      write into the live config in place, save the file, log each change
                 node.reconfigure()          components re-read their settings; tracks and history are kept

Persistence: every tuned value that differs from the startup base (defaults <- config.yaml <- command line)
is saved to a JSON file ({"rules.reach_m": 0.9, ...}) and loaded again at the next start.
Camera, display, model files and static lines stay startup-only.
"""
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

RULES = ("reach", "from_behind", "down", "pinned", "overhead")
BANDS = ("detect", "far", "approach", "close")


@dataclass
class Param:
    path: str                        # dotted path into the config, e.g. "rules.reach_m"
    kind: str                        # float | int | bool | choice | multi
    lo: Optional[float] = None
    hi: Optional[float] = None
    step: Optional[float] = None
    help: str = ""
    choices: Tuple[str, ...] = field(default_factory=tuple)


def _f(path, lo, hi, step, help):
    return Param(path, "float", lo, hi, step, help)


def _i(path, lo, hi, help):
    return Param(path, "int", lo, hi, 1, help)


def spec_str(spec):
    """Model spec (name or cascade list) -> 'a > b'."""
    return spec if isinstance(spec, str) else " > ".join(spec)


def spec_value(s):
    names = [n.strip() for n in s.split(">")]
    return names[0] if len(names) == 1 else names


def params(detector_specs=(), pose_specs=()):
    """All tunable settings. detector_specs / pose_specs: the model choices available (loaded at startup)."""
    ps = [
        _f("camera.focal_px", 200, 2000, 10, "Pinhole focal length in px (distance from box size)"),
        _f("person_height_m", 1.0, 2.2, 0.05, "Body size that turns pixels into metres"),
        _f("person_width_m", 0.2, 1.0, 0.05, "Body width, for depth when a box is cut at the top or bottom"),
        _f("beacon.absent_timeout_s", 0.5, 30, 0.5, "Beacon silent this long -> IDLE"),
        Param("roles.robot_is", "choice", help="Role rule when the second person appears",
              choices=("leftmost", "rightmost")),
        _f("tracker.gate", 0.2, 3.0, 0.1, "Max match distance, in track box heights"),
        _f("tracker.max_age_s", 0.2, 10, 0.1, "Drop a track not observed for this long"),
        _i("tracker.min_hits", 1, 10, "Observations before a track counts"),
        _f("tracker.vel_alpha", 0.05, 1.0, 0.05, "EMA weight of a new velocity sample"),
        _i("tracker.pose_min_kps", 1, 17, "Confident joints needed to move the box with the pose"),
        _f("bands.far_m", 1.0, 10, 0.1, "FAR above this robot-human gap"),
        _f("bands.close_m", 0.3, 5, 0.1, "CLOSE below this gap"),
        _f("bands.hysteresis_m", 0.0, 1.0, 0.05, "Band edge hysteresis"),
        _f("bands.fast_closing_mps", 0.0, 3.0, 0.05, "Closing faster than this -> one band closer"),
    ]
    for band in BANDS:
        if detector_specs:
            ps.append(Param(f"bands.rates.{band}.detector", "choice", help=f"Detector (or cascade) in {band.upper()}",
                            choices=tuple(detector_specs)))
        ps.append(_f(f"bands.rates.{band}.detector_hz", 0, 15, 0.5, f"Detector rate in {band.upper()}"))
        ps.append(_f(f"bands.rates.{band}.pose_hz", 0, 30, 1, f"Pose rate in {band.upper()}"))
    if pose_specs:
        ps.append(Param("pose.models", "choice", help="Pose model (or cascade)", choices=tuple(pose_specs)))
    ps += [
        _f("pose.crop_margin_x", 0, 1, 0.01, "Pose crop margin, fraction of box width"),
        _f("pose.crop_margin_y", 0, 1, 0.01, "Pose crop margin, fraction of box height"),
        _f("pose.min_score", 0.05, 0.95, 0.05, "A joint counts as visible at or above this"),
        _f("pose.max_age_s", 0.05, 5, 0.05, "Rules ignore poses older than this"),
        _f("pose.hourglass_gain", 0.5, 5, 0.1, "Hourglass heatmap peak x this = joint score"),
        Param("orientation.enabled", "bool", help="Run the orientation model on humans"),
        _f("orientation.min_prob", 0.25, 1.0, 0.05, "Ignore orientation answers below this probability"),
        Param("orientation.swap_left_right", "bool", help="Flip the model's left / right"),
        Param("rules.facing_source", "choice", help="from_behind: how 'facing away' is decided",
              choices=("auto", "keypoints", "orientation")),
        _f("predictor.window_s", 0.1, 3, 0.1, "History used to fit joint velocities"),
        _f("predictor.accel_window_s", 0.3, 3, 0.1, "History used to fit torso acceleration"),
        _f("predictor.horizon_s", 0.0, 3, 0.1, "Look-ahead"),
        _i("predictor.steps", 1, 10, "Look-ahead samples within the horizon"),
        Param("rules.enabled", "multi", help="Active rules", choices=RULES),
        _f("rules.reach_margin_m", 0.0, 1.5, 0.05, "reach: dilation of the robot arm hull"),
        _f("rules.depth_gate_m", 0.2, 5, 0.1, "contact rules: ignore pairs this far apart in depth"),
        _f("rules.reach_m", 0.1, 3, 0.05, "down: STOP within this body gap"),
        _f("rules.down_warn_m", 0.1, 5, 0.05, "down: WARN within this body gap"),
        _f("rules.down_tol_m", 0.0, 0.5, 0.01, "down: hip within this of knee height counts as down"),
        _f("rules.lying_aspect", 0.8, 3, 0.05, "down: box width / height above this counts as lying"),
        _f("rules.behind_m", 0.1, 5, 0.05, "from_behind: only counts this close"),
        _f("rules.closing_min_mps", 0.0, 2, 0.05, "Robot speed towards the human that counts as closing"),
        _f("rules.pin_line_m", 0.0, 3, 0.05, "pinned: human this close to a static line"),
        _f("rules.pin_corridor_m", 0.0, 3, 0.05, "pinned: and this close to the robot->line path"),
        _f("rules.pin_gap_m", 0.0, 5, 0.05, "pinned: with the robot this close"),
        _f("rules.overhead_tol_m", 0.0, 0.5, 0.01, "overhead: wrist above shoulder by this much"),
        _f("rules.drop_radius_m", 0.0, 5, 0.05, "overhead: human within this of a raised load"),
        _f("decision.stop_hold_s", 0.0, 10, 0.1, "STOP stays latched this long"),
        _f("decision.warn_hold_s", 0.0, 10, 0.1, "WARN stays latched this long"),
        _f("cascade.accept_score", 0.0, 1.0, 0.05, "Cheap detections below this escalate"),
        _f("cascade.lying_aspect", 0.8, 3, 0.05, "Lying-shaped box: skip pedestrian-trained detectors"),
        _f("cascade.watchdog_s", 0.2, 30, 0.1, "Full detector at least this often"),
        _f("cascade.upright_aspect", 0.2, 2, 0.05, "Cheap pose only for boxes narrower than this (w / h)"),
        _f("cascade.accept_kp", 0.0, 1.0, 0.05, "Mean body-joint confidence needed to keep a cheap pose"),
        _f("cascade.sensitivity_m", 0.0, 1.0, 0.05, "A rule this close to its threshold escalates"),
        _f("cascade.sudden_accel_mps2", 0.2, 20, 0.1, "Torso acceleration that counts as sudden motion"),
        Param("audio.enabled", "bool", help="Audio warnings"),
        _f("audio.repeat_s", 0.2, 10, 0.1, "Repeat the tone this often while the level holds"),
    ]
    return ps


# What changing a setting does (the UI shows this when hovering over the setting's name).
EFFECTS = {
    "camera.focal_px": "Scales every distance: higher makes people look farther away (larger gaps, later bands and "
                       "warnings); lower makes them look closer. Calibrate once with a checkerboard.",
    "person_height_m": "Assumed body size behind the monocular depth. Higher -> people are placed farther away and "
                       "all metre distances grow; lower -> closer and earlier warnings.",
    "beacon.absent_timeout_s": "How long the node keeps watching after the robot's beacon goes quiet. Longer rides out "
                               "beacon dropouts; shorter turns the camera off sooner.",
    "roles.robot_is": "Which of the first two people becomes the robot when roles lock. Only matters until the beacon "
                      "reports the robot's position.",
    "tracker.gate": "How far a detection may be from a track and still count as the same person. Larger survives fast "
                    "moves and falls but can swap people who pass close; smaller splits one person into new tracks.",
    "tracker.max_age_s": "How long an unseen person is kept. Longer bridges occlusion and missed detections; shorter "
                         "clears people who left sooner, but can forget someone who is still there.",
    "tracker.min_hits": "Detections before a person counts. Higher filters one-off false detections but delays "
                        "reacting to someone new.",
    "tracker.vel_alpha": "How fast the track velocity follows new motion. Higher reacts faster but is noisier "
                         "(affects predicted boxes and closing speed).",
    "tracker.pose_min_kps": "Confident joints needed before a pose moves the track box. Higher ignores poor poses; "
                            "lower lets partial poses keep the box current.",
    "bands.far_m": "Beyond this gap the node only runs the detector (FAR). Larger starts poses and rules earlier "
                   "and costs more power.",
    "bands.close_m": "Inside this gap the node runs poses at the full CLOSE rate. Larger gets full rate sooner.",
    "bands.hysteresis_m": "Dead zone around each band edge. Larger stops rates flapping at a boundary but changes band "
                          "later.",
    "bands.fast_closing_mps": "Closing faster than this treats the pair one band closer. Lower reacts to more "
                              "approaches with higher rates.",
    "pose.models": "Which pose model runs, or a cascade tried cheapest first. Only models loaded at startup.",
    "pose.crop_margin_x": "Extra width around the person box given to the pose model. More keeps outstretched arms in "
                          "the crop; less gives the model more pixels per joint.",
    "pose.crop_margin_y": "Extra height around the person box for the pose model (head and feet).",
    "pose.min_score": "Joint confidence needed to count as visible, everywhere (rules, tracker, drawing). Higher "
                      "drops doubtful joints, so rules see fewer; lower keeps more joints, including wrong ones.",
    "pose.max_age_s": "Rules ignore a pose older than this. Shorter avoids acting on stale poses; too short leaves "
                      "gaps between pose updates at low rates.",
    "predictor.window_s": "History used to fit joint velocities. Longer is smoother but reacts slower to a sudden "
                          "move; shorter reacts faster and is noisier.",
    "predictor.accel_window_s": "History used to estimate torso acceleration (sudden motion). Longer is steadier but "
                                "reacts later and blurs short jolts; at low pose rates it stretches back to 5 poses.",
    "predictor.horizon_s": "How far ahead dangers are predicted. Longer warns earlier and more often; 0 turns "
                           "prediction off (only current STOPs).",
    "predictor.steps": "Look-ahead samples within the horizon. More catches short-lived predicted contacts; costs "
                       "rule evaluations per frame.",
    "rules.enabled": "Rules that may raise WARN or STOP. A rule switched off never fires.",
    "rules.reach_margin_m": "reach: how far the robot's arm hull is grown. Larger stops earlier (more margin, more "
                            "false stops); smaller lets people come closer.",
    "rules.reach_m": "down: body gap (nearest human joint to robot body) that means STOP for a person on the floor. "
                     "Larger stops earlier.",
    "rules.down_warn_m": "down: body gap that means WARN for a person on the floor. Larger warns earlier.",
    "rules.down_tol_m": "down: how close hips may be to knee height and still count as down. Larger also treats "
                        "crouching or sitting as down.",
    "rules.lying_aspect": "down: box width / height above which a person counts as lying. Lower treats more "
                          "postures as lying.",
    "rules.behind_m": "from_behind: the gap within which a robot closing on someone facing away means STOP. Larger "
                      "stops earlier.",
    "rules.closing_min_mps": "Robot speed towards the human that counts as closing (from_behind, pinned). Lower "
                             "treats slower approaches as closing.",
    "rules.pin_line_m": "pinned: how close the human must be to a wall line. Larger counts more positions as "
                        "against the wall.",
    "rules.pin_corridor_m": "pinned: how close the human must be to the robot's path to the wall. Larger counts "
                            "more positions as trapped.",
    "rules.pin_gap_m": "pinned: robot-human gap within which pinning means STOP. Larger stops earlier.",
    "rules.overhead_tol_m": "overhead: how far the wrists must be above the shoulders to count as a raised load. "
                            "Lower counts smaller lifts.",
    "rules.drop_radius_m": "overhead: a human within this of a robot holding a load up means STOP. Larger stops "
                           "earlier.",
    "decision.stop_hold_s": "How long STOP stays after its cause is gone. Longer avoids stop-go flicker; shorter "
                            "resumes sooner.",
    "decision.warn_hold_s": "How long WARN stays after its cause is gone.",
    "cascade.accept_score": "Cheap detections below this score send the frame to the next detector. Higher "
                            "escalates more (safer, more power); lower trusts the cheap model more.",
    "cascade.lying_aspect": "Box width / height above which a person counts as lying for the cascade: the cheap "
                            "(pedestrian-trained) detector is skipped. Lower skips it more often.",
    "cascade.watchdog_s": "The full detector runs at least this often, to catch people the cheap one never "
                          "detected. Longer saves power; keep it below the time the robot needs to cover ~2 m.",
    "cascade.upright_aspect": "The cheap pose model is only tried on boxes narrower than this (width / height). "
                              "Lower uses the full pose model more often.",
    "cascade.accept_kp": "Mean body-joint confidence a cheap pose needs to be kept. Higher escalates more.",
    "cascade.sensitivity_m": "When a rule is within this distance of changing its decision, both cascades use their "
                             "full models. Larger escalates more often.",
    "cascade.sudden_accel_mps2": "A person (or the robot) whose torso accelerates faster than this gets the full "
                                 "models on the next frame: lunges, falls, abrupt starts and stops. Lower escalates "
                                 "more often (walking already peaks around 1-2 m/s^2); higher only reacts to violent "
                                 "moves.",
    "person_width_m": "Assumed body width, used for depth only when a person's box is cut off at the top or "
                      "bottom of the frame (someone close to the camera). Higher places them farther away.",
    "rules.depth_gate_m": "reach, down, pinned and overhead only fire when robot and human are within this "
                          "distance in depth: people who only overlap in the image (one near the camera, one far "
                          "behind) cannot touch. Larger is more cautious; ignored when a depth is unreliable.",
    "pose.hourglass_gain": "Hourglass heatmap peaks are ~0.1-0.6 on a clear person, not probabilities. Joint score = "
                           "peak x this gain (capped at 1). Higher counts more joints as visible; calibrate on real "
                           "footage.",
    "orientation.enabled": "Classify each human's facing direction (left / right / front / back) at the pose rate, "
                           "~1.8 ms per person on the DPU. Used by from_behind (see facing source).",
    "orientation.min_prob": "Orientation answers below this probability are treated as unknown.",
    "orientation.swap_left_right": "The model's 'left' may mean the person's own left rather than facing image-left. "
                                   "Check once with someone in profile; flip here if from_behind reacts to the "
                                   "wrong side.",
    "rules.facing_source": "How from_behind decides 'facing away from the robot'. auto: orientation model when it "
                           "is confident, else face keypoints. keypoints: no face points = facing away (can give "
                           "false STOPs with poses that lack face points). orientation: the model only.",
    "audio.enabled": "Warning tones on the node's audio output.",
    "audio.repeat_s": "How often the tone repeats while WARN or STOP holds.",
}
for _band in BANDS:
    EFFECTS[f"bands.rates.{_band}.detector"] = (f"Detector used in {_band.upper()}, or a cascade tried cheapest first. "
                                                "Only models loaded at startup.")
    EFFECTS[f"bands.rates.{_band}.detector_hz"] = (f"Detector calls per second in {_band.upper()}. Higher finds new "
                                                   "or moved people sooner; costs DPU time and power. 0 = none.")
    EFFECTS[f"bands.rates.{_band}.pose_hz"] = (f"Pose estimates per second per person in {_band.upper()}. Higher "
                                               "gives rules fresher joints; costs DPU time. 0 = no poses, no rules.")


def get(cfg, path):
    for k in path.split("."):
        cfg = cfg[k]
    return cfg


def put(cfg, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        cfg = cfg[k]
    cfg[keys[-1]] = value


def _ui(p, value):
    """Config value -> what the UI shows (model specs as 'a > b')."""
    return spec_str(value) if p.path.endswith((".detector", "pose.models")) else value


def _config(p, value):
    if p.path == "pose.models":
        return [n.strip() for n in value.split(">")]      # always a list in the config
    return spec_value(value) if p.path.endswith(".detector") else value


class Tuning:
    def __init__(self, cfg, base, path=None, log=print, detector_specs=(), pose_specs=()):
        """cfg: the live config (changed in place). base: startup values without the tuning file (reset target)."""
        self.cfg, self.base, self.path, self.log = cfg, base, path, log
        self.params = {p.path: p for p in params(detector_specs, pose_specs)}
        self.lock = threading.Lock()
        self.pending = {}

    # ---------------------------------------------------------------- validation

    def validate(self, path, value):
        """-> config value, or raise ValueError."""
        p = self.params.get(path)
        if p is None:
            raise ValueError("not tunable")
        if p.kind in ("float", "int"):
            if isinstance(value, bool):
                raise ValueError("expected a number")
            v = float(value)
            if not p.lo <= v <= p.hi:
                raise ValueError(f"out of range [{p.lo:g}, {p.hi:g}]")
            return int(round(v)) if p.kind == "int" else v
        if p.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError("expected true or false")
            return value
        if p.kind == "choice":
            if value not in p.choices:
                raise ValueError(f"expected one of {list(p.choices)}")
            return _config(p, value)
        if p.kind == "multi":
            if not isinstance(value, list) or any(v not in p.choices for v in value):
                raise ValueError(f"expected a list from {list(p.choices)}")
            return [c for c in p.choices if c in value]
        raise ValueError(f"unknown kind {p.kind}")

    def _cross_check(self, changes):
        def val(path):
            return changes.get(path, self.pending.get(path, get(self.cfg, path)))
        if val("bands.close_m") >= val("bands.far_m"):
            return {"bands.close_m": "must be below bands.far_m"}
        return {}

    # ---------------------------------------------------------------- web thread

    def describe(self):
        """-> list of dicts for the UI: definition, current value, startup default, pending value."""
        with self.lock:
            pending = dict(self.pending)
        out = []
        for p in self.params.values():
            group = p.path.split(".")[0] if "." in p.path else "camera"     # person_height_m: distance model
            d = {"path": p.path, "group": group, "kind": p.kind, "lo": p.lo, "hi": p.hi,
                 "step": p.step, "help": p.help, "effect": EFFECTS.get(p.path, p.help), "choices": list(p.choices),
                 "value": _ui(p, get(self.cfg, p.path)), "default": _ui(p, get(self.base, p.path))}
            if p.path in pending:
                d["pending"] = _ui(p, pending[p.path])
            out.append(d)
        return out

    def stage(self, changes):
        """{path: ui value} -> ({path: config value} staged, {path: error})."""
        ok, errors = {}, {}
        for path, value in changes.items():
            try:
                ok[path] = self.validate(path, value)
            except (ValueError, TypeError) as e:
                errors[path] = str(e)
        with self.lock:
            errors.update(self._cross_check(ok))
            ok = {k: v for k, v in ok.items() if k not in errors}
            self.pending.update(ok)
        return ok, errors

    def reset(self, paths=None):
        """Stage the startup value for `paths` (default: every tunable setting)."""
        paths = list(self.params) if not paths else paths
        return self.stage({p: _ui(self.params[p], get(self.base, p)) for p in paths if p in self.params})

    # ---------------------------------------------------------------- main loop

    def apply_pending(self):
        """Write staged values into the live config, save, log. -> [(path, old, new)] actually changed."""
        with self.lock:
            pending, self.pending = self.pending, {}
        changed = []
        for path, new in pending.items():
            old = get(self.cfg, path)
            if old != new:
                put(self.cfg, path, new)
                changed.append((path, old, new))
                self.log(f"[tune] {path}: {_ui(self.params[path], old)} -> {_ui(self.params[path], new)}")
        if changed:
            self.save()
        return changed

    # ---------------------------------------------------------------- persistence

    def overrides(self):
        return {p: get(self.cfg, p) for p in self.params if get(self.cfg, p) != get(self.base, p)}

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.overrides(), f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


def load_file(path, cfg, log=print):
    """Apply a saved tuning file onto cfg (startup). Unknown or invalid entries are skipped with a message."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        saved = json.load(f)
    t = Tuning(cfg, cfg, detector_specs=(), pose_specs=())
    applied = {}
    for key, value in saved.items():
        if key.endswith((".detector", "pose.models")):
            applied[key] = value                        # model choices are checked when the models load
            continue
        try:
            applied[key] = t.validate(key, value)
        except ValueError as e:
            log(f"[tune] {path}: skipping {key}={value!r}: {e}")
    for key, value in applied.items():
        put(cfg, key, value)
    return applied
