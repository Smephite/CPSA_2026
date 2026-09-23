"""Guardian settings: defaults below, overridden by the `guardian:` section of the repo's config.yaml."""
import copy

DEFAULTS = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "focal_px": 550.0,           # pinhole focal length in px (webcam ~60 deg HFOV at 640 px)
    },
    "person_height_m": 1.7,          # body size used to turn pixels into metres (monocular)
    "beacon": {
        "absent_timeout_s": 2.0,     # beacon silent this long -> IDLE
    },
    "roles": {
        "robot_is": "leftmost",      # leftmost | rightmost: role rule when the second person appears
    },
    "tracker": {
        "gate": 1.0,                 # max match distance, in units of the track's box height
        "max_age_s": 1.5,            # drop a track not observed for this long
        "min_hits": 2,               # observations before a track counts (and can get a role)
        "vel_alpha": 0.5,            # EMA weight of a new velocity sample
        "pose_min_kps": 5,           # confident keypoints needed to refresh the box from the pose
    },
    "bands": {                       # distance bands between robot and nearest human (metres)
        "far_m": 3.0,
        "close_m": 1.5,
        "hysteresis_m": 0.2,
        "fast_closing_mps": 0.8,     # closing faster than this -> one band closer
        "rates": {                   # per band: which detector, detector rate, pose rate (Hz)
            "detect":   {"detector": "yolov3_voc", "detector_hz": 3.0, "pose_hz": 0.0},
            "far":      {"detector": "yolov3_voc", "detector_hz": 2.0, "pose_hz": 0.0},
            "approach": {"detector": "yolov3_voc", "detector_hz": 2.0, "pose_hz": 5.0},
            "close":    {"detector": "yolov3_voc", "detector_hz": 2.0, "pose_hz": 15.0},
        },
    },
    "pose": {
        "crop_margin_x": 0.25,
        "crop_margin_y": 0.12,
        "min_score": 0.3,            # keypoint counts as visible at or above this score
        "max_age_s": 0.5,            # rules ignore poses older than this
    },
    "predictor": {
        "window_s": 0.6,             # history used to fit joint velocities
        "horizon_s": 1.0,            # look-ahead
        "steps": 4,                  # look-ahead samples within the horizon
    },
    "rules": {
        "reach_margin_m": 0.3,       # dilation of the robot arm hull
        "reach_m": 0.8,              # 'within reach' for the posture rule
        "behind_m": 1.5,             # approach-from-behind only counts this close
        "closing_min_mps": 0.15,     # robot speed towards the human that counts as closing
        "down_warn_m": 1.5,          # vulnerable posture: warn inside this distance
        "down_tol_m": 0.10,          # hip within this of knee height counts as down
        "lying_aspect": 1.3,         # box width / height above this counts as lying
        "pin_line_m": 0.8,           # human this close to a static line ...
        "pin_corridor_m": 0.5,       # ... and this close to the robot->line path ...
        "pin_gap_m": 1.5,            # ... with the robot this close -> pinned
        "overhead_tol_m": 0.05,      # wrist above shoulder by this much -> load overhead
        "drop_radius_m": 1.0,
        "static_lines": [],          # [[x1, y1, x2, y2], ...] in normalised image coords (0..1)
        "enabled": ["reach", "from_behind", "down", "pinned", "overhead"],
    },
    "decision": {
        "stop_hold_s": 2.0,          # STOP stays latched this long after the last STOP condition
        "warn_hold_s": 1.0,
    },
    "audio": {
        "enabled": True,
        "device": None,              # aplay -D device (board: the DisplayPort/HDMI card, see README)
        "repeat_s": 1.5,
    },
    "display": {
        "width": 1280,
        "height": 720,
        "dp_pixel_format": "rgb",    # rgb | bgr for PYNQ DisplayPort (colours looked swapped with bgr)
    },
    "power": {
        "rails": [],                 # names from pynq.get_rails() to sum; empty = all with a power sensor
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(overrides=None):
    """Defaults <- config.yaml `guardian:` <- `overrides` (dict)."""
    try:
        from utils.config import CONFIG
        file_cfg = CONFIG.get("guardian", {}) or {}
    except Exception:
        file_cfg = {}
    return _merge(_merge(DEFAULTS, file_cfg), overrides)
