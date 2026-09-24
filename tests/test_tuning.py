import copy
import json
import urllib.error
import urllib.request

import pytest

import main
from utils import settings as gcfg
from dashboard import tuning as gtune
from dashboard.sinks import MjpegSink
from utils.types import Level
from tests.test_scenario import first


@pytest.fixture
def tune(cfg, tmp_path):
    base = copy.deepcopy(cfg)
    return gtune.Tuning(cfg, base, str(tmp_path / "tuning.json"), log=lambda m: None,
                        detector_specs=["yolov3_voc", "cheap > yolov3_voc"], pose_specs=["movenet"])


def test_every_param_exists_in_defaults_and_is_described(cfg):
    for p in gtune.params(["a"], ["b"]):
        gtune.get(cfg, p.path)
        assert gtune.EFFECTS.get(p.path), p.path


@pytest.mark.parametrize("path, value, expect", [
    ("rules.reach_m", 0.9, 0.9),
    ("predictor.steps", 3.4, 3),
    ("audio.enabled", False, False),
    ("roles.robot_is", "rightmost", "rightmost"),
    ("rules.enabled", ["down", "reach"], ["reach", "down"]),              # kept in canonical order
    ("bands.rates.close.detector", "cheap > yolov3_voc", ["cheap", "yolov3_voc"]),
])
def test_validate_accepts(tune, path, value, expect):
    assert tune.validate(path, value) == expect


@pytest.mark.parametrize("path, value", [
    ("rules.reach_m", 99.0), ("rules.reach_m", "x"), ("rules.reach_m", True), ("audio.enabled", 1),
    ("roles.robot_is", "middle"), ("rules.enabled", ["jump"]), ("camera.index", 1), ("bands.rates.close.detector", "x"),
])
def test_validate_rejects(tune, path, value):
    with pytest.raises((ValueError, TypeError)):
        tune.validate(path, value)


def test_close_must_stay_below_far(tune):
    ok, errors = tune.stage({"bands.close_m": 3.5})
    assert not ok and "bands.close_m" in errors
    ok, errors = tune.stage({"bands.far_m": 5.0, "bands.close_m": 3.5})
    assert not errors


def test_changes_apply_only_in_the_main_loop_and_persist(tune, cfg, tmp_path):
    tune.stage({"rules.reach_m": 1.1, "rules.enabled": ["down"]})
    assert cfg["rules"]["reach_m"] == 0.8                                  # staged, not applied
    changed = tune.apply_pending()
    assert {c[0] for c in changed} == {"rules.reach_m", "rules.enabled"}
    assert cfg["rules"]["reach_m"] == 1.1
    saved = json.loads((tmp_path / "tuning.json").read_text())
    assert saved == {"rules.reach_m": 1.1, "rules.enabled": ["down"]}      # only what differs from base

    fresh = gcfg._merge(gcfg.DEFAULTS, {})
    gtune.load_file(str(tmp_path / "tuning.json"), fresh, log=lambda m: None)
    assert fresh["rules"]["reach_m"] == 1.1 and fresh["rules"]["enabled"] == ["down"]

    tune.reset()
    tune.apply_pending()
    assert cfg["rules"]["reach_m"] == 0.8 and json.loads((tmp_path / "tuning.json").read_text()) == {}


def test_load_file_skips_bad_entries(cfg, tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"rules.reach_m": 50, "no.such": 1, "decision.stop_hold_s": 3.0}))
    msgs = []
    applied = gtune.load_file(str(f), cfg, log=msgs.append)
    assert applied == {"decision.stop_hold_s": 3.0} and len(msgs) == 2 and cfg["rules"]["reach_m"] == 0.8


def test_describe_shows_cascades_as_text(tune, cfg):
    cfg["bands"]["rates"]["close"]["detector"] = ["cheap", "yolov3_voc"]
    d = {p["path"]: p for p in tune.describe()}
    assert d["bands.rates.close.detector"]["value"] == "cheap > yolov3_voc"
    assert d["bands.rates.close.detector"]["default"] == "yolov3_voc"


def test_live_rule_toggle_changes_the_running_node(tmp_path):
    """Switching `reach` off mid-run (before beat 2) removes its warning; tracks survive the reconfigure."""
    args = main.parse_args(["--fast", "--no-log", "--no-audio", "--port", "0", "--tuning", ""])
    cfg, base = main.configs(args)
    node, _, scenario, _, _ = main.build(args, cfg)
    node.events.echo = False
    t = main.make_tuning(args, node, base, lambda m: None)
    snaps, staged = [], False
    while True:
        if not staged and node.clock.now() >= 10.0:
            t.stage({"rules.enabled": ["from_behind", "down", "pinned", "overhead"]})
            staged = True
            n_tracks = len(node.tracker.tracks)
        if t.apply_pending():
            node.reconfigure()
            assert len(node.tracker.tracks) == n_tracks
        s = node.step()
        snaps.append((s.t, s.state, s.level, tuple(s.rules)))
        if s.t >= scenario.duration:
            break
    assert first(snaps, Level.WARN, "reach", 10.0, 16.0) is None
    assert first(snaps, Level.STOP, "down", 23.0, 30.0) is not None


def _req(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_api(tune, cfg):
    sink = MjpegSink(0, tuning=tune)
    base = f"http://127.0.0.1:{sink.port}"
    try:
        code, page = _req(base + "/")
        assert code == 200 and b"Guardian Node" in page and b"/api/params" in page
        code, body = _req(base + "/api/params")
        assert code == 200 and any(p["path"] == "rules.reach_m" for p in json.loads(body))
        code, body = _req(base + "/api/params", {"rules.reach_m": 1.2})
        assert code == 200 and "rules.reach_m" in json.loads(body)["ok"]
        code, body = _req(base + "/api/params", {"rules.reach_m": -3})
        assert code == 422 and "rules.reach_m" in json.loads(body)["errors"]
        tune.apply_pending()
        assert cfg["rules"]["reach_m"] == 1.2
        code, _ = _req(base + "/api/reset", {"paths": ["rules.reach_m"]})
        assert code == 200
        tune.apply_pending()
        assert cfg["rules"]["reach_m"] == 0.8
    finally:
        sink.close()


@pytest.mark.parametrize("flag", ["--silent", "--no-audio"])
def test_silent_mode_disables_audio(flag):
    from actuators.audio import NullAudio
    args = main.parse_args(["--fast", "--no-log", "--port", "0", "--tuning", "", flag])
    cfg, _ = main.configs(args)
    node, *_ = main.build(args, cfg)
    assert args.no_audio and cfg["audio"]["enabled"] is False
    assert any(isinstance(a, NullAudio) for a in node.actuators.actuators)
