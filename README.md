# Guardian Node

A safety supervisor for robots working among people, running on an AMD Kria KV260. A fixed camera watches the
robot and the people near it, independently of the robot's own sensors, and decides **NONE / WARN / STOP**.
The decision is shown on an HDMI dashboard and played as a warning tone; the robot link only logs its commands.

The node is dark until a robot announces itself (a beacon, simulated for now). Then a YOLO person detector
finds everyone, MoveNet estimates each person's pose on their crop, a predictor extrapolates the joints about
one second ahead, and five deterministic geometric rules decide. How often each model runs depends on how close
the robot is to the nearest person.

This repository is a fork of [fcabecciaw/CPSA_2026](https://github.com/fcabecciaw/CPSA_2026) and keeps its
module structure. The upstream stereotypy system is in [`legacy/`](legacy/README.md).

| | |
|---|---|
| Board | AMD Kria KV260 (Zynq UltraScale+), DPU B4096, Ubuntu 22.04 + PYNQ-DPU 2.5 (Vitis AI 2.5 models only) |
| Models | YOLOv3-VOC (person detection) and MoveNet Lightning (17 keypoints), Vitis AI 2.5 model zoo |
| Laptop | Everything except the DPU runs on a laptop against a synthetic scenario (Python 3.10, `uv`) |
| Status | Laptop demo and 110 tests green. First live board run done; see [Status](#status) |

---

## How it works

```
 sensors/                VIDEO_pipeline/                 core/                          actuators/
┌──────────────┐ Frame  ┌───────────────────────┐ tracks ┌───────────────────────────┐ Decision ┌──────────────┐
│ camera       ├───────►│ detector (YOLO)       ├───────►│ scheduler  (distance band)│─────────►│ audio        │
│ beacon       │ Beacon │ tracker + roles       │ poses  │ predictor  (look-ahead)   │          │ robot link   │
│ power meter  ├───────►│ pose (MoveNet, crops) │        │ rules      (5 predicates) │          │ event diary  │
└──────────────┘ State  │ world  (fake 3D)      │        │ decision latch            │          └──────────────┘
                        └───────────────────────┘        └─────────────┬─────────────┘
                                                                       │ Snapshot
                                                                       ▼
                                                           dashboard/  views ─► HDMI · web stream · mp4
                                                                       tuning web UI ─► live settings
```

One call to `GuardianNode.step()` (in `core/guardian_node.py`) is one frame:

1. **Beacon.** No robot announced for `beacon.absent_timeout_s` → **IDLE**: camera off, no inference.
2. **Detect** at the current band's rate → the tracker keeps ids, velocities, and roles
   (the leftmost person is the robot until a real beacon reports the robot's position).
3. **Band.** The floor gap between the robot and the nearest human sets the band: FAR (> 3 m), APPROACH, CLOSE (< 1.5 m).
   The band decides which detector and pose model run, and how often. **DETECT** until APPROACH, then **TRACK**.
4. **Pose** on each person's crop at the band's pose rate (none in FAR).
5. **Predict** joint positions over the next second from the pose history.
6. **Rules** on the current and the predicted poses. STOP now → STOP; STOP only in the future → WARN with a time to contact.
7. **Latch** the decision (STOP held 2 s) → actuators; everything the dashboard needs → `Snapshot`.

---

## Repository structure

The module boxes follow upstream CPSA_2026 (`sensors/`, `VIDEO_pipeline/`, `core/`, `actuators/`, `utils/`,
`main.py`). Each box's `__init__.py` states what it is responsible for, its interface, and what it must not do.

```
main.py                   entry point: builds every box, wires them, runs the loop (like upstream main.py)
config.yaml               site settings: logging + optional `guardian:` overrides of utils/settings.py
run.sh                    board launcher: root + PYNQ environment + main.py --source webcam --backend dpu --hdmi

sensors/                  INPUTS: what the node measures
  camera.py               WebcamCamera (V4L2), ScenarioCamera (synthetic frames)
  beacon.py               AlwaysBeacon, ScheduledBeacon, ManualBeacon (simulated; BLE fits the same interface)
  power.py                HwmonPower (KV260 INA260 via sysfs), PynqRailsPower, NullPower
  scenario.py             DemoScenario: the four-beat demo world, with ground truth

VIDEO_pipeline/           PERCEPTION: who is where, in what pose
  YOLO/                   yolo.py (pure pre/post-processing) + the YOLOv3-VOC xmodel
  MOVENET/                movenet.py (pure pre/post-processing) + the MoveNet xmodel and model-zoo prototxt
  dpu.py                  board backends: PYNQ overlay, one vart runner per model
  replay.py               laptop stand-ins reading the scenario's ground truth
  cascade.py              cheapest model first, escalate when unsure
  tracking.py             persistent ids, velocities, robot/human roles
  world.py                monocular depth from box size, floor positions, gaps

core/                     DECISION: the only place NONE / WARN / STOP is decided
  guardian_node.py        the per-frame loop and IDLE / DETECT / TRACK (upstream: EventDispatcher)
  scheduler.py            distance band -> models and rates
  predictor.py            joint velocities, future poses, torso acceleration
  rules.py                the five rules, combine, DecisionLatch

actuators/                OUTPUTS: acting on a decision
  actuator_manager.py     hands each Decision to every actuator (upstream: ActuatorManager)
  audio.py                warning tones through the HDMI screen (aplay)
  robot_link.py           slow / stop / resume, logged only
  event_diary.py          WARN / STOP lines in upstream's event diary

dashboard/                DISPLAY: draws Snapshots, never decides
  views.py                camera overlay | virtual scene (avatars on the joints) | status strip
  sinks.py                HDMI (PYNQ DisplayPort), MJPEG web stream + tuning API, mp4
  tuning.py, webui.py     live-tunable settings and their web page

utils/                    shared building blocks
  types.py                the data types the boxes exchange (Frame, Detection, Decision, Snapshot, ...)
  settings.py             every Guardian setting with its default and a comment
  geometry.py, clock.py, event_log.py
  config.py, logger.py, lock.py     upstream's config.yaml loader and log files

tests/                    110 tests: decoders, geometry, tracker, rules, scheduler, cascade, tuning, full scenario
tools/power_experiment.py board power protocol (docs/model_survey.md §4.4) -> CSV
docs/                     model_survey.md (models per stage and band), img/ (reviewed demo frames)
legacy/                   upstream CPSA_2026 code and README that Guardian does not use
AGENTS.md                 working notes: design decisions, pitfalls, open work
```

### Interfaces between the boxes

| Box | Provides | Consumes |
|---|---|---|
| sensors | `Camera.read() -> Frame`, `BeaconSource.poll(t) -> BeaconState`, `PowerMeter.read() -> W` | hardware / scenario |
| VIDEO_pipeline | `Detector.detect(frame) -> [Detection]`, `PoseEstimator.estimate(frame, box) -> (17, 3)`, `Tracker`, `WorldModel` | `Frame` |
| core | `GuardianNode.step() -> Snapshot`, a `Decision` per frame | all of the above |
| actuators | `update(decision, t)` on each actuator | `Decision` |
| dashboard | `Dashboard.render(snapshot) -> image`, `Sink.show(image)` | `Snapshot` |

The types live in [`utils/types.py`](utils/types.py). Keypoints are `(17, 3)` arrays of `(x, y, score)` in COCO
order, in frame pixels; names ending in `_m` are metres. Only `core/rules.py` turns model output into a decision.

---

## Quick start

### Laptop (no FPGA)

```sh
uv sync                                               # Python 3.10, numpy 1.26.4, OpenCV 4.11: as on the board
uv run pytest -q                                      # 110 tests, ~35 s
uv run python main.py                                 # demo in real time; dashboard + tuning on http://localhost:8080/
uv run python main.py --fast --no-audio --port 0 --tuning '' --record out/demo.mp4 --snapshots out/snaps
uv run python main.py --fast --cascade --no-audio --port 0 --tuning ''      # same demo through the model cascades
uv run python main.py --help
```

### Board (KV260)

```sh
git clone -b guardian-node https://github.com/Smephite/CPSA_2026.git guardian-node && cd guardian-node
./run.sh --power                  # webcam -> DPU -> HDMI dashboard (+ http://kria:8080/), board power on the dashboard
./run.sh --power --silent         # same, silent mode: no sound, visual warnings only
./run.sh --power --duration 45 --snapshots out/live --snapshot-every 3 --tuning ''
sudo systemctl start gdm          # afterwards: HDMI output stops the desktop, this brings it back
```

- Shut down Jupyter kernels first: only one process may own the DPU and the webcam.
- `run.sh` becomes root and sources PYNQ's environment (`/etc/environment`, `/etc/profile.d/pynq_venv.sh`).
- No extra packages: the code needs numpy, OpenCV and PyYAML, which PYNQ's venv has.

---

## Demo scenario

The synthetic scenario plays the pitch's choreography with two people, one playing the robot. On the laptop,
`VIDEO_pipeline/replay.py` reads its ground truth in place of the DPU models.

| Beat | Time | Decision |
|---|---|---|
| no beacon | 0–3 s | IDLE: camera off |
| 1: robot passes far away | 3–9 s | DETECT only, no danger |
| 2: robot approaches from the front | 9–16 s | WARN `reach` at 14.5 s (predicted), no STOP |
| 3: human turns away, robot keeps closing | 16–19.5 s | STOP `from_behind` at 17.2 s |
| 4: human falls, robot approaches | 23–30 s | WARN `down` at 23.5 s, STOP `down` at 25.7 s |
| beacon gone | 30 s | IDLE at 32.1 s |

`tests/test_scenario.py` checks this sequence per beat, for the full models and for the cascade.

| Beat 1 | Beat 2 |
|---|---|
| ![beat 1](docs/img/beat1_far.png) | ![beat 2](docs/img/beat2_front.png) |
| **Beat 3** | **Beat 4** |
| ![beat 3](docs/img/beat3_behind.png) | ![beat 4](docs/img/beat4_fallen_stop.png) |

The dashboard shows the camera with boxes, roles, skeletons and the robot's reach hull (left); a virtual floor
scene with avatars on the estimated joints, predicted positions and danger labels (right); and the state, band,
distances, active dangers with time to contact, model rates and the power trace (bottom).

---

## Rules

All five compare distances on the two skeletons; thresholds are in `utils/settings.py` under `rules`.

| Rule | STOP when | WARN when |
|---|---|---|
| `reach` | a human joint is inside the robot's arm hull dilated by `reach_margin_m` | STOP predicted within the look-ahead |
| `from_behind` | the robot closes within `behind_m` on a human facing away from it | predicted |
| `down` | a lying or crouching human has a joint within `reach_m` of the robot's body | within `down_warn_m`, or predicted |
| `pinned` | a human is between a closing robot and a configured wall line | predicted |
| `overhead` | the robot's wrists are above its shoulders with a human within `drop_radius_m` | predicted |

`reach`, `down`, `pinned` and `overhead` need physical contact, so they stay silent when robot and human are
more than `depth_gate_m` apart in depth (people who only overlap in the image). When a depth is unreliable
(a box cut off on both axes) the gate is not applied.

**Distances are monocular.** Depth comes from box size and an assumed body height (`person_height_m`) and
focal length (`camera.focal_px`, not yet calibrated). A box cut off at the top or bottom of the frame uses its
width instead (`person_width_m`).

---

## Distance bands and model cascades

| Band | Robot–human gap | Detector | Pose |
|---|---|---|---|
| (no pair yet) | – | 3 Hz | off |
| FAR | > 3 m | 2 Hz | off |
| APPROACH | 1.5–3 m | 2 Hz | 5 Hz |
| CLOSE | < 1.5 m | 2 Hz | 15 Hz |

Band edges have hysteresis (`bands.hysteresis_m`); closing faster than `fast_closing_mps` moves one band closer.
Each band names its own detector, so a specialised model can be chosen by a deterministic selector (distance)
rather than one general model everywhere.

A band's detector and `pose.models` may also be a **cascade**, cheapest first (`VIDEO_pipeline/cascade.py`).
A cheap result is only accepted when it is confident and matches the tracker; "nobody there" is never trusted.
The full model runs directly on a 3 s watchdog, near a rule threshold, on sudden torso acceleration, for lying
people, and when the facing rule needs face points. `--cascade` runs the survey's pairs (RefineDet-ped → OFA-YOLO,
SPnet → MoveNet); these have laptop stand-ins only, no DPU backend yet. See [`docs/model_survey.md`](docs/model_survey.md).

---

## Configuration and live tuning

- **Defaults**: every setting, with a comment, in [`utils/settings.py`](utils/settings.py).
- **Site overrides**: a `guardian:` section in `config.yaml`, same nesting.
- **Live**: `http://<node>:8080/` shows the stream next to every tunable setting (range, default, what a change
  does). Changes apply at the next frame, are logged, and are saved to `guardian_tuning.json` (git-ignored;
  `--tuning ''` turns this off). Command-line flags still win for their run.
- Layering: defaults ← `config.yaml` ← tuning file ← command line.
- **Silent mode**: `--silent` (or `--no-audio`) plays no sound at all; warnings stay visual. `audio.enabled`
  can also be switched off live in the web UI, or for good in `config.yaml` (`guardian: {audio: {enabled: false}}`).
- **No authentication** on the web UI: use it on a trusted lab network only.

---

## Board notes

| Measured on the KV260 (2026-09-24) | |
|---|---|
| Board power, camera on, between detections | ≈ 5.3 W (INA260 on the SOM supply) |
| Board power during a YOLOv3 call | peaks ≈ 9.1 W |
| YOLOv3-VOC call | ≈ 84 ms; loop ≈ 7 fps with YOLO at 2–3 Hz |
| MoveNet on a crop | ≈ 7–10 ms DPU (earlier pose demo) |

- **Power**: PYNQ's libsensors does not initialise on this image, so `sensors/power.py` reads the INA260 from
  sysfs (`/sys/class/hwmon/*/power1_input`). It is one sensor for the whole SOM: per-block power is only
  measurable as differences between states (`tools/power_experiment.py`).
- **Several models on one overlay**: every xir graph must stay referenced, or earlier runners segfault
  (`VIDEO_pipeline/dpu.py` keeps them).
- **HDMI**: colours (`display.dp_pixel_format`) and the audio device (`audio.device`, for `aplay -D`) still need
  checking on the monitor.
- **First live run** (an office, people seated and standing): the pipeline ran end to end. It showed depth
  errors for boxes cut by the frame and false `reach` STOPs between people far apart in depth, now handled by
  the cut-box depth and the depth gate. The robot role still goes to the leftmost person.

---

## Extending

- **A model**: implement `detect(frame)` or `estimate(frame, box)` in `VIDEO_pipeline/dpu.py` and add it to
  `DETECTORS` / `POSES`; add a stand-in to `VIDEO_pipeline/replay.py` for the laptop; name it in the band table.
  Check the xmodel's DPU fingerprint first (`DPUCZDX8G_ISA1_B4096`, Vitis AI 2.5).
- **A rule**: add `rule_<name>(self, p: Pair) -> Level` to `core/rules.py`, list it in `rules.enabled`, add a
  positive and a negative case to `tests/test_rules.py`. Future poses and time to contact come for free.
- **An actuator**: a class with `configure(cfg)` and `update(decision, t)` in `actuators/`, added to the
  `ActuatorManager` list in `main.py`.
- **A sensor**: implement the interface in `sensors/__init__.py` (e.g. a real BLE beacon with `poll(t)`) and
  select it in `main.py`.
- **A setting**: default + comment in `utils/settings.py`; for live tuning also a `Param` and a help text in
  `dashboard/tuning.py` (a test checks every tunable setting has both).

---

## Status

Done: the full pipeline on the laptop, 110 tests, the model survey, and a first live run on the board with the
webcam, YOLOv3 and MoveNet on the DPU, HDMI output and the power trace.

Next (details in [`AGENTS.md`](AGENTS.md)): the demo setup on the board (two people standing 2–4 m from the
camera), HDMI colours and audio, the power experiment, focal-length calibration, robot role from the beacon,
and DPU backends for the survey's cheaper models.

## Origin and credits

Forked from [CPSA_2026](https://github.com/fcabecciaw/CPSA_2026) by Francesco Urru (frarvo) and fcabecciaw,
MIT licensed. Guardian keeps its module structure, its logger and event diary, and its YOLO / MoveNet models;
the stereotypy pipeline, IMU sensing and physical actuators are in [`legacy/`](legacy/README.md).
