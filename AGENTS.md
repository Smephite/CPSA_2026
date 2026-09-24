# AGENTS.md: Guardian Node

Notes for coding agents (and humans) working on the Guardian Node in this repo: what it is, how it is built, what was decided and why, how to check a change, and what is still open. User-facing docs are in `README.md` (section "Guardian Node"). The model research is in `docs/model_survey.md`.

## What it is

Guardian Node is an infrastructure-side safety supervisor on an AMD Kria KV260. A fixed camera, independent of the robot, watches the robot and nearby people and decides NONE / WARN / STOP. Output is audiovisual only:
- an HDMI dashboard: camera overlay, a "virtual scene" floor view, and a status strip;
- warning tones through the HDMI screen;
- a robot link that only logs `slow` / `stop` / `resume`.

In the demo, a person plays the robot. The beacon that says "a robot is near" is simulated (`sensors/, actuators/, utils/clock.py, utils/event_log.py`), and its interface is kept open for a real BLE listener.

The upstream CPSA_2026 code (`main.py`, `core/`, `actuators/`, `VIDEO_pipeline/*_thread.py`) is left in place but not used. The upstream IMU pipeline was removed from the Guardian path.

## Current status (2026-09-24)

- **Runs live on the KV260** (`~/guardian-node`, `./run.sh --power --no-audio`): webcam → DPU → HDMI dashboard + web
  UI, 0 errors in a 60 s run after the empty-crop fix. DETECT ≈ 18 fps (camera-bound), TRACK ≈ 16 fps.
- All ten catalog models (7 detectors, MoveNet, Hourglass, orientation) load together and are switchable live.
- **Last work stream: recording and sensing, for offline evaluation.**
  1. Load display: per-core CPU, DPU busy, GPU state, PS/PL temperatures, and an *estimated* CPU / DPU power split
     (`sensors/system.py`). The coefficients in `power.model` are placeholders until `tools/power_calibration.py`
     runs on an idle board (Guardian stopped, ~2.5 min).
  2. Raw clip recording with a live HDMI / web view (`record.sh` → `tools/record.py`): one webcam, two webcams
     (`--right`, stereo, unsynchronised), or the RealSense F200's depth (`--depth`, raw `.u16`).
     Replay through the full pipeline: `./run.sh --source video --video clip.avi [--loop | --fast]`
     (`VideoFileCamera`, recorded frame times).
  3. RealSense F200 colour | depth demo (`./tools/board.sh tools/depth_demo.py`).
- **Tested on the board:** recording (webcam, F200 colour + depth), the depth demo (headless), the load monitor.
  **Not yet:** the HDMI live view of `record.py` / `depth_demo.py`, a stereo recording, and replaying a recorded
  clip through the DPU models.
- **Hardware on the bench changes:** the Trust webcam (USB 2.0, YUYV only), a "GENERAL WEBCAM" (MJPEG up to 1080p)
  and the RealSense F200 have all been plugged in at times. Check `v4l2-ctl --list-devices` before assuming
  `/dev/video0` is the webcam you expect.

## Run and check

```sh
uv sync                                          # Python 3.10 to match the board's PYNQ venv; numpy 1.26.4, opencv-headless 4.11
uv run pytest -q                                 # 176 tests, ~25 s; must stay green
uv run python main.py --fast --no-audio --port 0 --tuning '' --record out/demo.mp4 --snapshots out/snaps
uv run python main.py --fast --no-audio --port 0 --tuning '' --cascade      # same demo through the model cascades
uv run python main.py                   # real time, dashboard + tuning UI on http://localhost:8080/
```

Expected decisions in the fast demo (full models and `--cascade` agree within ~0.2 s):

| Beat | Time | Decision |
|---|---|---|
| 1: robot passes far away | 3–9 s | none (DETECT only) |
| 2: robot approaches from the front | 9–16 s | WARN `reach` at 14.5 s (predicted), no STOP |
| 3: human turns away, robot keeps closing | 16–19.5 s | STOP `from_behind` at 17.2 s |
| 4: human falls (21.4–22 s), robot approaches | 23–30 s | WARN `down` at 23.5 s, then STOP `down` at 25.7 s |
| beacon gone | 30 s | IDLE at 32.1 s |

`tests/test_scenario.py` asserts this sequence per beat window. After any change to views, look at the frames themselves (`out/snaps/*.png`); `docs/img/` holds one reviewed frame per beat.

Pass `--tuning ''` for reproducible runs. Otherwise `guardian_tuning.json`, written by the web UI, changes the settings.

Board: `./run.sh` becomes root, sources the PYNQ environment and runs `main.py --source webcam --backend dpu --hdmi`
(extra flags are passed on; a later `--source` wins). `./tools/board.sh <script.py>` does the same for any script,
`record.sh` for `tools/record.py`. Power: `tools/power_experiment.py` (protocol), `tools/power_calibration.py` (split).

## How a frame flows (`core/guardian_node.py`, `GuardianNode.step`)

```
beacon.poll ──absent > absent_timeout_s──► IDLE: camera off, no inference, state reset
camera.read (webcam, or the synthetic scenario)
detector cascade at the band's rate ──► tracker: ids, velocities, roles (leftmost = robot until the beacon says otherwise)
pair = robot + nearest human → floor gap (monocular depth from box size) → closing speed
scheduler: gap → band FAR / APPROACH / CLOSE (hysteresis; fast closing moves one band closer)
pose cascade on each person's crop at the band's pose rate (none in FAR)
predictor: per-joint velocities (LS over 0.6 s) → future poses at 0.25..1.0 s; torso acceleration (quadratic LS)
rules on now + each future pose: reach · from_behind · down · pinned · overhead
    STOP now → STOP; STOP only in a future pose → WARN with time to contact
combine (worst wins) → latch (STOP held 2 s, WARN 1 s) → audio, robot link, event log
snapshot → Dashboard → HDMI / MJPEG / mp4
```

Only the camera, perception backend, clock and beacon differ between laptop and board. All timing uses the node clock, so `--fast` (`SimClock`) runs the same logic as fast as the CPU allows.

## Code map

The layout follows upstream CPSA_2026's module boxes; the full tree is in README.md ("Repository structure").
Each box's `__init__.py` documents its interface. Boxes only talk through the types in `utils/types.py`:

| Box | Files | Interface |
|---|---|---|
| `sensors/` | `camera.py` (Webcam, VideoFile, Depth, Scenario), `beacon.py`, `power.py`, `system.py`, `scenario.py` | `Camera.read() -> Frame`, `BeaconSource.poll(t) -> BeaconState`, `PowerMeter.read() -> W`, `SystemMonitor.read(t, dpu_ms) -> SystemLoad` |
| `VIDEO_pipeline/` | `YOLO/yolo.py`, `MOVENET/movenet.py` (pure decoders), `dpu.py` (board), `replay.py` (laptop), `cascade.py`, `tracking.py`, `world.py` | `Detector.detect(frame)`, `PoseEstimator.estimate(frame, box)`, `Tracker`, `WorldModel` |
| `core/` | `guardian_node.py` (per-frame loop, states; `reconfigure()`), `scheduler.py`, `predictor.py`, `rules.py` (rules, `body_gap_m`, `near_threshold`, depth gate, `combine`, `DecisionLatch`) | `GuardianNode.step() -> Snapshot`; a `Decision` per frame to the actuators |
| `actuators/` | `actuator_manager.py`, `audio.py`, `robot_link.py`, `event_diary.py` | `update(decision, t)`, `configure(cfg)` |
| `dashboard/` | `views.py`, `sinks.py` (MJPEG + `/api/params`, `/api/reset`; mp4; DisplayPort), `tuning.py`, `webui.py` | `Dashboard.render(snapshot) -> image`, `Sink.show(image)` |
| `utils/` | `types.py`, `settings.py` (all defaults, one comment each), `geometry.py`, `clock.py`, `event_log.py`; upstream `config.py`, `logger.py`, `lock.py` | – |
| top level | `main.py` (CLI, config layering `configs`, wiring `build`, `DEMO_CASCADE`, loop), `run.sh`, `record.sh` | – |
| `tools/` | `record.py` (raw clips), `depth_demo.py`, `power_calibration.py`, `power_experiment.py`, `board.sh` | standalone scripts; reuse the boxes, not `GuardianNode` |

`legacy/` holds the upstream code Guardian does not use (reference only, not runnable).

## Design decisions (and why)

**Perception**
- YOLO gates MoveNet: the pose model runs on each person's crop, not the whole frame.
- Only Vitis AI **2.5** xmodels work: 3.5 models did not load on the PYNQ-DPU 2.5 runtime.
- On the DPU today: YOLOv3-VOC and MoveNet only. Every other model in the survey has estimated numbers only.
- MoveNet decoding follows the model zoo's prototxt: letterbox 192, (x − 127.5) / 127.5, centre-weighted single-pose decode. The upstream decoder fed [0, 1] pixels and took a per-joint argmax, which made its poses unusable.

**Distance and rules**
- Distances are monocular: depth = focal_px × person_height_m / box size. `focal_px` (550) is assumed, not calibrated.
- **`down` uses the body gap** (`RuleEngine.body_gap_m`): the smallest distance from any visible human joint to the hull of all visible robot joints. The centre-to-centre gap and the arm hull never reached STOP for a lying person, because the arms are above them.
- Rules are deterministic, and only `core/rules.py` decides. No model output reaches the outputs directly.
- A predicted STOP becomes a WARN with a time to contact.

**Bands and model choice**
- The bands (FAR > 3 m, APPROACH, CLOSE < 1.5 m) set the detector and pose rates.
- Each band can name its own detector: specialised models chosen by a deterministic selector, like altitude-specific drone models.

**Cascades** (`VIDEO_pipeline/cascade.py`)
- A band's detector and `pose.models` may be a list, cheapest first.
- A cheap result is accepted only when it is confident and agrees with the tracker.
- **"Nobody there" is never trusted:** a confirmed track without a matching detection escalates.
- Escalation reasons:
  - before the cheap model runs (it is then skipped): watchdog, near threshold, sudden motion, lying box, face needed, not upright;
  - after it ran: missing track, low score, low keypoints.
- The watchdog forces the full detector every 3 s. It catches people the cheap model never detected, and it is valid while the robot closes at ≤ ≈0.6 m/s from the 3 m band (the owner's call: humans are slow).
- The "near threshold" trigger asks whether more certainty could change the decision (a rule within 0.2 m of its threshold, or a predicted STOP).
- The sudden-motion trigger: torso acceleration from a quadratic fit over the pose history, above 2 m/s². Walking stays below ≈1.7 m/s²; falls reach 2.3–3.2.
- `--cascade` uses the survey's pairs (RefineDet-ped 0.96 → OFA-YOLO, SPnet → MoveNet). On the laptop these are stand-ins; they have no DPU backend yet.
- **Why the cascade matters** (`test_cheap_models_alone_are_unsafe`): the cheap models alone give a false STOP in beat 2 (SPnet has no face points) and lose the fallen human in beat 4.

**Live tuning**
- 58 settings are live-tunable from the web UI: every number, the rule switches, and the model per band (only models loaded at startup).
- The web thread only validates and queues; the main loop applies the change at the next frame and calls `node.reconfigure()`. Tracks and history survive.
- Every change is logged with its old and new value. Changes are saved to `guardian_tuning.json` (git-ignored).
- Layering order: defaults ← `config.yaml` ← tuning file ← command-line flags.
- **No authentication, by the owner's choice** (lab use): anyone who can reach port 8080 can change thresholds or switch rules off.

**Demo and hardware scope**
- Audiovisual only: no relay, no other hardware.
- The power-down experiment stays in scope: is unloading the PL in IDLE worth it?

## Pitfalls found so far

- **Power telemetry:** the KV260 has one power sensor, the INA260 (`ina260_u14`, hwmon, µW) on the SOM supply.
  AMS gives voltages and temperatures only, the DA9130/DA9131 PMICs nothing, PYNQ's libsensors fails to start.
  Any CPU / DPU / FPGA split is a fitted estimate: label it "est." everywhere.
- **GPU:** Mali-400, OpenGL ES 2.0 only, no OpenCL (the `xilinx.icd` is XRT for the PL), runtime-suspended.
  Nothing useful to offload; the dashboard render cost is better cut on the CPU (see Performance below).
- **USB:** all four board ports share one USB 2.0 link (USB 3 devices get their own). Two YUYV webcams at 30 fps do
  not fit: the second fails to start. At 15 fps requested, the Trust delivered ~10-11 fps and the GENERAL WEBCAM ~23.
  Frames from two webcams are not synchronised: pair them by the recorded times.
- **RealSense F200** (`8086:0a66`, plain UVC, no librealsense needed): colour `/dev/video0` (YUYV ≤ 1080p), depth
  `/dev/video2` (`Z16`, 640x480, ≤ 60 fps; open with `CAP_PROP_CONVERT_RGB 0`, set the fps explicitly). The first 1-3
  depth reads fail after opening (retried in `DepthCamera`). Range ~0.2-1.2 m, dead in sunlight: 0-2 % coverage in
  the office scene. Depth unit 1/32 mm is from the docs, **not measured** (`depth.unit_mm`).
- **Recording throughput:** the SD card writes ~17 MB/s; raw depth at 30 fps is 18 MB/s, so depth records at 15 fps.
  16-bit PNG (38-120 ms/frame) and zlib (9-83 ms) were too slow on the ARM.

- **Performance (profiled on the board, 2026-09-24):** the ARM was the bottleneck, not the DPU. Rules were pure-Python
  point/segment loops (226 ms per TRACK frame); now vectorised. Drawing + HDMI + JPEG (~110 ms) moved to the output
  thread; the detector to its own thread; DPU I/O is int8. Live: DETECT 7 -> 18 fps, TRACK ~2 -> ~16 fps. Next hot
  spots: `dashboard/views.py` drawing (80-130 ms, alpha overlays copy the full image), MoveNet decode (17 ms,
  float64 over 48x48x17), preprocessing. Measure with `--profile` before optimising.
- **Threads:** only on a live node (`RealClock`); `--fast`, `--record` and the tests stay single-threaded and
  deterministic. The output thread reads Track objects the main loop updates; fields are replaced, not mutated
  in place, so a frame may mix two updates but never crashes.

- **Models checked on the board (2026-09-24):** all ten catalog models (`VIDEO_pipeline/catalog.py`, measured DPU
  times there) load together on one overlay (≈ 400 MB) and are all loaded at startup, so the UI can switch any band's
  detector (every model alone + cheap -> full pairs) and the pose model live. Decoders were checked against real board
  output on `test_person.jpg`: all seven detectors box the person, Hourglass joints land on the body, orientation
  says "front". Unverified: orientation's left/right convention (`orientation.swap_left_right`) and
  `pose.hourglass_gain` (2.0) on real footage. **SPnet is not usable as a cheap pose stage:** its xmodel has two DPU subgraphs with a CPU
  average-pool between them (PYNQ's `load_model` accepts one), and it regresses 14 joint coordinates with no
  confidence per joint, so the cascade cannot tell a good cheap pose from a bad one.

- **Live office run (2026-09-24):** boxes cut off by the frame gave wrong depths, and `reach` fired STOP between a
  person near the camera and people metres behind (image overlap only). Fixed with cut-box depth
  (`WorldModel.cut`, `person_width_m`) and the depth gate (`rules.depth_gate_m`). Still open: the robot role
  goes to the leftmost person, and the band flips often with noisy real distances.

- **Replay stand-ins can be too kind.** `ReplayPose` used to return ground truth for any crop. That made the cheap-only run look safe. It is now crop-aware: below 60 % coverage it returns low-confidence keypoints. Be sure a stand-in isn't hiding the failure a test is meant to catch.
- **Scenario artefacts look like features.** A one-frame "teleport" fall produced fake joint velocities (an early predicted WARN) and no measurable acceleration. The fall is now animated. Robot keyframes are piecewise linear, so reversals are genuine one-frame velocity jumps.
- **`from_behind` facing is judged on the floor plane** (`RuleEngine.facing_away`). Live in the office, the old rule
  read "back to the camera" (no face points, or orientation "back") as "back to the robot" and gave a STOP although the
  seated people faced the robot deeper in the room. Now the facing label (orientation model or face keypoints,
  `rules.facing_source`) becomes a floor direction and the robot must be within +-60 degrees behind the person.
- **Board (from earlier sessions):**
  - Keep every xir graph referenced when loading several models, or earlier runners segfault (`DpuModels._graphs`).
  - Only one process may own the DPU and the webcam, so shut down Jupyter kernels first.
  - PYNQ needs `/etc/environment` and `/etc/profile.d/pynq_venv.sh` sourced, as root.
- **Latency:** a YOLOv3 call blocks the single Python thread for ≈110 ms, longer than one 15 Hz pose period.

## Conventions

- Commits: `sw: <lowercase imperative>`, no body, signed, explicit paths staged. Ask before pushing anything but `guardian-node`.
- Branch `guardian-node` on the fork `Smephite/CPSA_2026`; upstream is `fcabecciaw/CPSA_2026`.
- Keep `VIDEO_pipeline/{YOLO/yolo,MOVENET/movenet}.py` pure (no board imports), so the decoders stay testable. Board-only imports are lazy.
- New settings: add a default with a comment in `utils/settings.py`. If they should be tunable, add a `Param` and an `EFFECTS` entry in `dashboard/tuning.py`; `test_every_param_exists_in_defaults_and_is_described` enforces this. Components read settings in `configure(cfg)`, so they change live.
- Mark unmeasured numbers as estimates, as the survey does ([zoo] / [ours] / [est]).

## Open work

**Next (from the last work stream)**
1. Record a clip set (reach, from behind, fall, walk past; two people) with `record.sh`, replay one through the DPU
   with `--fast` to confirm the path end to end, and check the HDMI live views of `record.py` / `depth_demo.py`.
2. Clip labels (`rule: LEVEL t0-t1`) and an evaluation tool: replay every clip, report missed and false
   warnings / stops, sweep one threshold. Proposed, not started.
3. Run `tools/power_calibration.py` on an idle board and put the fit into `power.model`.
4. Measure the F200 depth unit (object at a known distance at the crosshair).
5. Stereo (two webcams): calibration clip with a checkerboard, then triangulate MoveNet joints to replace the
   monocular depth: detect on the left image only, MoveNet on the matching crop in both (epipolar search),
   `cv2.triangulatePoints`, monocular depth as fallback. ~20 cm baseline gives about ±0.2 m at 3 m.

**Needs the board**
1. Check HDMI colours (`display.dp_pixel_format`) and the HDMI audio device for `aplay -D` (the README's `hw:0,3` is
   a placeholder).
2. Run `tools/power_experiment.py`. So far it has only been dry-run against a fake `pynq`.
3. Controlled two-person test of the `from_behind` STOPs seen in the office; robot role from the beacon instead of
   "leftmost".
4. Calibrate:
   - the cascade accept thresholds and `sudden_accel_mps2`, using recorded clips and real pose jitter;
   - `camera.focal_px`, with a checkerboard;
   - `orientation.swap_left_right` and `pose.hourglass_gain` on real footage.
5. Dashboard render cost (80-130 ms) and MoveNet decode (17 ms) on the ARM.

**Can be done on the laptop**
1. `from_behind`: treat missing face points as "unknown", not "facing away", and let the orientation model or the full pose decide.
2. "Already fallen" scenario: show the watchdog finding a person the cheap detector never saw.
3. Survey items not yet followed:
   - band hysteresis ±0.3 m with a 0.5 s dwell (now 0.2 m, no dwell);
   - robot poses at 5 Hz;
   - pose-derived boxes in CLOSE.
4. The watchdog timer is shared by all bands; make it per model once bands use different full models.
5. `rules.static_lines` (walls for `pinned`) cannot be edited in the UI.
6. Confirm the `[tune]` event-log lines on a live run.
7. Web UI authentication, if the node ever leaves a trusted network.
