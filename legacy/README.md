# legacy: upstream CPSA_2026 code that Guardian Node does not use

Kept for reference: it is the original system this fork started from (stereotypy detection with BlueCoin IMUs,
YOLO + MoveNet threads, and physical actuators). It is **not runnable from here**: its imports expect the upstream
layout (`utils.config` getters, `utils.event_queue`, ...), whose originals are copied into `legacy/utils/`.

| Upstream file | What it did | Guardian counterpart |
|---|---|---|
| `main.py` | wire sensors, dispatcher, threads, dashboard | `../main.py` |
| `core/event_dispatcher.py` | tag events -> video stages + actuation | `../core/guardian_node.py` |
| `core/actuation_policy.py` | pick actuator + variation per tag | `../core/rules.py` + `../actuators/actuator_manager.py` |
| `actuators/` (MetaMotion, BT speaker, LED strip) | physical feedback | `../actuators/` (audio, robot link, event diary) |
| `VIDEO_pipeline/*/..._thread.py`, `shared/person_roi_state.py` | camera-owning YOLO / MoveNet threads, ROI hand-off | `../VIDEO_pipeline/` (models, cascade, tracker) |
| `utils/video_dashboard.py` | OpenCV window | `../dashboard/` |
| `utils/event_queue.py`, `utils/audio_paths.py`, `triggers/` | IMU event queue, speaker audio, constant tag | not needed |
| `xmutil_load_dpu.sh`, `DPU_FIRMWARE/` | load the DPU with xmutil | `../VIDEO_pipeline/dpu.py` (PYNQ `dpu.bit`) |
| `test_scripts/`, `config.yaml`, `utils/config.py` | upstream tests and device settings | `../tests/`, `../utils/settings.py` |

The IMU pipeline itself (BlueCoin sensors, C feature extraction, classifier) was removed in commit `cb78d2a`;
see git history or upstream `fcabecciaw/CPSA_2026`.
