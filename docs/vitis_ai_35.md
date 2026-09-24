# Guardian Node: Vitis AI 3.5 on the KV260?

*Date:* 2026-09-24. *Scope:* whether moving from PYNQ-DPU 2.5 / VART 2.5 to Vitis AI 3.5 would give us better person-detection or pose models, and what the move would cost. For the 2.5 model options, see [model_survey.md](model_survey.md).

*Labels:* **[fact]** was checked in a source (the numbers refer to §7) or on the board. **[ours]** was measured or inspected on our board or on our files. **[est]** is an estimate or hypothesis that has not been verified.

## TL;DR

- **Vitis AI 3.5 adds nothing for the KV260 at runtime.** The 3.5 release made no DPUCZDX8G IP, TRD or board-image updates [fact 1]. None of the 103 models in the 3.5 zoo has a KV260 build: every `model.yaml` offers only GPU, VEK280 or V70 downloads [fact 2]. For Zynq UltraScale+, the 3.5 documentation itself sends users to the **3.0** image, runtime and zoo [fact 3].
- **"3.5 models do not load" has a more specific cause.** The 3.0 zoo's `kv260` xmodels are compiled for fingerprint **0x101000056010407**. Our DPU has **0x101000016010407** [ours]. The two differ only in bit 30, which is the save-argmax feature [fact 4], and VART rejects any fingerprint that does not match exactly [fact 5].
- **Our working MoveNet xmodel was compiled with `xcompiler 3.5.0`** for 0x101000016010407, and it runs on VART 2.5 [ours: strings in `kv260_MoveNet_int.xmodel`]. So xmodels compiled with the 3.x tools for *our* fingerprint already load on the current runtime.
- **The upstream firmware `kv260-benchmark-b4096` is a Vitis AI 2.5 DPU.** It uses DPUCZDX8G v4.0.0, was built with Vitis 2022.1, and has fingerprint 0x101000016010407 [fact 6, 7]. That is the same DPU as PYNQ-DPU's `dpu.bit`, so switching to the xmutil route gains nothing.
- **Recommendation:** do not change the runtime before the demo. After the demo, recompile selected 3.0/3.5 zoo models for our fingerprint on an x86 host (route A below). This needs no changes on the board.

## 1. Runtime path: what would have to change

### Current state [ours, via ssh]

| Item | Value |
|---|---|
| OS / kernel | Ubuntu 22.04.4, `5.15.0-1027-xilinx-zynqmp` |
| XRT | `xrt 2.13.479` (2022.1 generation) |
| VART / XIR | `libvart 2.5.0`, `libxir 2.5.0` (system-wide in `/usr/lib`); `/etc/vart.conf` points to `/usr/lib/dpu.xclbin` |
| PYNQ | `pynq 3.0.1`, `pynq_dpu 2.5` in `/usr/local/share/pynq-venv` |
| Firmware apps | only `k26-starter-kits` in `/lib/firmware/xilinx`. The `benchmark-b4096` package is not installed. |
| Ubuntu archive | `vitis-ai-runtime` / `vitis-ai-library` exist only as version **2.0** (jammy universe) |

### Facts on the 3.x side

| Component | Status | Src |
|---|---|---|
| DPUCZDX8G IP | v4.1 in 3.0, and still 4.1 in 3.5 ("not updated"). Vitis AI 2.5 used v4.0. | 1, 8 |
| 3.5 zoo builds for `zcu102 & zcu104 & kv260` | **0 of 103** (GPU 103, VEK280 95, V70 85) | 2 |
| 3.0 zoo builds for `zcu102 & zcu104 & kv260` | 133 entries | 9 |
| Official ZU+ runtime and image | 3.5's MPSoC quickstart links to the `xilinx-kv260-dpu-v2022.2-v3.0.0.img.gz` image (PetaLinux) and to `vitis_ai_runtime_r3.0.0`. The 3.5 target runtime is RPMs for PetaLinux 2023.1 only (`board_setup/vek280`). | 3, 10 |
| DPU-PYNQ | The last release is **2.5.1** (Nov 2022); PyPI's highest version is 2.5.1; the repo was archived in Aug 2025. An unreleased branch, **`design_contest_3.5`** (Nov 2023, "updates to 3.5 for kria only"), ships `pynqdpu.dpu.kv260_som.3.5.0.bit` with `SAVE_ARGMAX_ENABLE`. Its installer (`amd/Kria-RoboticsAI` `install_update_kr260_to_vitisai35.sh`) downloads `vai3.5_kr260.zip`, runs its VART 3.5 `.deb` `setup.sh`, copies prebuilt XRT libraries (`lack_lib`) and `xbutil2` into `/usr/lib` and `/usr/bin`, reinstalls `pynq_dpu` into pynq-venv, and deletes the 2.5 `pynq-dpu` notebooks. The log in that README shows fingerprint 0x101000056010407. | 11, 12, 13 |
| Fingerprint check | `DpuRunnerBaseImp::check_fingerprint` requires `model_fingerprint == dpu_fingerprint`. It can be disabled with `XLNX_ENABLE_FINGERPRINT_CHECK=0`. | 5 |

**Upstream firmware [fact 6, 7; ours].** The xclbin in `legacy/DPU_FIRMWARE/` contains one CU, `DPUCZDX8G_1` (the aclk clock is set to 300 MHz in the link options). It was built with `v++ 2022.1` against `xilinx_kv260_ispMipiRx_vcu_DP_202210_1`, using `kria-vitis-platforms/.../overlays/examples/benchmark`. That overlay vendors `DPUCZDX8G_v4_0_0`, which is the Vitis AI **2.5** IP. Even the 2023.1 branch still ships v4.0.0. The upstream course example (`fcabecciaw/cpsa-example`) compiles with the **Vitis AI 3.5** docker, using `arch_kv260_benchmark_b4096.json = {"fingerprint":"0x101000016010407"}`, and runs on this firmware.

## 2. Can 2.5 and 3.5 coexist?

| Route | What changes on the board | Current models | Jupyter/PYNQ demos | Coexistence |
|---|---|---|---|---|
| **A. Recompile 3.x zoo models for 0x101000016010407** (on the host) | nothing | untouched | untouched | trivially coexists [fact: our MoveNet was built with 3.5] |
| **B. DPU-PYNQ `design_contest_3.5`** installed in place | VART 3.5 and XRT libraries overwritten in `/usr/lib`, `pynq_dpu` replaced in pynq-venv, notebooks purged | 0x...16... xmodels fail the exact fingerprint check on the 0x...56... bitstream, so all three must be recompiled (or the check disabled) [fact 5 + est] | the 2.5 `pynq-dpu` notebooks are removed. The other notebooks should keep working, but only if the XRT library swap works [est]. | **No.** One `libvart` set sits in `/usr/lib`, so a separate venv does not isolate the C++ `.so` files. Side-by-side use would need a container or a prefix install with `LD_LIBRARY_PATH` [est]. |
| **C. Vitis AI 3.0 PetaLinux image** on a second SD card | whole OS | would need recompiling for 0x...56... [est] | none (no PYNQ, no Ubuntu) | only by swapping SD cards |
| **D. `xmutil loadapp kv260-benchmark-b4096`** | new apt package plus dfx-mgr loading | work (same fingerprint) | risk from mixing dfx-mgr and PYNQ bitstream loading [est] | pointless: it is the same 2.5 DPU |

Switching bitstreams (`DpuOverlay("x.bit")`) takes seconds and is not the problem. The problem is the system-wide runtime libraries.

## 3. What the 3.x zoos add for us

The KV260 figures come from the **3.0** zoo sheet (C++ Vitis AI Library, 1x B4096 @ 300 MHz, single thread) [fact 14]. Because 3.5 has no ZU+ builds, the 3.0 xmodels (fingerprint 0x...56...) are the only public KV260 builds.

| Model | Input | GOPs | COCO mAP float → quant | KV260 fps (1 thr) | Public quantized model for recompiling? | NumPy decode |
|---|---|---|---|---|---|---|
| YOLOX-Nano | 416² | 1.0 | 0.22 → 0.21 | 198 | **yes** (QAT `YOLOX_0_int.xmodel` in the GPU zip) | anchor-free, 3 scales, 3,549 cells: low effort |
| YOLOv5-nano | 640² | 4.6 | 0.270 → 0.262 | 78.1 | no (GPL stub; retrain or re-quantize) | as OFA-YOLO, 25.2k candidates: medium |
| SSDLite-MobileNetV2 | 300² | 1.5 | 0.217 → 0.209 | 113 | TF package | SSD priors: medium |
| YOLOv5s6 | 1280² | 17 | 0.436 → 0.420 | 11.4 | no | 4 scales: medium |
| YOLOv6m | 640² | 82.4 | 0.483 → 0.475 | 18.1 | copyleft repo (3.5 code; builds for VEK280/V70 only) | anchor-free: medium |
| YOLOv5-large | 640² | 109.6 | 0.472 → 0.455 | 9.3 | no | medium |
| YOLOv4-CSP | 640² | 121 | 0.470 → 0.463 | 7.9 | copyleft repo | medium |
| EfficientDet-D2 | 768² | 11.1 | 0.413 → **0.327** | 3.8 | TF package | BiFPN/anchors, large quantization drop: skip |
| YOLOv7 (3.5 only) | 640² | 104.8 | 0.512 → 0.479 | builds for VEK280/V70 only | copyleft repo | medium |
| MoveNet (3.0/3.5) | 192² | 0.5 | 0.797 → 0.798 | 102 | same model we already run | done |

Things to note:
- **YOLOv8** exists only as a Vitis AI Library sample (`yolov8m_pt`). It has no zoo entry and no public model [fact 2].
- **Nothing new for pose, re-ID or orientation.** The 3.0/3.5 model lists contain no person-specific models, and the Caffe and Darknet families from 2.5 are gone (RefineDet-pedestrian, SSD-pedestrian, YOLOv2) [fact 2, 9]. **The 2.5 zoo is richer for our use case.**
- Compared with our 2.5 options: YOLOv2-VOC pruned (7.8 G, 16 ms [ours]), RefineDet-ped 0.92 (10.1 G, 0.649 person AP) and OFA-YOLO 0.5 (24.6 G, COCO 0.378). Only **YOLOX-Nano** (≈5 ms, but COCO 0.21) and **YOLOv5n** (≈13 ms, COCO 0.262) are cheaper. Both are weak on the partially visible, seated or lying people that motivate a better detector [est]. The accurate 3.x models (mAP 0.45–0.48) run at 8–18 fps, which is slower than our YOLOv3 [fact 14].
- Every new model needs its DPU and CPU subgraphs checked with `xir`, and its CPU part (sigmoid, concat, decode) re-implemented in NumPy (see the HANDOFF pattern).

## 4. Effort and risk

| Route | Steps | Time [est] | What can go wrong | Rollback |
|---|---|---|---|---|
| **A** | Get an x86 host with docker (not available on this laptop: no docker/podman, and the ETH machines have no sudo). Pull `xilinx/vitis-ai-pytorch-cpu` (3.5) or the 3.0 image (≈10+ GB). Run `vai_c_xir -x YOLOX_0_int.xmodel -a arch.json` with fingerprint 0x101000016010407. Copy the result, check it with `xir`, and write the decoder. | 0.5 day to a first inference, +0.5–1 day for the decoder and tuning | unsupported ops end up in CPU subgraphs; the 3.0 QAT xmodel may not re-compile cleanly with the 3.5 compiler (then use the 3.0 docker) | delete the file |
| **B** | Image the SD card, run the 3.5 installer, recompile all 3 models for 0x...56..., then re-test the app and all notebooks | 1–2 days, plus debugging | The installer is KR260-targeted and unreleased, from an archived repo. It puts prebuilt XRT 2.15 libraries over the apt XRT 2.13 while the kernel zocl driver stays 2022.1. `apt upgrade` (565 pending) can revert files. The Jupyter demos can break. | restore from a `dd` image or a second SD card (write it before starting) |
| **C** | Flash the 3.0 PetaLinux image on a new card and port the app (no PYNQ, no apt) | 2–4 days | Python dependencies, camera and display stack, audio, the demo notebooks all need porting | swap the card back |
| **D** | install `xlnx-firmware-kv260-benchmark-b4096`, `xmutil loadapp` | 1–2 h | dfx-mgr (`/tmp/dfx-mgrd.socket` already returns permission denied for the ubuntu user) can clash with PYNQ's bitstream loading | `xmutil unloadapp`, reboot |

## 5. Recommendation

- **Now (demo in days): no.** Vitis AI 3.5 brings no new DPU, no KV260 xmodels, and no person-specific models. The only 3.5-on-Kria path (route B) replaces the shared runtime, forces recompiling all three working models, and puts the Jupyter setup at risk.
- **Later: route A only, when there is a concrete model to test.** The smallest experiment needs zero board changes. On an x86 docker host, recompile the YOLOX-Nano QAT xmodel from `pt_yolox-nano_coco_416_416_1G_3.0.zip` with `{"fingerprint":"0x101000016010407"}`. Then `scp` it and, in pynq-venv, call `DpuOverlay("dpu.bit").load_model(...)`. Print the DPU and CPU subgraphs and time one `execute_async`. If it loads and runs in about 5 ms, the question "can we use the 3.x zoo" is answered with yes; the remaining work is per-model decoding. Before investing in YOLOv5, first compare the person recall of YOLOX-Nano and YOLOv5n with RefineDet-ped on our clips.
- **Do not bother with the upstream xmutil route.** It is the same Vitis AI 2.5 DPU.
- **Correct the HANDOFF note** "3.5 models do not load" (not changed here) [ours + est]. The likely failure was a 3.0 `kv260` build (0x...56...) or a 3.5 VEK280/V70 build. 3.x-compiled xmodels for 0x...16... do load, as MoveNet shows. Which file was tried earlier is not recorded, so this is an inference.

## 6. Open uncertainties

- Whether the 3.5 compiler accepts the 3.0 QAT `*_int.xmodel` (the 3.0 docker is the fallback) [est].
- Whether a 0x...16... model would run correctly on the 0x...56... contest bitstream with the check disabled. It is plausible, because the only difference is the argmax-save feature, but it is untested [est].

## 7. Sources

1. Vitis AI 3.5 release notes ("No DPU IP updates / TRD / pre-build board image updates" for DPUCZDX8G): https://github.com/Xilinx/Vitis-AI/blob/v3.5/docsrc/source/docs/reference/release_notes.rst
2. 3.5 model list, all `model.yaml` files (fetched via `gh api repos/Xilinx/Vitis-AI/contents/model_zoo/model-list?ref=v3.5`): https://github.com/Xilinx/Vitis-AI/tree/v3.5/model_zoo/model-list ; the YOLOv8 sample: https://github.com/Xilinx/Vitis-AI/tree/v3.5/examples/vai_library/samples/yolov8
3. 3.5 MPSoC quickstart (links to the 3.0 image and runtime): https://github.com/Xilinx/Vitis-AI/blob/v3.5/docsrc/source/docs/quickstart/mpsoc.rst
4. Fingerprint bit fields (`SV_AM = bit 30`): https://github.com/Xilinx/Vitis-AI/blob/v3.5/src/vai_runtime/target_factory/src/create_target_DPUCZDX8G_ISA1.cpp
5. Exact-match fingerprint check: https://github.com/Xilinx/Vitis-AI/blob/v3.5/src/vai_runtime/vart/dpu-runner/src/dpu_runner_base_imp.cpp (l. 695–739)
6. xclbin metadata: `strings legacy/DPU_FIRMWARE/kv260-benchmark-b4096.xclbin`; benchmark overlay DPU IP: https://github.com/Xilinx/kria-vitis-platforms/tree/xlnx_rel_v2022.1/kv260/overlays/dpu_ip
7. Upstream course example (VAI 3.5 compile, fingerprint 0x101000016010407, xmutil benchmark-b4096): https://github.com/fcabecciaw/cpsa-example/blob/main/files/board_execution.md , `files/arch_kv260_benchmark_b4096.json`
8. Version compatibility table: https://github.com/Xilinx/Vitis-AI/blob/v3.5/docsrc/source/docs/reference/version_compatibility.rst
9. 3.0 model list: https://github.com/Xilinx/Vitis-AI/tree/v3.0/model_zoo/model-list ; the 3.0 `kv260` xmodels checked here: `yolox_nano_pt-…-r3.0.0`, `movenet_ntd_pt-…-r3.0.0` (fingerprint read from the xmodel bytes)
10. 3.5 target runtime script: https://github.com/Xilinx/Vitis-AI/blob/v3.5/board_setup/vek280/target_vart_setup.sh
11. DPU-PYNQ releases and branches: https://github.com/Xilinx/DPU-PYNQ/releases , https://github.com/Xilinx/DPU-PYNQ/tree/design_contest_3.5 (commit 3d3678d)
12. Kria 3.5 installer: https://github.com/amd/Kria-RoboticsAI/blob/main/files/scripts/install_update_kr260_to_vitisai35.sh
13. Fingerprint 0x101000056010407 in the 3.5 PYNQ log: https://github.com/amd/Kria-RoboticsAI/blob/main/README.md (§4)
14. Zoo performance sheets: `ModelZoo_VAI3.0_Github.xlsx` (https://github.com/Xilinx/Vitis-AI/tree/v3.0/docsrc/source/docs/reference) and `ModelZoo_Github.xlsx` (v3.5, VEK280/V70 only); copyleft models: https://github.com/Xilinx/Vitis-AI-Copyleft-Model-Zoo
