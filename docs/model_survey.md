# Guardian Node: DPU Model Survey (Vitis AI 2.5, KV260)

*Scope:* person detectors, single-person pose estimators and auxiliary person models that can run on the KV260 (K26 SOM, one DPUCZDX8G_ISA1_B4096 @ 300 MHz, PYNQ-DPU 2.5 / VART 2.5), driven from one Python thread with NumPy pre- and post-processing on the Cortex-A53.

*Conventions:*
- **[zoo]** means a number taken from the Vitis AI model zoo README (v2.5, or v2.0 where the v2.5 README has no row for the model). All zoo throughput and latency figures were measured with the C++ Vitis AI Library on the stated board.
- **[ours]** means measured on our board.
- **[est]** means an estimate made in this report. Every [est] states its assumptions and must be checked on the board.

---

## 1. Executive summary

**Main findings**

1. The 2.5 zoo contains several **person-only detectors** with a `zcu102 & zcu104 & kv260` build. These are the Caffe RefineDet pedestrian family (0.8/0.92/0.96 pruned) and SSD-pedestrian. They are 2.6–13x cheaper on the DPU than YOLOv3-VOC. RefineDet's 360x480 input has exactly our camera's 4:3 aspect ratio, so no letterbox is needed.
2. For accurate, high-resolution detection, **OFA-YOLO 640** (native 640 input, COCO mAP 0.378–0.421 quantized) is the strongest 2.5 option. It is 1.3–1.9x faster than YOLOv3 on the DPU [zoo].
3. On the pose side, **MoveNet stays the primary model**. **SPnet** (0.55 GOPs, 1.8 ms [zoo]) is a cheap fallback for upright persons, and the **person-orientation** classifier (0.56 GOPs, ~1.4–1.6 ms [zoo]) can feed the "approach from behind" rule directly.
4. At room scale, "far" people are not small in the COCO sense. With an assumed focal length of ≈500 px [est], a 1.7 m person at 8 m is ≈105 px tall in the 640x480 frame. The main reason to use a high-resolution or accurate detector when FAR is recall on *partially visible, seated or lying* people, not tiny objects.
5. Energy scales mainly with DPU-busy time. The recommended schedule should use **roughly 3–10x less DPU energy per second** than running YOLOv3 continuously [est] (§4).

**Recommended configuration** (all [est] until verified on the board)

| State / band | Detector (rate) | Pose / auxiliary (rate) | DPU busy [est] |
|---|---|---|---|
| IDLE (no beacon) | none, camera off | none | 0 ms/s |
| DETECT (beacon seen, no person located yet) | OFA-YOLO 640 pruned-0.5 (24.6G) @ 3 Hz | none | ≈120–135 ms/s |
| FAR (> 3 m) | OFA-YOLO 640 pruned-0.5 @ 2 Hz (or RefineDet-ped 0.8 @ 3 Hz) | none | ≈80–110 ms/s |
| APPROACH (1.5–3 m) | RefineDet-ped 0.92 (10.1G) @ 5 Hz | MoveNet @ 5 Hz per person; orientation on humans @ 5 Hz | ≈200 ms/s |
| CLOSE (< 1.5 m) | RefineDet-ped 0.96 (5.1G) @ 3–5 Hz, plus pose-derived boxes in between | MoveNet on human @ 15 Hz, on robot @ 5 Hz; orientation @ 5–15 Hz | ≈250–300 ms/s |

**Demo-safe configuration.** This uses only the two models already verified on the board: YOLOv3-VOC 416 and MoveNet 192.

| Band | YOLOv3-VOC | MoveNet | DPU busy [est] |
|---|---|---|---|
| DETECT / FAR | 2 Hz | none | ≈180 ms/s |
| APPROACH | 3 Hz | 5 Hz per person | ≈370 ms/s |
| CLOSE | 2 Hz, with pose-derived boxes in between | 15 Hz human, 5 Hz robot | ≈350 ms/s DPU, ≈800 ms/s including ARM work |

In the demo-safe CLOSE band, every YOLOv3 call blocks the single Python thread for ≈110 ms, which is longer than one 15 Hz frame period (67 ms). There will be a visible pose gap unless the call is overlapped with ARM work (`execute_async` followed by a later `wait`).

---

## 2. Candidate models (all have a `zcu102 & zcu104 & kv260` xmodel in the v2.5 `model-list/*/model.yaml`)

### 2.1 How the Python-pipeline numbers are estimated

The only calibration points are our two measured models:

- **YOLOv3-VOC:** zoo KV260 end-to-end latency is 69–72 ms [zoo]. We measure 80–100 ms of DPU time alone [ours], so **k ≈ 1.1–1.4**.
- **MoveNet:** zoo end-to-end latency is ≈9.7 ms (102.8 fps) [zoo]. We measure 7–10 ms of DPU time [ours], so **k ≈ 0.7–1.0**.

From these:

- **Estimated DPU time** = k x zoo latency, with **k = 1.1–1.3** [est]. The zoo latency includes C++ pre/post-processing, so this is conservative for large models.
- **ARM time** is estimated per decoder type from our measured YOLOv3 decode (15–20 ms) and MoveNet pre-processing and decode (12–15 ms).
- **"Pipeline fps"** = 1000 / (DPU + ARM). It excludes camera capture, the rule engine and drawing. For YOLOv3 we measured ≈108 ms of DPU+ARM but only 6–7 fps live, so real live rates will be about **25–40 % lower** than pipeline fps.

### 2.2 Person detectors

Accuracy notes:
- VOC mAP is mAP@0.5 over 20 classes. COCO mAP is mAP@[.5:.95] over 80 classes. The two are **not comparable**.
- Person-only AP (the "COCO val person" rows) is not comparable to either.
- The zoo does not publish person-class AP for the multi-class models. We have to measure it on our own clips.

| Model id | FW | Input (HxW) | GOPs | Accuracy (float → quant) | KV260 C++ [zoo] | DPU / ARM / pipeline fps [est] | Decode in NumPy | Pros / cons for us |
|---|---|---|---|---|---|---|---|---|
| tf_yolov3_voc_416_416_65.63G | TF1 | 416x416 | 65.6 | VOC07 mAP 0.785 → 0.774 | 14.5 fps (72 ms) | **80–100 / 15–20 / ≈9 [ours]; live 6–7** | 3-scale YOLO, already implemented | Verified on the board. Most expensive per frame. VOC person class. |
| tf2_yolov3_coco_416_416_65.9G | TF2 | 416x416 | 65.9 | COCO 0.377 → 0.331 | 14.1 fps | 80–100 / 20–25 / 8–10 | Same decoder, 80 classes | No gain over VOC, and a 4.6-point quantization drop. |
| dk_yolov4_coco_416_416_60.1G | Darknet | 416x416 | 60.1 | COCO 0.395 → 0.373 (v2.0) | 13.8 fps (72 ms) | 80–95 / 20 / ≈9 | YOLO decode (reuse, COCO anchors) | Better COCO accuracy than YOLOv3 at the same cost. |
| dk_yolov4_coco_416_416_0.36_38.2G | Darknet | 416x416 | 38.2 | COCO 0.381 → 0.359 | 18.6 fps (53.7 ms) | 60–70 / 20 / 11–12 | YOLO decode (reuse) | Drop-in YOLO replacement at ≈0.6x the GOPs. |
| tf_yolov4_coco_416_416_60.3G | TF1 | 416x416 | 60.3 | COCO 0.477 → 0.393 | 14.4 fps | 75–95 / 20 / ≈9 | YOLO decode | 8.4-point quantization drop. Use the dk_ version instead. |
| pt_OFA-yolo_coco_640_640_48.88G | PyTorch | 640x640 | 48.9 | COCO 0.436 → 0.421 | 17.95 fps | 60–70 / 20–30 / 10–12 | YOLOv5-style decode over 3 scales (≈25k candidates); medium effort | Best quantized COCO mAP in the zoo. Native 640 input keeps full camera resolution (pad only). |
| pt_OFA-yolo_coco_640_640_0.3_34.72G | PyTorch | 640x640 | 34.7 | COCO 0.420 → 0.401 | 23.0 fps | 48–56 / 20–30 / 12–14 | as above | Good balance of accuracy and cost. |
| pt_OFA-yolo_coco_640_640_0.5_24.62G | PyTorch | 640x640 | 24.6 | COCO 0.392 → 0.378 | 29.2 fps | 38–45 / 20–30 / 13–17 | as above | **Recommended for FAR and DETECT.** Still more accurate than YOLOv3-COCO. |
| cf_refinedet_coco_360_480_0.8_25G | Caffe | 360x480 | 25.0 | COCO-person val 0.679 → 0.679 (v2.0) | 33.8 fps (29.6 ms) | 30–36 / 8–12 / 21–25 | Two-step (ARM + ODM) SSD-style decode over ≈10.8k priors [est], 2 classes; medium-high effort | Person-only, trained on COCO persons. 4:3 input, so resize only. |
| cf_refinedet_coco_360_480_0.92_10.10G | Caffe | 360x480 | 10.1 | 0.649 → 0.649 | 64.6 fps (15.5 ms) | 16–19 / 8–12 / 32–40 | as above | **Recommended for APPROACH.** |
| cf_refinedet_coco_360_480_0.96_5.08G | Caffe | 360x480 | 5.08 | 0.612 → 0.611 | 89.3 fps (11.2 ms) | 11–14 / 8–12 / 38–50 | as above | **Recommended for CLOSE**, where people are large. |
| cf_refinedet_coco_360_480_123G | Caffe | 360x480 | 123 | 0.693 → 0.704 | 9.0 fps (111 ms) | ≈120–140 / 10 / ≈7 | as above | Too slow for the small accuracy gain. |
| cf_ssdpedestrian_coco_360_640_0.97_5.9G | Caffe | 360x640 | 5.9 | COCO-person val 0.590 → 0.586 (trained on COCO person + CrowdHuman) | 80.4 fps (12.4 ms) | 12–15 / 8–12 / 37–47 | SSD priors, 2 classes | CrowdHuman training helps with occlusion. Input is 16:9, so it needs a letterbox or crop from 4:3. |
| dk_yolov2_voc_448_448_0.77_7.82G | Darknet | 448x448 | 7.82 | VOC07 0.758 → 0.748 | 90.2 fps (11.1 ms) | 11–14 / 5–8 / 45–60 | Single-scale YOLOv2 region decode; low effort | Cheapest YOLO-family model. Easy to decode. VOC classes. |
| dk_yolov2_voc_448_448_34G (0.66 / 0.71 pruned: 11.6G / 9.9G) | Darknet | 448x448 | 34 / 11.6 / 9.9 | 0.774 / 0.761 / 0.754 (quant) | 27.2 / 67 / 76.6 fps | ≈40–48 / 13–17 / 12–15 ms DPU | as above | Pruned variants are a cheap near-range detector. |
| tf_ssdmobilenetv1_coco_300_300_2.47G | TF1 | 300x300 | 2.47 | COCO 0.208 → 0.210 | 119 fps | 9–11 / 6–10 / 48–65 | SSD anchors (1917) x 91 classes | Cheap, but low mAP. 300 px input loses detail. |
| tf_ssdlite_mobilenetv2_coco_300_300_1.5G | TF1 | 300x300 | 1.5 | 0.217 → 0.209 | 111.5 fps | 9–11 / 6–10 / 48–65 | as above | Cheapest general-purpose detector. |
| tf_ssdmobilenetv2_coco_300_300_3.75G | TF1 | 300x300 | 3.75 | 0.215 → 0.211 | 87.7 fps | 12–15 / 6–10 / 40–50 | as above | Dominated by ssdlite. |
| tf_ssdinceptionv2_coco_300_300_9.62G | TF1 | 300x300 | 9.62 | 0.239 → 0.236 | 42.1 fps | 25–29 / 6–10 / 26–32 | as above | Dominated by RefineDet 0.92. |
| pt_FairMOT_mixed_640_480_0.5_36G | PyTorch | 640x480 (HxW order to verify) | 36 | MOT: MOTA 59.1 → 58.1 %, IDF1 62.5 → 60.5 % | 24.2 fps (41–44 ms) | 45–55 / 6–10 / 16–20 | CenterNet heatmap peaks + box regression + re-ID embedding; medium effort | Person-only. Matches camera resolution. Gives **identity embeddings for free**. Trained on upright pedestrians. |
| dk_yolov3_cityscapes_256_512_0.9_5.46G | Darknet | 256x512 | 5.46 | Cityscapes 0.552 → 0.530 | 95.5 fps | 11–13 / 8–10 | YOLO decode | Dashcam domain with a 2:1 aspect. Class list must be verified. Not recommended. |

**Rejected:**
- dk_tiny-yolov3_416 is a "commodity detection" model trained on a private dataset (VOC-style 0.965). It has no person class that we know of.
- tf_efficientdet-d2 runs at 3.3 fps, and quantization drops mAP from 0.413 to 0.327.
- tf_ssdresnet50v1_fpn (178G, 3 fps), tf_mlperf_resnet34 (433G) and tf_refinedet_VOC (81.9G) are too slow.
- dk_yolov3_bdd (53.7G) is a dashcam model.
- pt_yolox_TT100K (73G) detects traffic signs only.
- Face detectors (cf_densebox_wider 0.49G / 1.11G, cf_retinaface 1.11G, pt_face-mask-detection) do not find bodies. A face being detected inside a person box could serve as a cheap "facing the camera" cue.

### 2.3 Pose estimators

| Model id | FW | Input (HxW) | GOPs | Accuracy (float → quant) | KV260 C++ [zoo] | DPU / ARM per crop [est] | Decode | Pros / cons |
|---|---|---|---|---|---|---|---|---|
| pt_movenet_coco_192_192_0.5G | PyTorch | 192x192 | 0.5 | 0.797 → 0.798 (COCO; the zoo labels this only "accuracy", probably OKS-AP on a single-person subset; to verify) | 102.8 fps; 351 fps multi-thread | **7–10 / 12–15 [ours]** | Centre heatmap + offsets; already implemented | 17 COCO keypoints including face points (useful for facing direction). Upstream MoveNet training includes fitness and yoga poses. Verified on the board. |
| cf_SPnet_aichallenger_224_128_0.54G | Caffe | 224x128 (portrait) | 0.55 | PCKh@0.5 0.900 → 0.896 (AI Challenger) | 567 fps (1.76 ms) (v2.0) | 2–3 / 2–4 | Per-joint heatmap argmax; very easy | 14 keypoints with no face points. The portrait input suits standing people and suits lying people poorly. Good cheap fallback. |
| cf_hourglass_mpii_256_256_10.2G | Caffe | 256x256 | 10.2 | PCKh@0.5 0.872 → 0.866 (MPII) | 19.0 fps (52.7 ms) | 55–65 / 3–5 | Heatmap argmax | MPII covers many activities, including lying, but it is 6x slower than MoveNet. |
| cf_openpose_aichallenger_368_368_0.3_189.7G | Caffe | 368x368 | 189.7 | OKS 0.451 → 0.443 | 3.9 fps (258 ms) | 280–310 / 50–200 (PAF grouping) | Part affinity fields; high effort | Multi-person, so no detector would be needed, but it runs at under 3 fps in Python. Rejected. |

### 2.4 Auxiliary person models

| Model id | Input | GOPs | Accuracy (quant) | KV260 C++ [zoo] | DPU / ARM [est] | Use for Guardian Node |
|---|---|---|---|---|---|---|
| pt_person-orientation_224_112_558M | 224x112 | 0.56 | 0.929 (private dataset; number of orientation bins to verify) | 712.7 fps (≈1.4 ms) | 2 / 1–2 | Direct input to the "approach from behind" rule, cross-checked against the MoveNet shoulder and face keypoints. |
| pt_personreid-res18_market1501_176_80_1.1G | 176x80 | 1.1 | mAP 0.746, Rank-1 0.893 (Market1501) | 399.8 fps | 3 / 1–2 | Keeps human and robot identities apart across detector gaps. Runs only when tracks are ambiguous. |
| pt_personreid-res50_market1501_256_128_5.3G | 256x128 | 5.3 | mAP 0.869, Rank-1 0.948 | 115.8 fps | 10–11 / 2 | More accurate but 3.5x the cost. The pruned res50 variants have **no** KV260 build. |
| cf_reid_market1501_160_80_0.95G | 160x80 | 0.95 | mAP 0.559, Rank-1 0.776 | 340 fps (v2.0) | 3 / 1–2 | Dominated by res18. |

The robot is a person wearing a beacon. Re-ID is therefore only an identity aid. The **role** (robot or human) must still come from the beacon association, which is deterministic and safer.

---

## 3. Multi-modal scheduling design space

The DPU is a single core and all work happens on one Python thread. The DPU budget below is per second of wall-clock time. ARM work adds roughly 0.5–1.5x the DPU time, depending on the decoder [est].

| Option | Description | Pros | Cons / risk | Runtime impact [est] |
|---|---|---|---|---|
| A. Detector per band | Accurate detector (OFA-YOLO or RefineDet-0.8) when far; cheap detector (RefineDet-0.96 or YOLOv2-0.77) when close | Spends accuracy where people are hard to detect | Each model needs its own decoder and validation. Switching is cheap if all runners are preloaded (§5). | CLOSE detector cost falls from ≈90 to ≈12 ms per call |
| B. Rate scaling | Detector at 2 Hz FAR, 5 Hz APPROACH, 3–5 Hz CLOSE | Linear energy savings | Latency to the first detection of a new entrant is up to one period (500 ms FAR), which must fit inside the 1 s look-ahead | DPU time scales linearly with rate |
| C. ROI/crop cascade | Run the detector on a crop around the last known person or robot box, or on a 2x2 tile set when far | Higher effective resolution for small or lying persons | Can miss new entrants outside the ROI, so a periodic full-frame pass is needed | Tile pass = 4x detector cost at low rate; ROI pass = one call |
| D. Detect-then-track | Detector runs at low rate; a cheap IoU or centroid tracker (NumPy) associates boxes in between | Stable IDs for the rule engine. Almost no DPU cost. | Identity swaps under occlusion. Mitigated by Re-ID or the beacon. | ≈0 DPU, <1 ms ARM |
| E. Pose-derived boxes | In CLOSE, the next MoveNet crop comes from the previous keypoints' bounding box (plus a margin); the detector is skipped while keypoint confidence is high | Removes most detector calls when close | Drift if confidence is miscalibrated. A second person entering the frame is missed until the next full-frame detection, so a **forced detector call every ≤ 300 ms** is needed. | Detector calls in CLOSE drop from 15 Hz to 3–5 Hz |
| F. Asymmetric pose rates | Human at 15 Hz, robot at 5 Hz (the robot's plan is known and it moves slowly) | ≈33 % fewer MoveNet calls with two actors | The robot is a human actor and might move unexpectedly | Saves ≈10 calls/s, about 90 ms DPU and 130 ms ARM |
| G. Conditional auxiliaries | Orientation only when the human is within the facing-rule zone; Re-ID only on track ambiguity | Near-zero cost when not needed | More state-machine complexity | ≈2 ms per call |
| H. Async overlap | `execute_async` on the DPU while decoding the previous result on the ARM | Hides ARM time and gives up to ≈1.6x throughput | More complex code. PYNQ-DPU runners must support concurrent jobs (to verify). | Pipeline time moves from DPU+ARM towards max(DPU, ARM) |
| I. Closing-speed bump | Fast closing moves the node one band closer | Safety margin | Oscillation at band edges, so hysteresis is needed (e.g. ±0.3 m and a 0.5 s dwell) | Temporarily raises the budget |

**DPU budget per band for the recommended configuration.** This assumes two actors: one human and the robot.

| Band | Calls/s | DPU ms/s [est] | ARM ms/s [est] | Headroom (of 1000 ms) |
|---|---|---|---|---|
| IDLE | 0 | 0 | ≈0 (beacon polling only) | all |
| DETECT | OFA-YOLO-0.5 x3 | 115–135 | 60–90 | ≈75 % |
| FAR | OFA-YOLO-0.5 x2 | 76–90 | 40–60 | ≈85 % |
| APPROACH | RefineDet-0.92 x5, MoveNet x10, orientation x5 | 80–95 + 70–100 + 10 = 160–205 | 40–60 + 120–150 + 10 = 170–220 | ≈60 % |
| CLOSE | RefineDet-0.96 x5, MoveNet x20 (15 human + 5 robot), orientation x15 | 55–70 + 140–200 + 30 = 225–300 | 40–60 + 240–300 + 25 = 305–385 | ≈30–45 % |
| Baseline: YOLOv3 at full rate, always on | YOLOv3 x6.5 | 520–650 | 100–130 | ≈20 % |
| Current pipeline: YOLOv3 + MoveNet every frame, 2 actors | 6 + 12 calls | 600–710 | 250–300 | ≈0 % |

---

## 4. Energy

### 4.1 Published figures (for scale only; not our workload)

- KV260 boards are typically reported at **≈5 W** during DPU inference. The survey cited below attributes this to RetinaNet, MultiTaskV3 and segmentation studies.
- A multi-task ADAS network on KV260 with a B4096 DPU was reported at **7.19 W** at 25.4 fps.
- A KR260 user benchmark (same K26 SOM) measured **5.07 W** for B4096 @ 150 MHz with an external current checker.
- A pruned-YOLO study on KV260 reports **3.5 W** at 15 fps.

None of these separates idle power from DPU-active power. We therefore **assume ΔP_DPU ≈ 2.5 W (plausible range 1.5–3.5 W) [est]** above the "overlay loaded, idle" baseline while the DPU is busy. We also assume **≈0.4 W [est]** for one fully busy A53 core running Python.

### 4.2 Energy per inference ≈ ΔP x t_DPU [est]

| Model | t_DPU (ms) | E_DPU per call (mJ) at 2.5 W [est] | Relative to YOLOv3 |
|---|---|---|---|
| tf_yolov3_voc (65.6G) | 90 [ours] | ≈225 | 1.00 |
| dk_yolov4 pruned (38.2G) | 65 | ≈160 | 0.72 |
| OFA-YOLO-0.5 (24.6G) | 42 | ≈105 | 0.47 |
| RefineDet-ped 0.8 (25G) | 33 | ≈83 | 0.37 |
| RefineDet-ped 0.92 (10.1G) | 17 | ≈43 | 0.19 |
| RefineDet-ped 0.96 (5.08G) | 12.5 | ≈31 | 0.14 |
| SSD-pedestrian (5.9G) | 13.5 | ≈34 | 0.15 |
| MoveNet (0.5G) | 9 [ours] | ≈23 | 0.10 |
| SPnet (0.55G) | 2.5 | ≈6 | 0.03 |
| Person-orientation (0.56G) | 2 | ≈5 | 0.02 |

Energy is not proportional to GOPs for small models. MoveNet has 1/130 of YOLOv3's GOPs but uses 1/10 of its DPU time, because fixed per-call overhead and memory-bound layers dominate. The DPU-time ratio is the better proxy.

### 4.3 Per-band average DPU+ARM power above the idle baseline [est]

| Band | DPU W [est] | ARM W [est] | Total ΔW [est] |
|---|---|---|---|
| IDLE | 0 | ≈0 | ≈0 (plus the option to unload the PL, §4.4) |
| DETECT | 0.30 | 0.03 | ≈0.33 |
| FAR | 0.21 | 0.02 | ≈0.23 |
| APPROACH | 0.46 | 0.08 | ≈0.54 |
| CLOSE | 0.66 | 0.14 | ≈0.80 |
| YOLOv3 always on (6.5 Hz) | 1.46 | 0.05 | ≈1.5 |
| Current pipeline (YOLOv3 + MoveNet each frame) | 1.6–1.8 | 0.11 | ≈1.8 |

Example care-home day [est]: IDLE 60 %, FAR 20 %, APPROACH 12 %, CLOSE 8 %. This averages about **0.18 W above baseline, against ≈1.5 W for YOLOv3 always on**, a ≈8x reduction in inference energy. In the busiest band (CLOSE) the schedule still uses about half the always-on figure. The camera (≈0.5–1 W for a USB webcam [est]) and the board baseline (several W) are not included. They will dominate total power, which is why camera-off in IDLE matters as much as model choice.

### 4.4 Measurement protocol (planned board experiment)

Use `pynq.get_rails()` together with `pynq.DataRecorder` at ≥10 Hz, and cross-check with `xmutil platformstats -p`. Record the PS and PL rails and the total separately. Run each step ≥60 s and repeat it 3 times, reporting mean ± std.

1. **Idle baseline:** Linux booted, no overlay, no camera.
2. **Overlay loaded, idle:** `DpuOverlay` loaded, all runners created, no inference. This gives the PL static cost and the memory footprint (`free -m`).
3. **Camera on:** capture at 15 fps and 30 fps, no inference.
4. **Each model at continuous rate** (a tight loop over a fixed input). Record average power P_run and calls/s. Then calculate **E_call = (P_run − P_step2) / calls_per_s**, and also ΔP x t_DPU, where t_DPU is timed around `execute_async`/`wait`.
5. **Duty-cycled schedules:** run each band configuration from §3 and compare the measured average power with the §4.3 predictions.
6. **Overlay load and unload time:** measure the time and energy of `DpuOverlay(...)` plus runner creation, and of freeing the PL. Unloading the PL in IDLE is worthwhile only if (P_step2 − P_step1) x the expected IDLE duration is greater than the reload energy, **and** the reload latency (expected to be seconds [est]) is shorter than the time between beacon detection and the robot reaching the 3 m band.
7. **Thermal check:** record the temperature over 10 min in CLOSE, since the KV260 fan affects total power.

---

## 5. Specialisation over generalisation (deterministic model selection)

**The principle.** A deterministic, cheaply computed signal (like altitude for drones) selects a model that only has to be good in one regime. Specialised models can be smaller and more accurate within their regime than one general model. The cost is a validation burden for each regime and a dependency on the selector being right.

### 5.1 Selectors available to Guardian Node

| Selector | Source | Cost | Reliability / failure mode |
|---|---|---|---|
| Robot–human floor distance band | Box bottom edge on the ground plane (camera calibration), or box height | ≈0 | Box height fails for lying or seated people; the ground-contact point fails under occlusion. Use both and take the more conservative (closer) band. |
| Apparent person size (box height in px) | Last detection | 0 | Directly measures what the detector sees, so it is the most relevant selector for detector choice |
| Node state (IDLE / DETECT / TRACK) | State machine | 0 | Deterministic |
| Closing speed | Track history | ≈0 | Noisy at low detection rates. Needs filtering. |
| Number of people | Tracker | 0 | Undercounts at low rate |
| Scene type (corridor, room, warehouse aisle) | Commissioning config | 0 | Static and reliable. Sets priors (e.g. corridor = long distances). |
| Lighting / time of day | Frame mean luminance, clock | <1 ms | Reliable |
| Posture (standing, seated, lying) | Last pose (hip–shoulder angle, box aspect ratio) | <1 ms | Circular: a wrong pose selects the wrong pose model |

### 5.2 Selector → regime → model table

| Selector → regime | Detector | Pose | Auxiliary | Rate | Rationale |
|---|---|---|---|---|---|
| State = IDLE | none | none | none | 0 | No robot, so no hazard |
| State = DETECT or new-entrant scan | OFA-YOLO-0.5 (640, general) | none | none | 3 Hz | Maximum recall on unknown scenes |
| Distance > 3 m, or box height < 120 px | OFA-YOLO-0.5, or RefineDet-0.8 (person-only) | none | none | 2 Hz | Accuracy matters and time is not critical |
| Distance 1.5–3 m, or box height 120–250 px | RefineDet-ped 0.92 | MoveNet | orientation (human only) | 5 Hz | Medium-size persons; the pose rules start |
| Distance < 1.5 m, or box height > 250 px | RefineDet-ped 0.96, plus pose-derived boxes | MoveNet (human), MoveNet at low rate (robot) | orientation | pose 15 Hz, detector 3–5 Hz | Large persons, so a cheap detector suffices |
| Posture = lying/fallen (box aspect w/h > 1.2) | OFA-YOLO (COCO includes lying people), not the pedestrian models | MoveNet on a crop rotated upright, or Hourglass as a second opinion | none | ≥5 Hz | Pedestrian detectors and SPnet are biased towards upright people |
| Posture = upright and CLOSE with a CPU/DPU budget alarm | RefineDet-0.96 | SPnet (2.5 ms) | orientation (needed because SPnet has no face points) | 15 Hz | Degraded-mode fallback |
| People count ≥ 3 | RefineDet-0.92 at every band | MoveNet on the nearest 2 humans | Re-ID when tracks cross | 5 Hz | Keeps the budget bounded |
| Scene = corridor (commissioning) | OFA-YOLO-0.3 or full 48.9G for FAR | as above | none | 2 Hz | Long sight lines put people at smaller pixel sizes |
| Low light | OFA-YOLO (best general recall) at every band | MoveNet with a raised confidence threshold | none | as band | The cheap detectors' behaviour in low light is unverified. Stay general. |

### 5.3 Which specialised models exist now, and which would need training

**Available in the 2.5 zoo:**
- Person-only detectors: RefineDet-ped x4 and SSD-pedestrian. These are specialised by class, and by distance or size in the sense that their pruning levels trade small-object recall for cost.
- A single-person crop pose model (MoveNet, SPnet) as opposed to a multi-person one (OpenPose).
- Orientation and Re-ID models, run on demand.

**Would need our own training:** out of scope this week. Recorded here as the path forward.

- **Fallen/lying-person detector:** fine-tune RefineDet-ped or OFA-YOLO with fall datasets added.
- **Care-home-domain person detector:** wheelchairs, walkers, beds, occlusion by furniture.
- **Rotation-robust or lying-pose MoveNet variant.**
- **Size-specialised detectors:** e.g. a 320 px model trained only on people taller than 200 px for CLOSE, which could be ≈1–3 GOPs [est].

The toolchain is the Vitis AI 2.5 docker (not 3.x): train or fine-tune in float, then quantize with `vai_q_pytorch`, `vai_q_tensorflow` or Caffe `vai_q_caffe` (a quantization-aware training option exists), then compile with `vai_c_xir` against the `arch.json` of **DPUCZDX8G_ISA1_B4096** that matches our PYNQ-DPU 2.5 overlay fingerprint.

Expected gains [est]:
- A single-class person head reduces the decode to one class, which saves ARM time.
- Domain fine-tuning typically recovers much of the domain-gap loss, but by an amount we cannot predict without data.
- Size-specialised models cut CLOSE detector cost by 2–5x relative to RefineDet-0.96.

### 5.4 Pros, cons and safe fallbacks

**Pros**
- Accuracy per GOP goes up in each regime.
- Energy scales with the situation (§4.3).
- Switching is nearly free if every runner is created at boot on the one overlay. No PL reconfiguration is needed and there is only a Python dict lookup.

**Cons**
- Weight memory: the sum of the xmodels is an estimated ≈150–250 MB for all recommended models (YOLOv3 alone is roughly 60+ MB of int8 weights [est]). This fits in the K26's 4 GB but must be measured.
- Each model needs its own decoder, tests and validation in its regime, so the validation matrix grows as models × regimes.
- Behaviour at regime boundaries needs attention.

**Failure modes and mitigations**
1. **The selector picks the wrong regime.** Example: a depth error puts a close person in FAR, so there is no pose and only a 2 Hz detector. The rule is that **errors must bias toward the more protective band**. Take the minimum distance over the box-height and ground-contact estimates, and let the closing-speed bump override.
2. **Boundary oscillation.** Use hysteresis (±0.3 m, ≥0.5 s dwell). Within ±0.3 m of a boundary, run **both** regimes' detectors alternately, or take the union of their boxes.
3. **Specialised model misses in its own regime.** Example: the pedestrian detector misses a fallen person. Run a **general-model watchdog**: OFA-YOLO or YOLOv3 at ≥1 Hz in every non-IDLE band, and treat any person it finds that the specialised path missed as a fault. Escalate one band and log the event.
4. **Posture selector is circular** (a wrong pose picks the wrong pose model). Use the box aspect ratio from the detector as the posture selector, not the pose output.

---

## 6. Constraints and risks

| Risk | Detail | Mitigation / what to verify on the board |
|---|---|---|
| Runtime version | Only 2.5 xmodels load. 3.5 models failed. | Download only `*_2.5` models with the `zcu102 & zcu104 & kv260` build. Check each xmodel's DPU fingerprint against the overlay (`xir` subgraph attribute `dpu_fingerprint`) before relying on it. |
| Multiple runners | PYNQ-DPU's `load_model` exposes a single `overlay.runner` | Create additional runners directly with `vart.Runner.create_runner(subgraph, "run")` after loading the overlay. Verify that ≥4 runners can coexist, and check memory (`free -m`) and creation time. |
| CPU subgraphs | Some Caffe and TF xmodels may split into several DPU subgraphs, or leave softmax, prior-box or concat work on the CPU | Inspect `graph.get_root_subgraph().toposort_child_subgraph()` for each candidate. Prefer models with exactly one DPU subgraph for the Python path. |
| Quantization drops | tf_yolov4 −8.4 points, tf2_yolov3 −4.6, efficientdet −8.6. RefineDet, OFA-YOLO, MoveNet and SPnet show ≤1.5 points. | Prefer models with small reported drops. Re-measure person recall on our own clips. |
| Metric comparability | VOC mAP@0.5 vs COCO mAP@[.5:.95] vs COCO-person AP vs PCKh vs OKS | Build a small labelled test set (a few hundred frames from our camera and room: standing, seated, lying, occluded) and measure person recall at a fixed false-positive rate. |
| Domain gap: lying/fallen | Pedestrian models (RefineDet-ped from COCO person; SSD-ped with CrowdHuman; FairMOT on MOT) and SPnet's portrait input are biased towards upright people. COCO and VOC contain some lying and seated people. MPII (Hourglass) covers diverse activities. MoveNet's upstream training covers fitness and yoga poses but is reported to struggle with non-upright orientations. | Posture-selected path (§5.2), general-model watchdog, rotated-crop MoveNet. Test with staged falls. |
| Domain gap: wheelchairs, walkers, beds, occlusion | No zoo model was trained on care-home data | Test explicitly. CrowdHuman training (SSD-ped) may help with occlusion. |
| ARM-bound decode | SSD/RefineDet prior decoding and YOLOv5-style decoding over ≈25k candidates in NumPy | Vectorise, pre-compute priors once, filter by confidence before decoding boxes. Budget ≤12 ms. |
| Single thread | A 110 ms YOLOv3 call stalls 15 Hz pose | Async overlap (option H), and keep heavy detectors out of CLOSE. |
| Camera distortion and FOV | The distance-from-box-height estimate depends on the focal length (≈500 px assumed) | Calibrate once with a checkerboard, and also measure person height in px at 1, 2, 3, 5 and 8 m. |

---

## 7. On-board verification checklist (in priority order)

1. Load RefineDet-ped 0.92/0.96, OFA-YOLO-0.5, SPnet and person-orientation alongside YOLOv3 and MoveNet as separate runners. Record memory use and creation time.
2. Time `execute_async`+`wait` for each model and fit k (§2.1).
3. Implement and time the RefineDet and OFA-YOLO decoders in NumPy.
4. Measure person recall on our clips for YOLOv3-VOC, OFA-YOLO-0.5 and RefineDet 0.8/0.92/0.96, covering standing, seated, lying and occluded people.
5. Run the power protocol in §4.4.
6. Check the orientation model's output bins and its agreement with MoveNet shoulder and face keypoints.

---

## 8. Sources

- Vitis AI 2.5 model zoo README (accuracy tables, KV260 performance table): https://github.com/Xilinx/Vitis-AI/blob/v2.5/model_zoo/README.md
- Vitis AI 2.5 model list (per-model `model.yaml`, board builds): https://github.com/Xilinx/Vitis-AI/tree/v2.5/model_zoo/model-list
- Vitis AI 2.0 model zoo README (Caffe/Darknet accuracies, KV260 latency column): https://github.com/Xilinx/Vitis-AI/blob/v2.0/models/AI-Model-Zoo/README.md
- DPUCZDX8G product guide PG338: https://docs.amd.com/r/en-US/pg338-dpu
- KR260 B4096 benchmark with power (5.07 W @ 150 MHz): https://www.hackster.io/iotengineer22/benchmark-architectures-of-the-dpu-with-kr260-699f19
- KV260 ≈5 W inference survey: https://www.emergentmind.com/topics/amd-kria-kv260-system-on-module-som
- KV260 B4096 multi-task ADAS, 7.19 W @ 25.4 fps: https://www.researchgate.net/figure/Performance-comparison-of-DPU-with-and-without-optimized_tbl4_372850891
- KV260 on-board power measurement discussion: https://adaptivesupport.amd.com/s/question/0D54U00006sZWz7SAG/kv260-onboard-power-consumption-measurement-resolution-and-sampling-frequency?language=en_US
- Kria KV260 AI benchmark repo: https://github.com/Xilinx/kria-kv260-ai-benchmark
- PYNQ-DPU: https://github.com/Xilinx/DPU-PYNQ
