"""YOLOv3-family pre- and post-processing, pure numpy/OpenCV: YOLOv3-VOC, YOLOv4-COCO and OFA-YOLO (v5 style).

Same math as PYNQ's pynq-dpu/dpu_yolov3.ipynb (letterbox 416, RGB / 255, three heads, anchor masks,
correct_boxes), vectorised. The cps26 yolo_webcam version of this decoder matched PYNQ's `evaluate()`
box for box. Upstream CPSA_2026 feeds BGR without letterbox and decodes only the 13x13 head; not used here.
"""
import cv2
import numpy as np

from utils.types import Detection

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
PERSON = VOC_CLASSES.index("person")          # 14 (the brief's "class 0" is COCO numbering)
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"]
COCO_PERSON = 0

ANCHORS = np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                   np.float32).reshape(-1, 2)
ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]   # coarsest grid -> large anchors, finest -> small
# YOLOv4 COCO (zoo prototxt of dk_yolov4_coco_416_416_0.36_38.2G); OFA-YOLO uses the YOLOv3/v5 set above
YOLOV4_ANCHORS = np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                          np.float32).reshape(-1, 2)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(image_bgr, size=416, fill=128, scale=1.0 / 255.0):
    """BGR image -> (1, size, size, 3) float32 RGB * scale, padded with `fill` (PYNQ YOLOv3: 128, / 255)."""
    ih, iw = image_bgr.shape[:2]
    k = min(size / iw, size / ih)
    nw, nh = int(iw * k), int(ih * k)
    canvas = np.full((size, size, 3), fill, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * scale)[None]


def decode(outputs, image_hw, input_size=416, score_thresh=0.5, nms_iou=0.45, classes=(PERSON,),
           anchors=ANCHORS, class_names=VOC_CLASSES, style="v3"):
    """Three YOLOv3 heads (any order, each (1, g, g, 75)) -> list of Detection for `classes`.

    Boxes are undone from the letterbox into image pixels, then NMS per class.
    """
    ih, iw = image_hw
    k = min(input_size / iw, input_size / ih)
    new_w, new_h = round(iw * k), round(ih * k)
    off_x, off_y = (input_size - new_w) / 2 / input_size, (input_size - new_h) / 2 / input_size
    sx, sy = input_size / new_w, input_size / new_h
    num_classes = len(class_names)

    heads = sorted(outputs, key=lambda o: o.shape[1])          # 13, 26, 52
    boxes, scores, cls = [], [], []
    for out, mask in zip(heads, ANCHOR_MASK):
        g = out.shape[1]
        p = out.reshape(g, g, len(mask), 5 + num_classes)
        obj = _sigmoid(p[..., 4])
        gy, gx, a = np.nonzero(obj >= score_thresh)            # class score = obj * p(class) <= obj
        if len(a) == 0:
            continue
        cand = p[gy, gx, a]
        cls_scores = obj[gy, gx, a][:, None] * _sigmoid(cand[:, 5:])
        ci, cc = np.nonzero(cls_scores >= score_thresh)
        keep = np.isin(cc, classes)
        ci, cc = ci[keep], cc[keep]
        if len(ci) == 0:
            continue
        c, gyi, gxi, ai = cand[ci], gy[ci], gx[ci], a[ci]
        if style == "v5":                                      # YOLOv5 / OFA-YOLO box parametrisation
            bx = (_sigmoid(c[:, 0]) * 2 - 0.5 + gxi) / g
            by = (_sigmoid(c[:, 1]) * 2 - 0.5 + gyi) / g
            bw = (_sigmoid(c[:, 2]) * 2) ** 2 * anchors[mask][ai, 0] / input_size
            bh = (_sigmoid(c[:, 3]) * 2) ** 2 * anchors[mask][ai, 1] / input_size
        else:                                                   # YOLOv3 / YOLOv4
            bx = (_sigmoid(c[:, 0]) + gxi) / g
            by = (_sigmoid(c[:, 1]) + gyi) / g
            bw = np.exp(c[:, 2]) * anchors[mask][ai, 0] / input_size
            bh = np.exp(c[:, 3]) * anchors[mask][ai, 1] / input_size
        bx, by = (bx - off_x) * sx, (by - off_y) * sy
        bw, bh = bw * sx, bh * sy
        boxes.append(np.stack([(bx - bw / 2) * iw, (by - bh / 2) * ih, (bx + bw / 2) * iw, (by + bh / 2) * ih], 1))
        scores.append(cls_scores[ci, cc])
        cls.append(cc)
    if not boxes:
        return []
    boxes, scores, cls = np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, iw)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ih)

    dets = []
    for c in np.unique(cls):
        m = cls == c
        b, s = boxes[m], scores[m]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).tolist()
        idx = cv2.dnn.NMSBoxes(xywh, s.tolist(), score_thresh, nms_iou)
        for i in np.asarray(idx).flatten():
            dets.append(Detection(box=b[i].astype(np.float64), score=float(s[i]), label=class_names[int(c)]))
    return dets


def encode_for_test(boxes_xyxy, image_hw, input_size=416, cls=PERSON, logit=6.0):
    """Inverse of `decode` for unit tests: build three head tensors that decode to the given boxes."""
    ih, iw = image_hw
    k = min(input_size / iw, input_size / ih)
    new_w, new_h = round(iw * k), round(ih * k)
    off_x, off_y = (input_size - new_w) / 2 / input_size, (input_size - new_h) / 2 / input_size
    sx, sy = input_size / new_w, input_size / new_h
    grids = [13, 26, 52]
    heads = [np.full((1, g, g, 75), -12.0, np.float32) for g in grids]
    for (x1, y1, x2, y2) in boxes_xyxy:
        bx, by = (x1 + x2) / 2 / iw / sx + off_x, (y1 + y2) / 2 / ih / sy + off_y
        bw, bh = (x2 - x1) / iw / sx, (y2 - y1) / ih / sy
        g, head, mask = grids[0], heads[0], ANCHOR_MASK[0]
        a = 2                                              # largest anchor of the 13x13 head
        gx, gy = int(bx * g), int(by * g)
        fx, fy = bx * g - gx, by * g - gy
        v = head[0, gy, gx].reshape(3, 25)
        v[a, 0] = np.log(fx / (1 - fx))
        v[a, 1] = np.log(fy / (1 - fy))
        v[a, 2] = np.log(bw * input_size / ANCHORS[mask][a, 0])
        v[a, 3] = np.log(bh * input_size / ANCHORS[mask][a, 1])
        v[a, 4] = logit
        v[a, 5 + cls] = logit
        head[0, gy, gx] = v.reshape(-1)
    return heads
