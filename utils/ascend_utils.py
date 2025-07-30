import numpy as np
import cv2
from ais_bench.infer.interface import InferSession

# ------------------ preprocessing utilities ------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(img, img_size=(640, 640)):
    img_lb, r, (dw, dh) = letterbox(img, img_size)
    blob = img_lb[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob, dtype=np.float16) / 255.0
    return np.expand_dims(blob, 0), r, dw, dh


# ------------------ postprocessing utilities ------------------
def safe_infer(session, blob, mode="static"):
    try:
        return session.infer([blob], mode=mode)
    except TypeError:
        return session.infer(blob, mode=mode)


def postprocess(pred_list, conf_thres=0.25, iou_thres=0.45):
    pred = pred_list[0]
    pred = pred[0]
    if pred.shape[0] == 85:
        pred = pred.T

    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_conf = pred[:, 5:]
    cls_ids = np.argmax(cls_conf, axis=1)
    cls_scores = cls_conf[np.arange(len(pred)), cls_ids]

    scores = obj_conf * cls_scores
    keep = (obj_conf >= conf_thres) & (scores >= conf_thres)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if len(boxes) == 0:
        return []

    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 0:2] -= boxes_xyxy[:, 2:4] / 2
    boxes_xyxy[:, 2:4] += boxes_xyxy[:, 0:2]

    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

    idxs = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(idxs) == 0:
        return []
    idxs = idxs.flatten()
    return [(boxes_xyxy[i], scores[i], int(cls_ids[i])) for i in idxs]


def draw(img, dets, names, r, dw, dh, colors=None):
    if colors is None:
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in names]
    for box, score, cid in dets:
        x1, y1, x2, y2 = box
        x1, x2 = (x1 - dw) / r, (x2 - dw) / r
        y1, y2 = (y1 - dh) / r, (y2 - dh) / r
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cid], 2)
        cv2.putText(img, f"{names[cid]} {score:.2f}", (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cid], 1)
    return img


class AscendPredictor:
    def __init__(self, model_path, device_id=0):
        self.session = InferSession(device_id=device_id, model_path=model_path)

    def infer(self, img, conf_thres=0.25, iou_thres=0.45):
        blob, r, dw, dh = preprocess(img)
        outputs = safe_infer(self.session, blob, mode="static")
        dets = postprocess(outputs, conf_thres, iou_thres)

        # —— 反向去掉 letterbox 的缩放与填充，直接得到原图坐标 ——
        dets_scaled = []
        for box, score, cid in dets:
            x1, y1, x2, y2 = box
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r
            dets_scaled.append(((x1, y1, x2, y2), score, cid))

        return dets_scaled, r, dw, dh
