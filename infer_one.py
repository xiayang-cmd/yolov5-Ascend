#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend .om │ YOLOv5‑style 单图推理 Demo
作者: 狄云  ‖  日期: 2025‑07‑15
使用方法:
    python3 infer_one.py -m your_model.om -i test.jpg
输出:
    result.jpg   # 带检测框的图片
"""

import argparse, os, sys, time
import numpy as np

# ---------- 依赖检测 ----------
try:
    from ais_bench.infer.interface import InferSession
except ImportError:
    print("❌ 无法导入 ais_bench，请先执行:   pip install ais_bench")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("❌ 无法导入 OpenCV，请先执行:   pip install opencv-python")
    sys.exit(1)
# --------------------------------

# 根据自己的数据集改类别名；示例为中国象棋 14 类
CLASSES = [
        "body1","body2","body3","body4","body5","body6",
        "body7","body8","body9","body10""body11","body12",
]
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# ------------------ 工具函数 ------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """保持长宽比缩放 + 填充至 new_shape，返回新图、缩放比 r、填充 (dw,dh)"""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_size = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_size[0], new_shape[0] - new_size[1]
    dw, dh = dw / 2, dh / 2                               # 对称填充
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_resized = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color)
    return img_resized, r, (dw, dh)

def preprocess(img):
    img_lb, r, (dw, dh) = letterbox(img)
    # BGR→RGB, HWC→CHW, 归一化至 [0,1]
    blob = img_lb[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob, dtype=np.float16) / 255.0
    return np.expand_dims(blob, 0), r, dw, dh            # N,C,H,W


def postprocess(pred_list, conf_thres=0.001, iou_thres=0.60):
    """
    Ascend 推理原始输出 → 置信度过滤 → NMS 去重。
    与 YOLOv5 val.py 完全对齐：
      • score = obj_conf × cls_conf
      • NMS 用 [x, y, w, h]
      • 返回 xyxy，便于后续还原到原图
    """
    pred = pred_list[0]            # list → ndarray
    pred = pred[0]                 # (B, …) → 去掉 batch
    if pred.shape[0] == 85:        # Ascend 可能输出 (85, num)
        pred = pred.T              # → (num, 85)

    boxes              = pred[:, :4]        # xywh
    obj_conf           = pred[:, 4]
    cls_conf           = pred[:, 5:]
    cls_ids            = np.argmax(cls_conf, axis=1)
    cls_scores         = cls_conf[np.arange(len(pred)), cls_ids]

    # ★ 1. 置信度乘积，跟 val.py 保持一致
    scores = obj_conf * cls_scores          # ←←← 关键

    # ★ 2. 双阈值筛选（官方实现也是先看 obj_conf，再看乘积）
    keep = (obj_conf >= conf_thres) & (scores >= conf_thres)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

    if len(boxes) == 0:
        return []

    # 至此 boxes 仍是 xywh；先转成 xyxy 备份一份
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 0:2] -= boxes_xyxy[:, 2:4] / 2
    boxes_xyxy[:, 2:4] += boxes_xyxy[:, 0:2]

    # ★ 3. 给 OpenCV NMS 用的 [x, y, w, h]
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

    # ★ 4. NMS
    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(),
        conf_thres, iou_thres
    )
    if len(idxs) == 0:
        return []

    idxs = idxs.flatten()
    return [(boxes_xyxy[i], scores[i], int(cls_ids[i])) for i in idxs]

def draw(img, dets, r, dw, dh):
    h0, w0 = img.shape[:2]
    for box, score, cid in dets:
        x1, y1, x2, y2 = box
        # 复原到原图坐标
        x1, x2 = (x1 - dw) / r, (x2 - dw) / r
        y1, y2 = (y1 - dh) / r, (y2 - dh) / r
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cid].tolist(), 2)
        cv2.putText(img, f"{CLASSES[cid]} {score:.2f}",
                    (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORS[cid].tolist(), 1)
    return img
# ------------------------------------------------

# ---------- 前向 ----------（兼容旧/新版 ais_bench）---------
def safe_infer(session, blob, mode="static"):
    """兼容 ais_bench 0.x / 1.x：优先新版 [blob]，失败再退 ndarray"""
    try:
        return session.infer([blob], mode=mode)         # 新版 ≥1.1.x
    except TypeError:
        return session.infer(blob, mode=mode)           # 旧版 ≤0.3.x

def main(args):
    # ---------- 文件检查 ----------
    if not os.path.isfile(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"❌ 图片文件不存在: {args.image}")
        sys.exit(1)

    # ---------- 推理会话 ----------
    try:
        session = InferSession(device_id=args.device, model_path=args.model)
    except Exception as e:
        print("❌ 创建 InferSession 失败，请确认:")
        print("   ① Ascend 驱动/CANN 是否正常 ② 型号与 .om 是否匹配")
        print("错误信息:", e)
        sys.exit(1)

    # ---------- 读图 & 预处理 ----------
    img = cv2.imread(args.image)
    if img is None:
        print("❌ OpenCV 无法读取图片，请检查路径与格式")
        sys.exit(1)

    blob, r, dw, dh = preprocess(img)

    # ---------- 前向 ----------
    t0 = time.time()
    try:
        outputs = safe_infer(session, blob, mode="static")
        print("[DEBUG] 推理输出类型:", type(outputs))
        if isinstance(outputs, list):
            print("[DEBUG] 输出 list 长度:", len(outputs))
            print("[DEBUG] 第一项类型:", type(outputs[0]), "shape:", outputs[0].shape)
        else:
            print("❌ 推理返回值不是 list 类型，实际类型是:", type(outputs))
            sys.exit(1)
    except Exception as e:
        print("❌ 推理失败，常见原因：输入尺寸不符 / Ascend 运行时异常")
        print("错误信息:", e)
        sys.exit(1)
    t1 = time.time()

    # ---------- 后处理 & 画框 ----------
    dets = postprocess(outputs, conf_thres=args.score, iou_thres=args.iou)
    img_out = draw(img, dets, r, dw, dh)
    cv2.imwrite(args.output, img_out)
    print(f"✅ 推理完成，耗时 {(t1 - t0)*1000:.2f} ms，结果已保存: {args.output}")
    if len(dets) == 0:
        print("⚠️  置信度阈值过高或模型未能识别出目标，可尝试调低 --score")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ascend .om │ YOLOv5 单图推理示例")
    parser.add_argument("-m", "--model",  default="yolov5.om",
                        help=".om 模型路径")
    parser.add_argument("-i", "--image",  default="test.jpg",
                        help="待推理图片 (jpg/png)")
    parser.add_argument("-o", "--output", default="result.jpg",
                        help="检测后保存图片名")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="NPU 设备号 (默认 0)")
    parser.add_argument("--score", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--iou",   type=float, default=0.45,
                        help="NMS IoU 阈值")
    args = parser.parse_args()
    main(args)

