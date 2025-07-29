# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import json
import sys
from pathlib import Path
from datetime import datetime
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.torch_utils import time_sync                                             # ----- 添加

# 类型映射关系
type_mapping = {
    "Stakebed_Truck": 4,
    "Flatbed_Truck": 2,
    "Cross_country_Vehicle": 1,
    "Building": 3
}

# 处理数据
def process_data(data):
    # 检查数据合法性
    if not data:
        return None

    # 1. 转换时间戳 (使用第一个条目的时间)
    time_str = data[0]["time"]
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    timestamp = int(dt.timestamp() * 1000)

    # 2. 初始化目标列表
    target_ids = []
    type_states = []
    type_pres = []
    boxes = []

    # 3. 提取每个目标的信息
    for entry in data:
        target_ids.append(entry["Target_ID"])

        # 处理目标类型映射
        state_name = entry["Target_Type_state"]
        type_states.append(type_mapping.get(state_name, 0))  # 未映射类型设为0

        # 处理置信度
        type_pres.append(round(entry["Target_Type_Pre"], 2))  # 保留2位小数
        arr = [int(x) for x in entry["Target_Identification"].split(",")]
        boxes.extend(arr)

    # 4. 处理评估时间
    time_str = data[0]["Recognition_Time"].replace("ms", "")
    evaluate_time = float(time_str)

    # 5. 创建新数据结构
    return {
        "time": timestamp,
        "Target_Num": len(data),
        "Target_ID": target_ids,
        "Target_Type_State": type_states,
        "Target_Type_Pre": type_pres,
        "Target_Identification_box": boxes
    }

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        speed=True  #adjust windowWaitKey

):
    # print("image size: ".imgsz)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs


    show_name = "IR"
    cv2.namedWindow(show_name, 0)  # allow window resize (Linux)                                                  # -----增加
    # cv2.resizeWindow(show_name, int(1920 * 5 / 10), int(1080 * 5 / 10))
    # cv2.moveWindow(show_name, int(1920 * 5 / 10), 0)
    cv2.resizeWindow(show_name, int(2560), int(1440))

    all_detection_info = []

    # 初始化JSON文件路径和首帧标记
    json_file_path = save_dir / 'detection_info.json'
    first_frame = True



    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()                                                                        # ----添加时间
        with dt[0]:
            print(im.shape)
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        t2 = time_sync()                                                                        # ----添加时间

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()                                                                        # ----添加时间

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()                                                                        # ----添加时间

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        for i, det in enumerate(pred):  # per image
            seen += 1
            # 每帧开始时重置目标信息列表
            target_ids = []
            target_types = []
            target_pres = []
            target_identifications = []
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            image_detection_info = []

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            image_detection_info.append(f"Timestamp: {timestamp}")  # 插入时间戳






            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                target_id_counter = 1  # 新增：用于记录目标的ID
                for *xyxy, conf, cls in reversed(det):
                    target_type = names[int(cls)]
                    target_class_id = target_id_counter  # 修改：使用按顺序递增的ID
                    target_id_counter += 1  # 新增：递增ID计数器
                    # 收集每个目标的属性
                    target_ids.append(str(target_class_id))
                    target_types.append(target_type)
                    target_pres.append(f"{conf.item():.4f}")
                    xyxy_str = ','.join(map(str, [int(x.item()) for x in xyxy]))
                    target_identifications.append(xyxy_str)  # 直接添加字符串
                    c = int(cls)
                    label = None if hide_labels else (f'{target_type}' if hide_conf else f'{target_type} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))


            # 生成当前帧的JSON数据
            frame_detections = []
            for idx, (target_id, target_type, target_pre, target_box) in enumerate(
                    zip(target_ids, target_types, target_pres, target_identifications)):
                detection = {
                    "time": timestamp,
                    "Target_Num":int(len(target_ids)),
                    "Target_ID": int(target_id),
                    "Target_Type_state": target_type,
                    "Target_Type_Pre": float(target_pre),
                    "Recognition_Time": f'{round((t4 - t1) * 1000, 2)}ms',
                    "Target_Identification": f"{', '.join(map(str, [int(x) for x in target_box.split(',')]))}"
                }
                frame_detections.append(detection)


                # 无目标时的占位数据
            if not frame_detections:
                    frame_detections.append({
                    "time": timestamp,
                    "Target_Num":-1,
                    "Target_ID": 0,
                    "Target_Type_state": "",
                    "Target_Type_Pre": "",
                    "Recognition_Time": "0ms",
                    "Target_Identification": ""
                })


            # 为当前帧生成唯一的JSON文件名
            frame_json_file_path = save_dir / f'result_{seen}.json'
            # 将当前帧的检测信息写入对应的JSON文件
            with open(frame_json_file_path, 'w', encoding='utf-8') as f:
                json.dump(frame_detections, f, ensure_ascii=False, indent=4, separators=(',', ': '))
                result_json = process_data(frame_detections)
                json_str = json.dumps(result_json)
                print(f"result:{json_str}")



            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    # cv2.namedWindow("detect result",0)                                                                # -----注释掉
                    # cv2.resizeWindow("detect result", int(1920 * 6 / 10), int(1080 * 6 / 10))
                    # cv2.moveWindow("detect result", 0, 0)
                cv2.imshow(show_name, im0)  
                if speed == True:                                                                      # -------改名
                    cv2.waitKey(60)  # 1 millisecond
                elif speed == False :
                    cv2.waitKey(1)



    print("run over.")
    return 0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'D:\yanjiusheng\zzzz\hangtain\5.9\5.9\uav_Ir_LV_noRunway\yolov5\runs\train\exp3\weights\last.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default= r'D:\yanjiusheng\zzzz\hangtain\5.9\5.9\uav_Ir_LV_noRunway\yolov5\data\images\1\uav_ir_crop.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/hangpai.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[960], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--speed', default=True, action='store_true', help='adjust detect speed')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
