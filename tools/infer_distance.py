#!/usr/bin/env python3
"""
Run YOLOX inference with distance estimation.

Usage:
    python tools/infer_distance.py \
        --ckpt runs/distance_training/best_ckpt.pth \
        --img test_images/img001.jpg \
        --output-dir runs/results
"""

import argparse
import os
import sys
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess


COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
STOP_SIGN_CLASS = 11  # index in COCO_CLASSES


def build_model(num_classes=80, depth=0.33, width=0.50):
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    return YOLOX(backbone, head)


def preprocess(img, input_size=(640, 640)):
    """Letterbox + normalize."""
    preproc = ValTransform(legacy=False)
    tensor_img, _ = preproc(img, None, input_size)
    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()
    return tensor_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--img", required=True, help="Image file or directory")
    parser.add_argument("--output-dir", default="runs/results")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stop-sign-only", action="store_true",
                        help="Only show stop sign detections")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build and load model
    model = build_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.to(args.device)
    model.eval()

    # Collect images
    if os.path.isdir(args.img):
        img_files = sorted([
            os.path.join(args.img, f) for f in os.listdir(args.img)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
    else:
        img_files = [args.img]

    for img_path in img_files:
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to load {img_path}"
        orig_h, orig_w = img.shape[:2]
        ratio = min(args.img_size / orig_h, args.img_size / orig_w)

        # Preprocess
        tensor = preprocess(img, (args.img_size, args.img_size)).to(args.device)

        # Inference
        with torch.no_grad():
            raw_output = model(tensor)

        # raw_output shape: [1, N_anchors, 4+1+80+1]
        # postprocess handles distance internally; surviving detections have
        # output columns: [x1, y1, x2, y2, obj_conf, class_conf, class_id, dist]
        results = postprocess(raw_output, num_classes=80,
                              conf_thre=args.conf, nms_thre=args.nms)

        if results[0] is None:
            print(f"{img_path}: No detections")
            continue

        bboxes = results[0][:, :4]    # x1, y1, x2, y2
        scores = results[0][:, 4] * results[0][:, 5]  # obj_conf * cls_conf
        classes = results[0][:, 6].int()
        distances = results[0][:, 7]  # distance in metres, appended by postprocess

        # Draw results
        vis_img = img.copy()
        for i in range(len(bboxes)):
            cls_id = classes[i].item()

            if args.stop_sign_only and cls_id != STOP_SIGN_CLASS:
                continue

            x1, y1, x2, y2 = bboxes[i].cpu().numpy() / ratio
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            score = scores[i].item()
            cls_name = COCO_CLASSES[cls_id]

            distance_m = distances[i].item()

            # Draw bounding box
            color = (0, 0, 255) if cls_id == STOP_SIGN_CLASS else (0, 255, 0)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if cls_id == STOP_SIGN_CLASS:
                label = f"{cls_name} {score:.2f} | {distance_m:.1f}m"
            else:
                label = f"{cls_name} {score:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_img, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            print(f"  {cls_name}: bbox=({x1},{y1},{x2},{y2}) "
                  f"conf={score:.3f} dist={distance_m:.1f}m")

        # Save
        out_name = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, vis_img)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()