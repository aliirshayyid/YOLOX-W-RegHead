#!/usr/bin/env python3
"""
Run YOLOX inference with distance estimation.

Works for any class that has a distance head — stop signs and traffic lights
are both supported out of the box.

Usage:
    # Run on all detections
    python tools/infer_distance.py \
        --ckpt runs/distance_training/best_ckpt.pth \
        --img test_images/img001.jpg \
        --output-dir runs/results

    # Show distance only for traffic lights, hide everything else
    python tools/infer_distance.py \
        --ckpt runs/traffic_light_finetune/best_dist_ckpt.pth \
        --img test_images/ \
        --output-dir runs/results \
        --target-class 9

    # Backward-compatible: stop-sign-only (equivalent to --target-class 11)
    python tools/infer_distance.py \
        --ckpt runs/distance_training/best_ckpt.pth \
        --img test_images/ \
        --output-dir runs/results \
        --stop-sign-only
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

STOP_SIGN_CLASS    = 11   # COCO index
TRAFFIC_LIGHT_CLASS = 9   # COCO index

# Per-class BGR colours for bounding boxes
# Any class not listed here gets the DEFAULT_COLOR
_CLASS_COLORS = {
    STOP_SIGN_CLASS:     (0,   0,   255),   # red
    TRAFFIC_LIGHT_CLASS: (0,   200, 255),   # amber/yellow
}
DEFAULT_COLOR = (0, 255, 0)   # green for all other classes

# Classes for which the distance estimate is meaningful and should always
# be printed next to the label
DISTANCE_CLASSES = {STOP_SIGN_CLASS, TRAFFIC_LIGHT_CLASS}


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=80, depth=0.33, width=0.50):
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head     = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    return YOLOX(backbone, head)


# ─────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────

def preprocess(img, input_size=(640, 640)):
    """Letterbox resize + convert to float tensor."""
    preproc = ValTransform(legacy=False)
    tensor_img, _ = preproc(img, None, input_size)
    return torch.from_numpy(tensor_img).unsqueeze(0).float()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOX inference with distance estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt",       required=True,  help="Trained checkpoint (.pth)")
    parser.add_argument("--img",        required=True,  help="Image file or directory")
    parser.add_argument("--output-dir", default="runs/results")
    parser.add_argument("--conf",       type=float, default=0.3,  help="Confidence threshold")
    parser.add_argument("--nms",        type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img-size",   type=int,   default=640)
    parser.add_argument("--depth",      type=float, default=0.33, help="Model depth factor")
    parser.add_argument("--width",      type=float, default=0.50, help="Model width factor")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Filtering ─────────────────────────────────────────────────────
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--target-class", type=int, default=None,
        help=(
            "Only display detections of this COCO class ID and annotate with "
            "distance.  E.g. 9 = traffic light, 11 = stop sign."
        ),
    )
    filter_group.add_argument(
        "--stop-sign-only", action="store_true",
        help="Shorthand for --target-class 11 (backward-compatible).",
    )

    args = parser.parse_args()

    # Resolve filter class
    if args.stop_sign_only:
        filter_class = STOP_SIGN_CLASS
    else:
        filter_class = args.target_class   # None means show everything

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    model = build_model(depth=args.depth, width=args.width)
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.to(args.device)
    model.eval()

    # ── Collect images ────────────────────────────────────────────────
    if os.path.isdir(args.img):
        img_files = sorted([
            os.path.join(args.img, f) for f in os.listdir(args.img)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
    else:
        img_files = [args.img]

    # ── Per-image inference ───────────────────────────────────────────
    for img_path in img_files:
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to load: {img_path}"
        orig_h, orig_w = img.shape[:2]
        ratio = min(args.img_size / orig_h, args.img_size / orig_w)

        tensor = preprocess(img, (args.img_size, args.img_size)).to(args.device)

        with torch.no_grad():
            raw_output = model(tensor)

        # raw_output: [1, N_anchors, 4+1+num_classes+1]
        # After postprocess, each detection row is:
        #   [x1, y1, x2, y2, obj_conf, cls_conf, class_id, distance_m]
        results = postprocess(
            raw_output, num_classes=80,
            conf_thre=args.conf, nms_thre=args.nms,
        )

        if results[0] is None:
            print(f"{img_path}: no detections")
            continue

        bboxes    = results[0][:, :4]
        scores    = results[0][:, 4] * results[0][:, 5]
        classes   = results[0][:, 6].int()
        distances = results[0][:, 7]

        vis_img = img.copy()

        for i in range(len(bboxes)):
            cls_id = classes[i].item()

            # Apply class filter if requested
            if filter_class is not None and cls_id != filter_class:
                continue

            x1, y1, x2, y2 = (bboxes[i].cpu().numpy() / ratio).astype(int)
            score      = scores[i].item()
            distance_m = distances[i].item()
            cls_name   = COCO_CLASSES[cls_id]
            color      = _CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

            # Show distance for known distance-aware classes, or whenever a
            # specific target class was requested (covers custom fine-tuned models)
            show_dist = (cls_id in DISTANCE_CLASSES) or (filter_class == cls_id)
            if show_dist:
                label = f"{cls_name} {score:.2f} | {distance_m:.1f}m"
            else:
                label = f"{cls_name} {score:.2f}"

            # Bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Label background + text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_img, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(
                vis_img, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

            dist_str = f"{distance_m:.3f}m" if show_dist else "n/a"
            print(
                f"  [{cls_name}] bbox=({x1},{y1},{x2},{y2}) "
                f"conf={score:.3f} dist={dist_str}"
            )

        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, vis_img)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()