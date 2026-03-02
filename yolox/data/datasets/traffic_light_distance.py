#!/usr/bin/env python3
"""
Dataset for traffic light detection with distance labels.

Labels format: [class_id, cx, cy, w, h, distance_m]

CSV columns (required): image_path, x_min, y_min, x_max, y_max, distance_m
CSV columns (optional): class, state

  - class:    ignored at load time; all boxes are mapped to COCO class 9
              (traffic light) internally.
  - state:    one of {"red", "yellow", "green", "unknown"}.  When present,
              it is stored as an integer in an extra column [6] appended to
              the label tensor:
                0 = unknown / not provided
                1 = red
                2 = yellow
                3 = green
              Pass include_state=True to expose this column; otherwise the
              tensor shape is identical to StopSignDistanceDataset (N×6).

Example CSV (with state):
    image_path,x_min,y_min,x_max,y_max,distance_m,state
    images/frame001.jpg,312,88,376,210,18.5,red
    images/frame002.jpg,410,100,474,230,21.0,green

Example CSV (without state):
    image_path,x_min,y_min,x_max,y_max,distance_m
    images/frame001.jpg,312,88,376,210,18.5
"""

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# COCO class index for "traffic light"
TRAFFIC_LIGHT_CLASS_ID = 9

_STATE_MAP = {
    "red": 1,
    "yellow": 2,
    "green": 3,
}


class TrafficLightDistanceDataset(Dataset):
    """
    Dataset that loads images + bounding boxes + distance (+ optional state)
    from a CSV file.

    Args:
        csv_path (str):       Path to the CSV annotation file.
        img_dir (str):        Root directory for images.
        img_size (tuple):     Target (H, W) after letterbox resize.
        include_state (bool): If True, labels have shape (N, 7) where column 6
                              is the integer state code {0,1,2,3}.
                              If False (default), shape is (N, 6) — identical
                              to StopSignDistanceDataset.
    """

    def __init__(self, csv_path, img_dir, img_size=(640, 640), include_state=False):
        self.img_dir = img_dir
        self.img_size = img_size          # (H, W)
        self.include_state = include_state

        # Parse CSV → group annotations by image path
        # Each entry: (x1, y1, x2, y2, dist_m, state_int)
        self.image_data: dict[str, list] = {}

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_state = "state" in fieldnames

            for row in reader:
                img_path = row["image_path"]

                state_int = 0
                if has_state and row.get("state", "").strip().lower() in _STATE_MAP:
                    state_int = _STATE_MAP[row["state"].strip().lower()]

                ann = (
                    float(row["x_min"]),
                    float(row["y_min"]),
                    float(row["x_max"]),
                    float(row["y_max"]),
                    float(row["distance_m"]),
                    state_int,
                )

                if img_path not in self.image_data:
                    self.image_data[img_path] = []
                self.image_data[img_path].append(ann)

        self.image_paths = list(self.image_data.keys())

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_rel_path = self.image_paths[idx]
        img_path = os.path.join(self.img_dir, img_rel_path)

        # ── Load image ──────────────────────────────────────────────
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to load image: {img_path}"

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = self.img_size

        # Letterbox resize (preserve aspect ratio, pad to target size)
        r = min(target_h / orig_h, target_w / orig_w)
        new_h = int(orig_h * r)
        new_w = int(orig_w * r)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        # BGR → RGB, HWC → CHW, float32
        padded = padded[:, :, ::-1].copy()
        padded = padded.transpose(2, 0, 1).astype(np.float32)

        # ── Build labels ─────────────────────────────────────────────
        anns = self.image_data[img_rel_path]
        n_cols = 7 if self.include_state else 6
        labels = np.zeros((len(anns), n_cols), dtype=np.float32)

        for i, (x1, y1, x2, y2, dist, state_int) in enumerate(anns):
            # Scale bounding box to letterboxed coordinates
            x1_s, x2_s = x1 * r, x2 * r
            y1_s, y2_s = y1 * r, y2 * r
            cx = (x1_s + x2_s) / 2.0
            cy = (y1_s + y2_s) / 2.0
            w  = x2_s - x1_s
            h  = y2_s - y1_s

            labels[i, :6] = [TRAFFIC_LIGHT_CLASS_ID, cx, cy, w, h, dist]
            if self.include_state:
                labels[i, 6] = float(state_int)

        return torch.from_numpy(padded), torch.from_numpy(labels)


# ──────────────────────────────────────────────────────────────────────
# Collate function (compatible with StopSignDistanceDataset)
# ──────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Pads label tensors to a uniform length within each batch.

    Returns:
        imgs   (Tensor): [B, 3, H, W]
        labels (Tensor): [B, max_labels, n_cols]  — zero-padded rows
    """
    imgs, labels_list = zip(*batch)
    imgs = torch.stack(imgs, 0)

    max_labels = max(l.shape[0] for l in labels_list)
    max_labels = max(max_labels, 1)   # ensure at least 1 row

    n_cols = labels_list[0].shape[1]  # 6 or 7 depending on include_state
    padded_labels = torch.zeros(len(labels_list), max_labels, n_cols)

    for i, labels in enumerate(labels_list):
        if labels.shape[0] > 0:
            padded_labels[i, :labels.shape[0], :] = labels

    return imgs, padded_labels