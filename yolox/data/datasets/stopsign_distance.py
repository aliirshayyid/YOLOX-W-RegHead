#!/usr/bin/env python3
"""
Dataset for stop sign detection with distance labels.
Labels format: [class_id, cx, cy, w, h, distance_m]
"""

import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class StopSignDistanceDataset(Dataset):
    """
    Simple dataset that loads images + bounding boxes + distance from a CSV.

    CSV columns: image_path, x_min, y_min, x_max, y_max, class, distance_m
    All bounding boxes are mapped to class 0 (single-class: stop_sign).
    """

    def __init__(self, csv_path, img_dir, img_size=(640, 640)):
        """
        Args:
            csv_path: path to CSV file with annotations
            img_dir: root directory for images
            img_size: (height, width) to resize images to
        """
        self.img_dir = img_dir
        self.img_size = img_size  # (H, W)

        # Parse CSV → group annotations by image
        self.image_data = {}  # {img_path: [(x1,y1,x2,y2,dist), ...]}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                ann = (
                    float(row["x_min"]),
                    float(row["y_min"]),
                    float(row["x_max"]),
                    float(row["y_max"]),
                    float(row["distance_m"]),
                )
                if img_path not in self.image_data:
                    self.image_data[img_path] = []
                self.image_data[img_path].append(ann)

        self.image_paths = list(self.image_data.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_rel_path = self.image_paths[idx]
        img_path = os.path.join(self.img_dir, img_rel_path)

        # Load image
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to load {img_path}"
        orig_h, orig_w = img.shape[:2]

        # Resize to target size
        target_h, target_w = self.img_size
        r = min(target_h / orig_h, target_w / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size (letterbox)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        # Convert to tensor: HWC BGR → CHW RGB, normalized to [0, 1]
        padded = padded[:, :, ::-1].copy()  # BGR → RGB
        padded = padded.transpose(2, 0, 1).astype(np.float32)  # HWC → CHW

        # Build labels: [class_id, cx, cy, w, h, distance_m]
        anns = self.image_data[img_rel_path]
        labels = np.zeros((len(anns), 6), dtype=np.float32)
        for i, (x1, y1, x2, y2, dist) in enumerate(anns):
            # Scale bounding box coordinates
            x1_s, x2_s = x1 * r, x2 * r
            y1_s, y2_s = y1 * r, y2 * r
            cx = (x1_s + x2_s) / 2.0
            cy = (y1_s + y2_s) / 2.0
            w = x2_s - x1_s
            h = y2_s - y1_s
            labels[i] = [0, cx, cy, w, h, dist]  # class 0 = stop_sign

        return torch.from_numpy(padded), torch.from_numpy(labels)


def collate_fn(batch):
    """
    Collate function that pads labels to the same size.
    Returns:
        imgs: [B, 3, H, W]
        labels: [B, max_labels, 6] with zero-padding
    """
    imgs, labels_list = zip(*batch)
    imgs = torch.stack(imgs, 0)

    max_labels = max(l.shape[0] for l in labels_list)
    max_labels = max(max_labels, 1)  # at least 1

    padded_labels = torch.zeros(len(labels_list), max_labels, 6)
    for i, labels in enumerate(labels_list):
        if labels.shape[0] > 0:
            padded_labels[i, :labels.shape[0], :] = labels

    return imgs, padded_labels