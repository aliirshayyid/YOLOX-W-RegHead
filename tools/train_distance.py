#!/usr/bin/env python3
"""
Train YOLOX distance head on stop sign images.

Usage:
    python tools/train_distance.py \
        --ckpt weights/yolox_s.pth \
        --csv data/stopsigns/annotations.csv \
        --img-dir data/stopsigns/ \
        --epochs 100 \
        --batch-size 8 \
        --lr 1e-3 \
        --output-dir runs/distance_training
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.utils import load_ckpt
from yolox.data.datasets.stopsign_distance import StopSignDistanceDataset, collate_fn


def build_model(num_classes=80, depth=0.33, width=0.50):
    """Build YOLOX-s with distance head."""
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    model = YOLOX(backbone, head)
    return model


def freeze_detection(model):
    """Freeze backbone + all head branches EXCEPT distance."""
    # Freeze backbone (PAFPN)
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Freeze detection head branches
    for name, param in model.head.named_parameters():
        if "dist_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({100*trainable/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Pretrained YOLOX checkpoint")
    parser.add_argument("--csv", required=True, help="CSV with distance annotations")
    parser.add_argument("--img-dir", required=True, help="Root dir for images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output-dir", default="runs/distance_training")
    parser.add_argument("--unfreeze-epoch", type=int, default=50,
                        help="Epoch to unfreeze backbone for end-to-end fine-tuning")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Config: {vars(args)}")

    # ── Build model ──
    model = build_model(num_classes=80)  # keep 80 classes to match pretrained weights

    # ── Load pretrained weights (partial load — skips distance layers automatically!) ──
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model = load_ckpt(model, ckpt)
    logger.info("Loaded pretrained weights (distance layers initialized randomly)")

    # ── Freeze everything except distance branch ──
    freeze_detection(model)
    model.to(args.device)
    model.train()

    # ── Dataset ──
    dataset = StopSignDistanceDataset(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_size=(args.img_size, args.img_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
    )
    logger.info(f"Dataset: {len(dataset)} images, {len(dataloader)} batches")

    # ── Optimizer (only distance params) ──
    dist_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(dist_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ──
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dist_loss = 0.0
        t0 = time.time()

        # Unfreeze backbone for end-to-end fine-tuning
        if epoch == args.unfreeze_epoch:
            logger.info(f"Epoch {epoch}: Unfreezing all parameters for fine-tuning")
            for param in model.parameters():
                param.requires_grad = True
            # Reset optimizer with all params and lower LR
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.01, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=args.lr * 0.001
            )

        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(imgs, targets)

            loss = outputs["total_loss"]
            dist_loss = outputs.get("dist_loss", 0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if isinstance(dist_loss, torch.Tensor):
                epoch_dist_loss += dist_loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        avg_dist = epoch_dist_loss / len(dataloader)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss: {avg_loss:.4f} | dist_loss: {avg_dist:.4f} | "
            f"lr: {lr_now:.6f} | time: {elapsed:.1f}s"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "best_ckpt.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch}, save_path)
            logger.info(f"  → Saved best checkpoint (loss={best_loss:.4f})")

        # Save latest
        save_path = os.path.join(args.output_dir, "latest_ckpt.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, save_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()