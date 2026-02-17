#!/usr/bin/env python3
"""
Train YOLOX distance head on stop sign images.

Only the distance regression branch is trained; the backbone and all
detection-head branches (cls, reg, obj) stay frozen for the entire run.

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


def freeze_all_except_distance(model):
    """Freeze backbone + all head branches EXCEPT distance — permanently.

    Only freezes requires_grad on parameters.  Does NOT touch train/eval mode
    here — that is handled in the training loop so that YOLOX.forward() still
    takes the training (loss-computation) code path.
    """
    # ── Freeze every parameter first ──
    for param in model.parameters():
        param.requires_grad = False

    # ── Selectively unfreeze only distance-branch parameters ──
    dist_param_names = []
    for name, param in model.named_parameters():
        if "dist_" in name:
            param.requires_grad = True
            dist_param_names.append(name)

    if not dist_param_names:
        logger.error(
            "No distance-branch parameters found (expected names containing 'dist_'). "
            "Check that YOLOXHead defines the distance regression layers."
        )
        raise RuntimeError("Distance head parameters not found in model.")

    # ── Report ──
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)")
    logger.info(f"Trainable layers ({len(dist_param_names)}):")
    for n in dist_param_names:
        logger.info(f"  {n}")


def set_frozen_bn_eval(model):
    """Set frozen BatchNorm / Dropout layers to eval mode.

    Call this AFTER model.train() each epoch so that:
      - model.training == True  →  YOLOX.forward() computes losses
      - frozen BN running_mean/running_var are NOT updated
      - frozen Dropout stays deterministic
    """
    frozen_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.SyncBatchNorm, nn.Dropout, nn.Dropout2d)
    for name, module in model.named_modules():
        if isinstance(module, frozen_types):
            # If none of this layer's own parameters are trainable, lock it
            has_trainable = any(p.requires_grad for p in module.parameters(recurse=False))
            if not has_trainable:
                module.eval()


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

    # ── Freeze everything except distance branch — this is permanent ──
    freeze_all_except_distance(model)
    model.to(args.device)

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

    # ── Optimizer (only distance params — the only trainable ones) ──
    dist_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(dist_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ──
    best_loss = float("inf")
    for epoch in range(args.epochs):
        # model.train() sets the whole model to training mode so that
        # YOLOX.forward() takes the loss-computation code path.
        # Then set_frozen_bn_eval() flips frozen BN/Dropout back to eval
        # so their running stats and behavior are preserved.
        model.train()
        set_frozen_bn_eval(model)

        epoch_loss = 0.0
        epoch_dist_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(imgs, targets)

            loss = outputs["total_loss"]
            dist_loss = outputs.get("dist_loss", 0.0)

            optimizer.zero_grad()
            loss.backward()

            # Safety check: verify frozen params got no gradients
            # (only on first batch of first epoch for efficiency)
            if epoch == 0 and batch_idx == 0:
                for name, param in model.named_parameters():
                    if not param.requires_grad and param.grad is not None:
                        logger.warning(f"Frozen param {name} unexpectedly has a gradient!")

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