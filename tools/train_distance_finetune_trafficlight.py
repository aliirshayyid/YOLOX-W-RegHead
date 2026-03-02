#!/usr/bin/env python3
"""
Train YOLOX with distance head for traffic light detection.

Mirrors train_distance_finetune.py (stop sign) but targets traffic lights
(COCO class 9).  All five fine-tuning modes are supported:

  Mode 1: "distance_only"   — Only the distance branch (dist_conv / dist_pred layers)
  Mode 2: "head_only"       — Distance + full detection head (cls, reg, obj)
  Mode 3: "head_and_neck"   — Distance + head + FPN/PAN neck
  Mode 4: "full"            — Entire network end-to-end
  Mode 5: "gradual"         — Progressive unfreezing on a user-defined schedule

Typical usage:
    # Fine-tune detection head + distance branch
    python tools/train_distance_finetune_traffic.py \\
        --ckpt weights/yolox_s.pth \\
        --csv data/traffic_lights/annotations.csv \\
        --img-dir data/traffic_lights/ \\
        --mode head_only \\
        --epochs 50 --lr 5e-4

    # Gradual unfreezing
    python tools/train_distance_finetune_traffic.py \\
        --ckpt weights/yolox_s.pth \\
        --csv data/traffic_lights/annotations.csv \\
        --img-dir data/traffic_lights/ \\
        --mode gradual \\
        --unfreeze-head-epoch 10 \\
        --unfreeze-neck-epoch 30 \\
        --unfreeze-backbone-epoch 60 \\
        --epochs 100 --lr 1e-3

CSV format (required columns):
    image_path, x_min, y_min, x_max, y_max, distance_m

CSV format (optional extra column):
    state   →  "red" | "yellow" | "green"  (logged but not used in loss)
"""

import argparse
import math
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
from yolox.data.datasets.traffic_light_distance import (
    TrafficLightDistanceDataset,
    collate_fn,
)


# ─────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=80, depth=0.33, width=0.50):
    """Build YOLOX-s architecture (with distance head)."""
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    return YOLOX(backbone, head)


# ─────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ─────────────────────────────────────────────────────────────────────

def _set_requires_grad(module, value: bool):
    for param in module.parameters():
        param.requires_grad = value


def freeze_everything(model):
    _set_requires_grad(model, False)


def unfreeze_distance_branch(model):
    """Unfreeze only layers whose name contains 'dist_'."""
    for name, param in model.named_parameters():
        if "dist_" in name:
            param.requires_grad = True


def unfreeze_detection_head(model):
    """Unfreeze the entire detection head (cls, reg, obj, dist, stems)."""
    _set_requires_grad(model.head, True)


def unfreeze_neck(model):
    """
    Unfreeze FPN/PAN neck layers inside YOLOPAFPN.
    Everything in backbone module EXCEPT the inner CSPDarknet backbone
    (i.e. parameters whose names do NOT start with 'backbone.').
    """
    for name, param in model.backbone.named_parameters():
        if not name.startswith("backbone."):
            param.requires_grad = True


def unfreeze_backbone(model):
    """Unfreeze the CSPDarknet backbone inside YOLOPAFPN."""
    _set_requires_grad(model.backbone.backbone, True)


def unfreeze_full_model(model):
    _set_requires_grad(model, True)


def apply_freeze_mode(model, mode):
    """Freeze everything first, then selectively unfreeze."""
    freeze_everything(model)

    if mode == "distance_only":
        unfreeze_distance_branch(model)
    elif mode == "head_only":
        unfreeze_detection_head(model)
    elif mode == "head_and_neck":
        unfreeze_detection_head(model)
        unfreeze_neck(model)
    elif mode == "full":
        unfreeze_full_model(model)
    elif mode == "gradual":
        unfreeze_distance_branch(model)   # starts like distance_only
    else:
        raise ValueError(f"Unknown mode: '{mode}'")


def log_trainable_status(model):
    trainable, frozen = 0, 0
    trainable_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)
        else:
            frozen += param.numel()
    total = trainable + frozen
    logger.info(
        f"Trainable: {trainable:,} / {total:,} params "
        f"({100 * trainable / total:.2f}%)"
    )
    if len(trainable_names) <= 30:
        for n in trainable_names:
            logger.info(f"  [trainable] {n}")
    else:
        logger.info(f"  {len(trainable_names)} trainable layers (too many to list)")


def set_frozen_modules_eval(model):
    """
    After model.train(), set frozen BN / Dropout layers back to eval.
    Prevents frozen running statistics from being corrupted by training data.
    """
    frozen_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.Dropout, nn.Dropout2d,
    )
    for _, module in model.named_modules():
        if isinstance(module, frozen_types):
            has_trainable = any(
                p.requires_grad for p in module.parameters(recurse=False)
            )
            if not has_trainable:
                module.eval()


# ─────────────────────────────────────────────────────────────────────
# Differential learning-rate optimizer
# ─────────────────────────────────────────────────────────────────────

def build_optimizer(model, base_lr, backbone_lr_scale=0.1, neck_lr_scale=0.5):
    """
    Create per-group learning rates:
      distance branch  →  base_lr
      detection head   →  base_lr
      FPN/PAN neck     →  base_lr * neck_lr_scale   (default 0.5×)
      CSPDarknet       →  base_lr * backbone_lr_scale (default 0.1×)
    """
    dist_params, head_params, neck_params, backbone_params = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "dist_" in name:
            dist_params.append(param)
        elif name.startswith("head."):
            head_params.append(param)
        elif name.startswith("backbone.backbone."):
            backbone_params.append(param)
        elif name.startswith("backbone."):
            neck_params.append(param)
        else:
            head_params.append(param)   # safe catch-all

    param_groups = []
    for params, name, lr in [
        (dist_params,     "distance_branch", base_lr),
        (head_params,     "detection_head",  base_lr),
        (neck_params,     "neck",            base_lr * neck_lr_scale),
        (backbone_params, "backbone",        base_lr * backbone_lr_scale),
    ]:
        if params:
            n = sum(p.numel() for p in params)
            logger.info(f"  Optimizer group '{name}': {n:,} params, lr={lr:.2e}")
            param_groups.append({"params": params, "lr": lr, "name": name})

    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


# ─────────────────────────────────────────────────────────────────────
# Gradual-unfreezing helpers
# ─────────────────────────────────────────────────────────────────────

def _add_new_params_to_optimizer(optimizer, model, lr, group_name="new"):
    existing_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    new_params = [
        p for _, p in model.named_parameters()
        if p.requires_grad and id(p) not in existing_ids
    ]
    if new_params:
        n_new = sum(p.numel() for p in new_params)
        logger.info(
            f"  Adding {n_new:,} new params → group='{group_name}', lr={lr:.2e}"
        )
        optimizer.add_param_group({"params": new_params, "lr": lr, "name": group_name})
    else:
        logger.info(f"  No new params for group '{group_name}'")


def maybe_unfreeze_gradual(model, optimizer, epoch, args):
    """
    Called once per epoch.  Returns True if the optimizer was modified
    (caller should rebuild the LR scheduler).
    """
    changed = False

    if epoch == args.unfreeze_head_epoch:
        logger.info(f"=== Epoch {epoch + 1}: Unfreezing detection head ===")
        unfreeze_detection_head(model)
        _add_new_params_to_optimizer(optimizer, model, args.lr, "detection_head")
        changed = True

    if epoch == args.unfreeze_neck_epoch:
        logger.info(f"=== Epoch {epoch + 1}: Unfreezing FPN/PAN neck ===")
        unfreeze_neck(model)
        _add_new_params_to_optimizer(optimizer, model, args.lr * 0.5, "neck")
        changed = True

    if epoch == args.unfreeze_backbone_epoch:
        logger.info(f"=== Epoch {epoch + 1}: Unfreezing backbone ===")
        unfreeze_backbone(model)
        _add_new_params_to_optimizer(optimizer, model, args.lr * 0.1, "backbone")
        changed = True

    if changed:
        log_trainable_status(model)

    return changed


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, device):
    """
    Evaluate on validation set.  Model stays in train() mode so that
    YOLOX.forward() computes losses, but ALL BatchNorm layers are switched
    to eval so that running statistics are never updated with val data.
    """
    model.train()

    bn_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
    )
    bn_states = {}
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            bn_states[name] = module.training
            module.eval()

    total_loss, total_dist_loss, n_batches = 0.0, 0.0, 0

    for imgs, targets in dataloader:
        imgs    = imgs.to(device)
        targets = targets.to(device)

        # Drop optional state column (col 6) if present — model expects N×6
        if targets.shape[-1] == 7:
            targets = targets[..., :6]

        outputs = model(imgs, targets)
        total_loss += outputs["total_loss"].item()
        dist_loss = outputs.get("dist_loss", 0.0)
        if isinstance(dist_loss, torch.Tensor):
            total_dist_loss += dist_loss.item()
        n_batches += 1

    # Restore BN training flags
    for name, module in model.named_modules():
        if isinstance(module, bn_types) and name in bn_states:
            module.train(bn_states[name])

    n = max(n_batches, 1)
    return total_loss / n, total_dist_loss / n


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOX + distance head for traffic light detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────
    parser.add_argument("--ckpt",    required=True, help="Pretrained YOLOX checkpoint (.pth)")
    parser.add_argument("--csv",     required=True, help="CSV with training annotations")
    parser.add_argument("--img-dir", required=True, help="Root directory for images")

    # ── Training ──────────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--img-size",    type=int,   default=640)
    parser.add_argument("--output-dir",  default="runs/traffic_light_finetune")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int,   default=2)

    # ── Fine-tuning mode ──────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["distance_only", "head_only", "head_and_neck", "full", "gradual"],
        default="head_only",
        help=(
            "Which network parts to train:\n"
            "  distance_only  — only the distance branch (safest)\n"
            "  head_only      — distance + detection head\n"
            "  head_and_neck  — distance + head + FPN/PAN neck\n"
            "  full           — entire network\n"
            "  gradual        — progressive unfreezing on a schedule"
        ),
    )

    # ── Differential LR ───────────────────────────────────────────────
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="LR multiplier for backbone (default 0.1× base LR)")
    parser.add_argument("--neck-lr-scale",     type=float, default=0.5,
                        help="LR multiplier for FPN/PAN neck (default 0.5× base LR)")

    # ── Gradual schedule ──────────────────────────────────────────────
    parser.add_argument("--unfreeze-head-epoch",     type=int, default=10)
    parser.add_argument("--unfreeze-neck-epoch",     type=int, default=30)
    parser.add_argument("--unfreeze-backbone-epoch", type=int, default=60)

    # ── Validation ────────────────────────────────────────────────────
    parser.add_argument("--val-csv",      default=None,
                        help="Optional validation CSV (same format as --csv)")
    parser.add_argument("--val-interval", type=int, default=5,
                        help="Run validation every N epochs")

    # ── Model variant ─────────────────────────────────────────────────
    parser.add_argument("--depth", type=float, default=0.33, help="Depth scaling factor")
    parser.add_argument("--width", type=float, default=0.50, help="Width scaling factor")

    # ── Loss / initialization ─────────────────────────────────────────
    parser.add_argument("--dist-weight", type=float, default=1.0,
                        help="Weight applied to distance loss during training")
    parser.add_argument("--mean-dist",   type=float, default=15.0,
                        help="Approximate mean distance (m) in your dataset — used "
                             "to initialise the distance head bias via inverse-softplus. "
                             "Traffic lights are typically farther away than stop signs.")
    parser.add_argument("--freeze-det-loss", action="store_true",
                        help="Backprop only through distance loss (experimental)")

    # ── State column ──────────────────────────────────────────────────
    parser.add_argument("--include-state", action="store_true",
                        help="If set, the 'state' column in the CSV is loaded and "
                             "passed through (col 6 of label tensor).  The state is "
                             "NOT used in the loss — it is available for downstream "
                             "analysis or future state-classification experiments.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Config: {vars(args)}")

    # ── Build model ──────────────────────────────────────────────────
    model = build_model(num_classes=80, depth=args.depth, width=args.width)

    # ── Load pretrained weights ──────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model = load_ckpt(model, ckpt)
    logger.info("Loaded pretrained weights (distance layers initialised randomly)")

    # ── Initialise distance-head bias to dataset mean ────────────────
    # Traffic lights are typically ~10–30 m away; default mean_dist is 15 m.
    inverse_softplus = math.log(math.exp(args.mean_dist) - 1.0)
    with torch.no_grad():
        for pred_layer in model.head.dist_preds:
            pred_layer.bias.fill_(inverse_softplus)
    logger.info(
        f"Initialised distance bias → inverse_softplus({args.mean_dist}) "
        f"= {inverse_softplus:.4f}"
    )

    # ── Distance loss weight ─────────────────────────────────────────
    model.head.dist_weight = args.dist_weight

    # ── Freeze strategy ──────────────────────────────────────────────
    logger.info(f"Applying freeze mode: '{args.mode}'")
    apply_freeze_mode(model, args.mode)
    log_trainable_status(model)
    model.to(args.device)

    # ── Datasets ─────────────────────────────────────────────────────
    train_dataset = TrafficLightDistanceDataset(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_size=(args.img_size, args.img_size),
        include_state=args.include_state,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(args.device != "cpu"),
    )
    logger.info(
        f"Train dataset: {len(train_dataset)} images, "
        f"{len(train_loader)} batches/epoch"
    )

    val_loader = None
    if args.val_csv:
        val_dataset = TrafficLightDistanceDataset(
            csv_path=args.val_csv,
            img_dir=args.img_dir,
            img_size=(args.img_size, args.img_size),
            include_state=args.include_state,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=(args.device != "cpu"),
        )
        logger.info(
            f"Val dataset: {len(val_dataset)} images, "
            f"{len(val_loader)} batches"
        )

    # ── Optimizer ────────────────────────────────────────────────────
    if args.mode in ("distance_only", "gradual"):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = build_optimizer(
            model, args.lr,
            backbone_lr_scale=args.backbone_lr_scale,
            neck_lr_scale=args.neck_lr_scale,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Training loop ────────────────────────────────────────────────
    best_total_loss = float("inf")
    best_dist_loss  = float("inf")

    for epoch in range(args.epochs):

        # Gradual mode: check for scheduled unfreezing
        if args.mode == "gradual":
            changed = maybe_unfreeze_gradual(model, optimizer, epoch, args)
            if changed:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs - epoch,
                    eta_min=args.lr * 0.01,
                )

        model.train()
        set_frozen_modules_eval(model)

        epoch_loss = epoch_dist = epoch_iou = epoch_cls = epoch_conf = 0.0
        t0 = time.time()

        for imgs, targets in train_loader:
            imgs    = imgs.to(args.device)
            targets = targets.to(args.device)

            # Drop the optional state column before passing to the model
            # (model.forward expects N×6 labels: cls, cx, cy, w, h, dist)
            if targets.shape[-1] == 7:
                targets = targets[..., :6]

            outputs = model(imgs, targets)

            if args.freeze_det_loss:
                # Experimental: backprop only through distance loss
                loss = outputs.get("dist_loss", outputs["total_loss"])
                if isinstance(loss, (int, float)):
                    loss = outputs["total_loss"]
            else:
                loss = outputs["total_loss"]

            dist_loss = outputs.get("dist_loss", 0.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=10.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            if isinstance(dist_loss, torch.Tensor):
                epoch_dist += dist_loss.item()

            def _item(v):
                return v.item() if isinstance(v, torch.Tensor) else 0.0

            epoch_iou  += _item(outputs.get("iou_loss",  0))
            epoch_cls  += _item(outputs.get("cls_loss",  0))
            epoch_conf += _item(outputs.get("conf_loss", 0))

        scheduler.step()

        n = len(train_loader)
        avg_loss  = epoch_loss / n
        avg_dist  = epoch_dist / n
        avg_iou   = epoch_iou  / n
        avg_cls   = epoch_cls  / n
        avg_conf  = epoch_conf / n
        elapsed   = time.time() - t0

        lr_info = ", ".join(
            f"{pg.get('name', 'default')}={pg['lr']:.2e}"
            for pg in optimizer.param_groups
        )

        logger.info(
            f"Epoch {epoch + 1:>4}/{args.epochs} | "
            f"loss: {avg_loss:.4f} | dist: {avg_dist:.4f} | "
            f"iou: {avg_iou:.4f} | cls: {avg_cls:.4f} | conf: {avg_conf:.4f} | "
            f"lr: [{lr_info}] | {elapsed:.1f}s"
        )

        # ── Validation ───────────────────────────────────────────────
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_loss, val_dist = validate(model, val_loader, args.device)
            logger.info(
                f"  [VAL] loss: {val_loss:.4f} | dist_loss: {val_dist:.4f}"
            )
            track_dist = val_dist
        else:
            track_dist = avg_dist

        # ── Save best (distance loss) ─────────────────────────────────
        if 0 < track_dist < best_dist_loss:
            best_dist_loss = track_dist
            save_path = os.path.join(args.output_dir, "best_dist_ckpt.pth")
            torch.save({
                "model":     model.state_dict(),
                "epoch":     epoch,
                "dist_loss": best_dist_loss,
                "mode":      args.mode,
            }, save_path)
            logger.info(
                f"  → Saved best distance ckpt (dist_loss={best_dist_loss:.4f})"
            )

        # ── Save best (total loss) ────────────────────────────────────
        if avg_loss < best_total_loss:
            best_total_loss = avg_loss
            save_path = os.path.join(args.output_dir, "best_ckpt.pth")
            torch.save({
                "model":      model.state_dict(),
                "epoch":      epoch,
                "total_loss": best_total_loss,
                "mode":       args.mode,
            }, save_path)
            logger.info(
                f"  → Saved best total ckpt (loss={best_total_loss:.4f})"
            )

        # ── Save latest (with full training state) ────────────────────
        save_path = os.path.join(args.output_dir, "latest_ckpt.pth")
        torch.save({
            "model":           model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "epoch":           epoch,
            "best_dist_loss":  best_dist_loss,
            "best_total_loss": best_total_loss,
            "mode":            args.mode,
        }, save_path)

    logger.info(
        f"Training complete! "
        f"Best dist_loss: {best_dist_loss:.4f} | "
        f"Best total_loss: {best_total_loss:.4f}"
    )


if __name__ == "__main__":
    main()