#!/usr/bin/env python3
"""
Train YOLOX with distance head — supports multiple fine-tuning strategies.

Unlike train_distance.py (which permanently freezes everything except the
distance branch), this script lets you choose HOW MUCH of the network to train:

  Mode 1: "distance_only"   — Same as train_distance.py: only distance branch
  Mode 2: "head_only"       — Distance + detection head (cls, reg, obj branches)
  Mode 3: "head_and_neck"   — Distance + detection head + FPN/PAN neck
  Mode 4: "full"            — Entire network end-to-end
  Mode 5: "gradual"         — Start with distance_only, then progressively
                               unfreeze head → neck → backbone on a schedule

Typical usage:
    # Fine-tune the full detection head + distance branch
    python tools/train_distance_finetune.py \
        --ckpt weights/yolox_s.pth \
        --csv data/stopsigns/annotations.csv \
        --img-dir data/stopsigns/ \
        --mode head_only \
        --epochs 50 --lr 5e-4

    # Gradual unfreezing: distance first, then head, then full
    python tools/train_distance_finetune.py \
        --ckpt weights/yolox_s.pth \
        --csv data/stopsigns/annotations.csv \
        --img-dir data/stopsigns/ \
        --mode gradual \
        --epochs 100 \
        --unfreeze-head-epoch 10 \
        --unfreeze-neck-epoch 30 \
        --unfreeze-backbone-epoch 60 \
        --lr 1e-3
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.utils import load_ckpt
from yolox.data.datasets.stopsign_distance import StopSignDistanceDataset, collate_fn


# ─────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=80, depth=0.33, width=0.50):
    """Build YOLOX-s with distance head."""
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    return YOLOX(backbone, head)


# ─────────────────────────────────────────────────────────────────────
# Freezing / unfreezing utilities
# ─────────────────────────────────────────────────────────────────────

def _set_requires_grad(module, value: bool):
    """Set requires_grad for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = value


def freeze_everything(model):
    """Freeze the entire model."""
    _set_requires_grad(model, False)


def unfreeze_distance_branch(model):
    """Unfreeze only the distance conv + prediction layers in the head."""
    for name, param in model.named_parameters():
        if "dist_" in name:
            param.requires_grad = True


def unfreeze_detection_head(model):
    """Unfreeze the full detection head (cls, reg, obj, dist, stems)."""
    _set_requires_grad(model.head, True)


def unfreeze_neck(model):
    """
    Unfreeze the FPN/PAN neck layers in the backbone module.
    In YOLOPAFPN, the neck consists of lateral_conv0, C3_p4, reduce_conv1,
    C3_p3, bu_conv2, C3_n3, bu_conv1, C3_n4, and the upsample layer.
    Everything except model.backbone.backbone (the CSPDarknet) is neck.
    """
    # Unfreeze everything in YOLOPAFPN except the CSPDarknet backbone
    for name, param in model.backbone.named_parameters():
        if not name.startswith("backbone."):
            # These are FPN/PAN neck parameters
            param.requires_grad = True


def unfreeze_backbone(model):
    """Unfreeze the CSPDarknet backbone (inside YOLOPAFPN)."""
    _set_requires_grad(model.backbone.backbone, True)


def unfreeze_full_model(model):
    """Unfreeze everything."""
    _set_requires_grad(model, True)


def apply_freeze_mode(model, mode):
    """
    Apply a freezing strategy based on mode string.

    Starting from a fully frozen model, selectively unfreeze parts.
    """
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
        # Gradual starts like distance_only; unfreezing happens during training
        unfreeze_distance_branch(model)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def log_trainable_status(model):
    """Log which parameter groups are trainable."""
    trainable = 0
    frozen = 0
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
    Set frozen BatchNorm and Dropout layers to eval mode.

    Call AFTER model.train() each epoch so that:
      - model.training == True → YOLOX.forward() computes losses
      - frozen BN running_mean/var are NOT updated
      - frozen Dropout stays deterministic
    """
    frozen_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.Dropout, nn.Dropout2d,
    )
    for name, module in model.named_modules():
        if isinstance(module, frozen_types):
            has_trainable = any(
                p.requires_grad for p in module.parameters(recurse=False)
            )
            if not has_trainable:
                module.eval()


# ─────────────────────────────────────────────────────────────────────
# Optimizer builder — supports differential learning rates
# ─────────────────────────────────────────────────────────────────────

def build_optimizer(model, base_lr, mode, backbone_lr_scale=0.1, neck_lr_scale=0.5):
    """
    Build optimizer with per-group learning rates.

    - Distance branch:  base_lr
    - Detection head:   base_lr
    - Neck (FPN/PAN):   base_lr * neck_lr_scale
    - Backbone:         base_lr * backbone_lr_scale

    Lower LR for pretrained layers prevents catastrophic forgetting.
    """
    param_groups = []

    # Group 1: Distance branch (always highest LR)
    dist_params = []
    # Group 2: Detection head (excluding distance)
    head_params = []
    # Group 3: Neck (FPN/PAN)
    neck_params = []
    # Group 4: Backbone (CSPDarknet)
    backbone_params = []

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
            # Catch-all (shouldn't happen, but just in case)
            head_params.append(param)

    if dist_params:
        param_groups.append({
            "params": dist_params,
            "lr": base_lr,
            "name": "distance_branch",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": base_lr,
            "name": "detection_head",
        })
    if neck_params:
        param_groups.append({
            "params": neck_params,
            "lr": base_lr * neck_lr_scale,
            "name": "neck",
        })
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": base_lr * backbone_lr_scale,
            "name": "backbone",
        })

    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        logger.info(f"  Optimizer group '{pg['name']}': {n_params:,} params, lr={pg['lr']:.2e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    return optimizer


# ─────────────────────────────────────────────────────────────────────
# Gradual unfreezing logic
# ─────────────────────────────────────────────────────────────────────

def maybe_unfreeze_gradual(model, optimizer, epoch, args):
    """
    For mode='gradual', check if we should unfreeze more layers at this epoch.
    When new layers are unfrozen, add them to the optimizer.

    Returns True if the optimizer was modified (caller should rebuild scheduler).
    """
    changed = False

    if epoch == args.unfreeze_head_epoch:
        logger.info(f"=== Epoch {epoch + 1}: Unfreezing detection head ===")
        unfreeze_detection_head(model)
        _add_new_params_to_optimizer(
            optimizer, model, args.lr, group_name="detection_head"
        )
        changed = True

    if epoch == args.unfreeze_neck_epoch:
        logger.info(f"=== Epoch {epoch + 1}: Unfreezing FPN/PAN neck ===")
        unfreeze_neck(model)
        _add_new_params_to_optimizer(
            optimizer, model, args.lr * 0.5, group_name="neck"
        )
        changed = True

    if epoch == args.unfreeze_backbone_epoch:
        logger.info(f"=== Epoch {epoch + 1}: Unfreezing backbone ===")
        unfreeze_backbone(model)
        _add_new_params_to_optimizer(
            optimizer, model, args.lr * 0.1, group_name="backbone"
        )
        changed = True

    if changed:
        log_trainable_status(model)

    return changed


def _add_new_params_to_optimizer(optimizer, model, lr, group_name="new"):
    """
    Find parameters that are requires_grad=True but NOT already tracked
    by the optimizer, and add them as a new param group.
    """
    # Collect all param ids already in optimizer
    existing_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            existing_ids.add(id(p))

    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in existing_ids:
            new_params.append(param)

    if new_params:
        n_new = sum(p.numel() for p in new_params)
        logger.info(f"  Adding {n_new:,} new params to optimizer (group='{group_name}', lr={lr:.2e})")
        optimizer.add_param_group({
            "params": new_params,
            "lr": lr,
            "name": group_name,
        })
    else:
        logger.info(f"  No new params to add for group '{group_name}'")


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, device):
    """
    Run one pass over the validation set and return average distance loss.
    The model is set to train mode (so YOLOX.forward computes losses),
    but no gradients are computed.

    All BN layers are forced to eval mode so that validation data does not
    corrupt running_mean / running_var (torch.no_grad() does NOT prevent this).
    """
    model.train()

    # Save each BN module's training flag, then set all to eval so that
    # running statistics are never updated with validation data.
    bn_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
    )
    bn_states = {}
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            bn_states[name] = module.training
            module.eval()

    total_dist_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
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

    avg_loss = total_loss / max(n_batches, 1)
    avg_dist = total_dist_loss / max(n_batches, 1)
    return avg_loss, avg_dist


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOX + distance head with flexible fine-tuning"
    )

    # Required
    parser.add_argument("--ckpt", required=True, help="Pretrained YOLOX checkpoint")
    parser.add_argument("--csv", required=True, help="CSV with distance annotations")
    parser.add_argument("--img-dir", required=True, help="Root dir for images")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output-dir", default="runs/distance_finetune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)

    # Fine-tuning mode
    parser.add_argument(
        "--mode",
        choices=["distance_only", "head_only", "head_and_neck", "full", "gradual"],
        default="head_only",
        help="Which parts of the network to train",
    )

    # Differential learning rate scales (for pretrained layers)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="LR multiplier for backbone (default: 0.1x base LR)")
    parser.add_argument("--neck-lr-scale", type=float, default=0.5,
                        help="LR multiplier for FPN/PAN neck (default: 0.5x base LR)")

    # Gradual unfreezing schedule (only used with --mode gradual)
    parser.add_argument("--unfreeze-head-epoch", type=int, default=10,
                        help="Epoch to unfreeze detection head (gradual mode)")
    parser.add_argument("--unfreeze-neck-epoch", type=int, default=30,
                        help="Epoch to unfreeze neck (gradual mode)")
    parser.add_argument("--unfreeze-backbone-epoch", type=int, default=60,
                        help="Epoch to unfreeze backbone (gradual mode)")

    # Validation
    parser.add_argument("--val-csv", default=None,
                        help="Optional validation CSV (same format as --csv)")
    parser.add_argument("--val-interval", type=int, default=5,
                        help="Run validation every N epochs")

    # Model variant
    parser.add_argument("--depth", type=float, default=0.33, help="Model depth factor")
    parser.add_argument("--width", type=float, default=0.50, help="Model width factor")

    # Loss weighting
    parser.add_argument("--dist-weight", type=float, default=1.0,
                        help="Weight for distance loss (model.head.dist_weight)")
    parser.add_argument("--mean-dist", type=float, default=12.0,
                        help="Approximate mean distance (m) of your dataset, used to "
                             "initialise the distance head bias via inverse-softplus")
    parser.add_argument("--freeze-det-loss", action="store_true",
                        help="Even in modes that unfreeze the head, only backprop "
                             "through the distance loss (experimental)")

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
    logger.info("Loaded pretrained weights (distance layers initialized randomly)")

    # ── Initialize distance bias to sensible value ───────────────────
    inverse_softplus = math.log(math.exp(args.mean_dist) - 1.0)
    with torch.no_grad():
        for pred_layer in model.head.dist_preds:
            pred_layer.bias.fill_(inverse_softplus)
    logger.info(
        f"Initialized distance bias to inverse_softplus({args.mean_dist}) = {inverse_softplus:.2f}"
    )

    # ── Set distance loss weight ─────────────────────────────────────
    model.head.dist_weight = args.dist_weight

    # ── Apply freezing strategy ──────────────────────────────────────
    logger.info(f"Applying freeze mode: '{args.mode}'")
    apply_freeze_mode(model, args.mode)
    log_trainable_status(model)
    model.to(args.device)

    # ── Dataset ──────────────────────────────────────────────────────
    train_dataset = StopSignDistanceDataset(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_size=(args.img_size, args.img_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"Train dataset: {len(train_dataset)} images, {len(train_loader)} batches")

    val_loader = None
    if args.val_csv:
        val_dataset = StopSignDistanceDataset(
            csv_path=args.val_csv,
            img_dir=args.img_dir,
            img_size=(args.img_size, args.img_size),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        logger.info(f"Val dataset: {len(val_dataset)} images, {len(val_loader)} batches")

    # ── Optimizer ────────────────────────────────────────────────────
    if args.mode in ("distance_only", "gradual"):
        # Simple optimizer — only trainable params at this point
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    else:
        # Differential LR optimizer
        optimizer = build_optimizer(
            model, args.lr, args.mode,
            backbone_lr_scale=args.backbone_lr_scale,
            neck_lr_scale=args.neck_lr_scale,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ────────────────────────────────────────────────
    best_loss = float("inf")
    best_dist_loss = float("inf")

    for epoch in range(args.epochs):
        # Gradual unfreezing: check if we need to unfreeze more
        if args.mode == "gradual":
            changed = maybe_unfreeze_gradual(model, optimizer, epoch, args)
            if changed:
                # Rebuild scheduler to account for new param groups
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs - epoch,
                    eta_min=args.lr * 0.01,
                )

        # Set model to train mode, then fix frozen BN/Dropout
        model.train()
        set_frozen_modules_eval(model)

        epoch_loss = 0.0
        epoch_dist_loss = 0.0
        epoch_iou_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_conf_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(imgs, targets)

            if args.freeze_det_loss:
                # Experimental: only backprop through distance loss
                loss = outputs.get("dist_loss", outputs["total_loss"])
                if isinstance(loss, (int, float)):
                    loss = outputs["total_loss"]
            else:
                loss = outputs["total_loss"]

            dist_loss = outputs.get("dist_loss", 0.0)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients when
            # training many layers
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=10.0,
            )

            optimizer.step()

            epoch_loss += loss.item()
            if isinstance(dist_loss, torch.Tensor):
                epoch_dist_loss += dist_loss.item()
            epoch_iou_loss += outputs.get("iou_loss", torch.tensor(0.0)).item() \
                if isinstance(outputs.get("iou_loss", 0), torch.Tensor) else 0
            epoch_cls_loss += outputs.get("cls_loss", torch.tensor(0.0)).item() \
                if isinstance(outputs.get("cls_loss", 0), torch.Tensor) else 0
            epoch_conf_loss += outputs.get("conf_loss", torch.tensor(0.0)).item() \
                if isinstance(outputs.get("conf_loss", 0), torch.Tensor) else 0

        scheduler.step()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_dist = epoch_dist_loss / n_batches
        avg_iou = epoch_iou_loss / n_batches
        avg_cls = epoch_cls_loss / n_batches
        avg_conf = epoch_conf_loss / n_batches
        elapsed = time.time() - t0

        # Collect all LRs
        lr_strs = []
        for pg in optimizer.param_groups:
            name = pg.get("name", "default")
            lr_strs.append(f"{name}={pg['lr']:.2e}")
        lr_info = ", ".join(lr_strs)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss: {avg_loss:.4f} | dist: {avg_dist:.4f} | "
            f"iou: {avg_iou:.4f} | cls: {avg_cls:.4f} | conf: {avg_conf:.4f} | "
            f"lr: [{lr_info}] | time: {elapsed:.1f}s"
        )

        # ── Validation ───────────────────────────────────────────────
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_loss, val_dist = validate(model, val_loader, args.device)
            logger.info(
                f"  [VAL] loss: {val_loss:.4f} | dist_loss: {val_dist:.4f}"
            )
            # Use val dist loss for best model selection when available
            track_dist = val_dist
        else:
            track_dist = avg_dist

        # ── Save best (by distance loss) ─────────────────────────────
        if track_dist < best_dist_loss and track_dist > 0:
            best_dist_loss = track_dist
            save_path = os.path.join(args.output_dir, "best_dist_ckpt.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "dist_loss": best_dist_loss,
                "mode": args.mode,
            }, save_path)
            logger.info(f"  → Saved best distance checkpoint (dist_loss={best_dist_loss:.4f})")

        # ── Save best (by total loss) ────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "best_ckpt.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "total_loss": best_loss,
                "mode": args.mode,
            }, save_path)
            logger.info(f"  → Saved best total checkpoint (loss={best_loss:.4f})")

        # ── Save latest ──────────────────────────────────────────────
        save_path = os.path.join(args.output_dir, "latest_ckpt.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_dist_loss": best_dist_loss,
            "best_loss": best_loss,
            "mode": args.mode,
        }, save_path)

    logger.info(
        f"Training complete! Best dist_loss: {best_dist_loss:.4f}, "
        f"Best total_loss: {best_loss:.4f}"
    )


if __name__ == "__main__":
    main()