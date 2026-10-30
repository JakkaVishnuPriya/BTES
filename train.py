# -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Training Script
Bhuvan Basemap (approx. 3m resolution), 256x256 tiles
UNet / UNet++ with resnet34 encoder (offline weights)

@author: saurav_kumar
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import TreeDataset
from model import build_model
from loss import get_loss
from metrics import calculate_metrics

try:
    import monitor
    _MONITOR_AVAILABLE = True
except Exception:
    _MONITOR_AVAILABLE = False


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # SEED
    # =====================================================

    random.seed(CFG.SEED)

    np.random.seed(CFG.SEED)

    torch.manual_seed(CFG.SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(CFG.SEED)

    # =====================================================
    # LIVE MONITOR
    # =====================================================

    if CFG.ENABLE_MONITOR and _MONITOR_AVAILABLE:

        monitor.start_monitor()

    # =====================================================
    # DATASET
    # =====================================================
    # Images and masks are stored in ONE pair of folders.
    # TreeDataset receives the image/mask folders plus the
    # deterministic split information from config.py.
    #
    # No separate train/val/test directories are required.
    # The split is made from CFG.IMAGE_DIR / CFG.MASK_DIR.

    train_dataset = TreeDataset(
        CFG.IMAGE_DIR,
        CFG.MASK_DIR,
        split="train",
        train_ratio=CFG.TRAIN_RATIO,
        val_ratio=CFG.VAL_RATIO,
        test_ratio=CFG.TEST_RATIO,
        seed=CFG.SPLIT_SEED,
        augment=CFG.AUGMENT
    )

    val_dataset = TreeDataset(
        CFG.IMAGE_DIR,
        CFG.MASK_DIR,
        split="val",
        train_ratio=CFG.TRAIN_RATIO,
        val_ratio=CFG.VAL_RATIO,
        test_ratio=CFG.TEST_RATIO,
        seed=CFG.SPLIT_SEED,
        augment=False
    )

    # =====================================================
    # DATALOADER
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY
    )

    print("\n========================================")
    print("DATASET")
    print("========================================")

    print("Model :", CFG.MODEL_NAME)

    print("Images:", CFG.IMAGE_DIR)
    print("Masks :", CFG.MASK_DIR)
    print(
        "Split : "
        f"{CFG.TRAIN_RATIO:.0%} train / "
        f"{CFG.VAL_RATIO:.0%} val / "
        f"{CFG.TEST_RATIO:.0%} test"
    )
    print("Train :", len(train_dataset))
    print("Val   :", len(val_dataset))

    # =====================================================
    # MODEL
    # =====================================================

    model = build_model()

    model = model.to(CFG.DEVICE)

    # =====================================================
    # RESUME / WARM-START FULL MODEL (OPTIONAL)
    # =====================================================

    if CFG.PRETRAIN_DIR is not None and os.path.exists(CFG.PRETRAIN_DIR):

        try:

            checkpoint = torch.load(
                CFG.PRETRAIN_DIR,
                map_location=CFG.DEVICE
            )

            # Support both a plain state_dict and a full checkpoint.
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(
                state_dict,
                strict=False
            )

            print("\n========================================")
            print("PRETRAINED WEIGHTS LOADED")
            print("========================================")

            print(CFG.PRETRAIN_DIR)

        except Exception as e:

            print("\n========================================")
            print("FAILED TO LOAD PRETRAINED WEIGHTS")
            print("========================================")

            print(e)

            print("\nTraining From Scratch")

    else:

        print("\nNo full-model checkpoint set (CFG.PRETRAIN_DIR).")

        print("Training From Scratch (encoder initialised from local ResNet34 weights).")

    # =====================================================
    # LOSS
    # =====================================================

    loss_fn = get_loss()

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.LR
    )

    # =====================================================
    # LR SCHEDULER
    # =====================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # =====================================================
    # HISTORY
    # =====================================================

    train_losses = []

    val_losses = []

    history_acc = []

    history_precision = []

    history_recall = []

    history_f1 = []

    history_iou = []

    best_iou = -1

    # =====================================================
    # EARLY STOPPING
    # =====================================================
    
    epochs_without_improvement = 0

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(CFG.EPOCHS):

        start = time.time()

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        train_loss = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{CFG.EPOCHS}"
        )

        for imgs, masks in train_bar:

            imgs = imgs.to(CFG.DEVICE)

            masks = masks.to(CFG.DEVICE)

            # =============================================
            # ZERO GRAD
            # =============================================

            optimizer.zero_grad()

            # =============================================
            # FORWARD
            # =============================================

            outputs = model(imgs)

            # =============================================
            # LOSS
            # =============================================

            loss = loss_fn(outputs, masks)

            # =============================================
            # BACKWARD
            # =============================================

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= len(train_loader)

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_loss = 0

        all_preds = []

        all_masks = []

        with torch.no_grad():

            for imgs, masks in val_loader:

                imgs = imgs.to(CFG.DEVICE)

                masks = masks.to(CFG.DEVICE)

                outputs = model(imgs)

                loss = loss_fn(outputs, masks)

                val_loss += loss.item()

                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())

                all_masks.append(masks.cpu().numpy())

        val_loss /= len(val_loader)

        # =================================================
        # CONCAT
        # =================================================

        all_preds = np.concatenate(all_preds)

        all_masks = np.concatenate(all_masks)

        # =================================================
        # METRICS
        # =================================================

        acc, precision, recall, f1, iou = calculate_metrics(
            all_preds,
            all_masks,
            threshold=CFG.THRESHOLD
        )

        # =================================================
        # LR SCHEDULER
        # =================================================

        scheduler.step(val_loss)

        # =================================================
        # SAVE HISTORY
        # =================================================

        train_losses.append(train_loss)

        val_losses.append(val_loss)

        history_acc.append(acc)

        history_precision.append(precision)

        history_recall.append(recall)

        history_f1.append(f1)

        history_iou.append(iou)

        # =================================================
        # LIVE MONITOR UPDATE
        # =================================================

        if CFG.ENABLE_MONITOR and _MONITOR_AVAILABLE:

            monitor.update_metrics(
                epoch + 1,
                train_loss,
                val_loss,
                acc,
                precision,
                recall,
                f1,
                iou,
                optimizer.param_groups[0]['lr']
            )

        # =================================================
        # SAVE BEST MODEL
        # =================================================

        # =====================================================
        # SAVE BEST MODEL + EARLY STOPPING
        # =====================================================
        
        if iou > (best_iou + CFG.MIN_DELTA):
        
            best_iou = iou
        
            epochs_without_improvement = 0
        
            torch.save(
                model.state_dict(),
                CFG.BEST_MODEL
            )
        
            print("\n========================================")
            print("BEST MODEL SAVED")
            print("========================================")
            print(f"Best IoU : {best_iou:.4f}")
        
        else:
        
            epochs_without_improvement += 1
        
            print(
                f"\nNo IoU improvement for "
                f"{epochs_without_improvement}/{CFG.PATIENCE} epochs."
            )

        end = time.time()

        # =================================================
        # LOG
        # =================================================

        print("\n========================================")

        print(f"Epoch {epoch+1}/{CFG.EPOCHS}")

        print("========================================")

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Accuracy   : {acc:.4f}")

        print(f"Precision  : {precision:.4f}")

        print(f"Recall     : {recall:.4f}")

        print(f"F1 Score   : {f1:.4f}")

        print(f"IoU        : {iou:.4f}")

        print(f"LR         : {optimizer.param_groups[0]['lr']:.8f}")

        print(f"Time       : {(end-start):.2f} sec")

        # =====================================================
        # EARLY STOPPING
        # =====================================================
        
        if (
            CFG.EARLY_STOPPING
            and epochs_without_improvement >= CFG.PATIENCE
        ):
        
            print("\n========================================")
            print("EARLY STOPPING TRIGGERED")
            print("========================================")
            print(f"No IoU improvement in the last {CFG.PATIENCE} epochs.")
            print(f"Best IoU : {best_iou:.4f}")
            print(f"Stopped at Epoch : {epoch+1}")
        
            break

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print("\n========================================")
    print("LOADING BEST MODEL")
    print("========================================")

    model.load_state_dict(
        torch.load(
            CFG.BEST_MODEL,
            map_location=CFG.DEVICE
        )
    )

    # =====================================================
    # LOSS CURVE
    # =====================================================

    plt.figure(figsize=(10, 5))

    plt.plot(train_losses, label="Train Loss")

    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(f"Loss Curve ({CFG.MODEL_NAME})")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(CFG.LOG_DIR, "loss_curve.png"), dpi=300)

    plt.show()

    # =====================================================
    # METRICS CURVE
    # =====================================================

    plt.figure(figsize=(10, 5))

    plt.plot(history_iou, label="IoU")

    plt.plot(history_f1, label="F1 Score")

    plt.plot(history_precision, label="Precision")

    plt.plot(history_recall, label="Recall")

    plt.xlabel("Epoch")

    plt.ylabel("Metric")

    plt.title(f"Validation Metrics ({CFG.MODEL_NAME})")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(CFG.LOG_DIR, "metrics_curve.png"), dpi=300)

    plt.show()

    print("\n========================================")
    print("TRAINING COMPLETED")
    print("========================================")

    return model


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()