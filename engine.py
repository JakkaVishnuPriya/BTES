import json
import torch
import torch.nn as nn
from tqdm import tqdm

from configs.config import Config
from utils.metrics import BinaryMetrics


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Executes single training epoch using Mixed Precision."""
    model.train()
    total_loss = 0.0
    metrics_evaluator = BinaryMetrics()

    accumulated_metrics = {
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
    }

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Quick metric estimation during training at 0.5 threshold
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            batch_metrics = metrics_evaluator.compute_all(
                probs, masks, threshold=Config.DEFAULT_THRESHOLD
            )
            for k in accumulated_metrics:
                accumulated_metrics[k] += batch_metrics[k]

        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in accumulated_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates validation set across multiple candidate thresholds.
    Finds and returns the best threshold based on highest Validation IoU.
    """
    model.eval()
    total_loss = 0.0
    metrics_evaluator = BinaryMetrics()

    all_probs = []
    all_masks = []

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                logits = model(images)
                loss = criterion(logits, masks)

            total_loss += loss.item()
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())

    avg_loss = total_loss / len(dataloader)

    # Concatenate all predictions and ground truths for threshold evaluation
    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    best_threshold = Config.DEFAULT_THRESHOLD
    best_iou = -1.0
    best_threshold_metrics = {}

    # Evaluate all defined thresholds
    for thresh in Config.THRESHOLDS:
        metrics = metrics_evaluator.compute_all(
            all_probs, all_masks, threshold=thresh
        )
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_threshold = thresh
            best_threshold_metrics = metrics

    best_threshold_metrics["loss"] = avg_loss
    best_threshold_metrics["best_threshold"] = best_threshold

    return best_threshold_metrics, best_threshold