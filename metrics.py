# -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Metrics

@author: saurav_kumar
"""

import numpy as np


def calculate_metrics(preds, masks, threshold=0.5):
    """
    preds, masks : numpy arrays, any shape, same shape.
    preds are probabilities in [0, 1]; masks are binary {0, 1}.
    """

    preds_bin = (preds >= threshold).astype(np.uint8)

    masks_bin = masks.astype(np.uint8)

    tp = np.sum((preds_bin == 1) & (masks_bin == 1))

    tn = np.sum((preds_bin == 0) & (masks_bin == 0))

    fp = np.sum((preds_bin == 1) & (masks_bin == 0))

    fn = np.sum((preds_bin == 0) & (masks_bin == 1))

    eps = 1e-7

    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    precision = tp / (tp + fp + eps)

    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    iou = tp / (tp + fp + fn + eps)

    return float(acc), float(precision), float(recall), float(f1), float(iou)
