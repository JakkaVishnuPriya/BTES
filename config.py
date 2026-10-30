# -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Config
Bhuvan Basemap, ~3m resolution, 256x256 tiles
mask value 1 = tree, 2 = background (converted to 1/0 in dataset.py)

Dataset:
    Images: G:\TreeDataset\tcd\tree_dataset_1\images
    Masks : G:\TreeDataset\tcd\tree_dataset_1\masks

Train / validation / test splits are created from these single
image and mask folders. No separate train/val/test folders are required.

@author: saurav_kumar
"""

import os
import torch


class CFG:

    # =========================================================
    # DATASET PATHS
    # =========================================================

    # Single image folder
    IMAGE_DIR = r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_resnet_34\dataset\images"

    # Single mask folder
    MASK_DIR = r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_resnet_34\dataset\masks"

    # ---------------------------------------------------------
    # TRAIN / VAL / TEST SPLIT
    # ---------------------------------------------------------
    # These splits are made from IMAGE_DIR and MASK_DIR.
    # No separate train/val/test folders are needed.
    #
    # Example:
    #   70% -> training
    #   15% -> validation
    #   15% -> testing
    # ---------------------------------------------------------

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Fixed seed so the same images remain in the same split
    # every time the program is run.
    SPLIT_SEED = 42

    # =========================================================
    # PRETRAINED MODEL
    # =========================================================

    # Full model checkpoint (.pth)
    # Set to None to train from scratch (only encoder initialization)
    # PRETRAIN_DIR = r"D:\BTES\Hyderabad\Bhuvan_Basemap\Tree-Segmentation\pretrain\best_model_unetpp_2307.pth"

    # =========================================================
    # ENCODER WEIGHTS (OFFLINE / AIR-GAPPED)
    # =========================================================

    LOCAL_ENCODER_WEIGHT = r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_resnet_34\encoder_weight\resnet34-b627a593.pth"

    # Set to a full-model .pth path to resume / warm-start. None = train from scratch.
    PRETRAIN_DIR = r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_resnet_34\pretrain\best_model_unetpp_1808.pth"

    # =========================================================
    # MODEL SELECTION
    # =========================================================

    # "unet" or "unetpp"
    MODEL_NAME = "unetpp"

    ENCODER = "resnet34"

    # Keep None when offline; LOCAL_ENCODER_WEIGHT is used instead.
    ENCODER_WEIGHTS = None

    ACTIVATION = None

    IN_CHANNELS = 3

    CLASSES = 1

    # =========================================================
    # TRAINING
    # =========================================================

    IMAGE_SIZE = 256

    BATCH_SIZE = 32

    EPOCHS = 200

    LR = 1e-4

    NUM_WORKERS = 0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SEED = 42

    PIN_MEMORY = True

    AUGMENT = True  # simple horizontal/vertical flips on train split

    # =========================================================
    # EARLY STOPPING
    # =========================================================

    EARLY_STOPPING = True

    PATIENCE = 20          # Stop if IoU does not improve for 20 epochs

    MIN_DELTA = 1e-4       # Minimum improvement in IoU

    # =========================================================
    # LOSS (Focal + Dice + Tversky)
    # =========================================================

    # Main loss weights
    FOCAL_WEIGHT = 0.30
    DICE_WEIGHT = 0.40
    TVERSKY_WEIGHT = 0.30

    # Focal Loss
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0

    # Tversky Loss
    # Higher alpha -> stronger penalty for false positives
    # Useful for reducing grass predicted as tree
    TVERSKY_ALPHA = 0.70
    TVERSKY_BETA = 0.30

    # Kept for compatibility with any older code.
    # Not used by the new Focal + Dice + Tversky loss.
    POS_WEIGHT = 2.0

    # =========================================================
    # INFERENCE
    # =========================================================

    THRESHOLD = 0.5

    MIN_TREE_AREA = 5     # pixels; discard blobs smaller than this after cleanup
    MIN_DISTANCE = 3      # min peak separation (px) used to split touching crowns

    # =========================================================
    # LIVE TRAINING MONITOR (WEB GUI)
    # =========================================================

    ENABLE_MONITOR = True

    MONITOR_PORT = 8050

    # =========================================================
    # OUTPUTS
    # =========================================================

    ROOT_OUT = r"G:\Vishnu_Priya_BTES\Segformer_buildings\Tree_Label\dataset\Tree-Segmentation\TreeExtraction_Results"

    WEIGHT_DIR = os.path.join(ROOT_OUT, "weights")
    OUTPUT_DIR = os.path.join(ROOT_OUT, "output")
    PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
    INSTANCE_DIR = os.path.join(OUTPUT_DIR, "instances")
    LOG_DIR = os.path.join(ROOT_OUT, "logs")

    BEST_MODEL = os.path.join(WEIGHT_DIR, f"best_model_{MODEL_NAME}.pth")

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)