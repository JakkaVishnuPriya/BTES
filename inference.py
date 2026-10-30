 # -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Inference
Predicts on CFG.TEST_DIR, then applies distance-transform watershed to
separate touching tree crowns into individual instances.

@author: saurav_kumar
"""

import os
import glob
import csv

import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine

import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, opening, closing, disk

from config import CFG
from model import build_model


# =========================================================
# HELPERS
# =========================================================

def _load_image(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))

    return img


def _get_reference_profile(img_path):
    """
    Reuse georeference (CRS + transform) from the companion mask tif
    if it exists (test tiles kept for evaluation). Falls back to a
    plain identity-transform profile otherwise.
    """

    base = os.path.splitext(img_path)[0]

    mask_path = base + "_mask.tif"

    if os.path.exists(mask_path):

        with rasterio.open(mask_path) as src:

            return src.profile.copy()

    h, w = CFG.IMAGE_SIZE, CFG.IMAGE_SIZE

    return {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": None,
        "transform": Affine.identity()
    }


def _separate_trees(binary_mask, min_distance=3, min_area=15):
    """
    Morphologically clean the binary tree mask, then split touching
    tree crowns using a distance-transform + watershed approach:

      1. closing/opening to remove speckle noise and fill tiny gaps
      2. remove blobs smaller than min_area
      3. Euclidean distance transform of the cleaned mask
      4. local maxima of the distance transform = crown-top proxies
      5. watershed on the negative distance transform, seeded by
         those maxima, restricted to the tree mask
    """

    # =========================================================
    # CLEAN UP NOISE
    # =========================================================

    mask = closing(binary_mask.astype(bool), disk(1))

    mask = opening(mask, disk(1))

    mask = remove_small_objects(mask, min_size=min_area)

    # =========================================================
    # DISTANCE TRANSFORM + LOCAL MAXIMA (TREE-TOP PROXY)
    # =========================================================

    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask
    )

    peak_mask = np.zeros_like(distance, dtype=bool)

    if len(coords) > 0:

        peak_mask[tuple(coords.T)] = True

    markers, _ = ndi.label(peak_mask)

    # =========================================================
    # WATERSHED SEGMENTATION (SPLIT TOUCHING CROWNS)
    # =========================================================

    instances = watershed(-distance, markers, mask=mask)

    return mask.astype(np.uint8), instances.astype(np.int32)


# =========================================================
# MAIN
# =========================================================

def run_inference():

    device = CFG.DEVICE

    model = build_model()

    print("\nLoading Best Model...")

    model.load_state_dict(
        torch.load(CFG.BEST_MODEL, map_location=device)
    )

    model = model.to(device)

    model.eval()

    image_paths = sorted(glob.glob(os.path.join(CFG.TEST_DIR, "*.png")))

    image_paths = [
        p for p in image_paths if "_mask" not in os.path.basename(p)
    ]

    print(f"\nFound {len(image_paths)} test tiles")

    summary_rows = []

    with torch.no_grad():

        for img_path in image_paths:

            name = os.path.splitext(os.path.basename(img_path))[0]

            img = _load_image(img_path)

            img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(img_t)

            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            binary = (prob >= CFG.THRESHOLD).astype(np.uint8)

            clean_mask, instances = _separate_trees(
                binary,
                min_distance=CFG.MIN_DISTANCE,
                min_area=CFG.MIN_TREE_AREA
            )

            profile = _get_reference_profile(img_path)

            # =====================================================
            # SAVE PROBABILITY MAP
            # =====================================================

            prob_profile = profile.copy()

            prob_profile.update(dtype="float32", count=1)

            with rasterio.open(
                os.path.join(CFG.PRED_DIR, f"{name}_prob.tif"), "w", **prob_profile
            ) as dst:

                dst.write(prob.astype(np.float32), 1)

            # =====================================================
            # SAVE CLEAN BINARY MASK
            # =====================================================

            bin_profile = profile.copy()

            bin_profile.update(dtype="uint8", count=1)

            with rasterio.open(
                os.path.join(CFG.PRED_DIR, f"{name}_binary.tif"), "w", **bin_profile
            ) as dst:

                dst.write(clean_mask.astype(np.uint8), 1)

            # =====================================================
            # SAVE INSTANCE-SEPARATED TREES
            # =====================================================

            inst_profile = profile.copy()

            inst_profile.update(dtype="int32", count=1)

            with rasterio.open(
                os.path.join(CFG.INSTANCE_DIR, f"{name}_instances.tif"), "w", **inst_profile
            ) as dst:

                dst.write(instances.astype(np.int32), 1)

            # =====================================================
            # SUMMARY
            # =====================================================

            n_trees = int(instances.max())

            tree_pixels = int(clean_mask.sum())

            summary_rows.append([name, n_trees, tree_pixels])

    # =========================================================
    # WRITE SUMMARY CSV
    # =========================================================

    csv_path = os.path.join(CFG.OUTPUT_DIR, "inference_summary.csv")

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["tile", "tree_count", "tree_pixel_count"])

        writer.writerows(summary_rows)

    print("\n========================================")
    print("INFERENCE COMPLETED")
    print("========================================")

    print(f"Predictions : {CFG.PRED_DIR}")

    print(f"Instances   : {CFG.INSTANCE_DIR}")

    print(f"Summary CSV : {csv_path}")


if __name__ == "__main__":

    run_inference()
