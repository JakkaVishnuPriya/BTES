# -*- coding: utf-8 -*-

"""
=====================================================================
BTES TREE EXTRACTION
RESNET34 UNET / UNET++
ADVANCED RGB + TEXTURE + MEAN-SHIFT + WATERSHED
=====================================================================

FINAL VERSION

Pipeline:

    RGB
      |
      v
    ResNet34 UNet++
      |
      v
    Probability
      |
      +---- RGB vegetation features
      |
      +---- Texture features
      |
      +---- Mean-Shift regions
      |
      v
    Refined tree candidate
      |
      v
    Distance transform
      |
      v
    Tree-center detection
      |
      v
    Marker-controlled watershed
      |
      v
    Individual tree regions
      |
      v
    FINAL OVERLAY

IMPORTANT:
    - Small components are NOT removed.
    - Watershed boundaries are preserved.
    - Pixels rejected by watershed are NOT restored.
    - Therefore gaps between neighboring trees can remain background.
    - Randomly selects 10 test images.
    - Shows one image at a time.
    - Close the current window to immediately see the next image.
    - Outputs are also saved.
=====================================================================
"""

import os
import random
import traceback

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from scipy import ndimage as ndi
from skimage.segmentation import watershed

from model import build_model
from config import CFG


# =================================================================
# 1. PATHS
# =================================================================

BASE_DIR = r"G:\TreeDataset\tcd"

TEST_IMAGES_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test",
    "images"
)

TEST_MASKS_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test",
    "masks"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "predictions",
    "tree_resnet34_watershed_10_random"
)


# =================================================================
# 2. DEVICE
# =================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =================================================================
# 3. TEST SETTINGS
# =================================================================

NUM_RANDOM_IMAGES = 10

RANDOM_SEED = 42

IMAGE_SIZE = (256, 256)


# =================================================================
# 4. MODEL THRESHOLDS
# =================================================================

THRESHOLD = 0.05

HIGH_THRESHOLD = 0.60

VERY_HIGH_THRESHOLD = 0.80


# =================================================================
# 5. RGB REFINEMENT WEIGHTS
# =================================================================

MODEL_WEIGHT = 0.65

VEGETATION_WEIGHT = 0.15

TEXTURE_WEIGHT = 0.10

COLOR_WEIGHT = 0.10


# =================================================================
# 6. MEAN-SHIFT SETTINGS
# =================================================================

USE_MEAN_SHIFT = True

MEAN_SHIFT_SP = 12

MEAN_SHIFT_SR = 25


# =================================================================
# 7. TREE CENTER SETTINGS
# =================================================================

MIN_TREE_CENTER_DISTANCE = 1

CENTER_MIN_DISTANCE = 6


# =================================================================
# 8. PRINT CONFIGURATION
# =================================================================

print("=" * 80)
print("BTES TREE EXTRACTION")
print("RESNET34 UNET / UNET++")
print("RGB + TEXTURE + MEAN-SHIFT + WATERSHED")
print("10 RANDOM TEST IMAGES")
print("=" * 80)

print()
print("Model       :", CFG.MODEL_NAME)
print("Encoder     :", CFG.ENCODER)
print("Classes     :", CFG.CLASSES)
print("Input       :", CFG.IN_CHANNELS)

print()
print("Test images :", TEST_IMAGES_DIR)
print("Test masks   :", TEST_MASKS_DIR)
print("Output       :", OUTPUT_DIR)

print()
print("Device      :", DEVICE)
print("Threshold   :", THRESHOLD)
print("High thresh :", HIGH_THRESHOLD)

print()
print("Small component removal: DISABLED")


# =================================================================
# 9. NORMALIZE 0-1
# =================================================================

def normalize_01(array):

    array = np.asarray(
        array,
        dtype=np.float32
    )

    minimum = float(
        np.min(array)
    )

    maximum = float(
        np.max(array)
    )

    if maximum - minimum < 1e-8:

        return np.zeros_like(
            array,
            dtype=np.float32
        )

    return (
        (array - minimum)
        /
        (maximum - minimum)
    ).astype(
        np.float32
    )


# =================================================================
# 10. RGB FEATURES
# =================================================================

def calculate_rgb_features(image_rgb):

    image_rgb = np.asarray(
        image_rgb,
        dtype=np.uint8
    )

    rgb = (
        image_rgb.astype(
            np.float32
        )
        /
        255.0
    )

    red = rgb[:, :, 0]

    green = rgb[:, :, 1]

    blue = rgb[:, :, 2]

    # -------------------------------------------------------------
    # Excess Green
    # -------------------------------------------------------------

    exg = (
        2.0 * green
        -
        red
        -
        blue
    )

    exg_norm = np.clip(
        (exg + 0.25) / 0.75,
        0.0,
        1.0
    )

    # -------------------------------------------------------------
    # Green dominance
    # -------------------------------------------------------------

    green_dominance = (
        green
        -
        (red + blue) / 2.0
    )

    green_dom_norm = np.clip(
        (green_dominance + 0.20)
        /
        0.40,
        0.0,
        1.0
    )

    # -------------------------------------------------------------
    # HSV
    # -------------------------------------------------------------

    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )

    hue = hsv[:, :, 0].astype(
        np.float32
    )

    saturation = (
        hsv[:, :, 1].astype(
            np.float32
        )
        /
        255.0
    )

    hue_green = np.exp(
        -(
            (hue - 60.0)
            /
            35.0
        ) ** 2
    )

    hsv_score = (
        0.65 * hue_green
        +
        0.35 * saturation
    )

    hsv_score = np.clip(
        hsv_score,
        0.0,
        1.0
    )

    # -------------------------------------------------------------
    # LAB
    # -------------------------------------------------------------

    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    )

    a = (
        lab[:, :, 1].astype(
            np.float32
        )
        -
        128.0
    )

    lab_green = np.clip(
        (-a + 5.0)
        /
        40.0,
        0.0,
        1.0
    )

    return {
        "exg": exg_norm,
        "green": green_dom_norm,
        "hsv": hsv_score,
        "lab_green": lab_green,
        "saturation": saturation
    }


# =================================================================
# 11. TEXTURE
# =================================================================

def calculate_texture_features(image_rgb):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    ).astype(
        np.float32
    )

    # -------------------------------------------------------------
    # Local standard deviation
    # -------------------------------------------------------------

    mean = cv2.GaussianBlur(
        gray,
        (0, 0),
        2.0
    )

    sq_mean = cv2.GaussianBlur(
        gray * gray,
        (0, 0),
        2.0
    )

    variance = np.maximum(
        sq_mean - mean * mean,
        0.0
    )

    local_std = np.sqrt(
        variance
    )

    scale = float(
        np.percentile(
            local_std,
            95
        )
    )

    if scale > 1e-6:

        texture_norm = np.clip(
            local_std / scale,
            0.0,
            1.0
        )

    else:

        texture_norm = np.zeros_like(
            local_std,
            dtype=np.float32
        )

    # -------------------------------------------------------------
    # Sobel gradient
    # -------------------------------------------------------------

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        gx,
        gy
    )

    gradient_scale = float(
        np.percentile(
            gradient,
            95
        )
    )

    if gradient_scale > 1e-6:

        gradient_norm = np.clip(
            gradient / gradient_scale,
            0.0,
            1.0
        )

    else:

        gradient_norm = np.zeros_like(
            gradient,
            dtype=np.float32
        )

    # -------------------------------------------------------------
    # Laplacian
    # -------------------------------------------------------------

    laplacian = np.abs(
        cv2.Laplacian(
            gray,
            cv2.CV_32F
        )
    )

    lap_scale = float(
        np.percentile(
            laplacian,
            95
        )
    )

    if lap_scale > 1e-6:

        lap_norm = np.clip(
            laplacian / lap_scale,
            0.0,
            1.0
        )

    else:

        lap_norm = np.zeros_like(
            laplacian,
            dtype=np.float32
        )

    texture_score = (
        0.50 * texture_norm
        +
        0.30 * gradient_norm
        +
        0.20 * lap_norm
    )

    texture_score = np.clip(
        texture_score,
        0.0,
        1.0
    )

    return {
        "texture": texture_score,
        "gradient": gradient_norm,
        "local_std": texture_norm
    }


# =================================================================
# 12. MEAN-SHIFT
# =================================================================

def calculate_mean_shift_regions(image_rgb):

    if not USE_MEAN_SHIFT:

        return np.zeros(
            image_rgb.shape[:2],
            dtype=np.int32
        )

    filtered = cv2.pyrMeanShiftFiltering(
        image_rgb,
        sp=MEAN_SHIFT_SP,
        sr=MEAN_SHIFT_SR,
        maxLevel=1
    )

    pixels = filtered.reshape(
        -1,
        3
    )

    quantized = (
        pixels // 8
    ).astype(
        np.int32
    )

    _, inverse = np.unique(
        quantized,
        axis=0,
        return_inverse=True
    )

    labels = inverse.reshape(
        image_rgb.shape[:2]
    )

    return labels.astype(
        np.int32
    )


# =================================================================
# 13. MEAN-SHIFT REFINEMENT
# =================================================================

def refine_with_mean_shift(
    probability,
    candidate,
    image_rgb
):

    if not USE_MEAN_SHIFT:

        return candidate.copy()

    labels = calculate_mean_shift_regions(
        image_rgb
    )

    result = candidate.copy()

    unique_labels = np.unique(
        labels
    )

    for region_id in unique_labels:

        region = (
            labels == region_id
        )

        if not np.any(region):

            continue

        region_probability = float(
            np.mean(
                probability[region]
            )
        )

        region_candidate_fraction = float(
            np.mean(
                candidate[region]
            )
        )

        # Strong model evidence:
        # preserve the region.

        if region_probability >= HIGH_THRESHOLD:

            result[region] = 1

            continue

        # Moderate candidate region.

        if region_candidate_fraction > 0.50:

            if region_probability >= 0.10:

                result[region] = 1

            else:

                result[region] = 0

    return result.astype(
        np.uint8
    )


# =================================================================
# 14. SEMANTIC REFINEMENT
# =================================================================

def semantic_refinement(
    probability,
    image_rgb
):

    features = calculate_rgb_features(
        image_rgb
    )

    textures = calculate_texture_features(
        image_rgb
    )

    exg = features["exg"]

    green = features["green"]

    hsv_score = features["hsv"]

    lab_green = features["lab_green"]

    texture = textures["texture"]

    gradient = textures["gradient"]

    # -------------------------------------------------------------
    # Vegetation score
    # -------------------------------------------------------------

    vegetation_score = (
        0.45 * exg
        +
        0.30 * green
        +
        0.15 * hsv_score
        +
        0.10 * lab_green
    )

    vegetation_score = np.clip(
        vegetation_score,
        0.0,
        1.0
    )

    # -------------------------------------------------------------
    # Combined score
    # -------------------------------------------------------------

    combined_score = (
        MODEL_WEIGHT * probability
        +
        VEGETATION_WEIGHT * vegetation_score
        +
        TEXTURE_WEIGHT * texture
        +
        COLOR_WEIGHT * hsv_score
    )

    combined_score = np.clip(
        combined_score,
        0.0,
        1.0
    )

    refined = np.zeros_like(
        probability,
        dtype=np.uint8
    )

    # -------------------------------------------------------------
    # Strong model prediction
    # -------------------------------------------------------------

    refined[
        probability >= HIGH_THRESHOLD
    ] = 1

    # -------------------------------------------------------------
    # Medium confidence
    # -------------------------------------------------------------

    medium_model = (
        (probability >= THRESHOLD)
        &
        (probability < HIGH_THRESHOLD)
    )

    medium_score = (
        combined_score >= 0.42
    )

    refined[
        medium_model
        &
        medium_score
    ] = 1

    # -------------------------------------------------------------
    # Very strong
    # -------------------------------------------------------------

    refined[
        probability >= VERY_HIGH_THRESHOLD
    ] = 1

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # NO SMALL COMPONENT REMOVAL.
    # -------------------------------------------------------------

    return (
        refined.astype(np.uint8),
        combined_score.astype(np.float32),
        vegetation_score.astype(np.float32),
        texture.astype(np.float32),
        gradient.astype(np.float32)
    )


# =================================================================
# 15. DISTANCE TRANSFORM
# =================================================================

def calculate_distance_map(mask):

    binary = (
        mask > 0
    )

    if not np.any(binary):

        return np.zeros(
            mask.shape,
            dtype=np.float32
        )

    distance = ndi.distance_transform_edt(
        binary
    )

    distance = ndi.gaussian_filter(
        distance.astype(
            np.float32
        ),
        sigma=1.0
    )

    return distance


# =================================================================
# 16. TREE CENTERS
# =================================================================

def detect_tree_centers(
    distance,
    probability,
    mask
):

    centers = np.zeros(
        mask.shape,
        dtype=np.int32
    )

    if not np.any(mask):

        return centers

    # -------------------------------------------------------------
    # Local maximum detection
    # -------------------------------------------------------------

    size = (
        CENTER_MIN_DISTANCE * 2
        +
        1
    )

    local_max = cv2.dilate(
        distance,
        np.ones(
            (
                size,
                size
            ),
            dtype=np.uint8
        )
    )

    maxima = (
        (distance >= local_max)
        &
        (distance >= MIN_TREE_CENTER_DISTANCE)
        &
        (mask > 0)
    )

    # Need at least moderate probability.

    maxima &= (
        probability >= 0.30
    )

    # -------------------------------------------------------------
    # Connected maxima
    # -------------------------------------------------------------

    marker_components, num = ndi.label(
        maxima,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8
        )
    )

    marker_id = 1

    for component_id in range(
        1,
        int(num) + 1
    ):

        component = (
            marker_components
            ==
            component_id
        )

        if not np.any(component):

            continue

        values = distance[
            component
        ]

        flat_indices = np.flatnonzero(
            component
        )

        if values.size == 0:

            continue

        best_index = flat_indices[
            int(
                np.argmax(values)
            )
        ]

        centers.reshape(-1)[
            best_index
        ] = marker_id

        marker_id += 1

    # -------------------------------------------------------------
    # If no maxima:
    #
    # one marker per connected region.
    # -------------------------------------------------------------

    if marker_id == 1:

        labels, num_regions = ndi.label(
            mask.astype(bool),
            structure=np.ones(
                (3, 3),
                dtype=np.uint8
            )
        )

        for region_id in range(
            1,
            int(num_regions) + 1
        ):

            region = (
                labels == region_id
            )

            if not np.any(region):

                continue

            region_distance = np.where(
                region,
                distance,
                0
            )

            flat = np.argmax(
                region_distance
            )

            centers.reshape(-1)[
                flat
            ] = marker_id

            marker_id += 1

    return centers


# =================================================================
# 17. WATERSHED
# =================================================================

def marker_controlled_watershed(
    image_rgb,
    probability,
    refined_mask
):

    if not np.any(refined_mask):

        return (
            np.zeros_like(
                refined_mask,
                dtype=np.uint8
            ),
            np.zeros_like(
                refined_mask,
                dtype=np.int32
            ),
            np.zeros_like(
                refined_mask,
                dtype=np.float32
            ),
            np.zeros_like(
                refined_mask,
                dtype=np.int32
            )
        )

    # -------------------------------------------------------------
    # Distance
    # -------------------------------------------------------------

    distance = calculate_distance_map(
        refined_mask
    )

    # -------------------------------------------------------------
    # Tree centers
    # -------------------------------------------------------------

    centers = detect_tree_centers(
        distance,
        probability,
        refined_mask
    )

    # -------------------------------------------------------------
    # RGB gradient
    # -------------------------------------------------------------

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    rgb_gradient = cv2.magnitude(
        gx,
        gy
    )

    rgb_gradient = normalize_01(
        rgb_gradient
    )

    # -------------------------------------------------------------
    # Probability gradient
    # -------------------------------------------------------------

    px = cv2.Sobel(
        probability,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    py = cv2.Sobel(
        probability,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    probability_gradient = cv2.magnitude(
        px,
        py
    )

    probability_gradient = normalize_01(
        probability_gradient
    )

    # -------------------------------------------------------------
    # Watershed elevation
    # -------------------------------------------------------------

    elevation = (
        0.65 * rgb_gradient
        +
        0.35 * probability_gradient
    )

    elevation += (
        0.10
        *
        (1.0 - probability)
    )

    elevation = elevation.astype(
        np.float32
    )

    # -------------------------------------------------------------
    # MARKER CONTROLLED WATERSHED
    # -------------------------------------------------------------

    labels = watershed(
        elevation,
        centers,
        mask=(
            refined_mask > 0
        ),
        watershed_line=True
    )

    labels = np.asarray(
        labels,
        dtype=np.int32
    )

    # -------------------------------------------------------------
    # VERY IMPORTANT:
    #
    # DO NOT RESTORE REJECTED PIXELS.
    #
    # Previously we had:
    #
    # final_mask[refined_mask > 0] = 1
    #
    # That was filling the gaps again.
    #
    # Now the final mask comes ONLY from watershed.
    # -------------------------------------------------------------

    final_mask = (
        labels > 0
    ).astype(
        np.uint8
    )

    return (
        final_mask,
        labels,
        distance,
        centers
    )


# =================================================================
# 18. FIND TEST IMAGES
# =================================================================

def get_test_images():

    extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    )

    files = []

    for filename in os.listdir(
        TEST_IMAGES_DIR
    ):

        path = os.path.join(
            TEST_IMAGES_DIR,
            filename
        )

        if not os.path.isfile(path):

            continue

        if not filename.lower().endswith(
            extensions
        ):

            continue

        if "_mask" in filename.lower():

            continue

        files.append(
            filename
        )

    files.sort()

    return files


# =================================================================
# 19. FIND MASK
# =================================================================

def find_mask(base_name):

    extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    )

    for extension in extensions:

        path = os.path.join(
            TEST_MASKS_DIR,
            base_name
            +
            "_mask"
            +
            extension
        )

        if os.path.isfile(path):

            return path

    return None


# =================================================================
# 20. INSTANCE VISUALIZATION
# =================================================================

def create_instance_visualization(
    labels
):

    height, width = labels.shape

    output = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    unique_labels = np.unique(
        labels
    )

    rng = np.random.default_rng(
        12345
    )

    for tree_id in unique_labels:

        if tree_id == 0:

            continue

        color = rng.integers(
            50,
            255,
            size=3
        ).astype(
            np.uint8
        )

        output[
            labels == tree_id
        ] = color

    return output


# =================================================================
# 21. TREE OVERLAY
# =================================================================

def create_tree_overlay(
    image_rgb,
    labels
):

    overlay = image_rgb.copy()

    # -------------------------------------------------------------
    # Fill each watershed tree with transparent red
    # -------------------------------------------------------------

    tree_mask = (
        labels > 0
    )

    if np.any(tree_mask):

        tree_pixels = (
            image_rgb[
                tree_mask
            ].astype(
                np.float32
            )
        )

        red = np.zeros_like(
            tree_pixels
        )

        red[:, 0] = 255

        blended = (
            0.50 * tree_pixels
            +
            0.50 * red
        )

        overlay[
            tree_mask
        ] = blended.astype(
            np.uint8
        )

    # -------------------------------------------------------------
    # Draw boundaries.
    #
    # The boundaries remain visible, including gaps.
    # -------------------------------------------------------------

    unique_labels = np.unique(
        labels
    )

    for tree_id in unique_labels:

        if tree_id == 0:

            continue

        region = (
            labels == tree_id
        ).astype(
            np.uint8
        )

        contours, _ = cv2.findContours(
            region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 255, 0),
            1
        )

    return overlay


# =================================================================
# 22. SAVE VISUALIZATION
# =================================================================

def visualize_and_save(
    filename,
    image_rgb,
    ground_truth,
    probability,
    raw_mask,
    vegetation_score,
    texture_score,
    refined_mask,
    distance,
    centers,
    final_mask,
    labels
):

    overlay = create_tree_overlay(
        image_rgb,
        labels
    )

    instance_image = create_instance_visualization(
        labels
    )

    tree_count = int(
        labels.max()
    )

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(
            20,
            14
        )
    )

    # -------------------------------------------------------------
    # 1 Original
    # -------------------------------------------------------------

    axes[0, 0].imshow(
        image_rgb
    )

    axes[0, 0].set_title(
        "1. Original RGB"
    )

    axes[0, 0].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 2 Ground Truth
    # -------------------------------------------------------------

    axes[0, 1].imshow(
        ground_truth,
        cmap="gray"
    )

    axes[0, 1].set_title(
        "2. Ground Truth"
    )

    axes[0, 1].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 3 Probability
    # -------------------------------------------------------------

    im = axes[0, 2].imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[0, 2].set_title(
        "3. UNet Probability"
    )

    axes[0, 2].axis(
        "off"
    )

    plt.colorbar(
        im,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04
    )

    # -------------------------------------------------------------
    # 4 Raw
    # -------------------------------------------------------------

    axes[0, 3].imshow(
        raw_mask,
        cmap="gray"
    )

    axes[0, 3].set_title(
        f"4. Raw Mask > {THRESHOLD}"
    )

    axes[0, 3].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 5 Vegetation
    # -------------------------------------------------------------

    axes[1, 0].imshow(
        vegetation_score,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[1, 0].set_title(
        "5. RGB Vegetation Score"
    )

    axes[1, 0].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 6 Texture
    # -------------------------------------------------------------

    axes[1, 1].imshow(
        texture_score,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[1, 1].set_title(
        "6. Texture Score"
    )

    axes[1, 1].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 7 Refined
    # -------------------------------------------------------------

    axes[1, 2].imshow(
        refined_mask,
        cmap="gray"
    )

    axes[1, 2].set_title(
        "7. Refined Candidate"
    )

    axes[1, 2].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 8 Distance
    # -------------------------------------------------------------

    axes[1, 3].imshow(
        distance,
        cmap="magma"
    )

    axes[1, 3].set_title(
        "8. Distance Transform"
    )

    axes[1, 3].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 9 Centers
    # -------------------------------------------------------------

    axes[2, 0].imshow(
        image_rgb
    )

    y, x = np.where(
        centers > 0
    )

    if len(x) > 0:

        axes[2, 0].scatter(
            x,
            y,
            s=30,
            marker="x"
        )

    axes[2, 0].set_title(
        f"9. Tree Centers ({len(x)})"
    )

    axes[2, 0].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 10 Final watershed
    # -------------------------------------------------------------

    axes[2, 1].imshow(
        final_mask,
        cmap="gray"
    )

    axes[2, 1].set_title(
        f"10. Watershed ({tree_count} trees)"
    )

    axes[2, 1].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 11 Individual regions
    # -------------------------------------------------------------

    axes[2, 2].imshow(
        instance_image
    )

    axes[2, 2].set_title(
        "11. Individual Tree Regions"
    )

    axes[2, 2].axis(
        "off"
    )

    # -------------------------------------------------------------
    # 12 FINAL OVERLAY
    # -------------------------------------------------------------

    axes[2, 3].imshow(
        overlay
    )

    axes[2, 3].set_title(
        "12. FINAL TREE OVERLAY"
    )

    axes[2, 3].axis(
        "off"
    )

    fig.suptitle(
        f"Tree Extraction - {filename}",
        fontsize=16
    )

    fig.tight_layout()

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    base_name = os.path.splitext(
        filename
    )[0]

    prefix = os.path.join(
        OUTPUT_DIR,
        base_name
    )

    # Complete figure

    fig.savefig(
        prefix
        +
        "_analysis.png",
        dpi=180,
        bbox_inches="tight"
    )

    # -------------------------------------------------------------
    # Final mask
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_final_mask.png",
        (
            final_mask * 255
        ).astype(
            np.uint8
        )
    )

    # -------------------------------------------------------------
    # Watershed labels
    # -------------------------------------------------------------

    np.save(
        prefix
        +
        "_labels.npy",
        labels
    )

    # -------------------------------------------------------------
    # Raw
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_raw_mask.png",
        (
            raw_mask * 255
        ).astype(
            np.uint8
        )
    )

    # -------------------------------------------------------------
    # Refined
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_refined_mask.png",
        (
            refined_mask * 255
        ).astype(
            np.uint8
        )
    )

    # -------------------------------------------------------------
    # Probability
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_probability.png",
        (
            probability * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )
    )

    # -------------------------------------------------------------
    # Vegetation
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_vegetation.png",
        (
            vegetation_score * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )
    )

    # -------------------------------------------------------------
    # Texture
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_texture.png",
        (
            texture_score * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )
    )

    # -------------------------------------------------------------
    # Instance visualization
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_individual_trees.png",
        cv2.cvtColor(
            instance_image,
            cv2.COLOR_RGB2BGR
        )
    )

    # -------------------------------------------------------------
    # FINAL OVERLAY
    # -------------------------------------------------------------

    cv2.imwrite(
        prefix
        +
        "_final_overlay.png",
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "Raw pixels     :",
        int(
            np.count_nonzero(
                raw_mask
            )
        )
    )

    print(
        "Refined pixels :",
        int(
            np.count_nonzero(
                refined_mask
            )
        )
    )

    print(
        "Final pixels   :",
        int(
            np.count_nonzero(
                final_mask
            )
        )
    )

    print(
        "Tree regions   :",
        tree_count
    )

    print()
    print(
        "Saved:"
    )

    print(
        prefix
        +
        "_analysis.png"
    )

    print(
        prefix
        +
        "_final_mask.png"
    )

    print(
        prefix
        +
        "_individual_trees.png"
    )

    print(
        prefix
        +
        "_final_overlay.png"
    )

    # -------------------------------------------------------------
    # SHOW
    # -------------------------------------------------------------

    plt.show(
        block=True
    )

    plt.close(
        fig
    )


# =================================================================
# 23. MAIN
# =================================================================

def main():

    # -------------------------------------------------------------
    # Folder checks
    # -------------------------------------------------------------

    if not os.path.isdir(
        TEST_IMAGES_DIR
    ):

        raise FileNotFoundError(
            "Test image folder not found:\n"
            +
            TEST_IMAGES_DIR
        )

    if not os.path.isdir(
        TEST_MASKS_DIR
    ):

        print()
        print(
            "WARNING: Test mask folder not found:"
        )

        print(
            TEST_MASKS_DIR
        )

    # -------------------------------------------------------------
    # Find images
    # -------------------------------------------------------------

    image_files = get_test_images()

    print()
    print(
        "Total test images:",
        len(image_files)
    )

    if len(image_files) == 0:

        raise RuntimeError(
            "No test images found."
        )

    # -------------------------------------------------------------
    # Random selection
    # -------------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    count = min(
        NUM_RANDOM_IMAGES,
        len(image_files)
    )

    selected_files = random.sample(
        image_files,
        count
    )

    selected_files.sort()

    print()
    print(
        "Randomly selected images:"
    )

    for filename in selected_files:

        print(
            "   ",
            filename
        )

    # -------------------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)

    model = build_model()

    print(
        "Model:",
        model.__class__.__name__
    )

    # -------------------------------------------------------------
    # CHECKPOINT
    # -------------------------------------------------------------

    checkpoint_path = CFG.BEST_MODEL

    print()
    print(
        "Checkpoint:"
    )

    print(
        checkpoint_path
    )

    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            +
            checkpoint_path
        )

    # -------------------------------------------------------------
    # LOAD CHECKPOINT
    # -------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

        if "epoch" in checkpoint:

            print(
                "Checkpoint epoch:",
                checkpoint["epoch"]
            )

        if "val_iou" in checkpoint:

            print(
                "Saved Val IoU:",
                checkpoint["val_iou"]
            )

        if "val_dice" in checkpoint:

            print(
                "Saved Val Dice:",
                checkpoint["val_dice"]
            )

    else:

        state_dict = checkpoint

    # -------------------------------------------------------------
    # Remove DataParallel prefix
    # -------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        cleaned_key = key.replace(
            "module.",
            ""
        )

        cleaned_state_dict[
            cleaned_key
        ] = value

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print()
    print(
        "Model loaded successfully."
    )

    # -------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This must match your training dataset.py.
    #
    # If your dataset.py uses another normalization,
    # change these values.
    # -------------------------------------------------------------

    transform = A.Compose(
        [
            A.Resize(
                height=IMAGE_SIZE[0],
                width=IMAGE_SIZE[1]
            ),

            A.Normalize(
                mean=(
                    0.485,
                    0.456,
                    0.406
                ),
                std=(
                    0.229,
                    0.224,
                    0.225
                )
            ),

            ToTensorV2()
        ]
    )

    # -------------------------------------------------------------
    # PROCESS
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("PROCESSING RANDOM TEST IMAGES")
    print("=" * 80)

    for index, filename in enumerate(
        selected_files,
        start=1
    ):

        print()
        print("-" * 80)
        print(
            f"[{index}/{len(selected_files)}] {filename}"
        )
        print("-" * 80)

        image_path = os.path.join(
            TEST_IMAGES_DIR,
            filename
        )

        base_name = os.path.splitext(
            filename
        )[0]

        try:

            # -----------------------------------------------------
            # READ
            # -----------------------------------------------------

            bgr = cv2.imread(
                image_path,
                cv2.IMREAD_COLOR
            )

            if bgr is None:

                raise RuntimeError(
                    "Could not read image."
                )

            image_rgb_original = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB
            )

            # -----------------------------------------------------
            # 256x256
            # -----------------------------------------------------

            image_rgb = cv2.resize(
                image_rgb_original,
                (
                    IMAGE_SIZE[1],
                    IMAGE_SIZE[0]
                ),
                interpolation=cv2.INTER_LINEAR
            )

            # -----------------------------------------------------
            # Transform
            # -----------------------------------------------------

            transformed = transform(
                image=image_rgb
            )

            tensor = transformed[
                "image"
            ].unsqueeze(
                0
            ).to(
                DEVICE
            )

            # -----------------------------------------------------
            # MODEL
            # -----------------------------------------------------

            with torch.no_grad():

                output = model(
                    tensor
                )

                probability = torch.sigmoid(
                    output
                )

            probability = (
                probability
                .detach()
                .cpu()
                .numpy()
            )

            probability = np.squeeze(
                probability
            ).astype(
                np.float32
            )

            probability = np.clip(
                probability,
                0.0,
                1.0
            )

            print()
            print(
                "Probability:"
            )

            print(
                f"min  = {probability.min():.4f}"
            )

            print(
                f"max  = {probability.max():.4f}"
            )

            print(
                f"mean = {probability.mean():.4f}"
            )

            # -----------------------------------------------------
            # RAW
            # -----------------------------------------------------

            raw_mask = (
                probability >= THRESHOLD
            ).astype(
                np.uint8
            )

            print()
            print(
                "Raw tree pixels:",
                int(
                    np.count_nonzero(
                        raw_mask
                    )
                )
            )

            # -----------------------------------------------------
            # RGB + TEXTURE
            # -----------------------------------------------------

            print(
                "Applying RGB + texture refinement..."
            )

            (
                refined_mask,
                combined_score,
                vegetation_score,
                texture_score,
                gradient_score
            ) = semantic_refinement(
                probability,
                image_rgb
            )

            print(
                "Refined pixels:",
                int(
                    np.count_nonzero(
                        refined_mask
                    )
                )
            )

            # -----------------------------------------------------
            # MEAN SHIFT
            # -----------------------------------------------------

            print(
                "Applying Mean-Shift..."
            )

            refined_mask = refine_with_mean_shift(
                probability,
                refined_mask,
                image_rgb
            )

            print(
                "Mean-Shift pixels:",
                int(
                    np.count_nonzero(
                        refined_mask
                    )
                )
            )

            # -----------------------------------------------------
            # WATERSHED
            # -----------------------------------------------------

            print(
                "Applying marker-controlled watershed..."
            )

            (
                final_mask,
                labels,
                distance,
                centers
            ) = marker_controlled_watershed(
                image_rgb,
                probability,
                refined_mask
            )

            print(
                "Watershed completed."
            )

            print(
                "Tree centers:",
                int(
                    np.count_nonzero(
                        centers
                    )
                )
            )

            print(
                "Individual tree regions:",
                int(
                    labels.max()
                )
            )

            print(
                "Final tree pixels:",
                int(
                    np.count_nonzero(
                        final_mask
                    )
                )
            )

            # -----------------------------------------------------
            # GROUND TRUTH
            # -----------------------------------------------------

            ground_truth = np.zeros(
                IMAGE_SIZE,
                dtype=np.uint8
            )

            mask_path = find_mask(
                base_name
            )

            if mask_path is not None:

                gt = cv2.imread(
                    mask_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if gt is not None:

                    gt = cv2.resize(
                        gt,
                        (
                            IMAGE_SIZE[1],
                            IMAGE_SIZE[0]
                        ),
                        interpolation=cv2.INTER_NEAREST
                    )

                    ground_truth = (
                        gt > 0
                    ).astype(
                        np.uint8
                    )

            # -----------------------------------------------------
            # VISUALIZE
            # -----------------------------------------------------

            visualize_and_save(
                filename=filename,
                image_rgb=image_rgb,
                ground_truth=ground_truth,
                probability=probability,
                raw_mask=raw_mask,
                vegetation_score=vegetation_score,
                texture_score=texture_score,
                refined_mask=refined_mask,
                distance=distance,
                centers=centers,
                final_mask=final_mask,
                labels=labels
            )

            print()
            print(
                "Closed."
            )

            print(
                "Next image..."
            )

        except Exception as error:

            print()
            print("=" * 80)
            print(
                "ERROR:",
                filename
            )
            print("=" * 80)

            print(
                str(error)
            )

            traceback.print_exc()

            print("=" * 80)

            continue

    print()
    print("=" * 80)
    print("10 RANDOM IMAGE TEST COMPLETE")
    print("=" * 80)

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 80)


# =================================================================
# RUN
# =================================================================

if __name__ == "__main__":

    main()