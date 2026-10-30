import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi
from skimage.segmentation import watershed


# ==========================================
# 1. IMAGE-GUIDED WATERSHED SEPARATION
# ==========================================
def apply_image_guided_watershed(mask, image_rgb):
    """
    Applies image-guided watershed:
    - Markers: Extracted from the predicted mask distance centers.
    - Surface: Gradient/Edge map calculated from the ORIGINAL RGB image.
    - Constraint: Constrained within the predicted mask region.
    """
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    core_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    CORE_RATIO = 0.50        # Core threshold ratio (0.45 - 0.60)
    MIN_DISTANCE = 4.0       # Minimum distance peak to generate unique cores
    MIN_INSTANCE_AREA = 15   # Filter out tiny residual pixels

    # 1. Clean small mask noise
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, open_kernel)

    # 2. Extract Image Gradient from ORIGINAL RGB Image
    # Convert image to grayscale and apply mild blur to smooth rooftop textures
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Compute Sobel edge gradient magnitude on the original image
    grad_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
    image_gradient = np.sqrt(grad_x**2 + grad_y**2)

    # 3. Generate Seeds/Markers using Distance Transform on Mask
    distance = ndi.distance_transform_edt(mask)
    if distance.max() == 0:
        return mask

    distance_smoothed = ndi.gaussian_filter(distance, sigma=1.0)
    base_labels, num_objects = ndi.label(mask)
    markers = np.zeros(mask.shape, dtype=np.int32)
    marker_id = 1

    for obj in range(1, num_objects + 1):
        obj_mask = (base_labels == obj)
        obj_distance = distance_smoothed * obj_mask
        max_distance = obj_distance.max()

        # Small/narrow buildings keep 1 single seed
        if max_distance < MIN_DISTANCE:
            markers[obj_mask] = marker_id
            marker_id += 1
            continue

        # Extract deep central core seeds
        core = (obj_distance >= max_distance * CORE_RATIO).astype(np.uint8)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, core_kernel)
        core_labels, num_core = ndi.label(core)

        if num_core == 0:
            markers[obj_mask] = marker_id
            marker_id += 1
            continue

        for c in range(1, num_core + 1):
            markers[core_labels == c] = marker_id
            marker_id += 1

    # 4. Perform Watershed using ORIGINAL IMAGE GRADIENT as the flooding surface
    labels = watershed(
        image_gradient,  # Uses real image pixel edges!
        markers,
        mask=mask,
        watershed_line=True,
    )

    # 5. Extract Separation Boundaries
    output = (mask * 255).astype(np.uint8)
    boundary = (labels == 0) & (mask > 0)

    boundary = cv2.dilate(
        boundary.astype(np.uint8),
        boundary_kernel,
        iterations=1,
    )

    output[boundary > 0] = 0

    # 6. Filter Out Micro-Artifacts
    clean = np.zeros_like(output)
    cc, n = ndi.label(output > 0)

    for i in range(1, n + 1):
        inst = (cc == i).astype(np.uint8)
        if inst.sum() < MIN_INSTANCE_AREA:
            continue

        contours, _ = cv2.findContours(
            inst,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            clean,
            contours,
            -1,
            255,
            thickness=cv2.FILLED,
        )

    return (clean > 0).astype(np.uint8)


# ==========================================
# 2. INFERENCE & POST-PROCESSING
# ==========================================
def predict_unetplusplus_with_watershed(
    image_path, 
    model_path, 
    threshold=0.10, 
    device='cuda', 
    image_size=(512, 512)
):
    print("Initializing UNet++ (EfficientNet-B4) architecture...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {model_path}")

    print(f"Loading trained weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        if "epoch" in checkpoint:
            print(f"   ► Checkpoint Epoch : {checkpoint['epoch']}")
        if "val_iou" in checkpoint:
            iou = checkpoint['val_iou']
            iou_pct = iou * 100 if iou <= 1.0 else iou
            print(f"   ► Saved Val IoU    : {iou_pct:.2f}%")
        if "val_dice" in checkpoint:
            dice = checkpoint['val_dice']
            dice_pct = dice * 100 if dice <= 1.0 else dice
            print(f"   ► Saved Val Dice   : {dice_pct:.2f}%")
    else:
        state_dict = checkpoint

    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("[LOAD REPORT] Weights loaded successfully with 0 missing/unexpected keys!")
    except RuntimeError as e:
        print("\n" + "="*80)
        print("ERROR: Model state dict key mismatch detected!")
        print("="*80 + "\n")
        raise e

    model.to(device)
    model.eval()

    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    image_rgb_resized = cv2.resize(image_rgb, (image_size[1], image_size[0]))

    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    transformed = val_transform(image=image_rgb)
    tensor_img = transformed['image'].unsqueeze(0).to(device)

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type):
            logits = model(tensor_img)

        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    print(f"Probability Range -> Min: {prob_map.min():.4f} | Max: {prob_map.max():.4f}")

    raw_mask = (prob_map > threshold).astype(np.uint8)

    print("Applying Original Image-Guided Watershed Building Separation...")
    watershed_mask = apply_image_guided_watershed(raw_mask, image_rgb_resized)

    return image_rgb_resized, prob_map, raw_mask, watershed_mask


# ==========================================
# 3. VISUALIZATION FUNCTION
# ==========================================
def visualize_prediction(image_rgb, raw_mask, watershed_mask, ground_truth_mask=None, save_path=None):
    if ground_truth_mask is not None:
        if ground_truth_mask.shape[:2] != image_rgb.shape[:2]:
            ground_truth_mask = cv2.resize(
                ground_truth_mask, 
                (image_rgb.shape[1], image_rgb.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    # Red Tint Overlay
    red_mask = np.zeros_like(image_rgb)
    red_mask[watershed_mask == 1] = [255, 0, 0]

    overlay = image_rgb.copy()
    mask_boolean = (watershed_mask == 1)
    overlay[mask_boolean] = (
        0.5 * image_rgb[mask_boolean] + 0.5 * red_mask[mask_boolean]
    ).astype(np.uint8)

    fig_cols = 5 if ground_truth_mask is not None else 4
    fig, axes = plt.subplots(1, fig_cols, figsize=(5 * fig_cols, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    col_idx = 1
    if ground_truth_mask is not None:
        axes[col_idx].imshow(ground_truth_mask, cmap="gray")
        axes[col_idx].set_title("Ground Truth Mask")
        axes[col_idx].axis("off")
        col_idx += 1

    axes[col_idx].imshow(raw_mask, cmap="gray")
    axes[col_idx].set_title("Raw UNet++ Output")
    axes[col_idx].axis("off")
    col_idx += 1

    axes[col_idx].imshow(watershed_mask, cmap="gray")
    axes[col_idx].set_title("Image-Guided Watershed")
    axes[col_idx].axis("off")
    col_idx += 1

    axes[col_idx].imshow(overlay)
    axes[col_idx].set_title("Refined Overlay")
    axes[col_idx].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to: {save_path}")

    plt.show()


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    BASE_DIR = r"G:\Vishnu_Priya_BTES\Building_detection\UNET++_with_efficientnet"
    
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_unetplusplus_ori.pth")

    SAMPLE_FILENAME = "img_39.png"
    TEST_IMAGE_PATH = os.path.join(BASE_DIR, "dataset", "val1", "images", SAMPLE_FILENAME)
    TEST_MASK_PATH = os.path.join(BASE_DIR, "dataset", "val1", "masks", SAMPLE_FILENAME)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(TEST_IMAGE_PATH):
        image_rgb, prob_map, raw_mask, watershed_mask = predict_unetplusplus_with_watershed(
            image_path=TEST_IMAGE_PATH,
            model_path=CHECKPOINT_PATH,
            threshold=0.18,
            device=DEVICE,
            image_size=(512, 512)
        )

        gt_mask = cv2.imread(TEST_MASK_PATH, cv2.IMREAD_GRAYSCALE) if os.path.exists(TEST_MASK_PATH) else None

        OUTPUT_PLOT_PATH = os.path.join(BASE_DIR, "predictions", f"result_image_guided_{SAMPLE_FILENAME}")

        visualize_prediction(
            image_rgb=image_rgb,
            raw_mask=raw_mask,
            watershed_mask=watershed_mask,
            ground_truth_mask=gt_mask,
            save_path=OUTPUT_PLOT_PATH
        )
    else:
        print("Check paths! Either checkpoint or test image was not found.")