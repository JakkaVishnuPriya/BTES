from pathlib import Path
import numpy as np
import rasterio
import onnxruntime as ort
import matplotlib.pyplot as plt


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(
    r"G:\Vishnu_Priya_BTES\Building_detection\DINOV3_onnx"
)

MODEL_PATH = BASE_DIR / "models" / "model.onnx"

INPUT_FOLDER = BASE_DIR / "input"

OUTPUT_FOLDER = BASE_DIR / "output_masks"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================
# SETTINGS
# ======================================================

# Exact input image
IMAGE_NAME = "img_114.png"

# Building probability threshold
THRESHOLD = 0.08

# DINOv3 building channel
BUILDING_CHANNEL = 1


# ======================================================
# INPUT IMAGE
# ======================================================

image_path = INPUT_FOLDER / IMAGE_NAME


if not image_path.exists():

    raise FileNotFoundError(
        f"Input image not found:\n{image_path}"
    )


print("=" * 60)
print("DINOv3 ONNX BUILDING SEGMENTATION")
print("=" * 60)

print(
    "Input image :",
    image_path
)

print(
    "File type   :",
    image_path.suffix.lower()
)


# ======================================================
# LOAD IMAGE
# ======================================================

with rasterio.open(image_path) as src:

    image = src.read()

    height = src.height
    width = src.width

    input_crs = src.crs
    input_transform = src.transform


print(
    "Image Shape :",
    image.shape
)

print(
    "Data Type   :",
    image.dtype
)


# ======================================================
# CHECK IMAGE
# ======================================================

if image.ndim != 3:

    raise ValueError(
        f"Expected image with shape "
        f"(bands, height, width), "
        f"but received {image.shape}"
    )


if image.shape[0] < 3:

    raise ValueError(
        "Input image must contain at least "
        "3 bands."
    )


# ======================================================
# RGB IMAGE
# ======================================================

image_rgb = image[:3]


# ======================================================
# PREPROCESSING
# ======================================================

image_float = (
    image_rgb.astype(np.float32) / 255.0
)


# ======================================================
# DINOv3 NORMALIZATION
# ======================================================

mean = np.array(
    [
        0.4297,
        0.4002,
        0.3433
    ],
    dtype=np.float32
).reshape(3, 1, 1)


std = np.array(
    [
        0.2056,
        0.1674,
        0.1599
    ],
    dtype=np.float32
).reshape(3, 1, 1)


image_normalized = (
    image_float - mean
) / std


# ======================================================
# ADD BATCH DIMENSION
# ======================================================

model_input = np.expand_dims(
    image_normalized,
    axis=0
)


print(
    "Input Shape :",
    model_input.shape
)


# ======================================================
# LOAD ONNX MODEL
# ======================================================

print("\nLoading ONNX model...")


session = ort.InferenceSession(
    str(MODEL_PATH)
)


input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name


print(
    "Model Input  :",
    input_name
)

print(
    "Model Output :",
    output_name
)


# ======================================================
# RUN INFERENCE
# ======================================================

print("\nRunning inference...")


output = session.run(
    [output_name],
    {
        input_name: model_input
    }
)[0]


print(
    "Output Shape :",
    output.shape
)


# ======================================================
# CHECK OUTPUT
# ======================================================

if output.ndim != 4:

    raise ValueError(
        f"Unexpected model output shape: "
        f"{output.shape}"
    )


if BUILDING_CHANNEL >= output.shape[1]:

    raise ValueError(
        f"Building channel {BUILDING_CHANNEL} "
        f"does not exist. "
        f"Model has {output.shape[1]} channels."
    )


# ======================================================
# OUTPUT STATISTICS
# ======================================================

print(
    "\n========== OUTPUT STATISTICS ==========\n"
)


for i in range(output.shape[1]):

    channel = output[0, i]

    print(
        f"Channel {i}"
    )

    print(
        f" Min : {channel.min():.4f}"
    )

    print(
        f" Max : {channel.max():.4f}"
    )

    print(
        f" Mean: {channel.mean():.4f}"
    )

    print(
        f" Std : {channel.std():.4f}"
    )

    print(
        "-" * 40
    )


# ======================================================
# SIGMOID
# ======================================================

sigmoid = (
    1.0 /
    (
        1.0 +
        np.exp(-output)
    )
)


# ======================================================
# BUILDING PROBABILITY MAP
# ======================================================

building_probability = sigmoid[
    0,
    BUILDING_CHANNEL
]


# ======================================================
# BINARY BUILDING MASK
# ======================================================

binary_mask = (
    building_probability >= THRESHOLD
).astype(np.uint8)


# ======================================================
# MASK STATISTICS
# ======================================================

building_pixels = int(
    np.sum(binary_mask == 1)
)

total_pixels = int(
    binary_mask.size
)

building_percentage = (
    building_pixels /
    total_pixels
) * 100


print(
    "\n========== BINARY MASK STATISTICS =========="
)

print(
    "Building Channel     :",
    BUILDING_CHANNEL
)

print(
    "Threshold            :",
    THRESHOLD
)

print(
    "Building Pixels      :",
    building_pixels
)

print(
    "Total Pixels         :",
    total_pixels
)

print(
    "Building Coverage    :",
    f"{building_percentage:.2f}%"
)


# ======================================================
# DELETE OLD OUTPUT FILE
# ======================================================

def remove_old_file(path):

    if path.exists():

        try:

            path.unlink()

            print(
                "Removed old file:",
                path.name
            )

        except PermissionError:

            raise PermissionError(
                f"\nCannot delete:\n{path}\n\n"
                "The file may be open in QGIS, "
                "ArcGIS, Windows Photos, or another "
                "application.\n"
                "Close the file and run the script again."
            )


# ======================================================
# OUTPUT FILE PATHS
# ======================================================

binary_mask_png = (
    OUTPUT_FOLDER /
    "img_114_binary_mask.png"
)

binary_mask_tif = (
    OUTPUT_FOLDER /
    "img_114_binary_mask.tif"
)

building_probability_png = (
    OUTPUT_FOLDER /
    "img_114_building_probability.png"
)

building_probability_tif = (
    OUTPUT_FOLDER /
    "img_114_building_probability.tif"
)

overlay_path = (
    OUTPUT_FOLDER /
    "img_114_overlay.png"
)

visualization_path = (
    OUTPUT_FOLDER /
    "img_114_visualization.png"
)


# ======================================================
# REMOVE OLD TIFF FILES
# ======================================================

print(
    "\nChecking old output files..."
)


remove_old_file(
    binary_mask_tif
)

remove_old_file(
    building_probability_tif
)


# ======================================================
# SAVE RAW LOGITS
# ======================================================

print(
    "\nSaving raw logits..."
)


for i in range(output.shape[1]):

    output_path = (
        OUTPUT_FOLDER /
        f"img_114_logits_{i}.png"
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        output[0, i],
        cmap="viridis"
    )

    plt.title(
        f"Raw Logits Channel {i}"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ======================================================
# SAVE SIGMOID MAPS
# ======================================================

print(
    "Saving sigmoid maps..."
)


for i in range(sigmoid.shape[1]):

    output_path = (
        OUTPUT_FOLDER /
        f"img_114_sigmoid_{i}.png"
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        sigmoid[0, i],
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.title(
        f"Sigmoid Channel {i}"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ======================================================
# SAVE BINARY MASK PNG
# ======================================================

print(
    "Saving binary mask PNG..."
)


plt.imsave(
    binary_mask_png,
    binary_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)


# ======================================================
# CREATE CLEAN GEOTIFF PROFILE
# ======================================================

def create_geotiff_profile(
    dtype
):

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "compress": "lzw",
        "photometric": "MINISBLACK"
    }

    # Add CRS only when available
    if input_crs is not None:

        profile["crs"] = input_crs

    # Add transform only when available
    if input_transform is not None:

        profile["transform"] = input_transform

    return profile


# ======================================================
# SAVE BINARY MASK TIFF
# ======================================================

print(
    "Saving binary mask TIFF..."
)


mask_profile = create_geotiff_profile(
    "uint8"
)


with rasterio.open(
    binary_mask_tif,
    "w",
    **mask_profile
) as dst:

    dst.write(
        binary_mask,
        1
    )


print(
    "Binary mask TIFF saved successfully."
)


# ======================================================
# SAVE BUILDING PROBABILITY TIFF
# ======================================================

print(
    "Saving building probability TIFF..."
)


probability_profile = create_geotiff_profile(
    "float32"
)


with rasterio.open(
    building_probability_tif,
    "w",
    **probability_profile
) as dst:

    dst.write(
        building_probability.astype(
            np.float32
        ),
        1
    )


print(
    "Building probability TIFF saved successfully."
)


# ======================================================
# SAVE BUILDING PROBABILITY PNG
# ======================================================

print(
    "Saving building probability PNG..."
)


plt.imsave(
    building_probability_png,
    building_probability,
    cmap="viridis",
    vmin=0,
    vmax=1
)


# ======================================================
# PREPARE ORIGINAL IMAGE FOR DISPLAY
# ======================================================

original_image = np.transpose(
    image_rgb,
    (1, 2, 0)
)


# ======================================================
# NORMALIZE DISPLAY IMAGE
# ======================================================

if original_image.dtype == np.uint8:

    display_image = (
        original_image.astype(
            np.float32
        ) / 255.0
    )

else:

    original_min = float(
        original_image.min()
    )

    original_max = float(
        original_image.max()
    )

    if original_max > original_min:

        display_image = (
            original_image -
            original_min
        ) / (
            original_max -
            original_min
        )

    else:

        display_image = np.zeros_like(
            original_image,
            dtype=np.float32
        )


display_image = np.clip(
    display_image,
    0,
    1
)


# ======================================================
# CREATE RED BUILDING OVERLAY
# ======================================================

overlay = display_image.copy()


colored_mask = np.zeros_like(
    overlay
)


# Building pixels = RED

colored_mask[
    binary_mask == 1
] = [
    1.0,
    0.0,
    0.0
]


# Overlay transparency

alpha = 0.45


overlay = (
    overlay * (1 - alpha)
    +
    colored_mask * alpha
)


overlay = np.clip(
    overlay,
    0,
    1
)


# ======================================================
# SAVE OVERLAY
# ======================================================

print(
    "Saving overlay..."
)


plt.imsave(
    overlay_path,
    overlay
)


# ======================================================
# COMPLETE VISUALIZATION
# ======================================================

print(
    "Creating complete visualization..."
)


fig, axes = plt.subplots(
    1,
    4,
    figsize=(24, 6)
)


# ------------------------------------------------------
# ORIGINAL IMAGE
# ------------------------------------------------------

axes[0].imshow(
    display_image
)

axes[0].set_title(
    "Original Image",
    fontsize=14
)

axes[0].axis("off")


# ------------------------------------------------------
# PROBABILITY MAP
# ------------------------------------------------------

im = axes[1].imshow(
    building_probability,
    cmap="viridis",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "Building Probability",
    fontsize=14
)

axes[1].axis("off")


plt.colorbar(
    im,
    ax=axes[1],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------
# BINARY MASK
# ------------------------------------------------------

axes[2].imshow(
    binary_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    f"Binary Building Mask\n"
    f"Threshold = {THRESHOLD}",
    fontsize=14
)

axes[2].axis("off")


# ------------------------------------------------------
# OVERLAY
# ------------------------------------------------------

axes[3].imshow(
    overlay
)

axes[3].set_title(
    "Building Prediction Overlay",
    fontsize=14
)

axes[3].axis("off")


# ======================================================
# MAIN TITLE
# ======================================================

fig.suptitle(
    "DINOv3 Building Segmentation - img_114",
    fontsize=16
)


plt.tight_layout()


# ======================================================
# SAVE COMPLETE VISUALIZATION
# ======================================================

plt.savefig(
    visualization_path,
    dpi=200,
    bbox_inches="tight"
)


print(
    "Visualization saved successfully."
)


# Show result

plt.show()

plt.close()


# ======================================================
# VERIFY OUTPUT FILES
# ======================================================

print(
    "\nChecking generated files..."
)


output_files = [
    binary_mask_png,
    binary_mask_tif,
    building_probability_png,
    building_probability_tif,
    overlay_path,
    visualization_path
]


for file_path in output_files:

    if file_path.exists():

        print(
            "[OK] ",
            file_path.name
        )

    else:

        print(
            "[FAILED] ",
            file_path.name
        )


# ======================================================
# FINAL
# ======================================================

print(
    "\n" + "=" * 60
)

print(
    "PROCESSING COMPLETE"
)

print(
    "=" * 60
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_FOLDER
)

print(
    "\nBinary mask:"
)

print(
    binary_mask_tif
)

print(
    "\nOverlay:"
)

print(
    overlay_path
)

print(
    "\nVisualization:"
)

print(
    visualization_path
)

print(
    "\nFinished successfully."
)