import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# ============================================================
# 1. REPRODUCIBILITY + GPU
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# 2. PATHS
# ============================================================

# Dataset paths
IMAGE_DIR = r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_efficientnet_b4\dataset\images"
MASK_DIR = r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_efficientnet_b4\dataset\masks"

# All model/checkpoint files are saved here
CHECKPOINT_DIR = r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_efficientnet_b4\checkpoints"

# Starting weights requested by the user.
# These are used only when no interrupted-training checkpoint exists.
INITIAL_WEIGHTS_PATH = (
    r"G:\Vishnu_Priya_BTES\Tree_Detection\UNet++_with_efficientnet_b4\pretrain\best_tree_unetplusplus_1708.pth"
)

# Saved after EVERY COMPLETED EPOCH.
# Tomorrow, this checkpoint takes priority over INITIAL_WEIGHTS_PATH.
LAST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "last_tree_unetplusplus_checkpoint.pth"
)

# Best model path requested by the user.
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_tree_unetplusplus.pth"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

BATCH_SIZE = 32

# Total epochs for the complete training run.
# Yesterday's completed epochs are included in this total.
NUM_EPOCHS = 100

# Resume behavior:
# 1. If last_tree_unetplusplus_checkpoint.pth exists,
#    resume from the last COMPLETED epoch.
# 2. Otherwise load INITIAL_WEIGHTS_PATH.
RESUME_TRAINING = True

PATIENCE = 10

VAL_RATIO = 0.15

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-3

THRESHOLD = 0.40

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("TREE CANOPY SEMANTIC SEGMENTATION - UNET++")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Dataset      : {os.path.dirname(IMAGE_DIR)}")
print(f"Images       : {IMAGE_DIR}")
print(f"Masks        : {MASK_DIR}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Validation   : {VAL_RATIO * 100:.0f}%")
print(f"Initial Wts  : {INITIAL_WEIGHTS_PATH}")
print(f"Last Checkpt : {LAST_CHECKPOINT_PATH}")
print(f"Best Model   : {BEST_MODEL_PATH}")
print("=" * 70)


# ============================================================
# 4. FIND IMAGE/MASK PAIRS
# ============================================================

def get_png_files(folder):

    files = glob.glob(
        os.path.join(
            folder,
            "*.png"
        )
    )

    return sorted(files)


def create_pairs():

    image_paths = get_png_files(
        IMAGE_DIR
    )

    mask_paths = get_png_files(
        MASK_DIR
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            f"No PNG images found in:\n"
            f"{IMAGE_DIR}"
        )

    if len(mask_paths) == 0:

        raise RuntimeError(
            f"No PNG masks found in:\n"
            f"{MASK_DIR}"
        )

    print()
    print(
        f"Images found : {len(image_paths)}"
    )

    print(
        f"Masks found  : {len(mask_paths)}"
    )

    # --------------------------------------------------------
    # IMAGE / MASK NAMING
    #
    # Images:
    #     img_1.png
    #     img_10.png
    #     img_100.png
    #
    # Masks:
    #     img_1_mask.png
    #     img_10_mask.png
    #     img_100_mask.png
    #
    # Therefore, remove "_mask" from the mask filename when
    # creating the lookup key.
    # --------------------------------------------------------

    mask_lookup = {}

    for mask_path in mask_paths:

        mask_filename = os.path.basename(
            mask_path
        )

        mask_stem, mask_ext = os.path.splitext(
            mask_filename
        )

        if mask_stem.endswith("_mask"):

            image_stem = mask_stem[:-5]

        else:

            # Also support masks whose filename is exactly
            # the same as the image filename.
            image_stem = mask_stem

        image_filename = (
            image_stem + mask_ext
        )

        mask_lookup[image_filename] = mask_path

    pairs = []

    missing_masks = []

    for image_path in image_paths:

        filename = os.path.basename(
            image_path
        )

        if filename in mask_lookup:

            pairs.append(
                (
                    image_path,
                    mask_lookup[filename]
                )
            )

        else:

            missing_masks.append(
                filename
            )

    print(
        f"Matched pairs: {len(pairs)}"
    )

    if len(pairs) > 0:

        print()
        print("Example image-mask pairs:")

        for image_path, mask_path in pairs[:5]:

            print(
                f"    {os.path.basename(image_path)}"
                f"  <->  "
                f"{os.path.basename(mask_path)}"
            )

    if missing_masks:

        print()
        print(
            f"WARNING: {len(missing_masks)} "
            f"images have no matching mask."
        )

        print(
            "First 10 missing masks:"
        )

        for filename in missing_masks[:10]:

            print(
                "   ",
                filename
            )

    if len(pairs) == 0:

        raise RuntimeError(
            "No matching image-mask pairs found."
        )

    return pairs


# ============================================================
# 5. TREE CANOPY DATASET
# ============================================================

class TreeCanopyDataset(Dataset):

    def __init__(
        self,
        pairs,
        transform=None
    ):

        self.pairs = pairs

        self.transform = transform


    def __len__(self):

        return len(
            self.pairs
        )


    def __getitem__(self, index):

        image_path, mask_path = (
            self.pairs[index]
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise ValueError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise ValueError(
                f"Could not read mask:\n"
                f"{mask_path}"
            )


        # ----------------------------------------------------
        # MASK:
        #
        # Saved PNG:
        #
        # 0   = background
        # 255 = tree
        #
        # Training:
        #
        # 0.0 = background
        # 1.0 = tree
        # ----------------------------------------------------

        mask = (
            mask > 0
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------

        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented[
                "image"
            ]

            mask = augmented[
                "mask"
            ]


        # ----------------------------------------------------
        # MASK SHAPE
        #
        # Required:
        #
        # [1, H, W]
        # ----------------------------------------------------

        if mask.ndim == 2:

            mask = mask.unsqueeze(
                0
            )

        elif mask.ndim == 3:

            if mask.shape[-1] == 1:

                mask = mask.permute(
                    2,
                    0,
                    1
                )

        mask = mask.float()


        return image, mask


# ============================================================
# 6. AUGMENTATIONS
# ============================================================

def get_transforms():

    train_transform = A.Compose([

        A.HorizontalFlip(
            p=0.5
        ),

        A.VerticalFlip(
            p=0.5
        ),

        A.RandomRotate90(
            p=0.5
        ),

        A.Affine(
            scale=(
                0.90,
                1.10
            ),
            translate_percent=(
                -0.0625,
                0.0625
            ),
            rotate=(
                -30,
                30
            ),
            p=0.5
        ),

        A.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05,
            p=0.30
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

    ])


    val_transform = A.Compose([

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

    ])


    return (
        train_transform,
        val_transform
    )


# ============================================================
# 7. TREE CANOPY LOSS
# ============================================================

class TreeCanopyLoss(nn.Module):

    def __init__(
        self,
        edge_weight=2.0,
        smooth=1e-6
    ):

        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss(
                reduction="none"
            )
        )

        self.edge_weight = (
            edge_weight
        )

        self.smooth = smooth

        self.register_buffer(
            "kernel",
            torch.ones(
                (1, 1, 3, 3),
                dtype=torch.float32
            )
        )


    def forward(
        self,
        logits,
        targets
    ):

        # ----------------------------------------------------
        # BCE
        # ----------------------------------------------------

        bce_loss = self.bce(
            logits,
            targets
        )


        # ----------------------------------------------------
        # EDGE WEIGHTING
        # ----------------------------------------------------

        kernel = self.kernel.to(
            targets.device
        )

        with torch.no_grad():

            dilated = torch.clamp(

                F.conv2d(
                    targets,
                    kernel,
                    padding=1
                ),

                0,
                1
            )

            eroded = (
                1.0
                -
                torch.clamp(

                    F.conv2d(
                        1.0 - targets,
                        kernel,
                        padding=1
                    ),

                    0,
                    1
                )
            )

            edge_mask = (
                dilated
                -
                eroded
            )


        weights = (
            1.0
            +
            self.edge_weight
            *
            edge_mask
        )


        weighted_bce = (
            bce_loss
            *
            weights
        ).mean()


        # ----------------------------------------------------
        # DICE
        # ----------------------------------------------------

        probs = torch.sigmoid(
            logits
        )

        probs_flat = (
            probs.reshape(-1)
        )

        targets_flat = (
            targets.reshape(-1)
        )

        intersection = (
            probs_flat
            *
            targets_flat
        ).sum()


        dice = (

            2.0
            *
            intersection
            +
            self.smooth

        ) / (

            probs_flat.sum()
            +
            targets_flat.sum()
            +
            self.smooth

        )


        dice_loss = (
            1.0 - dice
        )


        # ----------------------------------------------------
        # FINAL LOSS
        # ----------------------------------------------------

        return (
            0.5 * weighted_bce
            +
            0.5 * dice_loss
        )


# ============================================================
# 8. METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    threshold=0.40,
    smooth=1e-6
):

    probs = torch.sigmoid(
        logits
    )

    predictions = (
        probs > threshold
    ).float()


    intersection = (
        predictions
        *
        targets
    ).sum()


    prediction_sum = (
        predictions.sum()
    )

    target_sum = (
        targets.sum()
    )


    union = (
        prediction_sum
        +
        target_sum
        -
        intersection
    )


    iou = (
        intersection
        +
        smooth
    ) / (
        union
        +
        smooth
    )


    dice = (

        2.0
        *
        intersection
        +
        smooth

    ) / (

        prediction_sum
        +
        target_sum
        +
        smooth

    )


    correct = (
        predictions == targets
    ).sum()


    total = targets.numel()


    accuracy = (
        correct.float()
        /
        total
    )


    return (
        iou.item(),
        dice.item(),
        accuracy.item()
    )


# ============================================================
# 9. TRAINING FUNCTION
# ============================================================

def train_model(

    model,

    train_loader,

    val_loader,

    criterion,

    optimizer,

    scheduler,

    scaler,

    save_dir,

    num_epochs,

    patience,

    device

):

    # ============================================================
    # RESUME FROM LAST COMPLETED EPOCH / INITIAL WEIGHTS
    # ============================================================

    start_epoch = 0
    best_val_iou = 0.0
    early_stop_counter = 0

    # History is stored inside the last-epoch checkpoint so it
    # survives stopping and restarting the training tomorrow.
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_acc": [],
        "val_acc": []
    }

    best_model_path = BEST_MODEL_PATH

    # ============================================================
    # 1. RESUME FROM LAST COMPLETED EPOCH
    # ============================================================

    if RESUME_TRAINING and os.path.exists(LAST_CHECKPOINT_PATH):

        print()
        print("=" * 70)
        print("RESUMING FROM LAST COMPLETED EPOCH")
        print("=" * 70)
        print(f"Checkpoint: {LAST_CHECKPOINT_PATH}")

        checkpoint = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if (
            scaler is not None
            and checkpoint.get("scaler_state_dict") is not None
        ):
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = int(
            checkpoint.get("epoch", 0)
        )

        best_val_iou = float(
            checkpoint.get(
                "best_val_iou",
                checkpoint.get("val_iou", 0.0)
            )
        )

        early_stop_counter = int(
            checkpoint.get("early_stop_counter", 0)
        )

        saved_history = checkpoint.get("history")

        if isinstance(saved_history, dict):
            history = saved_history

        print(
            f"Last completed epoch : {start_epoch}"
        )
        print(
            f"Best Val IoU         : "
            f"{best_val_iou * 100:.2f}%"
        )
        print(
            f"Continuing from      : Epoch {start_epoch + 1}"
        )
        print(
            f"Training until       : Epoch {num_epochs}"
        )
        print("=" * 70)

    # ============================================================
    # 2. OTHERWISE LOAD THE USER-SPECIFIED STARTING WEIGHTS
    # ============================================================

    elif RESUME_TRAINING and os.path.exists(INITIAL_WEIGHTS_PATH):

        print()
        print("=" * 70)
        print("STARTING FROM SPECIFIED WEIGHTS")
        print("=" * 70)
        print(f"Initial weights: {INITIAL_WEIGHTS_PATH}")

        initial_checkpoint = torch.load(
            INITIAL_WEIGHTS_PATH,
            map_location=device
        )

        # Supports both a complete checkpoint and a plain state_dict.
        if "model_state_dict" in initial_checkpoint:

            model.load_state_dict(
                initial_checkpoint["model_state_dict"]
            )

            # If the supplied checkpoint contains an epoch,
            # continue counting from that epoch.
            start_epoch = int(
                initial_checkpoint.get("epoch", 0)
            )

            best_val_iou = float(
                initial_checkpoint.get(
                    "best_val_iou",
                    initial_checkpoint.get("val_iou", 0.0)
                )
            )

        else:

            model.load_state_dict(
                initial_checkpoint
            )

        print("Initial model weights loaded successfully.")
        print(
            f"Starting from epoch {start_epoch + 1}"
        )
        print(
            f"Training until epoch {num_epochs}"
        )
        print("=" * 70)

    else:

        print()
        print("=" * 70)
        print("NO CHECKPOINT FOUND")
        print("=" * 70)
        print(
            "Training starts from randomly initialized weights."
        )
        print("=" * 70)

    print()
    print("Starting training...")

    if start_epoch >= num_epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch}. "
            f"NUM_EPOCHS is {num_epochs}."
        )
        print("No additional training required.")
        return history

    for epoch in range(
        start_epoch,
        num_epochs
    ):


        # ====================================================
        # TRAIN
        # ====================================================

        model.train()


        train_loss = 0.0

        train_iou = 0.0

        train_dice = 0.0

        train_acc = 0.0

        train_batches = 0


        progress = tqdm(

            train_loader,

            desc=(

                f"Epoch "
                f"{epoch + 1:02d}/"
                f"{num_epochs:02d} [Train]"

            )

        )


        for images, masks in progress:


            images = images.to(

                device,

                non_blocking=True

            )


            masks = masks.to(

                device,

                non_blocking=True

            )


            optimizer.zero_grad(
                set_to_none=True
            )


            # ------------------------------------------------
            # GPU mixed precision
            # ------------------------------------------------

            if device == "cuda":

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    logits = model(
                        images
                    )

                    loss = criterion(
                        logits,
                        masks
                    )

            else:

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    masks
                )


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            if device == "cuda":

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                optimizer.step()


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            with torch.no_grad():

                iou, dice, acc = (
                    calculate_metrics(
                        logits,
                        masks,
                        THRESHOLD
                    )
                )


            train_loss += (
                loss.item()
            )

            train_iou += iou

            train_dice += dice

            train_acc += acc

            train_batches += 1


            progress.set_postfix(

                loss=(
                    f"{loss.item():.4f}"
                ),

                iou=(
                    f"{iou * 100:.2f}%"
                )

            )


        avg_train_loss = (
            train_loss
            /
            max(train_batches, 1)
        )


        avg_train_iou = (
            train_iou
            /
            max(train_batches, 1)
        )


        avg_train_dice = (
            train_dice
            /
            max(train_batches, 1)
        )


        avg_train_acc = (
            train_acc
            /
            max(train_batches, 1)
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        val_loss = 0.0

        val_iou = 0.0

        val_dice = 0.0

        val_acc = 0.0

        val_batches = 0


        with torch.no_grad():

            progress = tqdm(

                val_loader,

                desc=(

                    f"Epoch "
                    f"{epoch + 1:02d}/"
                    f"{num_epochs:02d} [Val]"

                )

            )


            for images, masks in progress:


                images = images.to(

                    device,

                    non_blocking=True

                )


                masks = masks.to(

                    device,

                    non_blocking=True

                )


                if device == "cuda":

                    with torch.amp.autocast(
                        device_type="cuda"
                    ):

                        logits = model(
                            images
                        )

                        loss = criterion(
                            logits,
                            masks
                        )

                else:

                    logits = model(
                        images
                    )

                    loss = criterion(
                        logits,
                        masks
                    )


                iou, dice, acc = (
                    calculate_metrics(
                        logits,
                        masks,
                        THRESHOLD
                    )
                )


                val_loss += (
                    loss.item()
                )

                val_iou += iou

                val_dice += dice

                val_acc += acc

                val_batches += 1


        avg_val_loss = (
            val_loss
            /
            max(val_batches, 1)
        )


        avg_val_iou = (
            val_iou
            /
            max(val_batches, 1)
        )


        avg_val_dice = (
            val_dice
            /
            max(val_batches, 1)
        )


        avg_val_acc = (
            val_acc
            /
            max(val_batches, 1)
        )


        scheduler.step()


        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch + 1}/{num_epochs}"
        )

        print(
            f"Train Loss : "
            f"{avg_train_loss:.4f}"
        )

        print(
            f"Train IoU  : "
            f"{avg_train_iou * 100:.2f}%"
        )

        print(
            f"Train Dice : "
            f"{avg_train_dice * 100:.2f}%"
        )

        print(
            f"Train Acc  : "
            f"{avg_train_acc * 100:.2f}%"
        )

        print("-" * 70)

        print(
            f"Val Loss   : "
            f"{avg_val_loss:.4f}"
        )

        print(
            f"Val IoU    : "
            f"{avg_val_iou * 100:.2f}%"
        )

        print(
            f"Val Dice   : "
            f"{avg_val_dice * 100:.2f}%"
        )

        print(
            f"Val Acc    : "
            f"{avg_val_acc * 100:.2f}%"
        )

        print("=" * 70)

        # ====================================================
        # SAVE HISTORY FOR TRAINING/VALIDATION GRAPHS
        # ====================================================

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_iou"].append(avg_train_iou)
        history["val_iou"].append(avg_val_iou)
        history["train_dice"].append(avg_train_dice)
        history["val_dice"].append(avg_val_dice)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)


        # ====================================================
        # BEST MODEL
        # ====================================================

        if avg_val_iou > best_val_iou:

            improvement = (
                avg_val_iou
                -
                best_val_iou
            )


            best_val_iou = (
                avg_val_iou
            )


            early_stop_counter = 0


            checkpoint = {

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "scaler_state_dict":
                    scaler.state_dict() if scaler is not None else None,

                "val_loss":
                    avg_val_loss,

                "val_iou":
                    avg_val_iou,

                "val_dice":
                    avg_val_dice,

                "val_accuracy":
                    avg_val_acc,

                "history":
                    history

            }


            torch.save(

                checkpoint,

                best_model_path

            )


            print()
            print(
                "🎉 NEW BEST TREE MODEL SAVED!"
            )

            print(
                f"Val IoU: "
                f"{avg_val_iou * 100:.2f}%"
            )

            print(
                f"Val Dice: "
                f"{avg_val_dice * 100:.2f}%"
            )

            print(
                f"Saved to:"
            )

            print(
                best_model_path
            )


        else:

            early_stop_counter += 1


            print()
            print(

                f"No improvement: "
                f"{early_stop_counter}/"
                f"{patience}"

            )


            if (
                early_stop_counter
                >=
                patience
            ):

                print()
                print(
                    "Early stopping triggered."
                )


        # ========================================================
        # SAVE LAST COMPLETED EPOCH
        # ========================================================
        # This is saved after every completed epoch, regardless
        # of whether validation IoU improved.
        #
        # Therefore, if training is stopped today, tomorrow it
        # resumes from the actual last completed epoch instead
        # of going back to an older BEST model.
        latest_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":
                scaler.state_dict() if scaler is not None else None,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_iou": avg_train_iou,
            "val_iou": avg_val_iou,
            "train_dice": avg_train_dice,
            "val_dice": avg_val_dice,
            "train_accuracy": avg_train_acc,
            "val_accuracy": avg_val_acc,
            "best_val_iou": best_val_iou,
            "early_stop_counter": early_stop_counter,
            "history": history
        }

        torch.save(
            latest_checkpoint,
            LAST_CHECKPOINT_PATH
        )

        print(
            f"Last-epoch checkpoint saved: "
            f"{LAST_CHECKPOINT_PATH}"
        )

        # Stop after saving the checkpoint if patience was reached.
        if early_stop_counter >= patience:
            break


    return history


# ============================================================
# 10. PLOT TRAINING HISTORY
# ============================================================

def plot_training_history(history, save_dir):

    if not history or len(history.get("epoch", [])) == 0:
        print("No training history available for plotting.")
        return

    epochs = history["epoch"]

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_validation_loss.png"),
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        np.array(history["train_iou"]) * 100,
        label="Train IoU"
    )
    plt.plot(
        epochs,
        np.array(history["val_iou"]) * 100,
        label="Validation IoU"
    )
    plt.xlabel("Epoch")
    plt.ylabel("IoU (%)")
    plt.title("Training vs Validation IoU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_validation_iou.png"),
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # DICE
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        np.array(history["train_dice"]) * 100,
        label="Train Dice"
    )
    plt.plot(
        epochs,
        np.array(history["val_dice"]) * 100,
        label="Validation Dice"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Dice (%)")
    plt.title("Training vs Validation Dice")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_validation_dice.png"),
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        np.array(history["train_acc"]) * 100,
        label="Train Accuracy"
    )
    plt.plot(
        epochs,
        np.array(history["val_acc"]) * 100,
        label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_validation_accuracy.png"),
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # IoU + DICE COMBINED
    # --------------------------------------------------------
    plt.figure(figsize=(11, 7))
    plt.plot(
        epochs,
        np.array(history["train_iou"]) * 100,
        label="Train IoU"
    )
    plt.plot(
        epochs,
        np.array(history["val_iou"]) * 100,
        label="Validation IoU"
    )
    plt.plot(
        epochs,
        np.array(history["train_dice"]) * 100,
        label="Train Dice"
    )
    plt.plot(
        epochs,
        np.array(history["val_dice"]) * 100,
        label="Validation Dice"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_validation_metrics.png"),
        dpi=200
    )
    plt.close()

    print()
    print("=" * 70)
    print("TRAINING GRAPHS SAVED")
    print("=" * 70)
    print(
        os.path.join(save_dir, "training_validation_loss.png")
    )
    print(
        os.path.join(save_dir, "training_validation_iou.png")
    )
    print(
        os.path.join(save_dir, "training_validation_dice.png")
    )
    print(
        os.path.join(save_dir, "training_validation_accuracy.png")
    )
    print(
        os.path.join(save_dir, "training_validation_metrics.png")
    )
    print("=" * 70)


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------
    # Find all image-mask pairs
    # --------------------------------------------------------

    all_pairs = (
        create_pairs()
    )


    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.Random(
        SEED
    ).shuffle(
        all_pairs
    )


    # --------------------------------------------------------
    # 85% TRAIN / 15% VALIDATION
    # --------------------------------------------------------

    total = len(
        all_pairs
    )


    val_count = max(

        1,

        int(
            total
            *
            VAL_RATIO
        )

    )


    train_count = (
        total
        -
        val_count
    )


    train_pairs = (
        all_pairs[
            :train_count
        ]
    )


    val_pairs = (
        all_pairs[
            train_count:
        ]
    )


    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Total images : {total}"
    )

    print(
        f"Training     : "
        f"{len(train_pairs)}"
    )

    print(
        f"Validation   : "
        f"{len(val_pairs)}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform, val_transform = (
        get_transforms()
    )


    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = (
        TreeCanopyDataset(

            train_pairs,

            transform=train_transform

        )
    )


    val_dataset = (
        TreeCanopyDataset(

            val_pairs,

            transform=val_transform

        )
    )


    # --------------------------------------------------------
    # DataLoaders
    #
    # num_workers=0 is intentional for Windows stability.
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=(
            DEVICE == "cuda"
        ),

        drop_last=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            DEVICE == "cuda"
        )

    )


    # --------------------------------------------------------
    # UNET++
    # --------------------------------------------------------

    print()
    print(
        "Building UNet++..."
    )


    model = smp.UnetPlusPlus(

        encoder_name="efficientnet-b4",

        # Start without downloading weights.
        # We can add verified pretrained weights later.

        encoder_weights=None,

        in_channels=3,

        classes=1

    ).to(DEVICE)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = (
        TreeCanopyLoss(
            edge_weight=2.0
        ).to(DEVICE)
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = (

        torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=NUM_EPOCHS,

            eta_min=1e-6

        )

    )


    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    if DEVICE == "cuda":

        scaler = (
            torch.amp.GradScaler(
                "cuda"
            )
        )

    else:

        scaler = None


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = train_model(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        scaler=scaler,

        save_dir=CHECKPOINT_DIR,

        num_epochs=NUM_EPOCHS,

        patience=PATIENCE,

        device=DEVICE

    )


    # Generate graphs after training completes.
    plot_training_history(
        history,
        CHECKPOINT_DIR
    )


    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)