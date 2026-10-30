import os
import random
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = Path(r"G:\Segformer_buildings\Tree_Label\dataset\data\train")
VAL_DIR = Path(r"G:\Segformer_buildings\Tree_Label\dataset\data\val")

# Move 30% of IMAGE + MASK pairs
VAL_PERCENT = 0.30

# Set a fixed seed so the same random split can be reproduced.
RANDOM_SEED = 42

# Your images/masks are PNG, but these are also supported.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ============================================================
# FUNCTIONS
# ============================================================

def find_mask(train_dir: Path, image_file: Path):
    """
    For:
        img_123.png

    find:
        img_123_mask.png
    """

    base = image_file.stem

    possible_masks = [
        train_dir / f"{base}_mask.png",
        train_dir / f"{base}_mask.jpg",
        train_dir / f"{base}_mask.jpeg",
        train_dir / f"{base}_mask.tif",
        train_dir / f"{base}_mask.tiff",
    ]

    for mask_file in possible_masks:
        if mask_file.is_file():
            return mask_file

    return None


def main():

    print("=" * 70)
    print("RANDOM 30% TRAIN -> VALIDATION SPLIT")
    print("=" * 70)

    print()
    print("Train folder:")
    print(TRAIN_DIR)

    print()
    print("Validation folder:")
    print(VAL_DIR)

    # ========================================================
    # CHECK TRAIN FOLDER
    # ========================================================

    if not TRAIN_DIR.exists():
        print()
        print("ERROR: Train folder does not exist!")
        print(TRAIN_DIR)
        print()
        print("Please check the path.")
        return

    if not TRAIN_DIR.is_dir():
        print()
        print("ERROR: Train path is not a folder!")
        print(TRAIN_DIR)
        return

    # ========================================================
    # CREATE VAL FOLDER
    # ========================================================

    VAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # FIND IMAGE FILES
    # ========================================================

    all_files = [
        f for f in TRAIN_DIR.iterdir()
        if f.is_file()
    ]

    image_files = []

    for file in all_files:

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Never treat a mask as an image.
        if file.stem.lower().endswith("_mask"):
            continue

        image_files.append(file)

    image_files.sort(
        key=lambda x: x.name.lower()
    )

    print()
    print("Images found in train:", len(image_files))

    if len(image_files) == 0:
        print()
        print("ERROR: No image files found.")
        return

    # ========================================================
    # BUILD VALID IMAGE/MASK PAIRS
    # ========================================================

    pairs = []
    missing_masks = []

    for image_file in image_files:

        mask_file = find_mask(
            TRAIN_DIR,
            image_file
        )

        if mask_file is None:
            missing_masks.append(
                image_file.name
            )
            continue

        pairs.append(
            (
                image_file,
                mask_file
            )
        )

    print(
        "Complete image/mask pairs:",
        len(pairs)
    )

    print(
        "Images without masks:",
        len(missing_masks)
    )

    # ========================================================
    # STOP IF NO PAIRS
    # ========================================================

    if len(pairs) == 0:
        print()
        print("ERROR: No complete image/mask pairs found.")
        return

    # ========================================================
    # RANDOMIZE
    # ========================================================

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        pairs
    )

    # ========================================================
    # CALCULATE 30%
    # ========================================================

    move_count = int(
        len(pairs) * VAL_PERCENT
    )

    # Make sure at least one pair is moved when pairs exist.
    if move_count < 1:
        move_count = 1

    selected_pairs = pairs[
        :move_count
    ]

    print()
    print(
        "Validation percentage:",
        f"{VAL_PERCENT * 100:.0f}%"
    )

    print(
        "Pairs to move:",
        move_count
    )

    # ========================================================
    # MOVE PAIRS
    # ========================================================

    moved = 0
    skipped = 0

    print()
    print("Moving pairs...")
    print("-" * 70)

    for image_file, mask_file in selected_pairs:

        image_destination = (
            VAL_DIR / image_file.name
        )

        mask_destination = (
            VAL_DIR / mask_file.name
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if image_destination.exists():
            print(
                "SKIP - image already exists in val:",
                image_file.name
            )
            skipped += 1
            continue

        if mask_destination.exists():
            print(
                "SKIP - mask already exists in val:",
                mask_file.name
            )
            skipped += 1
            continue

        try:

            # Move image
            shutil.move(
                str(image_file),
                str(image_destination)
            )

            # Move corresponding mask
            shutil.move(
                str(mask_file),
                str(mask_destination)
            )

            moved += 1

            if moved <= 10 or moved % 50 == 0:
                print(
                    f"Moved {moved}: "
                    f"{image_file.name} + "
                    f"{mask_file.name}"
                )

        except Exception as e:

            print()
            print(
                "ERROR while moving:",
                image_file.name
            )

            print(
                "Reason:",
                str(e)
            )

            # If image moved but mask failed,
            # try to put image back into train.
            if (
                image_destination.exists()
                and
                not image_file.exists()
            ):

                try:
                    shutil.move(
                        str(image_destination),
                        str(image_file)
                    )
                except Exception:
                    pass

            skipped += 1

    # ========================================================
    # FINAL COUNTS
    # ========================================================

    remaining_images = [
        f for f in TRAIN_DIR.iterdir()
        if (
            f.is_file()
            and
            f.suffix.lower() in IMAGE_EXTENSIONS
            and
            not f.stem.lower().endswith("_mask")
        )
    ]

    val_images = [
        f for f in VAL_DIR.iterdir()
        if (
            f.is_file()
            and
            f.suffix.lower() in IMAGE_EXTENSIONS
            and
            not f.stem.lower().endswith("_mask")
        )
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print()
    print("Pairs originally available :", len(pairs))
    print("Requested 30%              :", move_count)
    print("Pairs actually moved       :", moved)
    print("Pairs skipped               :", skipped)
    print("Missing masks               :", len(missing_masks))

    print()
    print("Images remaining in TRAIN  :", len(remaining_images))
    print("Images in VAL              :", len(val_images))

    print()
    print("TRAIN:")
    print(TRAIN_DIR)

    print()
    print("VAL:")
    print(VAL_DIR)

    # ========================================================
    # MISSING MASK REPORT
    # ========================================================

    if missing_masks:

        print()
        print("=" * 70)
        print("WARNING: IMAGES WITHOUT MASKS")
        print("=" * 70)

        for name in missing_masks[:30]:
            print("  ", name)

        if len(missing_masks) > 30:
            print(
                f"  ... and {len(missing_masks) - 30} more"
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()