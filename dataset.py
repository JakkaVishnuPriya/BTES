"""
Tree Semantic Segmentation Dataset

Expected files:

    G:\TreeDataset\tcd\tree_dataset_1\
        images/
            img_1.png
            img_2.png
            ...
        masks/
            img_1_mask.png
            img_2_mask.png
            ...

Train / validation / test splits are created in memory from these
two folders. No separate train/val/test folders are required.

Mask labels:
    0 = background
    1 = tree

Image size:
    256 x 256
"""

import os
import random

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


# ============================================================
# DATASET
# ============================================================

class TreeDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        split="train",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        augment=False
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.split = split.lower()
        self.augment = augment

        if self.split not in ("train", "val", "test"):
            raise ValueError(
                "split must be one of: 'train', 'val', 'test'"
            )

        total_ratio = train_ratio + val_ratio + test_ratio

        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
            )

        if not os.path.isdir(self.image_dir):
            raise RuntimeError(
                f"Image directory does not exist:\n{self.image_dir}"
            )

        if not os.path.isdir(self.mask_dir):
            raise RuntimeError(
                f"Mask directory does not exist:\n{self.mask_dir}"
            )

        # ----------------------------------------------------
        # Find ALL image/mask pairs first.
        # ----------------------------------------------------

        all_samples = []

        self._find_separate_folders(
            self.image_dir,
            self.mask_dir,
            samples=all_samples
        )

        if len(all_samples) == 0:
            raise RuntimeError(
                "\nNo valid image/mask pairs found.\n"
                f"Images: {self.image_dir}\n"
                f"Masks : {self.mask_dir}\n"
                "Expected matching files such as:\n"
                "  images/img_1.png\n"
                "  masks/img_1_mask.png"
            )

        # ----------------------------------------------------
        # Deterministic 70/15/15 split.
        # No files are copied or moved.
        # ----------------------------------------------------

        rng = random.Random(seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)

        total = len(indices)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if self.split == "train":
            selected_indices = indices[:train_end]
        elif self.split == "val":
            selected_indices = indices[train_end:val_end]
        else:
            selected_indices = indices[val_end:]

        self.samples = [all_samples[i] for i in selected_indices]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Split '{self.split}' contains zero samples."
            )

        print()
        print("=" * 60)
        print("TreeDataset")
        print("=" * 60)
        print("Images :", self.image_dir)
        print("Masks  :", self.mask_dir)
        print("Split  :", self.split)
        print("Total  :", total)
        print("Samples:", len(self.samples))
        print("Augmentation:", self.augment)
        print("Mask labels:", "0 = background, 1 = tree")
        print("=" * 60)



    # ========================================================
    # FIND IMAGE/MASK PAIRS
    # ========================================================

    def _find_samples(self):

        # ----------------------------------------------------
        # Case 1:
        #
        # root/
        #     images/
        #     masks/
        # ----------------------------------------------------

        images_dir = os.path.join(
            self.root_dir,
            "images"
        )

        masks_dir = os.path.join(
            self.root_dir,
            "masks"
        )


        if (
            os.path.isdir(images_dir)
            and
            os.path.isdir(masks_dir)
        ):

            self._find_separate_folders(
                images_dir,
                masks_dir
            )

            return


        # ----------------------------------------------------
        # Case 2:
        #
        # root/
        #     img_1.png
        #     img_1_mask.png
        # ----------------------------------------------------

        self._find_same_folder(
            self.root_dir
        )


    # ========================================================
    # SEPARATE IMAGE/MASK FOLDERS
    # ========================================================

    def _find_separate_folders(
        self,
        images_dir,
        masks_dir,
        samples=None
    ):

        if samples is None:
            samples = self.samples

        image_files = sorted(
            os.listdir(
                images_dir
            )
        )


        for filename in image_files:

            # -----------------------------------------------
            # Ignore directories
            # -----------------------------------------------

            image_path = os.path.join(
                images_dir,
                filename
            )


            if not os.path.isfile(
                image_path
            ):

                continue


            # -----------------------------------------------
            # Only supported image files
            # -----------------------------------------------

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                continue


            # -----------------------------------------------
            # Never treat *_mask as an image
            # -----------------------------------------------

            if filename.lower().endswith(
                "_mask.png"
            ):

                continue

            if filename.lower().endswith(
                "_mask.jpg"
            ):

                continue

            if filename.lower().endswith(
                "_mask.jpeg"
            ):

                continue

            if filename.lower().endswith(
                "_mask.tif"
            ):

                continue

            if filename.lower().endswith(
                "_mask.tiff"
            ):

                continue


            # -----------------------------------------------
            # Base name
            #
            # img_511.png
            #     ↓
            # img_511
            # -----------------------------------------------

            base_name = os.path.splitext(
                filename
            )[0]


            # -----------------------------------------------
            # Expected PNG mask
            #
            # img_511_mask.png
            # -----------------------------------------------

            mask_filename = (
                base_name
                +
                "_mask.png"
            )


            mask_path = os.path.join(
                masks_dir,
                mask_filename
            )


            # -----------------------------------------------
            # If PNG mask exists
            # -----------------------------------------------

            if os.path.isfile(
                mask_path
            ):

                samples.append(
                    (
                        image_path,
                        mask_path
                    )
                )

                continue


            # -----------------------------------------------
            # Also support TIFF mask if an old dataset
            # is being used.
            # -----------------------------------------------

            found_mask = False


            for ext in (
                ".tif",
                ".tiff",
                ".jpg",
                ".jpeg"
            ):

                alternative_mask = os.path.join(
                    masks_dir,
                    base_name
                    +
                    "_mask"
                    +
                    ext
                )


                if os.path.isfile(
                    alternative_mask
                ):

                    samples.append(
                        (
                            image_path,
                            alternative_mask
                        )
                    )

                    found_mask = True

                    break


            # -----------------------------------------------
            # Missing mask
            # -----------------------------------------------

            if not found_mask:

                print(
                    "WARNING: Missing mask for:",
                    filename
                )


    # ========================================================
    # SAME FOLDER
    # ========================================================

    def _find_same_folder(
        self,
        directory
    ):

        if not os.path.isdir(
            directory
        ):

            return


        files = sorted(
            os.listdir(
                directory
            )
        )


        for filename in files:

            image_path = os.path.join(
                directory,
                filename
            )


            if not os.path.isfile(
                image_path
            ):

                continue


            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                continue


            # -----------------------------------------------
            # Ignore mask files
            # -----------------------------------------------

            if "_mask" in filename.lower():

                continue


            # -----------------------------------------------
            # Base name
            # -----------------------------------------------

            base_name = os.path.splitext(
                filename
            )[0]


            # -----------------------------------------------
            # First look for PNG mask
            # -----------------------------------------------

            mask_filename = (
                base_name
                +
                "_mask.png"
            )


            mask_path = os.path.join(
                directory,
                mask_filename
            )


            if os.path.isfile(
                mask_path
            ):

                self.samples.append(
                    (
                        image_path,
                        mask_path
                    )
                )

                continue


            # -----------------------------------------------
            # Fallback extensions
            # -----------------------------------------------

            found_mask = False


            for ext in (
                ".tif",
                ".tiff",
                ".jpg",
                ".jpeg"
            ):

                alternative_mask = os.path.join(
                    directory,
                    base_name
                    +
                    "_mask"
                    +
                    ext
                )


                if os.path.isfile(
                    alternative_mask
                ):

                    self.samples.append(
                        (
                            image_path,
                            alternative_mask
                        )
                    )

                    found_mask = True

                    break


            if not found_mask:

                print(
                    "WARNING: Missing mask for:",
                    filename
                )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.samples
        )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def _load_image(
        self,
        image_path
    ):

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )


        if image is None:

            raise RuntimeError(
                "\n"
                "Could not read image:\n"
                f"{image_path}"
            )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        return image


    # ========================================================
    # LOAD MASK
    # ========================================================

    def _load_mask(
        self,
        mask_path
    ):

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )


        if mask is None:

            raise RuntimeError(
                "\n"
                "Could not read mask:\n"
                f"{mask_path}"
            )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Your masks are:
        #
        # 0 = background
        # 1 = tree
        #
        # Keep them as 0/1.
        # ----------------------------------------------------

        unique_values = np.unique(
            mask
        )


        # ----------------------------------------------------
        # If mask is 0/1, keep it.
        # ----------------------------------------------------

        if set(
            unique_values.tolist()
        ).issubset(
            {0, 1}
        ):

            mask = (
                mask > 0
            ).astype(
                np.float32
            )


        # ----------------------------------------------------
        # If an old mask uses 0/255,
        # convert it to 0/1.
        # ----------------------------------------------------

        elif set(
            unique_values.tolist()
        ).issubset(
            {0, 255}
        ):

            mask = (
                mask > 0
            ).astype(
                np.float32
            )


        # ----------------------------------------------------
        # If there are unexpected values,
        # convert everything > 0 to tree.
        #
        # This keeps the segmentation binary.
        # ----------------------------------------------------

        else:

            mask = (
                mask > 0
            ).astype(
                np.float32
            )


        return mask


    # ========================================================
    # AUGMENTATION
    # ========================================================

    def _augment(
        self,
        image,
        mask
    ):

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if random.random() < 0.5:

            image = np.fliplr(
                image
            ).copy()

            mask = np.fliplr(
                mask
            ).copy()


        # ----------------------------------------------------
        # Vertical flip
        # ----------------------------------------------------

        if random.random() < 0.5:

            image = np.flipud(
                image
            ).copy()

            mask = np.flipud(
                mask
            ).copy()


        # ----------------------------------------------------
        # 90 degree rotation
        # ----------------------------------------------------

        if random.random() < 0.5:

            k = random.randint(
                1,
                3
            )


            image = np.rot90(
                image,
                k
            ).copy()


            mask = np.rot90(
                mask,
                k
            ).copy()


        return (
            image,
            mask
        )


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        image_path, mask_path = (
            self.samples[index]
        )


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = self._load_image(
            image_path
        )


        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask = self._load_mask(
            mask_path
        )


        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if (
            image.shape[0]
            !=
            mask.shape[0]
        ):

            raise RuntimeError(
                "\n"
                "Image/mask height mismatch:\n"
                f"Image: {image_path}\n"
                f"Mask : {mask_path}\n"
                f"Image shape: {image.shape}\n"
                f"Mask shape : {mask.shape}"
            )


        if (
            image.shape[1]
            !=
            mask.shape[1]
        ):

            raise RuntimeError(
                "\n"
                "Image/mask width mismatch:\n"
                f"Image: {image_path}\n"
                f"Mask : {mask_path}\n"
                f"Image shape: {image.shape}\n"
                f"Mask shape : {mask.shape}"
            )


        # ----------------------------------------------------
        # Resize ONLY if necessary
        #
        # Your generated dataset should already be 256x256.
        # ----------------------------------------------------

        if (
            image.shape[0] != 256
            or
            image.shape[1] != 256
        ):

            image = cv2.resize(
                image,
                (256, 256),
                interpolation=cv2.INTER_LINEAR
            )


            mask = cv2.resize(
                mask,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            )


        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.augment:

            image, mask = (
                self._augment(
                    image,
                    mask
                )
            )


        # ====================================================
        # IMAGE NORMALIZATION
        # ====================================================

        image = (
            image.astype(
                np.float32
            )
            /
            255.0
        )


        # ----------------------------------------------------
        # ImageNet normalization
        #
        # Standard for ResNet34 encoder.
        # ----------------------------------------------------

        mean = np.array(
            [
                0.485,
                0.456,
                0.406
            ],
            dtype=np.float32
        )


        std = np.array(
            [
                0.229,
                0.224,
                0.225
            ],
            dtype=np.float32
        )


        image = (
            image - mean
        ) / std


        # ====================================================
        # HWC -> CHW
        # ====================================================

        image = np.transpose(
            image,
            (
                2,
                0,
                1
            )
        )


        # ====================================================
        # MASK
        # H,W -> 1,H,W
        # ====================================================

        mask = np.expand_dims(
            mask,
            axis=0
        )


        # ====================================================
        # TORCH TENSORS
        # ====================================================

        image = torch.from_numpy(
            image.copy()
        ).float()


        mask = torch.from_numpy(
            mask.copy()
        ).float()


        return (
            image,
            mask
        )