import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import safetensors.torch

# Enable CuDNN benchmark for faster training on fixed-size inputs
torch.backends.cudnn.benchmark = True


# ==============================================================================
# 1. DATASET CLASS WITH BACKGROUND FILTERING
# ==============================================================================
class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, is_train=True):
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
        
        image_paths_set = set()
        mask_paths_set = set()
        
        for ext in extensions:
            image_paths_set.update(glob.glob(os.path.join(image_dir, ext)))
            mask_paths_set.update(glob.glob(os.path.join(mask_dir, ext)))

        self.image_paths = sorted(list(image_paths_set))
        self.mask_paths = sorted(list(mask_paths_set))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in '{image_dir}'. Please verify your paths!")

        self.transform = transform
        valid_img_paths = []
        valid_mask_paths = []

        print(f"Filtering dataset ({'Train' if is_train else 'Val'}). Total unique images found: {len(self.image_paths)}...")
        
        for img_p, msk_p in zip(self.image_paths, self.mask_paths):
            mask = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                continue
                
            building_pixel_count = np.count_nonzero(mask > 0)
            
            if is_train:
                if building_pixel_count > 0:
                    valid_img_paths.append(img_p)
                    valid_mask_paths.append(msk_p)
            else:
                valid_img_paths.append(img_p)
                valid_mask_paths.append(msk_p)

        self.image_paths = valid_img_paths
        self.mask_paths = valid_mask_paths
        print(f"Retained {len(self.image_paths)} images after background filtering.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Could not read image at {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask at {self.mask_paths[idx]}")
            
        mask = (mask > 0).astype(np.float32)  # Binary mask [0.0, 1.0]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # Shape: (1, H, W)

        return image, mask


# ==============================================================================
# 2. ALBUMENTATIONS DATA TRANSFORMS
# ==============================================================================
def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


# ==============================================================================
# 3. EDGE-WEIGHTED LOSS FUNCTION
# ==============================================================================
class EdgeWeightedBuildingLoss(nn.Module):
    def __init__(self, edge_weight=3.0, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.edge_weight = edge_weight
        self.smooth = smooth
        self.register_buffer('kernel', torch.ones((1, 1, 3, 3), dtype=torch.float32))

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        kernel = self.kernel.to(targets.device)
        
        with torch.no_grad():
            dilated = torch.clamp(nn.functional.conv2d(targets, kernel, padding=1), 0, 1)
            eroded = 1.0 - torch.clamp(nn.functional.conv2d(1.0 - targets, kernel, padding=1), 0, 1)
            edge_mask = dilated - eroded
        
        weights = 1.0 + (self.edge_weight * edge_mask)
        weighted_bce = (bce_loss * weights).mean()
        
        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs.sum() + targets_flat.sum() + self.smooth)
        
        return 0.5 * weighted_bce + 0.5 * dice_loss


# ==============================================================================
# 4. DETAILED METRICS COMPUTATION (IoU, Dice, Accuracy)
# ==============================================================================
def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    intersection = (pred_mask * true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    union = total - intersection
    
    # 1. IoU (Jaccard Index)
    iou = (intersection + smooth) / (union + smooth)
    
    # 2. Dice Coefficient
    dice = (2.0 * intersection + smooth) / (total + smooth)
    
    # 3. Pixel Accuracy
    correct_pixels = (pred_mask == true_mask).sum()
    total_pixels = true_mask.numel()
    accuracy = correct_pixels / total_pixels

    return iou.item(), dice.item(), accuracy.item()


# ==============================================================================
# 5. TRAINING AND VALIDATION PIPELINE
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, save_dir, num_epochs=35, patience=10, device='cuda'):
    best_val_iou = 0.0
    early_stop_counter = 0

    os.makedirs(save_dir, exist_ok=True)
    best_model_pth = os.path.join(save_dir, "best_unetplusplus.pth")

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs:02d} [Train]"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_iou_list = []
        val_dice_list = []
        val_acc_list = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs:02d} [Val]  "):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, masks)
                
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.40).float()
                
                for p, t in zip(preds, masks):
                    iou, dice, acc = calculate_metrics(p, t)
                    val_iou_list.append(iou)
                    val_dice_list.append(dice)
                    val_acc_list.append(acc)

        avg_val_loss = val_loss / len(val_loader)
        mean_val_iou = np.mean(val_iou_list) * 100
        mean_val_dice = np.mean(val_dice_list) * 100
        mean_val_acc = np.mean(val_acc_list) * 100

        # --- CHECK IF MODEL IMPROVED ---
        if mean_val_iou > best_val_iou:
            iou_gain = mean_val_iou - best_val_iou
            best_val_iou = mean_val_iou
            early_stop_counter = 0  # Reset patience
            
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_iou': mean_val_iou,
                'val_dice': mean_val_dice,
                'val_accuracy': mean_val_acc
            }
            torch.save(checkpoint_data, best_model_pth)
            
            print("\n" + "=" * 60)
            print(f"🎉 NEW BEST MODEL SAVED AT EPOCH {epoch+1}!")
            print(f"   ► Val IoU      : {mean_val_iou:.2f}%  (+{iou_gain:.2f}% improvement)")
            print(f"   ► Val Accuracy : {mean_val_acc:.2f}%")
            print(f"   ► Val Dice     : {mean_val_dice:.2f}%")
            print(f"   ► Val Loss     : {avg_val_loss:.4f}  | Train Loss: {avg_train_loss:.4f}")
            print("=" * 60 + "\n")
        else:
            early_stop_counter += 1
            print(f"\nEpoch {epoch+1:02d} Summary:")
            print(f"   Val IoU: {mean_val_iou:.2f}% (Best: {best_val_iou:.2f}%) | Val Acc: {mean_val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")
            print(f"   No improvement counter: {early_stop_counter}/{patience}\n")
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without IoU improvement.")
                break


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    BASE_DIR = r"G:\Segformer_buildings"
    
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    TRAIN_IMG = os.path.join(DATASET_DIR, "train", "images")
    TRAIN_MASK = os.path.join(DATASET_DIR, "train", "masks")
    VAL_IMG = os.path.join(DATASET_DIR, "val", "images")
    VAL_MASK = os.path.join(DATASET_DIR, "val", "masks")
    
    LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 35
    PATIENCE = 10

    train_transform, val_transform = get_transforms()

    train_dataset = BuildingDataset(TRAIN_IMG, TRAIN_MASK, transform=train_transform, is_train=True)
    val_dataset = BuildingDataset(VAL_IMG, VAL_MASK, transform=val_transform, is_train=False)

    num_workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    print("Building UNet++ model with local weights...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(DEVICE)

    safetensors_path = os.path.join(LOCAL_MODEL_DIR, "model.safetensors")
    bin_path = os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin")

    try:
        if os.path.exists(safetensors_path):
            print(f"Loading local weights from: {safetensors_path}")
            state_dict = safetensors.torch.load_file(safetensors_path)
        elif os.path.exists(bin_path):
            print(f"Loading local weights from: {bin_path}")
            state_dict = torch.load(bin_path, map_location=DEVICE)
        else:
            state_dict = None
            print(f"Warning: No weights file found in {LOCAL_MODEL_DIR}. Initializing backbone cleanly.")

        if state_dict is not None:
            model.encoder.load_state_dict(state_dict, strict=False)
            print("Local backbone weights loaded into UNet++ successfully!")

    except Exception as e:
        print(f"Notice: Initialization proceeding with standard architecture ({e}).")

    criterion = EdgeWeightedBuildingLoss(edge_weight=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    train_model(
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