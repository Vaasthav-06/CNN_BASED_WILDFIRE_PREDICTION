# src/uav_processor.py
"""
UAV / Aerial Fire Dataset Integration
Supports:
  1. FLAME dataset (drone fire/no-fire video frames)
  2. FiSmo / ForestFireDetection aerial datasets
  3. Sentinel-2 pseudo-UAV tiles at 10m resolution (fallback)
"""
import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from region_config import REGIONS

# ─────────────────────────────────────────────
# 1. FLAME DATASET LOADER
# ─────────────────────────────────────────────

def load_flame_dataset(flame_dir, img_size=256):
    """
    Load FLAME drone fire dataset (https://ieee-dataport.org/FLAME).
    """
    try:
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
    except ImportError:
        print("⚠️  torch / torchvision not installed. Run: pip install torch torchvision")
        return None, None

    img_size = int(img_size)

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(flame_dir, "Training")
    val_path   = os.path.join(flame_dir, "Validation")

    if not os.path.exists(train_path):
        print(f"⚠️  FLAME Training folder not found at {train_path}")
        return None, None

    train_ds = datasets.ImageFolder(train_path, transform=transform_train)
    val_ds   = datasets.ImageFolder(val_path,   transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"✅ FLAME dataset loaded: {len(train_ds)} train / {len(val_ds)} val images")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# 2. CONVERT NPY TILES → IMAGE FOLDER STRUCTURE
# ─────────────────────────────────────────────

def convert_tiles_to_imagefolder(region_name, tiles_dir, output_dir, bands_to_rgb=(0, 1, 2)):
    """
    Convert .npy satellite tiles to JPEG images in ImageFolder structure.
    bands_to_rgb: tuple of 3 band indices to use as R, G, B channels.
    We use (0,1,2) for NDVI, NBR, NDMI.
    """
    try:
        from PIL import Image
    except ImportError:
        print("⚠️  Pillow not installed. Run: pip install pillow")
        return

    for label in ('fire', 'no_fire'):
        src_folder = os.path.join(tiles_dir, 'pre_fire' if label == 'fire' else 'no_fire')
        dst_folder = os.path.join(output_dir, label)
        os.makedirs(dst_folder, exist_ok=True)

        if not os.path.exists(src_folder):
            continue

        npy_files = [f for f in os.listdir(src_folder) if f.endswith('.npy')]
        converted = int(0)

        for fname in npy_files:
            try:
                arr = np.load(os.path.join(src_folder, fname))  # (bands, H, W)
                
                # Check if the array has the required bands
                if arr.shape[0] < 3:
                    # Repeat the single band across 3 channels if missing data
                    arr_rgb = np.stack([arr[0]] * 3, axis=0)
                else:
                    arr_rgb = arr[list(bands_to_rgb)]  # Select 3 bands

                # Normalise raw spectral indices (-1 to 1) to standard 8-bit image (0-255)
                arr_rgb = arr_rgb.astype(float)
                for ch in range(3):
                    ch_min = arr_rgb[ch].min()
                    ch_max = arr_rgb[ch].max()
                    if ch_max > ch_min:
                        arr_rgb[ch] = (arr_rgb[ch] - ch_min) / (ch_max - ch_min) * 255.0
                    else:
                        arr_rgb[ch] = 0.0

                # Convert to standard integers for the image format
                arr_rgb = arr_rgb.astype(np.uint8).transpose(1, 2, 0)  # CHW → HWC
                
                img = Image.fromarray(arr_rgb, mode='RGB')
                out_name = fname.replace('.npy', '.jpg')
                img.save(os.path.join(dst_folder, out_name), quality=95)
                converted += 1
                
            except Exception as e:
                # Silently skip corrupted tiles to keep the pipeline moving
                pass

        print(f"  ✅ {label}: converted {converted} tiles → {dst_folder}")


# ─────────────────────────────────────────────
# 3. BATCH RUNNER
# ─────────────────────────────────────────────

def prepare_all_regions_for_cnn(base_data_dir="data/case_studies"):
    """
    For all 4 regions, convert .npy tiles to ImageFolder structure.
    """
    print("🛸 MULTI-REGION SATELLITE TO CNN PREPARATION")
    print("=" * 60)
    for region_name in REGIONS.keys():
        print(f"\n🌍 Preparing {REGIONS[region_name]['name']}...")
        tiles_dir  = f"{base_data_dir}/{region_name}/processed/tiles"
        output_dir = f"{base_data_dir}/{region_name}/processed/cnn_dataset"
        
        convert_tiles_to_imagefolder(region_name, tiles_dir, output_dir)

if __name__ == "__main__":
    prepare_all_regions_for_cnn()