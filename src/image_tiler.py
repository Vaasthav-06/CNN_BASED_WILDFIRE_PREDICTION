# src/image_tiler.py
import os
import sys
import shutil
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import re
import random

# =========================================================
# 0. Bulletproof Dynamic Path Setup
# =========================================================
# 1. Identify exactly where this script lives (the 'src' folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Identify the root folder (one level up from 'src')
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 3. Add BOTH folders to the Python system path
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Force the terminal to operate out of the root folder
os.chdir(ROOT_DIR)

# Python will now flawlessly find these files inside your src/ folder
from region_config import REGIONS
from region_utils import RegionManager

TILE_SIZE = int(256)   # Enforcing standard int for array dimensions
SCALE = int(20)        # Sentinel-2 resolution: 20 metres/pixel

def clean_tile_directories(output_dir):
    """
    Surgically removes the old pre_fire and no_fire directories and 
    recreates them empty to ensure old naming formats don't pollute the dataset.
    """
    print(f"  🧹 Cleaning old tiles from {output_dir}...")
    for subdir in ['pre_fire', 'no_fire']:
        dir_path = os.path.join(output_dir, subdir)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # Delete the folder and everything inside it
        os.makedirs(dir_path, exist_ok=True) # Recreate it fresh and empty

def find_nearest_satellite(satellite_dir, target_date=None):
    """
    Find the .tif file in satellite_dir that matches the target year.
    Files are expected to contain a year string YYYY in their names.
    """
    if not os.path.exists(satellite_dir):
        return None

    tif_files = [f for f in os.listdir(satellite_dir) if f.endswith('.tif')]
    if not tif_files:
        return None

    if target_date is None:
        return os.path.join(satellite_dir, tif_files[0])

    target_year = int(pd.to_datetime(target_date).year)

    for fname in tif_files:
        # Extract the year from the Earth Engine filename
        match = re.search(r'(\d{4})', fname)
        if match:
            try:
                file_year = int(match.group(1))
                if file_year == target_year:
                    return os.path.join(satellite_dir, fname)
            except Exception:
                continue

    # Fallback
    return os.path.join(satellite_dir, tif_files[0])

def extract_tile(raster_path, lat, lon, size=int(256)):
    """
    Extract a square patch (size x size pixels) centered at (lat, lon).
    Returns a numpy array of shape (bands, size, size), or None if out of bounds/corrupted.
    """
    try:
        with rasterio.open(raster_path) as src:
            # Convert geographic coordinates -> pixel coordinates
            col, row = ~src.transform * (lon, lat)
            col, row = int(col), int(row)

            half = int(size // int(2))
            window = Window(col - half, row - half, size, size)

            data = src.read(window=window)  # Shape: (bands, H, W)

            # 1. Reject tiles that were clipped at the image edge
            if data.shape[1] == size and data.shape[2] == size:
                
                # 2. Black Pixel / NoData Filter
                nodata_mask = (data[0] == int(0))
                nodata_ratio = np.sum(nodata_mask) / (size * size)
                
                if nodata_ratio > 0.30:
                    return None
                    
                return data
            else:
                return None

    except Exception as e:
        return None

def extract_fire_tiles(region_name, firms_df, satellite_dir, output_dir, days_before=int(14)):
    """
    For each FIRMS fire detection:
      - Find the Sentinel-2 yearly composite
      - Extract a TILE_SIZE x TILE_SIZE patch centered on the fire point
      - Save as a .npy file with the DATE in the filename for Phase 4 Joining
    """
    print(f"\n🛰️  Starting positive tile extraction for: {region_name}")
    success_count = int(0)
    skip_count = int(0)

    for idx, row in firms_df.iterrows():
        lat  = row["latitude"]
        lon  = row["longitude"]
        fire_date = pd.to_datetime(row["acq_date"])
        
        # 🚨 FORMAT DATE FOR FILENAME 🚨
        fire_date_str = fire_date.strftime("%Y-%m-%d")
        
        target_date = fire_date - pd.Timedelta(days=days_before)
        sat_file = find_nearest_satellite(satellite_dir, target_date)

        if sat_file is None:
            skip_count += 1
            continue

        tile = extract_tile(sat_file, lat, lon, TILE_SIZE)

        if tile is not None:
            save_path = f"{output_dir}/pre_fire/fire_{fire_date_str}_{idx}.npy"
            np.save(save_path, tile)
            success_count += 1
        else:
            skip_count += 1

    print(f"  ✅ Generated {success_count} fire tiles  |  ⏭️  Skipped {skip_count}")
    return success_count

def generate_random_date(start_year=int(2018), end_year=int(2026)):
    """Helper to pick a random day within our dataset timeframe."""
    start = pd.to_datetime(f"{start_year}-01-01")
    end = pd.to_datetime(f"{end_year}-01-01")
    delta = end - start
    random_days = random.randint(int(0), int(delta.days))
    return start + pd.Timedelta(days=random_days)

def extract_negative_tiles(region_name, satellite_dir, output_dir, n_samples=int(200)):
    """
    Generate no-fire (negative) tiles by picking random coordinates 
    and random dates within the region's bounding box and timeframe.
    """
    print(f"🛰️  Starting negative tile extraction for: {region_name}")
    manager = RegionManager(region_name)
    bounds = manager.bounds

    success = int(0)
    attempts = int(0)
    max_attempts = n_samples * int(5)

    while success < n_samples and attempts < max_attempts:
        attempts += 1
        
        rand_lat = random.uniform(bounds['lat_min'], bounds['lat_max'])
        rand_lon = random.uniform(bounds['lon_min'], bounds['lon_max'])
        
        # 🚨 GENERATE DATE FOR FILENAME 🚨
        random_date = generate_random_date()
        random_date_str = random_date.strftime("%Y-%m-%d")

        sat_file = find_nearest_satellite(satellite_dir, random_date)
        if sat_file is None:
            continue

        tile = extract_tile(sat_file, rand_lat, rand_lon, TILE_SIZE)
        
        if tile is not None:
            np.save(f"{output_dir}/no_fire/no_fire_{random_date_str}_{success}.npy", tile)
            success += 1

    print(f"  ✅ Generated {success} no-fire (negative) tiles")
    return success

def run_tiling_all_regions():
    """Batch tile extraction across all 4 regions."""
    print("✂️  MULTI-REGION SATELLITE TILE EXTRACTION (DATE INCLUDED)")
    print("=" * 70)
    
    for region_name in REGIONS.keys():
        csv_path = os.path.join(ROOT_DIR, "data", "case_studies", region_name, "raw", "firms", f"firms_{region_name}.csv")
        sat_dir  = os.path.join(ROOT_DIR, "data", "case_studies", region_name, "processed", "indices")
        out_dir  = os.path.join(ROOT_DIR, "data", "case_studies", region_name, "processed", "tiles")

        try:
            firms_df = pd.read_csv(csv_path)
            firms_df["acq_date"] = pd.to_datetime(firms_df["acq_date"])

            config = REGIONS[region_name]
            days_before = config['satellite_settings']['buffer_days_before_fire']

            # CLEAN OLD FILES FIRST
            clean_tile_directories(out_dir)

            extract_fire_tiles(region_name, firms_df, sat_dir, out_dir, days_before)
            
            n_negatives = int(len(firms_df) * 1.5)
            extract_negative_tiles(region_name, sat_dir, out_dir, n_samples=n_negatives)

        except FileNotFoundError:
            print(f"  ⚠️  FIRMS CSV not found for {region_name} at {csv_path}")

if __name__ == "__main__":
    run_tiling_all_regions()