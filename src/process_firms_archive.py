# src/process_firms_archive.py
import os
import sys
import pandas as pd
import glob

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.region_utils import RegionManager
from region_config import REGIONS

def process_all_archives(archive_dir):
    print("🔥 NASA FIRMS LOCAL ARCHIVE PROCESSOR (BATCH MODE)")
    print("=" * 70)

    # 1. Find all CSVs in the target folder
    csv_files = glob.glob(f"{archive_dir}/*.csv")
    if not csv_files:
        print(f"❌ Error: No CSV files found in {archive_dir}")
        return

    print(f"📂 Found {len(csv_files)} files to process:")
    for f in csv_files:
        print(f"   - {os.path.basename(f)}")

    # 2. Dictionary to hold combined data for each region
    regional_data = {region: [] for region in REGIONS.keys()}

    # 3. Loop through every NASA file
    for file_path in csv_files:
        print(f"\n📥 Processing {os.path.basename(file_path)}...")
        
        df = pd.read_csv(file_path)
        
        # Standardize numeric columns
        numeric_cols = ['latitude', 'longitude', 'brightness', 'frp', 'confidence']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if 'acq_date' in df.columns:
            df['acq_date'] = pd.to_datetime(df['acq_date'])

        # 4. Filter this specific file for all 4 regions
        for region_name in REGIONS.keys():
            manager = RegionManager(region_name)
            filtered = manager.filter_firms_data(df)
            
            if not filtered.empty:
                regional_data[region_name].append(filtered)
                print(f"   ➔ Found {len(filtered)} fires for {region_name}")

    print("\n💾 MERGING & SAVING REGIONAL DATA")
    print("-" * 70)
    
    # 5. Merge, clean, and save the final regional files
    for region_name, dfs in regional_data.items():
        if dfs:
            # Combine Archive + NRT + MODIS + VIIRS
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Drop duplicates in case Archive and NRT overlap on the edge dates
            combined_df = combined_df.drop_duplicates(subset=['latitude', 'longitude', 'acq_date', 'acq_time'])
            
            # Routing to correct folder
            out_dir = f"data/case_studies/{region_name}/raw/firms"
            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/firms_{region_name}.csv"
            
            combined_df.to_csv(out_path, index=False)
            
            config = REGIONS[region_name]
            print(f"✅ {config['name'].upper()}: Saved {len(combined_df):,} total unique detections to {out_path}")
        else:
            config = REGIONS[region_name]
            print(f"⚠️ {config['name'].upper()}: 0 detections found across all files.")

if __name__ == "__main__":
    # Points directly to the folder containing your 4 NASA files
    ARCHIVE_DIR = "data/master_archive"
    process_all_archives(ARCHIVE_DIR)