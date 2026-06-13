# src/data_loader.py
import os
import sys
import pandas as pd

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from region_config import REGIONS

def load_and_merge_tabular_data(region_name, force_refresh=False):
    """
    Loads FIRMS and Weather data for a given region and merges them based on the date.
    Features checkpointing to skip already processed regions and safe-fill for missing weather.
    """
    base_dir = f"data/case_studies/{region_name}/raw"
    firms_path = f"{base_dir}/firms/firms_{region_name}.csv"
    weather_path = f"{base_dir}/weather/daily_weather.csv"

    out_dir = f"data/case_studies/{region_name}/processed/tabular"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/merged_fire_weather_{region_name}.csv"

    # Checkpointing logic: Skip if already done
    if os.path.exists(out_path) and not force_refresh:
        print(f"  ⏭️  Merged data already exists. Skipping (Set force_refresh=True to overwrite).")
        return pd.read_csv(out_path)

    # Validate raw data exists
    if not os.path.exists(firms_path):
        print(f"  ❌ Missing FIRMS data at: {firms_path}")
        return None
    if not os.path.exists(weather_path):
        print(f"  ❌ Missing Weather data at: {weather_path}")
        return None

    try:
        # Load FIRMS Data
        firms_df = pd.read_csv(firms_path)
        firms_df = firms_df.dropna(subset=['acq_date'])
        firms_df['date'] = pd.to_datetime(firms_df['acq_date']).dt.date
        
        # Load Weather Data
        weather_df = pd.read_csv(weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

        # Merge based on date (Left join keeps ALL fire events)
        merged_df = pd.merge(firms_df, weather_df, on='date', how='left')
        
        # Create target variable (1 = fire occurred)
        merged_df['fire_occurred'] = int(1) # Enforcing standard int

        # SAFETY NET: Fill missing weather data with the nearest day's weather 
        # so absolutely NO fire points are dropped due to blank data.
        weather_cols = [col for col in weather_df.columns if col not in ['date', 'region']]
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        merged_df[weather_cols] = merged_df[weather_cols].ffill()
        # Drop any remaining NaNs at the very start (no data before first weather obs)
        merged_df = merged_df.dropna(subset=weather_cols, how='all')

        # Save processed dataset
        merged_df.to_csv(out_path, index=False)
        print(f"  ✅ Successfully merged and saved {len(merged_df)} records to {out_path}")
        return merged_df

    except Exception as e:
        print(f"  ⚠️ Error processing {region_name}: {e}")
        return None

def batch_process_all_regions():
    """Loops safely through all regions to merge data."""
    print("🔄 BATCH MERGING FIRMS & WEATHER DATA (Step 2A)")
    print("=" * 70)
    
    for region in REGIONS.keys():
        print(f"\n🌍 Processing {REGIONS[region]['name']}...")
        # force_refresh=True ensures the new ffill()/bfill() logic is applied over the old files
        load_and_merge_tabular_data(region, force_refresh=True) 

if __name__ == "__main__":
    batch_process_all_regions()