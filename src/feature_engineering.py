# src/feature_engineering.py
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from region_config import REGIONS
from src.spectral_indices import extract_pixel_values


# ─────────────────────────────────────────────
# 1. VAPOR PRESSURE DEFICIT (VPD)
# ─────────────────────────────────────────────

def add_vpd(df):
    """
    VPD = Saturation Vapor Pressure − Actual Vapor Pressure  (kPa)
    Uses mean daily temperature and mean daily relative humidity.
    """
    T  = df["temperature_2m_mean"].astype(float)
    RH = df["relative_humidity_2m_mean"].astype(float)

    # Tetens formula for SVP
    SVP = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    AVP = SVP * (RH / 100.0)
    df["vpd"] = (SVP - AVP).clip(lower=0)
    return df


# ─────────────────────────────────────────────
# 2. ROLLING WEATHER STATISTICS
# ─────────────────────────────────────────────

def add_rolling_features(df, windows=(7, 14, 30)):
    """
    Rolling features using ONLY past data — no look-ahead leakage.
    Uses .shift(1) so each row's window ends the DAY BEFORE that row.
    """
    df = df.copy().sort_values('date').reset_index(drop=True)

    weather_cols = [
        'temperature_2m_mean', 'temperature_2m_max',
        'relative_humidity_2m_mean',
        'precipitation_sum',
        'wind_speed_10m_max', 'wind_speed_10m_mean',
        'soil_moisture_0_1cm_mean',
        'vpd'
    ]
    existing_cols = [c for c in weather_cols if c in df.columns]

    for window in windows:
        for col in existing_cols:
            # .shift(1) ensures today's value is NOT included in today's window
            rolled = df[col].shift(1).rolling(window, min_periods=1)
            df[f"{col}_roll{window}d_mean"] = rolled.mean()
            if col in ('precipitation_sum', 'et0_fao_evapotranspiration_sum'):
                df[f"{col}_roll{window}d_sum"] = rolled.sum()
            if col in ('temperature_2m_max', 'wind_speed_10m_max', 'vpd'):
                df[f"{col}_roll{window}d_max"] = rolled.max()

    return df


# ─────────────────────────────────────────────
# 3. TEMPORAL FEATURES
# ─────────────────────────────────────────────

def add_temporal_features(df, region_name):
    """Add calendar, cyclical, and fire-season features using standard ints."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df['month']       = df['date'].dt.month.astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear.astype(int)
    df['year']        = df['date'].dt.year.astype(int)
    df['week']        = df['date'].dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Fire season flag (region-specific)
    config = REGIONS[region_name]
    fire_start = int(config['characteristics']['fire_season_start'])
    fire_end   = int(config['characteristics']['fire_season_end'])

    if fire_start <= fire_end:
        df['fire_season'] = df['month'].between(fire_start, fire_end).astype(int)
    else:  # Wraps around year
        df['fire_season'] = ((df['month'] >= fire_start) | (df['month'] <= fire_end)).astype(int)

    return df


# ─────────────────────────────────────────────
# 4. SATELLITE SPECTRAL INDEX FEATURES
# ─────────────────────────────────────────────

def add_satellite_features(df, region_name, index_dir):
    """
    Extracts NDVI, NBR, NDMI pixel values from the yearly raster files.
    """
    config = REGIONS[region_name]
    center_lat = config['center']['lat']
    center_lon = config['center']['lon']

    def find_nearest_raster(directory, target_date):
        """Finds the S2 yearly composite matching the target date."""
        if not os.path.exists(directory):
            return None
            
        files = [f for f in os.listdir(directory) if f.endswith('.tif')]
        if not files:
            return None

        target_year = int(pd.to_datetime(target_date).year)
        
        for fname in files:
            match = re.search(r'(\d{4})', fname)
            if match:
                try:
                    file_year = int(match.group(1))
                    if file_year == target_year:
                        return os.path.join(directory, fname)
                except Exception:
                    continue
                    
        return os.path.join(directory, files[0])

    ndvi_vals, nbr_vals, ndmi_vals = [], [], []

    for _, row in df.iterrows():
        # Since tabular is region-wide weather, we extract satellite data for the region's center
        date = row['date']
        raster = find_nearest_raster(index_dir, date)
        
        if raster:
            # We extract a large 5-pixel buffer (approx 100m) to get a generalized reading
            vals = extract_pixel_values(raster, center_lat, center_lon, buffer=5)
            # The indices are stacked as bands: 0:NDVI, 1:NBR, 2:NDMI, 3:BSI
            if vals is not None and len(vals) >= 3:
                ndvi_vals.append(float(vals[0]))
                nbr_vals.append(float(vals[1]))
                ndmi_vals.append(float(vals[2]))
            else:
                ndvi_vals.append(np.nan)
                nbr_vals.append(np.nan)
                ndmi_vals.append(np.nan)
        else:
            ndvi_vals.append(np.nan)
            nbr_vals.append(np.nan)
            ndmi_vals.append(np.nan)

    df = df.copy()
    df['ndvi'] = ndvi_vals
    df['nbr']  = nbr_vals
    df['ndmi'] = ndmi_vals

    if 'month' in df.columns:
        monthly_mean = df.groupby('month')['ndvi'].transform('mean')
        df['ndvi_anomaly'] = df['ndvi'] - monthly_mean

    if 'temperature_2m_max' in df.columns:
        df['fuel_dryness'] = (1.0 - df['ndmi'].fillna(0)) * df['temperature_2m_max']

    df['nbr_roll14d_mean'] = df['nbr'].rolling(14, min_periods=1).mean()
    df['nbr_deficit']      = df['nbr_roll14d_mean'] - df['nbr']

    return df


# ─────────────────────────────────────────────
# 5. LABEL CREATION
# ─────────────────────────────────────────────

def create_fire_labels(weather_df, firms_df, region_name, window_days=1):
    """
    Merge weather data with FIRMS fire detections.
    """
    weather_df = weather_df.copy()
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    weather_df['fire'] = int(0)

    if firms_df.empty:
        return weather_df

    firms_df = firms_df.copy()
    firms_df['acq_date'] = pd.to_datetime(firms_df['acq_date'])
    fire_dates = set(firms_df['acq_date'].dt.date)

    for idx, row in weather_df.iterrows():
        check_date = row['date'].date()
        for delta in range(-window_days, window_days + 1):
            candidate = check_date + timedelta(days=delta)
            if candidate in fire_dates:
                weather_df.at[idx, 'fire'] = int(1)
                break

    fire_pct = weather_df['fire'].mean() * 100
    print(f"  🔥 {region_name}: {weather_df['fire'].sum()} fire days "
          f"({fire_pct:.1f}% of dataset)")
    return weather_df


# ─────────────────────────────────────────────
# 6. MASTER PIPELINE
# ─────────────────────────────────────────────

def build_features_for_region(region_name):
    """
    Full feature engineering pipeline for one region.
    """
    print(f"\n{'='*60}")
    print(f"⚙️  Feature Engineering: {REGIONS[region_name]['name']}")
    print(f"{'='*60}")

    # Load raw data directly from the processed step 2A output if possible, otherwise raw
    firms_path   = f"data/case_studies/{region_name}/raw/firms/firms_{region_name}.csv"
    weather_path = f"data/case_studies/{region_name}/raw/weather/daily_weather.csv"
    index_dir    = f"data/case_studies/{region_name}/processed/indices"

    try:
        firms_df   = pd.read_csv(firms_path)
        firms_df['acq_date'] = pd.to_datetime(firms_df['acq_date'])
    except FileNotFoundError:
        print(f"  ⚠️  FIRMS file not found. Run download_firms.py first.")
        firms_df = pd.DataFrame()

    try:
        weather_df = pd.read_csv(weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
    except FileNotFoundError:
        print(f"  ❌ Weather file not found. Run weather_collector.py first.")
        return None

    # Build features
    df = add_vpd(weather_df)
    df = add_rolling_features(df)
    df = add_temporal_features(df, region_name)
    df = add_satellite_features(df, region_name, index_dir)
    df = create_fire_labels(df, firms_df, region_name)

    df['region'] = region_name

    # Drop rows where all key features are NaN
    key_features = ['temperature_2m_mean', 'relative_humidity_2m_mean',
                    'precipitation_sum', 'vpd']
    df = df.dropna(subset=key_features)

    # Save
    out_dir = f"data/case_studies/{region_name}/processed/tabular"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/features_{region_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  💾 Saved {len(df)} rows × {df.shape[1]} features → {out_path}")
    print(f"  📊 Class balance: {df['fire'].value_counts().to_dict()}")
    return df


def build_all_regions():
    """Run feature engineering for all 4 regions."""
    all_features = {}
    for region_name in REGIONS.keys():
        df = build_features_for_region(region_name)
        all_features[region_name] = df
    return all_features


if __name__ == "__main__":
    all_features = build_all_regions()
    print("\n✅ Feature engineering complete for all regions!")