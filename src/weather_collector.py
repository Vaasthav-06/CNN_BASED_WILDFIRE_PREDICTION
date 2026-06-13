# src/weather_collector.py
import os
import sys
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from region_config import REGIONS

# Setup cached + retry session to avoid hammering the API
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

class MultiRegionWeatherCollector:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

    def download_region_weather(self, region_name, start_date="2018-01-01", end_date="2026-06-01", force_refresh=False):
        """Download daily weather for a specific region from Open-Meteo."""
        config = REGIONS[region_name]
        center = config['center']
        
        # CORRECTED: Removed the '../' so it saves properly when run from the root terminal
        out_dir = f"data/case_studies/{region_name}/raw/weather"
        out_path = f"{out_dir}/daily_weather.csv"
        os.makedirs(out_dir, exist_ok=True)

        # Resuming / Checkpointing logic
        if os.path.exists(out_path) and not force_refresh:
            print(f"\n🌤️ Weather data for {config['name']} already exists. Skipping download.")
            print("   (Set force_refresh=True to overwrite).")
            return pd.read_csv(out_path)

        print(f"\n🌤️ Downloading weather: {config['name']}")
        print(f"   Location: ({center['lat']}, {center['lon']}) | Range: {start_date} to {end_date}")

        params = {
            "latitude": center['lat'],
            "longitude": center['lon'],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": config['critical_weather_vars'], # Dynamically pull vars from config
            "timezone": "Asia/Kolkata"
        }

        try:
            responses = openmeteo.weather_api(self.base_url, params=params)
            response = responses[0]
            hourly = response.Hourly()

            n_vals = int(hourly.Variables(0).ValuesAsNumpy().shape[0]) # Enforcing standard int
            time_index = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit='s', utc=True),
                periods=n_vals,
                freq='1h'
            ).tz_convert("Asia/Kolkata").tz_localize(None)
            
            # Dynamically build the dataframe based on the config requested variables
            hourly_data = {'datetime': time_index}
            for idx, var_name in enumerate(config['critical_weather_vars']):
                hourly_data[var_name] = hourly.Variables(idx).ValuesAsNumpy()

            hourly_df = pd.DataFrame(hourly_data)
            hourly_df['date'] = hourly_df['datetime'].dt.date

            # Define aggregation rules dynamically
            agg_rules = {}
            for var in config['critical_weather_vars']:
                if 'temperature' in var or 'humidity' in var:
                    agg_rules[var] = ['min', 'max', 'mean']
                elif 'precipitation' in var or 'evapotranspiration' in var:
                    agg_rules[var] = 'sum'
                elif 'wind_speed' in var:
                    agg_rules[var] = ['max', 'mean']
                elif 'wind_gusts' in var:
                    agg_rules[var] = 'max'
                elif 'soil_moisture' in var:
                    agg_rules[var] = 'mean'
                else:
                    agg_rules[var] = 'mean' # Default fallback

            daily_df = hourly_df.groupby('date').agg(agg_rules).reset_index()

            # Flatten multi-level column names (e.g. 'temperature_2m_mean')
            daily_df.columns = ['_'.join(col).strip('_') for col in daily_df.columns.values]
            
            daily_df['region'] = region_name
            daily_df['date'] = pd.to_datetime(daily_df['date'])

            # Save to disk
            daily_df.to_csv(out_path, index=False)
            print(f"  ✅ Retrieved {len(daily_df)} days → saved to {out_path}")
            return daily_df

        except Exception as e:
            print(f"  ❌ Error for {region_name}: {e}")
            return pd.DataFrame()

    def download_all_regions(self, start_date="2018-01-01", end_date="2026-06-01"):
        all_weather = {}
        for region_name in REGIONS.keys():
            all_weather[region_name] = self.download_region_weather(
                region_name, start_date, end_date
            )
        return all_weather


if __name__ == "__main__":
    print("🌤️ MULTI-REGION WEATHER DATA ACQUISITION")
    print("=" * 70)
    
    collector = MultiRegionWeatherCollector()
    
    # Executing the download for the 8.5 year date range
    regional_weather = collector.download_all_regions(start_date="2018-01-01", end_date="2026-06-01")

    print("\n📊 WEATHER DATA SUMMARY (8.5-Year Averages):")
    for region_name, df in regional_weather.items():
        if not df.empty:
            print(f"\n{REGIONS[region_name]['name']}:")
            # Calculate annual averages using standard int
            years = int(len(df['date'].dt.year.unique()))
            if 'temperature_2m_mean' in df.columns:
                print(f"  Avg Temp:      {df['temperature_2m_mean'].mean():.1f}°C")
            if 'precipitation_sum' in df.columns:
                print(f"  Annual Precip: {(df['precipitation_sum'].sum() / years):.0f} mm/year")