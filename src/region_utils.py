# src/region_utils.py

import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
import folium

# Root directory setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from region_config import REGIONS

class RegionManager:
    """
    🗺️ Manage geographic boundaries and data extraction for regions.
    """
    
    def __init__(self, region_name):
        self.region_name = region_name
        self.config = REGIONS[region_name]
        self.bounds = self.config['bounds']
        self.center = self.config['center']
        
        # Create boundary box for spatial filtering
        self.boundary_box = box(
            self.bounds['lon_min'],
            self.bounds['lat_min'],
            self.bounds['lon_max'],
            self.bounds['lat_max']
        )
    
    def filter_firms_data(self, firms_df):
        """
        🔥 Filters the master FIRMS dataset to only include fires within this specific region's bounding box.
        """
        filtered = firms_df[
            (firms_df['latitude'] >= self.bounds['lat_min']) &
            (firms_df['latitude'] <= self.bounds['lat_max']) &
            (firms_df['longitude'] >= self.bounds['lon_min']) &
            (firms_df['longitude'] <= self.bounds['lon_max'])
        ].copy()
        
        filtered['region'] = self.region_name
        filtered['region_name'] = self.config['name']
        
        return filtered
    
    def filter_weather_data(self, weather_df):
        """
        🌤️ Attaches region metadata to the weather dataframe.
        """
        weather_df['region'] = self.region_name
        return weather_df
    
    def create_grid(self, grid_size=0.05):
        """
        📍 Creates a spatial grid matrix covering the region (0.05° ≈ 5.5 km resolution).
        Useful for generating negative samples (no-fire zones) for the CNN model.
        """
        lats = np.arange(self.bounds['lat_min'], self.bounds['lat_max'], grid_size)
        lons = np.arange(self.bounds['lon_min'], self.bounds['lon_max'], grid_size)
        
        grid_points = []
        for lat in lats:
            for lon in lons:
                grid_points.append({
                    'grid_lat': lat,
                    'grid_lon': lon,
                    'region': self.region_name
                })
        
        return pd.DataFrame(grid_points)
    
    def get_fire_season_months(self):
        """
        🔥 Returns a list of standard integers representing the months in the active fire season.
        """
        # Strictly enforce standard int typing to prevent silent comparison errors
        start = int(self.config['characteristics']['fire_season_start'])
        end = int(self.config['characteristics']['fire_season_end'])
        
        if start <= end:
            return list(range(start, end + 1))
        else:  # Wraps around the New Year (e.g., Nov to Feb)
            return list(range(start, 13)) + list(range(1, end + 1))
    
    def create_folium_map(self):
        """
        🗺️ Generates an interactive Folium map focused on this specific region.
        """
        region_map = folium.Map(
            location=[self.center['lat'], self.center['lon']],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        folium.Rectangle(
            bounds=[
                [self.bounds['lat_min'], self.bounds['lon_min']],
                [self.bounds['lat_max'], self.bounds['lon_max']]
            ],
            color='red',
            fill=False,
            weight=3,
            popup=self.config['name']
        ).add_to(region_map)
        
        folium.Marker(
            location=[self.center['lat'], self.center['lon']],
            popup=f"{self.config['name']} (Center)",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(region_map)
        
        return region_map
    
    def get_region_summary(self):
        """
        📊 Prints a detailed tabular summary of the region's climate and topography.
        """
        print(f"\n{'='*60}")
        print(f"🌍 REGION: {self.config['name']}")
        print(f"{'='*60}")
        print(f"State: {self.config['state']} | District: {self.config['district']}")
        print(f"Area: {self.config['area_sq_km']:,} sq km")
        print(f"Elevation: {self.config['elevation_m']['min']}-{self.config['elevation_m']['max']} m")
        print(f"Forest Type: {self.config['forest_type']}")
        
        print(f"\n🔥 Fire Characteristics:")
        print(f"  Fire Season: Month {self.config['characteristics']['fire_season_start']} to Month {self.config['characteristics']['fire_season_end']}")
        print(f"  Avg Fires/Year: {self.config['fire_history']['avg_fires_per_year']}")
        print(f"  Avg Burned Area: {self.config['fire_history']['avg_burned_area_hectares']} hectares")
        
        print(f"\n🌤️ Climate:")
        print(f"  Annual Rainfall: {self.config['characteristics']['avg_annual_rainfall_mm']} mm")
        print(f"  Monsoon Months: {self.config['characteristics']['monsoon_months']}")
        print(f"  Dry Months: {self.config['characteristics']['dry_months']}")

def process_all_regions(func, *args, **kwargs):
    """
    Executes a given function across all 4 initialized region managers.
    """
    results = {}
    for region_name in REGIONS.keys():
        manager = RegionManager(region_name)
        results[region_name] = func(manager, *args, **kwargs)
    return results

if __name__ == "__main__":
    for region_name in REGIONS.keys():
        manager = RegionManager(region_name)
        manager.get_region_summary()