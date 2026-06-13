# region_config.py - Region-Specific Parameters

REGIONS = {
    # ============================================
    # 1️⃣ LAISONG RESERVED FOREST (Assam)
    # ============================================
    "laisong": {
        "name": "Laisong Reserved Forest",
        "state": "Assam",
        "district": "Dima Hasao",
        "center": {"lat": 25.8500, "lon": 92.9500},
        "bounds": {
            "lat_min": 25.75,
            "lat_max": 25.95,
            "lon_min": 92.85,
            "lon_max": 93.05
        },
        "area_sq_km": 450,
        "elevation_m": {"min": 1200, "max": 1800},
        "forest_type": "Tropical Deciduous Mixed Forest",
        "climate": "Humid Subtropical",
        
        # 🔥 ADDED: FIRMS sources for historical archive
        "firms_sources": ["VIIRS_SNPP_SP", "MODIS_SP"],
        
        # 🌍 Geographic characteristics
        "characteristics": {
            "monsoon_months": [5, 6, 7, 8, 9],  
            "dry_months": [11, 12, 1, 2, 3, 4],       
            "fire_season_start": 11,
            "fire_season_end": 4, # Changed from 2 to 4 (April)
            "avg_annual_rainfall_mm": 2000,
        },
        
        # 🔥 Historical fire data
        "fire_history": {
            "high_fire_years": [2019, 2020, 2022],
            "major_fire_month": 1,  # January
            "avg_fires_per_year": 15,
            "avg_burned_area_hectares": 200
        },
        
        # 🌤️ Critical weather parameters for this region
        "critical_weather_vars": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "soil_moisture_0_1cm",
            "et0_fao_evapotranspiration"
        ],
        
        # 📸 Satellite imagery parameters
        "satellite_settings": {
            "tile_size": 256,
            "buffer_days_before_fire": 14,
            "buffer_days_after_fire": 3,
            "preferred_sensors": ["Sentinel-2", "Sentinel-1"]
        }
    },
    
    # ============================================
    # 2️⃣ JYOTIKUCHI DHOPOLIA HILL (Guwahati)
    # ============================================
    "jyotikuchi": {
        "name": "Jyotikuchi Dhopolia Hill",
        "state": "Assam",
        "district": "Kamrup",
        "center": {"lat": 26.1667, "lon": 91.7667},
        "bounds": {
            "lat_min": 26.10,
            "lat_max": 26.23,
            "lon_min": 91.70,
            "lon_max": 91.83
        },
        "area_sq_km": 85,
        "elevation_m": {"min": 100, "max": 600},
        "forest_type": "Sub-Tropical Deciduous (Urban-Forest Interface)",
        "climate": "Humid Subtropical (Urban Modified)",
        
        # 🔥 ADDED: FIRMS sources for historical archive
        "firms_sources": ["VIIRS_SNPP_SP", "MODIS_SP"],
        
        "characteristics": {
            "monsoon_months": [5, 6, 7, 8, 9],
            "dry_months": [10, 11, 12, 1, 2, 3, 4], # Added 3 and 4
            "fire_season_start": 9, # Start in September (to catch the Sept/Oct spikes)
            "fire_season_end": 4,   # End in April 
            "avg_annual_rainfall_mm": 2200,
            "urban_influence": True, 
            "population_density_high": True
        },
        
        "fire_history": {
            "high_fire_years": [2018, 2019, 2021, 2023],
            "major_fire_month": 11,  # November
            "avg_fires_per_year": 8,
            "avg_burned_area_hectares": 50,
            "human_caused_incidents": 0.65  # 65% human-caused
        },
        
        "critical_weather_vars": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m"
        ],
        
        "satellite_settings": {
            "tile_size": 256,
            "buffer_days_before_fire": 7,  # Shorter buffer - rapid fire progression
            "buffer_days_after_fire": 2,
            "preferred_sensors": ["Sentinel-2"]
        }
    },
    
    # ============================================
    # 3️⃣ CORBETT NATIONAL PARK (Uttarakhand)
    # ============================================
    "corbett": {
        "name": "Corbett National Park",
        "state": "Uttarakhand",
        "district": "Nainital",
        "center": {"lat": 29.3900, "lon": 79.2800},
        "bounds": {
            "lat_min": 29.25,
            "lat_max": 29.50,
            "lon_min": 79.10,
            "lon_max": 79.45
        },
        "area_sq_km": 1318,
        "elevation_m": {"min": 300, "max": 2400},
        "forest_type": "Himalayan Subtropical & Temperate",
        "climate": "Humid Subtropical transitioning to Temperate",
        
        # 🔥 ADDED: FIRMS sources for historical archive
        "firms_sources": ["VIIRS_SNPP_SP", "MODIS_SP"],
        
        "characteristics": {
            "monsoon_months": [6, 7, 8, 9],
            "dry_months": [10, 11, 12, 1, 2, 3, 4],
            "fire_season_start": 2,
            "fire_season_end": 5,  # Longer fire season (elevation-dependent)
            "avg_annual_rainfall_mm": 1600,
            "elevation_variance": True  # Multiple elevation zones
        },
        
        "fire_history": {
            "high_fire_years": [2016, 2018, 2020, 2021],
            "major_fire_month": 4,  # April (pre-monsoon)
            "avg_fires_per_year": 25,
            "avg_burned_area_hectares": 400,
            "fire_behavior": "Complex (elevation-dependent)"
        },
        
        "critical_weather_vars": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",  # Important for high elevations
            "soil_moisture_0_1cm"
        ],
        
        "satellite_settings": {
            "tile_size": 256,
            "buffer_days_before_fire": 21,  # Longer buffer - seasonal patterns
            "buffer_days_after_fire": 5,
            "preferred_sensors": ["Sentinel-2"]
        }
    },
    
    # ============================================
    # 4️⃣ SIMILIPAL NATIONAL PARK (Odisha)
    # ============================================
    "similipal": {
        "name": "Similipal National Park",
        "state": "Odisha",
        "district": "Mayurbhanj",
        "center": {"lat": 22.2340, "lon": 86.4050},
        "bounds": {
            "lat_min": 22.05,
            "lat_max": 22.40,
            "lon_min": 86.15,
            "lon_max": 86.65
        },
        "area_sq_km": 2750,
        "elevation_m": {"min": 250, "max": 1150},
        "forest_type": "Tropical Dry Deciduous",
        "climate": "Tropical Dry",
        
        # 🔥 ADDED: FIRMS sources for historical archive
        "firms_sources": ["VIIRS_SNPP_SP", "MODIS_SP"],
        
        "characteristics": {
            "monsoon_months": [6, 7, 8, 9],
            "dry_months": [10, 11, 12, 1, 2, 3, 4, 5],  # Very long dry season!
            "fire_season_start": 1,
            "fire_season_end": 5,
            "avg_annual_rainfall_mm": 1450,
            "high_fire_susceptibility": True  # Dry deciduous = high burn risk
        },
        
        "fire_history": {
            "high_fire_years": [2018, 2019, 2021, 2023],
            "major_fire_month": 3,  # March
            "avg_fires_per_year": 35,  # HIGHEST of all 4 regions
            "avg_burned_area_hectares": 800,  # LARGEST burned areas
            "fire_frequency": "Very High"
        },
        
        "critical_weather_vars": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "soil_moisture_0_1cm"
        ],
        
        "satellite_settings": {
            "tile_size": 256,
            "buffer_days_before_fire": 30,  # Long buffer - seasonal peaks
            "buffer_days_after_fire": 7,    # Long recovery period
            "preferred_sensors": ["Sentinel-2"]
        }
    }
}

# Global parameters (apply to all regions)
GLOBAL_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "val_size": 0.1,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    
    # Ensemble weights
    "ensemble_weights": {
        "xgboost": 0.35,
        "catboost": 0.35,
        "cnn": 0.30
    },
    
    # FIRMS parameters
    "firms": {
        # 🔥 UPDATED: Using _SP tags for historical archive fetching
        "sources": ["VIIRS_SNPP_SP", "MODIS_SP"],
        "confidence_threshold": 30,
        "min_frp": 0.5
    }
}