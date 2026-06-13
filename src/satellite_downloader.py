# src/satellite_downloader.py
import os
import sys
import ee

# =========================================================
# ROOT DIRECTORY SETUP
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from region_config import REGIONS

# =========================================================
# Step 1: Connect to Google Earth Engine
# =========================================================
# If this is your first time running this, a browser window will pop up asking you to log in.
ee.Authenticate()   
ee.Initialize(project='beaming-sunset-498205-r9')

def get_sentinel2_indices(region_name, year, max_clouds=20):
    """
    Downloads and processes Sentinel-2 satellite data for a specific region and year.
    """
    config = REGIONS[region_name]
    bounds = config["bounds"]
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    roi = ee.Geometry.BBox(
        bounds["lon_min"], bounds["lat_min"],
        bounds["lon_max"], bounds["lat_max"]
    )

    def mask_s2_clouds(image):
        """Removes cloudy pixels so our vegetation data is clear."""
        qa = image.select("QA60")
        
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        
        mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
                 .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
        
        return image.updateMask(mask).divide(10000)

    def add_indices(img):
        """Calculates NDVI (Greenness), NBR (Burn), NDMI (Moisture), and BSI (Bare Soil)."""
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        nbr  = img.normalizedDifference(["B8", "B12"]).rename("NBR")
        ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
        
        bsi  = img.expression(
            "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
            {
                "SWIR": img.select("B11"),
                "RED":  img.select("B4"),
                "NIR":  img.select("B8"),
                "BLUE": img.select("B2")
            }
        ).rename("BSI")
        
        return img.addBands([ndvi, nbr, ndmi, bsi])

    print(f"  [{year}] Fetching Sentinel-2 data...")
    
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_clouds))
          .map(mask_s2_clouds))

    # SAFETY CHECK: Verify images exist before processing
    try:
        image_count = int(s2.size().getInfo())
        print(f"  [{year}] Found {image_count} usable images.")
        
        if image_count == 0:
            print(f"  ⚠️ WARNING: 0 images found for {year}. Skipping export.")
            return None
    except Exception as e:
        print(f"  ⚠️ Error checking image count: {e}. Skipping export.")
        return None

    s2_indexed = s2.map(add_indices)

    # ---------------------------------------------------------
    # Step 3: Export to Google Drive
    # ---------------------------------------------------------
    export_image = s2_indexed.median().select(["NDVI", "NBR", "NDMI", "BSI"])

    task = ee.batch.Export.image.toDrive(
        image=export_image,
        description=f"S2_indices_{region_name}_{year}",
        folder=f"wildfire_data_{region_name}", # Organized by region in your Drive
        region=roi,
        scale=20, # 20m resolution is the standard for Sentinel-2 SWIR bands
        crs="EPSG:4326",
        maxPixels=1e13
    )
    
    task.start()
    return task

def download_all_regions(start_year=2018, end_year=2025, max_clouds=25):
    """Export Sentinel-2 indices for all 4 regions year-by-year."""
    tasks = []
    
    for region_name in REGIONS.keys():
        print(f"\n🚀 Queuing exports for: {REGIONS[region_name]['name']}")
        
        for year in range(start_year, end_year + 1):
            task = get_sentinel2_indices(region_name, year, max_clouds)
            if task is not None:
                tasks.append(task)

    print(f"\n✅ {len(tasks)} export tasks started in Google Earth Engine.")
    print("Check your Google Drive folders in ~30-60 minutes.")
    print("You can monitor the progress at: https://code.earthengine.google.com/tasks")
    return tasks


if __name__ == "__main__":
    print("🛰️ MULTI-REGION SATELLITE DATA ACQUISITION")
    print("=" * 70)
    
    # We will loop through 2018 to 2025 to match our FIRMS and Weather data
    # (Note: Sentinel-2 SR harmonized data availability varies before 2019, 
    # so some early years might skip automatically if no images are found)
    active_tasks = download_all_regions(
        start_year=2018,
        end_year=2025,
        max_clouds=30 # Slightly higher cloud tolerance for annual medians
    )