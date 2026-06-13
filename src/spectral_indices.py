# src/spectral_indices.py
import rasterio
import numpy as np

def compute_ndvi(nir_band, red_band):
    """
    Normalized Difference Vegetation Index (NDVI)
    Measures healthy, green vegetation. 
    Formula: (NIR - Red) / (NIR + Red)
    """
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    
    # np.where prevents "divide by zero" errors if the pixel is completely blank (0)
    ndvi = np.where(
        (nir + red) > 0,
        (nir - red) / (nir + red),
        -9999 # Standard nodata value in geospatial datasets
    )
    return ndvi

def compute_nbr(nir_band, swir2_band):
    """
    Normalized Burn Ratio (NBR)
    Highlights burned areas and estimates fire severity.
    Formula: (NIR - SWIR2) / (NIR + SWIR2)
    """
    nir = nir_band.astype(float)
    swir2 = swir2_band.astype(float)
    
    return np.where(
        (nir + swir2) > 0, 
        (nir - swir2) / (nir + swir2), 
        -9999
    )

def compute_dnbr(nbr_pre, nbr_post):
    """
    Delta NBR (dNBR) = NBR_pre - NBR_post
    Calculates the absolute change in the landscape caused by the fire.
    """
    return nbr_pre - nbr_post

def classify_burn_severity(dnbr):
    """
    Converts the raw dNBR decimal value into the official USGS burn severity classes.
    0 = Regrowth, 5 = High Severity burn.
    """
    # Enforcing standard int typing
    severity = np.zeros_like(dnbr, dtype=int)
    
    severity[dnbr < -0.25]  = 0  # Enhanced regrowth (Vegetation grew back greener)
    severity[(dnbr >= -0.25) & (dnbr < 0.1)] = 1  # Unburned
    severity[(dnbr >= 0.1) & (dnbr < 0.27)] = 2  # Low severity
    severity[(dnbr >= 0.27) & (dnbr < 0.44)] = 3 # Moderate-low
    severity[(dnbr >= 0.44) & (dnbr < 0.66)] = 4 # Moderate-high
    severity[dnbr >= 0.66] = 5  # High severity (Total canopy destruction)
    
    return severity

def extract_pixel_values(raster_path, lat, lon, buffer=0):
    """
    Opens a satellite image (GeoTIFF) and extracts the exact pixel value at a given GPS coordinate.
    If buffer > 0, it takes the average of the surrounding pixels.
    """
    try:
        with rasterio.open(raster_path) as src:
            # Convert GPS coordinates (Lat/Lon) to Image Coordinates (Row/Col)
            row, col = src.index(lon, lat)
            
            # Define a small window around the coordinate
            window = rasterio.windows.Window(col - buffer, row - buffer, 2 * buffer + 1, 2 * buffer + 1)
            
            # Read the pixel data inside that window
            data = src.read(window=window)
            
            # Return the average value of those pixels
            return data.mean(axis=(1, 2))
            
    except IndexError:
        # Safety catch: If the NASA FIRMS fire coordinate is slightly outside 
        # the bounds of our downloaded satellite map, return None instead of crashing.
        return None