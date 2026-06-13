# api/firms_cache.py
"""
Loads the master FIRMS archive CSVs once at startup and exposes
per-region fire-point queries with an in-memory cache.

The two master archive files are at:
  data/master_archive/fire_archive_M-C61_758212.csv   (MODIS)
  data/master_archive/fire_archive_SV-C2_758213.csv   (VIIRS)
"""

import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("wildfire.firms")

# Region bounding boxes — duplicated here so this module has no circular imports
_BOUNDS = {
    "corbett":    {"lat_min": 29.25, "lat_max": 29.50, "lon_min": 79.10, "lon_max": 79.45},
    "jyotikuchi": {"lat_min": 26.10, "lat_max": 26.23, "lon_min": 91.70, "lon_max": 91.83},
    "laisong":    {"lat_min": 25.75, "lat_max": 25.95, "lon_min": 92.85, "lon_max": 93.05},
    "similipal":  {"lat_min": 22.05, "lat_max": 22.40, "lon_min": 86.15, "lon_max": 86.65},
}


class FIRMSCache:
    """
    Loads MODIS + VIIRS archive files once, merges them, and answers
    spatial queries per region. Cached as a class-level DataFrame so
    all API requests share one copy.
    """

    _df: Optional[pd.DataFrame] = None   # class-level cache

    def __init__(self):
        if FIRMSCache._df is None:
            FIRMSCache._df = self._load()

    def _load(self) -> pd.DataFrame:
        archive_dir = ROOT / "data" / "master_archive"
        frames = []
        for csv_file in archive_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                df["source_file"] = csv_file.stem
                frames.append(df)
                logger.info(f"FIRMS: loaded {len(df):,} rows from {csv_file.name}")
            except Exception as exc:
                logger.warning(f"FIRMS: could not load {csv_file}: {exc}")

        if not frames:
            logger.warning("FIRMS: no archive files found — heatmap will be empty")
            return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp"])

        combined = pd.concat(frames, ignore_index=True)

        # Normalise column names (MODIS uses 'latitude'/'longitude'; VIIRS same)
        combined.columns = [c.lower().strip() for c in combined.columns]

        # Keep only the columns we need downstream
        keep = ["latitude", "longitude", "acq_date", "frp", "confidence", "source_file"]
        keep = [c for c in keep if c in combined.columns]
        combined = combined[keep].copy()

        # Parse date
        combined["acq_date"] = pd.to_datetime(combined["acq_date"], errors="coerce")
        combined = combined.dropna(subset=["latitude", "longitude", "acq_date"])
        combined["latitude"]  = combined["latitude"].astype(float)
        combined["longitude"] = combined["longitude"].astype(float)

        logger.info(f"FIRMS cache ready: {len(combined):,} total fire points")
        return combined

    def get_region_points(
        self,
        region: str,
        days_back: int = 365 * 5,   # default: last 5 years
        max_points: int = 2000,      # cap to keep API response lean
    ) -> list[dict]:
        """
        Return fire points within the region bounding box.
        Each point: {lat, lon, frp, date}
        """
        if region not in _BOUNDS:
            return []

        b   = _BOUNDS[region]
        df  = FIRMSCache._df
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

        mask = (
            (df["latitude"]  >= b["lat_min"]) & (df["latitude"]  <= b["lat_max"]) &
            (df["longitude"] >= b["lon_min"]) & (df["longitude"] <= b["lon_max"]) &
            (df["acq_date"]  >= cutoff)
        )
        subset = df[mask].copy()

        if len(subset) > max_points:
            subset = subset.sample(max_points, random_state=42)

        subset = subset.sort_values("acq_date", ascending=False)

        return [
            {
                "lat":  round(float(row["latitude"]),  5),
                "lon":  round(float(row["longitude"]), 5),
                "frp":  round(float(row["frp"]), 2) if "frp" in row and pd.notna(row["frp"]) else 1.0,
                "date": row["acq_date"].strftime("%Y-%m-%d"),
            }
            for _, row in subset.iterrows()
        ]

    def get_monthly_fire_counts(self, region: str, years_back: int = 3) -> list[dict]:
        """
        Return monthly fire counts for the region for the sparkline.
        Returns [{month: 'YYYY-MM', count: N}, ...] sorted by month.
        """
        if region not in _BOUNDS:
            return []

        b  = _BOUNDS[region]
        df = FIRMSCache._df
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * years_back)

        mask = (
            (df["latitude"]  >= b["lat_min"]) & (df["latitude"]  <= b["lat_max"]) &
            (df["longitude"] >= b["lon_min"]) & (df["longitude"] <= b["lon_max"]) &
            (df["acq_date"]  >= cutoff)
        )
        subset = df[mask].copy()
        if subset.empty:
            return []

        subset["month"] = subset["acq_date"].dt.to_period("M").astype(str)
        counts = subset.groupby("month").size().reset_index(name="count")
        counts = counts.sort_values("month")
        return counts.to_dict(orient="records")