# api/schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from enum import Enum


class RegionName(str, Enum):
    corbett    = "corbett"
    jyotikuchi = "jyotikuchi"
    laisong    = "laisong"
    similipal  = "similipal"


class TabularPredictRequest(BaseModel):
    """
    Weather + environmental features for a single prediction.
    All fields mirror the columns produced by feature_engineering.py.
    Only the core weather features are required; derived/rolling features
    are computed server-side so the client stays simple.
    """
    region: RegionName

    # Core weather observations (required)
    temperature_2m_mean:        float = Field(..., ge=-20,  le=60,   description="°C")
    temperature_2m_max:         float = Field(..., ge=-20,  le=60,   description="°C")
    relative_humidity_2m_mean:  float = Field(..., ge=0,    le=100,  description="%")
    precipitation_sum:          float = Field(..., ge=0,    le=500,  description="mm")
    wind_speed_10m_max:         float = Field(..., ge=0,    le=150,  description="km/h")
    wind_speed_10m_mean:        float = Field(..., ge=0,    le=150,  description="km/h")
    soil_moisture_0_1cm_mean:   float = Field(..., ge=0,    le=1,    description="m³/m³")
    vpd:                        float = Field(..., ge=0,    le=10,   description="kPa – vapour pressure deficit")

    # Optional contextual fields (used when present, ignored when absent)
    month:                      Optional[int]   = Field(None, ge=1, le=12)
    day_of_year:                Optional[int]   = Field(None, ge=1, le=366)
    ndvi_mean:                  Optional[float] = Field(None, ge=-1, le=1)
    nbr_mean:                   Optional[float] = Field(None, ge=-1, le=1)
    frp_mean:                   Optional[float] = Field(None, ge=0)
    fire_count_7d:              Optional[int]   = Field(None, ge=0)

    @field_validator("region", mode="before")
    @classmethod
    def normalise_region(cls, v):
        return v.lower() if isinstance(v, str) else v


class ModelName(str, Enum):
    xgboost   = "xgboost"
    catboost  = "catboost"
    lightgbm  = "lightgbm"
    ensemble  = "ensemble"      # averaged prediction across all three


class TabularPredictResponse(BaseModel):
    region:          str
    model_used:      str
    fire_probability: float = Field(..., ge=0, le=1, description="P(fire) from 0 to 1")
    fire_risk_label: Literal["Low", "Moderate", "High", "Extreme"]
    confidence:      float = Field(..., ge=0, le=1)
    feature_importances: Optional[dict[str, float]] = None


class RegionRiskResponse(BaseModel):
    region:           str
    region_name:      str
    fire_probability: float
    fire_risk_label:  Literal["Low", "Moderate", "High", "Extreme"]
    last_updated:     str


class AllRegionsRiskResponse(BaseModel):
    regions:    list[RegionRiskResponse]
    timestamp:  str


class HealthResponse(BaseModel):
    status:         Literal["ok", "degraded"]
    models_loaded:  dict[str, bool]
    regions:        list[str]
    version:        str = "1.0.0"


# ── Phase 3 additions ──────────────────────────────────────────────────────

class FirePoint(BaseModel):
    lat:  float
    lon:  float
    frp:  float = 1.0       # Fire Radiative Power (MW)
    date: str               # YYYY-MM-DD


class RegionFirePointsResponse(BaseModel):
    region:      str
    points:      list[FirePoint]
    total:       int
    days_back:   int


class MonthlyFireCount(BaseModel):
    month: str    # "YYYY-MM"
    count: int


class RegionTimelineResponse(BaseModel):
    region:   str
    timeline: list[MonthlyFireCount]


class RegionGeoFeature(BaseModel):
    """GeoJSON-style feature for one region polygon."""
    type:       str = "Feature"
    region:     str
    name:       str
    state:      str
    area_sq_km: int
    center:     dict        # {lat, lon}
    bounds:     dict        # {lat_min, lat_max, lon_min, lon_max}
    fire_season_start: int
    fire_season_end:   int
    avg_fires_per_year: int
    forest_type:       str


class AllRegionsGeoResponse(BaseModel):
    features: list[RegionGeoFeature]