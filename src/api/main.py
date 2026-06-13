# src/api/main.py

import sys
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

# ── ROOT is two levels up from src/api/main.py ────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # → wildfire-prediction-system/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / ".env")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from src.region_config import REGIONS

from src.api.schemas import (
    TabularPredictRequest,
    TabularPredictResponse,
    AllRegionsRiskResponse,
    RegionRiskResponse,
    HealthResponse,
    ModelName,
    RegionFirePointsResponse,
    FirePoint,
    RegionTimelineResponse,
    MonthlyFireCount,
    AllRegionsGeoResponse,
    RegionGeoFeature,
)
from src.api.predictor import WildfirePredictor
from src.api.firms_cache import FIRMSCache

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("wildfire.api")


# ── Lifespan: load models once, share via app.state ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading wildfire prediction models & FIRMS cache …")
    app.state.predictor = WildfirePredictor()
    app.state.firms_cache = FIRMSCache()  # Initialize the FIRMS cache engine
    
    loaded = sum(app.state.predictor.load_status.values())
    total  = len(app.state.predictor.load_status)
    logger.info(f"Models ready: {loaded}/{total} loaded.")
    yield
    logger.info("API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Wildfire Prediction API",
    description=(
        "Serves wildfire fire-risk predictions from trained multimodal models "
        "across four Indian forest regions: Corbett, Jyotikuchi, Laisong, Similipal."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this to your frontend domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────
def _get_predictor(request: Request) -> WildfirePredictor:
    return request.app.state.predictor


# ── Routes: Prediction & Status ────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(request: Request):
    """
    Returns model load status and available regions.
    Call this first to verify the API is serving correctly.
    """
    predictor = _get_predictor(request)
    all_ok = all(predictor.load_status.values())
    return HealthResponse(
        status        = "ok" if all_ok else "degraded",
        models_loaded = predictor.load_status,
        regions       = list(predictor.feature_columns.keys()),
    )


@app.get("/regions/risk", response_model=AllRegionsRiskResponse, tags=["Risk Map"])
def get_all_regions_risk(request: Request):
    """
    Returns current fire-risk scores for all four regions.
    Risk is computed from median historical weather conditions.
    This is the primary endpoint for the live map view.
    """
    predictor = _get_predictor(request)
    try:
        region_results = predictor.predict_all_regions()
        return AllRegionsRiskResponse(
            regions   = [RegionRiskResponse(**r) for r in region_results],
            timestamp = datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.error(f"/regions/risk failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/tabular", response_model=TabularPredictResponse, tags=["Prediction"])
def predict_tabular(
    body:    TabularPredictRequest,
    request: Request,
    model:   ModelName = Query(ModelName.ensemble, description="Which model to use"),
    explain: bool      = Query(False, description="Include feature importances"),
):
    """
    Predict fire risk from a set of weather and environmental features.

    Send a JSON body with the required weather fields for a specific region.
    The `model` query parameter selects which trained model to use.
    Set `explain=true` to receive top-10 feature importances alongside the prediction.
    """
    predictor = _get_predictor(request)

    features = body.model_dump(exclude={"region"})
    features = {k: v for k, v in features.items() if v is not None}

    try:
        result = predictor.predict_tabular(
            region             = body.region.value,
            features           = features,
            model_name         = model.value,
            return_importances = explain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"/predict/tabular failed: {exc}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return TabularPredictResponse(**result)


@app.get("/regions/{region}/risk", response_model=RegionRiskResponse, tags=["Risk Map"])
def get_region_risk(region: str, request: Request):
    """
    Fire-risk score for a single region using median historical conditions.
    """
    predictor = _get_predictor(request)
    try:
        all_results = predictor.predict_all_regions()
        match = next((r for r in all_results if r["region"] == region.lower()), None)
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=f"Region '{region}' not found. "
                       f"Valid regions: {list(predictor.feature_columns.keys())}",
            )
        return RegionRiskResponse(**match)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"/regions/{region}/risk failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Routes: Geospatial & FIRMS Data ────────────────────────────────────────

@app.get("/regions/geo", response_model=AllRegionsGeoResponse, tags=["Risk Map"])
def get_regions_geo(request: Request):
    """
    Returns bounding box polygons and metadata for all regions.
    Used to draw region rectangles on the map.
    """
    features = []
    for key, cfg in REGIONS.items():
        ch = cfg.get("characteristics", {})
        fh = cfg.get("fire_history", {})
        features.append(RegionGeoFeature(
            region             = key,
            name               = cfg["name"],
            state              = cfg["state"],
            area_sq_km         = cfg["area_sq_km"],
            center             = cfg["center"],
            bounds             = cfg["bounds"],
            fire_season_start  = int(ch.get("fire_season_start", 1)),
            fire_season_end    = int(ch.get("fire_season_end", 5)),
            avg_fires_per_year = int(fh.get("avg_fires_per_year", 0)),
            forest_type        = cfg.get("forest_type", ""),
        ))
    return AllRegionsGeoResponse(features=features)


@app.get(
    "/regions/{region}/firms",
    response_model=RegionFirePointsResponse,
    tags=["Risk Map"],
)
def get_region_firms(
    region:    str,
    request:   Request,
    days_back: int = 365 * 5,
):
    """
    Historical FIRMS fire points for a region (heatmap data).
    Use days_back to control how far back to go (default 5 years).
    """
    region = region.lower()
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
    
    cache  = request.app.state.firms_cache
    points = cache.get_region_points(region, days_back=days_back)
    
    return RegionFirePointsResponse(
        region    = region,
        points    = [FirePoint(**p) for p in points],
        total     = len(points),
        days_back = days_back,
    )


@app.get(
    "/regions/{region}/timeline",
    response_model=RegionTimelineResponse,
    tags=["Risk Map"],
)
def get_region_timeline(region: str, request: Request):
    """
    Monthly fire counts for the last 3 years (sparkline data).
    """
    region = region.lower()
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
    
    cache    = request.app.state.firms_cache
    timeline = cache.get_monthly_fire_counts(region)
    
    return RegionTimelineResponse(
        region   = region,
        timeline = [MonthlyFireCount(**t) for t in timeline],
    )