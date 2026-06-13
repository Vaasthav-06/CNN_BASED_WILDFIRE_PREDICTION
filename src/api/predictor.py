# src/api/predictor.py

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

# ── ROOT is two levels up from src/api/predictor.py ───────────────────────
ROOT = Path(__file__).resolve().parents[2]   # → wildfire-prediction-system/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.region_config import REGIONS

logger = logging.getLogger("wildfire.predictor")

# ── Risk label thresholds ──────────────────────────────────────────────────
def _risk_label(prob: float) -> str:
    if prob < 0.25:   return "Low"
    if prob < 0.50:   return "Moderate"
    if prob < 0.75:   return "High"
    return "Extreme"


class WildfirePredictor:

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir  = Path(models_dir) if models_dir else ROOT / "saved_models"
        self.tabular_dir = self.models_dir / "tabular"

        self.models:          dict = {}
        self.load_status:     dict = {"xgboost": False, "catboost": False, "lightgbm": False}
        self.feature_columns: dict = {}   # region → list[str]
        self.medians:         dict = {}   # region → {col: median_val}

        self._load_models()
        self._load_regional_schemas()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_models(self):
        candidates = {
            "xgboost":  "xgboost_model.pkl",
            "catboost": "catboost_model.pkl",
            "lightgbm": "lightgbm_model.pkl",
        }
        for name, filename in candidates.items():
            path = self.tabular_dir / filename
            if not path.exists():
                logger.warning(f"Model not found: {path}")
                continue
            try:
                self.models[name]      = joblib.load(path)
                self.load_status[name] = True
                logger.info(f"Loaded {name} from {path}")
            except Exception as exc:
                logger.error(f"Failed to load {name}: {exc}")

    def _load_regional_schemas(self):
        for region in REGIONS.keys():
            csv_path = (
                ROOT / "data" / "case_studies" / region
                / "processed" / "tabular" / f"features_{region}.csv"
            )
            if not csv_path.exists():
                logger.warning(f"Feature CSV not found for {region}: {csv_path}")
                continue
            try:
                df = pd.read_csv(csv_path)

                # Store numeric medians for missing-feature fallback
                self.medians[region] = df.median(numeric_only=True).to_dict()

                # Prefer the column order the model was actually trained on
                if self.load_status["xgboost"]:
                    xgb_model = self.models["xgboost"]
                    if hasattr(xgb_model, "feature_names_in_"):
                        self.feature_columns[region] = list(xgb_model.feature_names_in_)
                    elif hasattr(xgb_model, "feature_names"):
                        self.feature_columns[region] = list(xgb_model.feature_names)

                # Fallback: derive from CSV columns
                if region not in self.feature_columns:
                    drop_cols = {"fire", "image_path", "region", "date"} | {
                        c for c in df.columns if c.startswith("Unnamed")
                    }
                    self.feature_columns[region] = [
                        c for c in df.columns if c not in drop_cols
                    ]

                logger.info(
                    f"Feature schema loaded for {region}: "
                    f"{len(self.feature_columns[region])} cols"
                )
            except Exception as exc:
                logger.warning(f"Error loading schema for {region}: {exc}")

    # ── Public predict API ─────────────────────────────────────────────────

    def predict_tabular(
        self,
        region:             str,
        features:           dict,
        model_name:         str  = "xgboost",
        return_importances: bool = False,
    ) -> dict:
        """
        Single-row prediction.

        Parameters
        ----------
        region            : one of the REGIONS keys
        features          : {col_name: value} — missing cols filled with median
        model_name        : "xgboost" | "catboost" | "lightgbm" | "ensemble"
        return_importances: include top-10 feature importances in response
        """
        if region not in REGIONS:
            raise ValueError(f"Unknown region: '{region}'.")

        # ── Ensemble: average the three models ────────────────────────────
        if model_name == "ensemble":
            probas = []
            for name in ("xgboost", "catboost", "lightgbm"):
                if not self.load_status.get(name):
                    continue
                try:
                    p = self.predict_tabular(region, features, model_name=name)
                    probas.append(p["fire_probability"])
                except Exception as exc:
                    logger.warning(f"Ensemble: {name} failed: {exc}")
            if not probas:
                raise RuntimeError("No models available for ensemble prediction.")
            proba = float(np.mean(probas))
            return {
                "region":           region,
                "model_used":       "ensemble (XGBoost + CatBoost + LightGBM)",
                "fire_probability": round(proba, 4),
                "fire_risk_label":  _risk_label(proba),
                "confidence":       round(min(1.0, abs(proba - 0.5) * 2), 4),
            }

        # ── Single model ───────────────────────────────────────────────────
        if not self.load_status.get(model_name):
            raise RuntimeError(f"Model '{model_name}' is not loaded.")
        if region not in self.feature_columns:
            raise ValueError(f"No feature schema for region '{region}'.")

        model          = self.models[model_name]
        expected_cols  = self.feature_columns[region]
        region_medians = self.medians.get(region, {})

        # Build the feature vector in training column order,
        # filling any missing field with the training-set median
        row = [
            features.get(col, region_medians.get(col, 0.0))
            for col in expected_cols
        ]
        arr = np.array([row], dtype=np.float32)

        # Hard-align to whatever shape the model expects
        expected_n = getattr(model, "n_features_in_", arr.shape[1])
        if arr.shape[1] < expected_n:
            arr = np.hstack([arr, np.zeros((1, expected_n - arr.shape[1]), dtype=np.float32)])
        elif arr.shape[1] > expected_n:
            arr = arr[:, :expected_n]

        proba = float(model.predict_proba(arr)[0][1])

        response = {
            "region":           region,
            "model_used":       model_name,
            "fire_probability": round(proba, 4),
            "fire_risk_label":  _risk_label(proba),
            "confidence":       round(min(1.0, abs(proba - 0.5) * 2), 4),
        }

        if return_importances and hasattr(model, "feature_importances_"):
            imps      = model.feature_importances_
            n         = min(len(expected_cols), len(imps))
            top10     = sorted(
                zip(expected_cols[:n], map(float, imps[:n])),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            response["feature_importances"] = {k: round(v, 6) for k, v in top10}

        return response

    def predict_all_regions(self) -> list[dict]:
        """
        Fire-risk estimate for every region using median weather conditions.
        Used to populate the live map.
        """
        results = []
        for region, cfg in REGIONS.items():
            if region not in self.feature_columns:
                logger.warning(f"predict_all_regions: skipping {region} — no schema")
                continue
            try:
                pred = self.predict_tabular(
                    region     = region,
                    features   = {},          # all-median fallback
                    model_name = "ensemble",
                )
                results.append({
                    "region":           region,
                    "region_name":      cfg["name"],   # ← needed by RegionRiskResponse
                    "fire_probability": pred["fire_probability"],
                    "fire_risk_label":  pred["fire_risk_label"],
                    "last_updated":     pd.Timestamp.now(tz="UTC").isoformat(),
                })
            except Exception as exc:
                logger.error(f"predict_all_regions failed for {region}: {exc}")
                results.append({
                    "region":           region,
                    "region_name":      cfg.get("name", region),
                    "fire_probability": 0.0,
                    "fire_risk_label":  "Low",
                    "last_updated":     pd.Timestamp.now(tz="UTC").isoformat(),
                })
        return results