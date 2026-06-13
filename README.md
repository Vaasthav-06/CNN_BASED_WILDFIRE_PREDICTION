# 🔥 WILDFIRE_CNN

A machine learning system that predicts wildfire risk across four Indian forest regions — before fires start. It combines satellite imagery, weather data, and vegetation health signals, then feeds everything into a three-stage AI pipeline: gradient-boosted trees, convolutional neural networks, and a multimodal ensemble that fuses both.

> 💡 **Four Case-Study Regions:** Corbett National Park · Jyotikuchi Dhopolia Hill · Laisong Reserved Forest · Similipal National Park

---

## 🎯 What Does This Project Do?

Given weather and satellite conditions for a region, the system:

- **Predicts next-day fire probability** for each of the four study regions
- **Outputs a risk score (0–1)** and fire/no-fire classification per location
- **Breaks down predictions by region** so you can see where the model is most and least confident
- **Serves predictions live** through a FastAPI backend and React dashboard that reads historical FIRMS fire points on an interactive map

The full training pipeline runs in three phases — tabular models first, then CNNs on satellite image tiles, then a fused ensemble that combines both signals.

---

## 📊 Where Does the Data Come From?

| Source | What It Tells Us | Coverage |
|---|---|---|
| **NASA FIRMS (VIIRS S-NPP C2)** | Exactly when and where past wildfires happened | 2018–2026 |
| **Open-Meteo Weather API** | Daily temperature, humidity, wind, precipitation, soil moisture | 2018–present |
| **Satellite Image Tiles** | Visual patches of each study region (fire vs. no-fire) | Per-region |


---

## 🧠 How the Pipeline Works

### Step 1 — Feature Engineering

Before any model training, `src/feature_engineering.py` turns raw weather readings into a richer set of predictors. Key engineered features include:

- **VPD** (Vapor Pressure Deficit) — how dry the air is
- **NDVI / NBR** (vegetation greenness and burn ratio from satellite bands)
- **NDVI anomaly & NBR deficit** — how much vegetation has changed from its normal state
- **Fuel dryness** — a composite measure of how combustible the ground-level vegetation is
- **Fire season flag** — binary indicator for peak fire months per region
- **Soil moisture** at 0–1 cm depth

### Step 2 — Phase 1: Tabular Models

Three gradient-boosted tree models are trained on the engineered feature set, benchmarked side-by-side:

- **LightGBM**
- **XGBoost**
- **CatBoost**

All three use `region` as a categorical feature so they can learn region-specific fire dynamics. Training uses 5-fold cross-validation with a spatial hash split to avoid data leakage between train and validation. After training, each model produces a per-region breakdown table showing sample counts, fire prevalence, and AUC for each of the four regions.

### Step 3 — Phase 2: CNN Models (Staged Progression)

 The CNN training follows a three-stage progression:

**Stage 1+2 — Baseline Comparison**
ResNet50 and EfficientNet-B0 are trained and benchmarked. A comparison table prints validation accuracy, average epoch time, and overfitting gap (train accuracy − val accuracy) for each.

**Stage 3 — Conditional Scaling**
Scaling is triggered automatically when:
- Best baseline val_acc < 0.85, **or**
- Overfitting gap > 0.05 (5%)

When triggered, the system scales to EfficientNet-B2 (moderate underfit, val_acc > 0.75) or EfficientNet-B3 (severe underfit). All CNN models use early stopping with patience = 7 epochs and a minimum improvement delta of 0.001.

> 💡 **Why this order?** The staged progression is a hard requirement: always establish baselines before scaling up. It documents exactly which model capacity the data actually needs.

### Step 4 — Phase 3: Multimodal Ensemble

Two fusion strategies combine the tabular and image branches:

**Late Fusion (Stacking)** — A logistic regression meta-learner is trained on the fire-risk probabilities produced by the CNN branch and the tabular OOF (out-of-fold) probabilities. The two modalities are decoupled: the tabular branch runs on its full CSV while the CNN branch runs on image-loader samples, since there is no shared row index between them.

Both fusion paths are evaluated on a held-out validation set for AUC, F1, and accuracy.

---


---

## 📁 Project Structure

```
wildfire-prediction-system/
│
├── src/
│   ├── api/
│   │   ├── main.py                    # FastAPI app, routes, lifespan startup
│   │   ├── predictor.py               # WildfirePredictor — loads models, runs inference
│   │   ├── firms_cache.py             # In-memory FIRMS fire-point cache
│   │   ├── schemas.py                 # Pydantic request/response schemas
│   │   └── .env.example               # Environment variable template
│   │
│   ├── models/
│   │   ├── tabular_models.py          # TabularModelTrainer (LightGBM · XGBoost · CatBoost)
│   │   ├── cnn_models.py              # CNNTrainer + GAPExtractor (ResNet50 · EfficientNet)
│   │   ├── ensemble.py                # MultimodalEnsemble (Early Fusion · Late Fusion)
│   │   └── test_models.py             # Smoke tests for all three model classes
│   │
│   ├── data_loader.py                 # Loads raw weather + FIRMS CSVs per region
│   ├── feature_engineering.py         # VPD, NDVI, NBR, fuel dryness, fire season flag
│   ├── image_tiler.py                 # Cuts satellite rasters into fire/no-fire tile patches
│   ├── region_config.py               # Bounding boxes, fire seasons, names for all regions
│   ├── region_utils.py                # RegionManager — spatial filtering helpers
│   ├── satellite_downloader.py        # Downloads satellite imagery per region
│   ├── spectral_indices.py            # Computes NDVI, NBR, and other band ratios
│   ├── weather_collector.py           # Pulls weather from Open-Meteo per region
│   ├── process_firms_archive.py       # Splits master FIRMS CSVs into per-region files
│   ├── prepare_combined_dataset.py    # Merges four region CSVs → one global CSV
│   ├── prepare_combined_cnn_dataset.py# Merges four image tile sets → one ImageFolder
│   └── train_global_model.py          # Full three-phase pipeline on pooled data
│
├── train_all_models.py                # Per-region three-phase training entry point
│
├── notebooks/
│   ├── 02_eda_tabular_by_region.ipynb # 15-section EDA with 15+ charts per region
│   └── 10_visualization_fire_maps.ipynb
│
├── make_grids.py                      # Stitches output images into summary grid PNGs
├── make_pdf.py                        # Compiles all visualizations into a single PDF report
│
├── data/
│   ├── case_studies/
│   │   ├── corbett/
│   │   ├── jyotikuchi/
│   │   ├── laisong/
│   │   └── similipal/
│   │       ├── raw/firms/             # Per-region FIRMS CSVs
│   │       ├── raw/weather/           # Per-region daily weather CSVs
│   │       └── processed/
│   │           ├── tabular/           # Engineered feature CSVs
│   │           └── cnn_dataset/       # fire/ and no_fire/ image tile folders
│   ├── combined/                      # Global pooled data (generated)
│   └── master_archive/                # Full MODIS + VIIRS archive CSVs
│
├── saved_models/                      # Trained .pkl (tabular) and .pt (CNN) files
│   └── global/                        # Global model weights
│
└── outputs/
    ├── case_studies/<region>/visualizations/
    ├── comparative_analysis/          # Cross-region benchmark charts
    ├── global/                        # Global model outputs
    ├── ensemble/                      # Fusion training curves and metrics
    └── summary_grids/                 # Grid PNGs and compiled PDF report
```


## 💡 Key Design Decisions

**Region as a predictor, not just a label.** In the global model, the `region` column is passed as a categorical feature to all three gradient-boosted tree models. This means a single global model can still distinguish between the dry deciduous patterns of Corbett and the moist semi-evergreen patterns of Similipal without needing separate weights.

**Staged CNN progression is a hard constraint.** Baselines (ResNet50 + EfficientNet-B0) are always trained and benchmarked before any scaling decision is made. Scaling to B2 or B3 is triggered by data — specifically, val_acc < 0.85 or an overfitting gap > 0.05 — not by default.

**Decoupled fusion.** The tabular CSV (daily aggregated weather and FIRMS signals) and the CNN image tiles (per-point spatial patches) share no common row index. Early fusion therefore uses CNN GAP features only. Late fusion uses a meta-learner on CNN probabilities and tabular out-of-fold probabilities, evaluated independently.

---

## 🚀 Future Steps

- Compare global model AUC against per-region models across all four regions
- Build a per-tile feature CSV (one tabular row per image tile, keyed by date and location) to enable true early fusion with tabular + image concatenation
- Add forest cover density as a predictor to improve performance in densely vegetated areas
- Expand to additional fire-prone regions across South and Southeast Asia

---

*Built to give communities and responders an early read on fire risk — before the smoke starts.*
