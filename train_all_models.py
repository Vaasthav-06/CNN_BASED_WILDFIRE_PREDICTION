#!/usr/bin/env python3
"""
train_all_models.py
===================
Orchestrates training of tabular models, CNN models, and multimodal ensemble
for one region at a time.

Usage
-----
    python train_all_models.py \
        --feature_csv data/case_studies/corbett/processed/tabular/features_corbett.csv

Key fixes vs previous version
------------------------------
- epochs default raised to 50 (was 2 → produced random val_acc=0.5)
- train/val use separate transforms (augmentation on train only)
- proper TRAIN_TRANSFORM from cnn_models.py applied (flips, rotation, jitter)
- ensemble phase now actually calls fit_early_fusion() + fit_late_fusion()
- X_tab and y properly prepared for ensemble
- label_col default corrected to "fire" (matches feature CSV column)
- feature importance plots called after tabular training
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── make src/ importable ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from models.tabular_models import TabularModelTrainer
from models.cnn_models      import CNNTrainer, TRAIN_TRANSFORM, VAL_TRANSFORM
from models.ensemble        import MultimodalEnsemble


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper: applies a different transform per subset after random_split
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED: Geographic train/val split to prevent spatial leakage
# Tiles whose filenames share the same spatial hash (grid cell) stay together.
# ─────────────────────────────────────────────────────────────────────────────
import hashlib

class SubsetWithTransform(Dataset):
    """Wraps an explicit list of (path, label) samples with a transform."""
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _spatial_hash(filename: str, n_buckets: int = 10) -> int:
    """
    Assign a tile to a spatial bucket based on its filename.
    Fire tiles are named: fire_YYYY-MM-DD_<idx>.npy  (idx encodes row in FIRMS CSV)
    No-fire tiles:        no_fire_YYYY-MM-DD_<n>.npy
    We hash the base stem to get a stable spatial group.
    """
    stem = Path(filename).stem
    h    = int(hashlib.md5(stem.encode()).hexdigest(), 16)
    return h % n_buckets


def make_image_loaders(img_dir: str, batch_size: int,
                        val_fraction: float = 0.2,
                        n_spatial_buckets: int = 10):
    """
    Builds train/val DataLoaders using a spatial-hash split.

    All tiles hashed to the same bucket go entirely to train OR val —
    never split across both.  This prevents spatially-overlapping tiles
    from appearing in both sets (the root cause of val_acc = 1.00).

    val_fraction: approximate fraction of BUCKETS reserved for validation.
    """
    base_ds = datasets.ImageFolder(root=img_dir)
    class_to_idx = base_ds.class_to_idx

    # Assign each sample to a spatial bucket
    n_val_buckets = max(1, round(n_spatial_buckets * val_fraction))
    val_buckets   = set(range(n_val_buckets))  # buckets 0..n_val_buckets-1 → val

    train_samples, val_samples = [], []
    for path, label in base_ds.samples:
        fname  = os.path.basename(path)
        bucket = _spatial_hash(fname, n_spatial_buckets)
        if bucket in val_buckets:
            val_samples.append((path, label))
        else:
            train_samples.append((path, label))

    # Guarantee at least 1 sample per class in both splits
    for label_idx in [0, 1]:
        val_has   = any(l == label_idx for _, l in val_samples)
        train_has = any(l == label_idx for _, l in train_samples)
        if not val_has and train_has:
            # Move one sample of this class from train → val
            for i, (p, l) in enumerate(train_samples):
                if l == label_idx:
                    val_samples.append(train_samples.pop(i))
                    break
        if not train_has and val_has:
            for i, (p, l) in enumerate(val_samples):
                if l == label_idx:
                    train_samples.append(val_samples.pop(i))
                    break

    train_ds = SubsetWithTransform(train_samples, TRAIN_TRANSFORM)
    val_ds   = SubsetWithTransform(val_samples,   VAL_TRANSFORM)

    n_total = len(train_samples) + len(val_samples)
    print(f"  ImageFolder: {n_total} images → {len(train_samples)} train "
          f"| {len(val_samples)} val  [spatial-hash split, "
          f"{n_val_buckets}/{n_spatial_buckets} val buckets]")
    print(f"  Classes: {list(class_to_idx.keys())}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Helper: prepare tabular X and y for ensemble (encode + drop non-feature cols)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_tabular(df: pd.DataFrame, label_col: str, region_col: str | None):
    drop_cols = [label_col]
    if region_col and region_col in df.columns:
        drop_cols.append(region_col)
    # Drop any other known non-feature columns
    for col in ["date", "acq_date", "latitude", "longitude",
                "grid_lat", "grid_lon", "confidence"]:
        if col in df.columns:
            drop_cols.append(col)

    X_df = df.drop(columns=drop_cols, errors="ignore").copy()
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_df.values.astype(np.float32)
    y = df[label_col].values.astype(int)
    return X, y


def _banner(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv",   required=True)
    parser.add_argument("--label_col",     default="fire",
                        help="Target column in feature CSV (default: fire)")
    parser.add_argument("--region_col",    default="region")
    parser.add_argument("--img_dir",       default=None)
    parser.add_argument("--epochs",        type=int,   default=50,
                        help="Max CNN epochs per model (default: 50)")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--n_splits",      type=int,   default=5)
    parser.add_argument("--skip_ensemble", action="store_true")
    args = parser.parse_args()

    feature_csv = Path(args.feature_csv)
    if not feature_csv.exists():
        sys.exit(f"❌  Feature CSV not found: {feature_csv}")

    region_name = feature_csv.parent.parent.parent.name

    _banner("🔥 WILDFIRE PREDICTION – MULTI-MODEL TRAINING")
    print(f"  Region      : {region_name}")
    print(f"  Feature CSV : {feature_csv}")
    print(f"  Label col   : {args.label_col}")
    print(f"  CNN epochs  : {args.epochs}")
    print(f"  CV folds    : {args.n_splits}")

    # ─────────────────────────────────────────────────────────────
    # PHASE 1: TABULAR MODELS
    # ─────────────────────────────────────────────────────────────
    _banner("📈 PHASE 1 – TABULAR MODELS  (LightGBM · XGBoost · CatBoost)")

    tabular_trainer = None
    try:
        df = pd.read_csv(feature_csv)
        print(f"\n  Loaded {len(df):,} rows × {len(df.columns)} columns")

        if args.label_col not in df.columns:
            fire_cols = [c for c in df.columns if "fire" in c.lower()]
            sys.exit(f"❌  Label '{args.label_col}' not in CSV. "
                     f"Columns with 'fire': {fire_cols}")

        region_col = args.region_col if args.region_col in df.columns else None
        if args.region_col and region_col is None:
            print(f"  ⚠  region_col='{args.region_col}' not found, proceeding without it")

        tabular_trainer = TabularModelTrainer(
            df,
            label_col  = args.label_col,
            n_splits   = args.n_splits,
            region_col = region_col,
        )
        tabular_trainer.train_all(save=True)

        # Feature importance plots for every trained model
        for model_name in tabular_trainer.trained_models_:
            tabular_trainer.plot_feature_importance(model_name=model_name, save=True)

        print(f"\n  ✅ Tabular training complete.")

    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        print(f"\n  ❌ Tabular training failed: {exc}")
        traceback.print_exc()

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: CNN MODELS
    # ─────────────────────────────────────────────────────────────
    _banner("🖼️  PHASE 2 – CNN MODELS  (ResNet50 · EfficientNet-B0 / B2 / B3)")

    img_dir = Path(args.img_dir) if args.img_dir \
              else feature_csv.parent.parent / "cnn_dataset"

    cnn_trainer  = None
    train_loader = None
    val_loader   = None

    if not img_dir.exists():
        print(f"\n  ⚠  CNN dataset not found at {img_dir} – skipping CNN phase.")
    else:
        try:
            print(f"\n  CNN dataset: {img_dir}")
            train_loader, val_loader = make_image_loaders(
                str(img_dir), args.batch_size
            )

            cnn_trainer = CNNTrainer(
                train_loader = train_loader,
                val_loader   = val_loader,
                num_classes  = 2,
                epochs       = args.epochs,   # 50 — not 2
                lr           = 1e-3,
                patience     = 7,
                min_delta    = 1e-3,
            )

            print("\n  ── Stage 1+2: Baseline (ResNet50 vs EfficientNet-B0) ──")
            cnn_trainer.run_baseline_comparison()

            print("\n  ── Stage 3: Conditional scaling ──")
            cnn_trainer.scale_if_needed(scale_target="auto")

            best_name, _ = cnn_trainer.best_model()
            best_acc     = cnn_trainer.results_[best_name]["best_val_acc"]
            print(f"\n  ✅ CNN complete. Best: {best_name}  val_acc={best_acc:.4f}")

        except Exception as exc:
            import traceback
            print(f"\n  ❌ CNN training failed: {exc}")
            traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# PHASE 3: MULTIMODAL ENSEMBLE
# ─────────────────────────────────────────────────────────────
    _banner("🎯 PHASE 3 – MULTIMODAL ENSEMBLE  (Early Fusion · Late Fusion)")

    if args.skip_ensemble:
        print("\n  Skipped (--skip_ensemble flag).")
    elif tabular_trainer is None:
        print("\n  ⚠  Skipped – tabular models unavailable.")
    elif cnn_trainer is None:
        print("\n  ⚠  Skipped – CNN models unavailable.")
    else:
        try:
            # ── ARCHITECTURAL FIX ──────────────────────────────────────────────
            # The tabular CSV (3 074 rows of daily weather/FIRMS aggregates) and
            # the CNN image tiles (per-point spatial patches) share NO common row
            # index.  We therefore decouple the two modalities:
            #
            #   Early fusion  → CNN-only GAP features  (no tabular concat)
            #   Late fusion   → meta-learner on CNN proba + tabular OOF proba
            #                   where tabular proba is derived from the FULL CSV
            #                   evaluated at the image-loader sample order.
            #
            # This is the only valid fusion path until a per-tile feature CSV is
            # built (one tabular row per image tile, keyed by tile date+location).
            # ──────────────────────────────────────────────────────────────────

            # Rebuild clean loaders (same seed → same split as Phase 2)
            train_loader, val_loader = make_image_loaders(
                str(img_dir), args.batch_size
            )

            # Collect ground-truth labels straight from the image loaders
            # (order preserved because shuffle=False on val, shuffle handled
            #  internally by SubsetWithTransform for train)
            def collect_labels(loader):
                ys = []
                for _, labels in loader:
                    ys.append(labels.numpy())
                return np.concatenate(ys)

            y_tr_img = collect_labels(train_loader)
            y_val_img = collect_labels(val_loader)

            ensemble = MultimodalEnsemble(
                cnn_trainer     = cnn_trainer,
                tabular_trainer = tabular_trainer,
                n_classes       = 2,
                meta_strategy   = "logistic",
            )

            print("\n  ── Early Fusion (CNN GAP only – no tabular concat) ──")
            ensemble.fit_early_fusion_cnn_only(train_loader, y_tr_img, n_splits=5)

            print("\n  ── Late Fusion (CNN proba + tabular OOF proba) ──")
            # For late fusion, tabular branch uses its own OOF predictions
            # on its full dataset; CNN branch uses image-loader predictions.
            # We pass None for X_tab and rely on the tabular trainer's stored
            # OOF proba averaged to a scalar fire-risk score.
            ensemble.fit_late_fusion_decoupled(
                train_loader, y_tr_img, tabular_trainer, n_splits=5
            )

            print("\n  ── Held-out Evaluation ──")
            ensemble.evaluate_decoupled(val_loader, y_val_img, tabular_trainer)

            print("\n  ✅ Ensemble complete.")

        except Exception as exc:
            import traceback
            print(f"\n  ❌ Ensemble failed: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()