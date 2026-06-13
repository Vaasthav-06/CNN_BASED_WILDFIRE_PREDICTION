#!/usr/bin/env python3
"""
train_global_model.py
======================
Trains a SINGLE set of models (tabular + CNN + ensemble) pooled across
all four case-study regions, instead of one model per region.

Architecture is identical to train_all_models.py — same classes, same
three phases — the only difference is the data source:

  Per-region  →  one features_<region>.csv + one cnn_dataset/
  Global      →  features_all_regions.csv  + combined cnn_dataset/

The combined CSVs / image dirs are built first by:
    python prepare_combined_dataset.py
    python prepare_combined_cnn_dataset.py

Or this script can auto-prepare them if you pass --auto_prepare.

Usage
-----
    # Minimal (auto-prepare data, then train all three phases)
    python src/train_global_model.py --auto_prepare

    # Full control
    python src/train_global_model.py \
        --feature_csv  data/combined/features_all_regions.csv \
        --img_dir      data/combined/cnn_dataset \
        --label_col    fire \
        --epochs       50 \
        --batch_size   32 \
        --n_splits     5
"""

import os
import sys
import argparse
import warnings
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── FIX: Add the project root (not src/) to the python path ──────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now these imports will work perfectly
from src.models.tabular_models import TabularModelTrainer
from src.models.cnn_models      import CNNTrainer, TRAIN_TRANSFORM, VAL_TRANSFORM
from src.models.ensemble        import MultimodalEnsemble

# ── default paths ─────────────────────────────────────────────────────────────
DEFAULT_FEATURE_CSV = ROOT / "data/combined/features_all_regions.csv"
DEFAULT_IMG_DIR     = ROOT / "data/combined/cnn_dataset"

# ── redirect outputs so global models don't overwrite per-region ones ─────────
import src.models.tabular_models as _tab_mod
import src.models.cnn_models     as _cnn_mod
import src.models.ensemble       as _ens_mod

for _mod, _sub in [
    (_tab_mod, "tabular_models"),
    (_cnn_mod, "cnn_models"),
    (_ens_mod, "ensemble"),
]:
    _mod.OUTPUTS    = ROOT / "outputs"    / "global" / _sub
    _mod.MODELS_DIR = ROOT / "saved_models" / "global" / _sub
    _mod.OUTPUTS.mkdir(parents=True, exist_ok=True)
    _mod.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers (identical logic to train_all_models.py)
# ─────────────────────────────────────────────────────────────────────────────
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
    stem = Path(filename).stem
    h    = int(hashlib.md5(stem.encode()).hexdigest(), 16)
    return h % n_buckets


def make_image_loaders(
    img_dir: str,
    batch_size: int,
    val_fraction: float = 0.2,
    n_spatial_buckets: int = 10,
):
    base_ds       = datasets.ImageFolder(root=img_dir)
    n_val_buckets = max(1, round(n_spatial_buckets * val_fraction))
    val_buckets   = set(range(n_val_buckets))

    train_samples, val_samples = [], []
    for path, label in base_ds.samples:
        fname  = os.path.basename(path)
        bucket = _spatial_hash(fname, n_spatial_buckets)
        (val_samples if bucket in val_buckets else train_samples).append((path, label))

    # Guarantee at least 1 sample per class in both splits
    for label_idx in [0, 1]:
        for primary, other in [(val_samples, train_samples), (train_samples, val_samples)]:
            if not any(l == label_idx for _, l in primary):
                for i, (p, l) in enumerate(other):
                    if l == label_idx:
                        primary.append(other.pop(i))
                        break

    train_ds = SubsetWithTransform(train_samples, TRAIN_TRANSFORM)
    val_ds   = SubsetWithTransform(val_samples,   VAL_TRANSFORM)

    n_total = len(train_samples) + len(val_samples)
    print(f"  ImageFolder: {n_total} images → {len(train_samples)} train "
          f"| {len(val_samples)} val  [spatial-hash split, "
          f"{n_val_buckets}/{n_spatial_buckets} val buckets]")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Tabular helpers
# ─────────────────────────────────────────────────────────────────────────────
def prepare_tabular(df: pd.DataFrame, label_col: str, region_col: str | None):
    drop_cols = [label_col]
    if region_col and region_col in df.columns:
        drop_cols.append(region_col)
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


def _print_per_region_breakdown(
    trainer: TabularModelTrainer,
    df: pd.DataFrame,
    label_col: str,
) -> None:
    if "region" not in df.columns:
        return

    print(f"\n{'═'*65}")
    print(f"  PER-REGION BREAKDOWN  (global model, held-out OOF predictions)")
    print(f"{'─'*65}")
    print(f"  {'Region':<16} {'N':>6}  {'Fire%':>7}  {'Best Model AUC':>14}")
    print(f"{'─'*65}")

    best_name, _ = trainer.best_model()
    oof_preds    = trainer.results_[best_name]["oof_preds"]

    from sklearn.metrics import roc_auc_score
    for region in sorted(df["region"].unique()):
        mask  = (df["region"] == region).values
        y_reg = df.loc[mask, label_col].values
        oof_r = oof_preds[mask]
        n     = int(mask.sum())
        fire_pct = float(y_reg.mean()) * 100
        try:
            auc = roc_auc_score(y_reg, oof_r)
            auc_str = f"{auc:.4f}"
        except Exception:
            auc_str = "  N/A  "
        print(f"  {region:<16} {n:>6}  {fire_pct:>6.1f}%  {auc_str:>14}")

    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────
def collect_labels(loader):
    ys = []
    for _, labels in loader:
        ys.append(labels.numpy())
    return np.concatenate(ys)


def _banner(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-prepare helper
# ─────────────────────────────────────────────────────────────────────────────
def _auto_prepare(label_col: str):
    import importlib, importlib.util

    def _run_module(script_name, **kwargs):
        spec = importlib.util.spec_from_file_location(
            script_name, ROOT / "src" / f"{script_name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    print("\n  [auto-prepare] Building combined tabular dataset …")
    tab_mod = _run_module("prepare_combined_dataset")
    tab_mod.prepare_combined(label_col=label_col)

    print("\n  [auto-prepare] Building combined CNN dataset …")
    cnn_mod = _run_module("prepare_combined_cnn_dataset")
    cnn_mod.prepare_combined_cnn()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train a single global wildfire model across all four regions."
    )
    parser.add_argument("--feature_csv",   default=str(DEFAULT_FEATURE_CSV))
    parser.add_argument("--img_dir",       default=None)
    parser.add_argument("--label_col",     default="fire")
    parser.add_argument("--region_col",    default="region")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--n_splits",      type=int,   default=5)
    parser.add_argument("--skip_cnn",      action="store_true")
    parser.add_argument("--skip_ensemble", action="store_true")
    parser.add_argument(
        "--auto_prepare", action="store_true",
        help="Run prepare_combined_dataset.py and prepare_combined_cnn_dataset.py "
             "automatically before training if the combined files are missing."
    )
    args = parser.parse_args()

    feature_csv = Path(args.feature_csv)
    img_dir     = Path(args.img_dir) if args.img_dir else DEFAULT_IMG_DIR

    if args.auto_prepare:
        if not feature_csv.exists() or not img_dir.exists():
            _auto_prepare(args.label_col)
        else:
            print("  [auto-prepare] Combined data already present — skipping.")

    if not feature_csv.exists():
        sys.exit(
            f"❌  Feature CSV not found: {feature_csv}\n"
            "    Run:  python src/prepare_combined_dataset.py\n"
            "    or pass --auto_prepare"
        )

    _banner("🌍 WILDFIRE PREDICTION – GLOBAL MODEL (ALL REGIONS POOLED)")
    print(f"  Feature CSV  : {feature_csv}")
    print(f"  Image dir    : {img_dir}")
    print(f"  Label col    : {args.label_col}")
    print(f"  CNN epochs   : {args.epochs}")
    print(f"  CV folds     : {args.n_splits}")
    print(f"  Outputs      : outputs/global/")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 – TABULAR MODELS
    # ─────────────────────────────────────────────────────────────────────────
    _banner("📈 PHASE 1 – TABULAR MODELS  (LightGBM · XGBoost · CatBoost)")

    tabular_trainer = None
    df_global       = None
    try:
        df_global = pd.read_csv(feature_csv)
        print(f"\n  Loaded {len(df_global):,} rows × {len(df_global.columns)} cols")
        print(f"  Regions present: {sorted(df_global['region'].unique()) if 'region' in df_global.columns else 'N/A'}")

        if args.label_col not in df_global.columns:
            fire_cols = [c for c in df_global.columns if "fire" in c.lower()]
            sys.exit(
                f"❌  Label '{args.label_col}' not in CSV. "
                f"Columns with 'fire': {fire_cols}"
            )

        region_col = args.region_col if args.region_col in df_global.columns else None
        if args.region_col and region_col is None:
            print(f"  ⚠  region_col='{args.region_col}' not found, "
                  "training without region feature.")

        tabular_trainer = TabularModelTrainer(
            df_global,
            label_col  = args.label_col,
            n_splits   = args.n_splits,
            region_col = region_col,   # used for per-region reporting
        )
        tabular_trainer.train_all(save=True)

        for model_name in tabular_trainer.trained_models_:
            tabular_trainer.plot_feature_importance(model_name=model_name, save=True)

        # ── Extra: per-region breakdown table ─────────────────────────────────
        _print_per_region_breakdown(tabular_trainer, df_global, args.label_col)

        print("  ✅  Tabular training complete.")

    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        print(f"\n  ❌  Tabular training failed: {exc}")
        traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 – CNN MODELS
    # ─────────────────────────────────────────────────────────────────────────
    cnn_trainer  = None
    train_loader = None
    val_loader   = None

    if args.skip_cnn:
        _banner("🖼️  PHASE 2 – CNN MODELS  [SKIPPED via --skip_cnn]")
    else:
        _banner("🖼️  PHASE 2 – CNN MODELS  (ResNet50 · EfficientNet-B0 / B2 / B3)")

        if not img_dir.exists():
            print(f"\n  ⚠  Combined CNN dataset not found at {img_dir}")
            print( "      Run:  python src/prepare_combined_cnn_dataset.py")
            print( "      or pass --auto_prepare  — skipping CNN phase.")
        else:
            try:
                print(f"\n  CNN dataset : {img_dir}")
                train_loader, val_loader = make_image_loaders(
                    str(img_dir), args.batch_size
                )

                cnn_trainer = CNNTrainer(
                    train_loader = train_loader,
                    val_loader   = val_loader,
                    num_classes  = 2,
                    epochs       = args.epochs,
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
                print(f"\n  ✅  CNN complete. Best: {best_name}  val_acc={best_acc:.4f}")

            except Exception as exc:
                import traceback
                print(f"\n  ❌  CNN training failed: {exc}")
                traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 – MULTIMODAL ENSEMBLE
    # ─────────────────────────────────────────────────────────────────────────
    _banner("🎯 PHASE 3 – MULTIMODAL ENSEMBLE  (Early Fusion · Late Fusion)")

    if args.skip_ensemble:
        print("\n  Skipped (--skip_ensemble).")
    elif tabular_trainer is None:
        print("\n  ⚠  Skipped – tabular models unavailable.")
    elif cnn_trainer is None:
        print("\n  ⚠  Skipped – CNN models unavailable.")
    else:
        try:
            # Rebuild loaders with same seed → same split
            train_loader, val_loader = make_image_loaders(
                str(img_dir), args.batch_size
            )

            y_tr_img  = collect_labels(train_loader)
            y_val_img = collect_labels(val_loader)

            ensemble = MultimodalEnsemble(
                cnn_trainer     = cnn_trainer,
                tabular_trainer = tabular_trainer,
                n_classes       = 2,
                meta_strategy   = "logistic",
            )

            print("\n  ── Early Fusion (CNN GAP only – decoupled) ──")
            ensemble.fit_early_fusion_cnn_only(train_loader, y_tr_img, n_splits=5)

            print("\n  ── Late Fusion (CNN proba + tabular OOF prior) ──")
            ensemble.fit_late_fusion_decoupled(
                train_loader, y_tr_img, tabular_trainer, n_splits=5
            )

            print("\n  ── Held-out Evaluation ──")
            ensemble.evaluate_decoupled(val_loader, y_val_img, tabular_trainer)

            print("\n  ✅  Ensemble complete.")

        except Exception as exc:
            import traceback
            print(f"\n  ❌  Ensemble failed: {exc}")
            traceback.print_exc()

    _banner("🏁 GLOBAL TRAINING COMPLETE")
    print("  Artefacts saved under:")
    print("    outputs/global/tabular_models/")
    print("    outputs/global/cnn_models/")
    print("    outputs/global/ensemble/")
    print("    saved_models/global/tabular/")
    print("    saved_models/global/cnn/")
    print("    saved_models/global/ensemble/")


if __name__ == "__main__":
    main()