"""
tabular_models.py
=================
Trains and benchmarks three gradient-boosting models on the engineered tabular
feature set:
  • XGBoost   – highly tunable, reliable baseline
  • CatBoost  – excellent with categoricals, strong built-in regularisation
  • LightGBM  – fastest on large numerical datasets, often ties for #1

All three are cross-validated, their metrics are printed side-by-side, and the
best single model (or an ensemble of all three) is returned for downstream use
in the multimodal ensemble.

Usage
-----
    from src.models.tabular_models import TabularModelTrainer
    trainer = TabularModelTrainer(feature_df, label_col="fire_risk")
    results  = trainer.train_all()
    best     = trainer.best_model()
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
OUTPUTS     = ROOT / "outputs" / "tabular_models"
MODELS_DIR  = ROOT / "saved_models" / "tabular"
OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: encode categoricals with LabelEncoder
# ─────────────────────────────────────────────────────────────────────────────
def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def _build_xgboost(n_classes: int, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    """XGBoost – tuned for binary or multiclass wildfire risk."""
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return xgb.XGBClassifier(
        objective          = objective,
        n_estimators       = 600,
        learning_rate      = 0.05,
        max_depth          = 6,
        min_child_weight   = 3,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        scale_pos_weight   = scale_pos_weight,   # handles class imbalance
        use_label_encoder  = False,
        eval_metric        = "logloss",
        tree_method        = "hist",             # fast CPU training
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = 0,
    )


def _build_catboost(cat_features: list[int] | None, n_classes: int) -> cb.CatBoostClassifier:
    """CatBoost – handles categoricals natively, excellent regularisation."""
    loss = "Logloss" if n_classes == 2 else "MultiClass"
    return cb.CatBoostClassifier(
        iterations          = 600,
        learning_rate       = 0.05,
        depth               = 6,
        l2_leaf_reg         = 3.0,
        border_count        = 128,
        loss_function       = loss,
        eval_metric         = "AUC" if n_classes == 2 else "Accuracy",
        cat_features        = cat_features or [],
        auto_class_weights  = "Balanced",        # built-in imbalance handling
        random_seed         = 42,
        verbose             = False,
    )


def _build_lightgbm(n_classes: int, scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
    """LightGBM – fastest on large numerical datasets, leaf-wise trees."""
    objective = "binary" if n_classes == 2 else "multiclass"
    return lgb.LGBMClassifier(
        objective          = objective,
        n_estimators       = 600,
        learning_rate      = 0.05,
        max_depth          = -1,                 # unlimited, controlled by num_leaves
        num_leaves         = 63,
        min_child_samples  = 20,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        scale_pos_weight   = scale_pos_weight,
        class_weight       = "balanced",
        random_state       = 42,
        n_jobs             = -1,
        verbose            = -1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer class
# ─────────────────────────────────────────────────────────────────────────────
class TabularModelTrainer:
    """
    Cross-validates XGBoost, CatBoost, and LightGBM on a tabular DataFrame,
    prints a benchmark table, and exposes the best model.

    Parameters
    ----------
    df         : pd.DataFrame with feature columns + label_col
    label_col  : target column name (binary or multiclass integer labels)
    n_splits   : number of stratified CV folds (default 5)
    region_col : optional column for per-region breakdown in reports
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        label_col:  str   = "fire_risk",
        n_splits:   int   = 5,
        region_col: str | None = "region",
    ):
        self.label_col  = label_col
        self.n_splits   = n_splits
        self.region_col = region_col

        # Split X / y
        drop_cols = [label_col] + ([region_col] if region_col and region_col in df.columns else [])
        self.regions = df[region_col].values if (region_col and region_col in df.columns) else None
        self.y       = df[label_col].values
        self.X_raw   = df.drop(columns=drop_cols)

        # Identify categorical column indices (for CatBoost)
        cat_cols           = list(self.X_raw.select_dtypes(include=["object","category"]).columns)
        self.cat_indices   = [self.X_raw.columns.get_loc(c) for c in cat_cols]

        # For XGBoost/LightGBM we need numeric-only input
        self.X_encoded = _encode_categoricals(self.X_raw).values.astype(np.float32)
        self.X_raw_np  = self.X_raw.copy()  # CatBoost can handle mixed types

        # Class statistics
        classes, counts    = np.unique(self.y, return_counts=True)
        self.n_classes     = len(classes)
        self.scale_pos_w   = float(counts[0] / counts[1]) if self.n_classes == 2 else 1.0
        print(f"[TabularTrainer] {len(self.X_encoded)} samples | "
              f"{self.X_encoded.shape[1]} features | "
              f"{self.n_classes} classes | "
              f"class dist: {dict(zip(classes.tolist(), counts.tolist()))}")

        self.results_: dict = {}
        self.trained_models_: dict = {}

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────
    def train_all(self, save: bool = True) -> dict:
        """Train and cross-validate all three models. Returns metrics dict."""
        cv    = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        specs = {
            "XGBoost":  (self._cv_xgboost,  self.X_encoded),
            "CatBoost": (self._cv_catboost,  self.X_raw_np),
            "LightGBM": (self._cv_lightgbm,  self.X_encoded),
        }
        for name, (fn, X) in specs.items():
            print(f"\n{'─'*60}")
            print(f"  Training {name} …")
            metrics = fn(X, cv)
            self.results_[name] = metrics
            print(f"  ✓ {name}  AUC={metrics['auc']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"Acc={metrics['accuracy']:.4f}  "
                  f"Time={metrics['train_time']:.1f}s")

        self._print_benchmark_table()
        self._plot_benchmark(save=save)
        self._plot_roc_curves(save=save)

        if save:
            self._save_models()

        return self.results_

    def best_model(self) -> tuple[str, object]:
        """Return (name, fitted_model) of the model with highest CV AUC."""
        if not self.results_:
            raise RuntimeError("Call train_all() first.")
        best_name = max(self.results_, key=lambda k: self.results_[k]["auc"])
        return best_name, self.trained_models_[best_name]

    def ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Simple averaging ensemble over all three trained models.
        X must be pre-encoded (numeric).
        """
        if len(self.trained_models_) < 3:
            raise RuntimeError("Call train_all() first.")
        probas = []
        for name, model in self.trained_models_.items():
            p = model.predict_proba(X)
            probas.append(p[:, 1] if self.n_classes == 2 else p)
        return np.mean(probas, axis=0)

    # ──────────────────────────────────────────────────────
    # Internal CV helpers
    # ──────────────────────────────────────────────────────
    def _cv_xgboost(self, X, cv) -> dict:
        model = _build_xgboost(self.n_classes, self.scale_pos_w)
        return self._run_cv(model, X, cv, "XGBoost")

    def _cv_catboost(self, X, cv) -> dict:
        model = _build_catboost(self.cat_indices, self.n_classes)
        return self._run_cv(model, X if isinstance(X, np.ndarray) else X.values, cv, "CatBoost")

    def _cv_lightgbm(self, X, cv) -> dict:
        model = _build_lightgbm(self.n_classes, self.scale_pos_w)
        return self._run_cv(model, X, cv, "LightGBM")

    def _run_cv(self, model, X, cv, name: str) -> dict:
        """Run stratified CV, collect metrics, retrain on full data."""
        auc_scores, f1_scores, acc_scores = [], [], []
        oof_preds = np.zeros(len(self.y))
        t0 = time.time()

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, self.y), 1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = self.y[tr_idx], self.y[val_idx]

            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)
            preds = model.predict(X_val)

            auc  = roc_auc_score(y_val, proba[:, 1] if self.n_classes == 2 else proba,
                                 multi_class="ovr" if self.n_classes > 2 else "raise")
            f1   = f1_score(y_val, preds, average="weighted")
            acc  = accuracy_score(y_val, preds)
            auc_scores.append(auc); f1_scores.append(f1); acc_scores.append(acc)

            if self.n_classes == 2:
                oof_preds[val_idx] = proba[:, 1]

        elapsed = time.time() - t0

        # Retrain on full dataset for deployment
        model.fit(X, self.y)
        self.trained_models_[name] = model

        return {
            "auc":        float(np.mean(auc_scores)),
            "auc_std":    float(np.std(auc_scores)),
            "f1":         float(np.mean(f1_scores)),
            "f1_std":     float(np.std(f1_scores)),
            "accuracy":   float(np.mean(acc_scores)),
            "acc_std":    float(np.std(acc_scores)),
            "train_time": elapsed,
            "oof_preds":  oof_preds,
            "model":      model,
        }

    # ──────────────────────────────────────────────────────
    # Feature importance
    # ──────────────────────────────────────────────────────
    def plot_feature_importance(
        self,
        model_name: str = "XGBoost",
        top_n: int = 20,
        save: bool = True,
    ) -> None:
        """Bar chart of top-N features for the specified model."""
        model = self.trained_models_.get(model_name)
        if model is None:
            raise ValueError(f"{model_name} not trained yet.")

        feat_names = list(self.X_raw.columns)
        if model_name == "XGBoost":
            importances = model.feature_importances_
        elif model_name == "CatBoost":
            importances = model.get_feature_importance()
        else:  # LightGBM
            importances = model.feature_importances_

        idx   = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh([feat_names[i] for i in idx], importances[idx], color="#E05C2A")
        ax.set_title(f"{model_name} – Top {top_n} Feature Importances", fontsize=14)
        ax.set_xlabel("Importance score")
        ax.invert_yaxis()
        plt.tight_layout()
        if save:
            path = OUTPUTS / f"feature_importance_{model_name.lower()}.png"
            fig.savefig(path, dpi=150)
            print(f"  Saved → {path}")
        plt.close(fig)

    # ──────────────────────────────────────────────────────
    # Reporting helpers
    # ──────────────────────────────────────────────────────
    def _print_benchmark_table(self) -> None:
        print(f"\n{'═'*65}")
        print(f"  {'Model':<12} {'AUC':>8} {'±':>5} {'F1':>8} {'±':>5} "
              f"{'Acc':>8} {'Time(s)':>8}")
        print(f"{'─'*65}")
        for name, m in self.results_.items():
            print(f"  {name:<12} {m['auc']:>8.4f} {m['auc_std']:>5.4f} "
                  f"{m['f1']:>8.4f} {m['f1_std']:>5.4f} "
                  f"{m['accuracy']:>8.4f} {m['train_time']:>8.1f}")
        best = max(self.results_, key=lambda k: self.results_[k]['auc'])
        print(f"{'─'*65}")
        print(f"  Best model by AUC: {best}  ({self.results_[best]['auc']:.4f})")
        print(f"{'═'*65}\n")

    def _plot_benchmark(self, save: bool = True) -> None:
        names   = list(self.results_.keys())
        metrics = ["auc", "f1", "accuracy"]
        labels  = ["ROC-AUC", "F1 (weighted)", "Accuracy"]
        colors  = ["#2E86AB", "#E05C2A", "#44BBA4"]

        x    = np.arange(len(names))
        w    = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            vals = [self.results_[n][metric] for n in names]
            errs = [self.results_[n][f"{metric[:3]}_std"] for n in names]
            ax.bar(x + i * w, vals, w, yerr=errs, label=label, color=color,
                   capsize=4, alpha=0.9)

        ax.set_xticks(x + w)
        ax.set_xticklabels(names, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Tabular Model Benchmark: XGBoost vs CatBoost vs LightGBM")
        ax.legend(loc="lower right")
        plt.tight_layout()
        if save:
            path = OUTPUTS / "tabular_benchmark_comparison.png"
            fig.savefig(path, dpi=150)
            print(f"  Saved → {path}")
        plt.close(fig)

    def _plot_roc_curves(self, save: bool = True) -> None:
        """Plot OOF ROC curves for each model (binary only)."""
        if self.n_classes != 2:
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = {"XGBoost": "#2E86AB", "CatBoost": "#E05C2A", "LightGBM": "#44BBA4"}
        for name, m in self.results_.items():
            oof = m["oof_preds"]
            auc = roc_auc_score(self.y, oof)
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(self.y, oof)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=colors[name], lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("OOF ROC Curves – Tabular Models")
        ax.legend()
        plt.tight_layout()
        if save:
            path = OUTPUTS / "tabular_roc_curves.png"
            fig.savefig(path, dpi=150)
            print(f"  Saved → {path}")
        plt.close(fig)

    def _save_models(self) -> None:
        for name, model in self.trained_models_.items():
            path = MODELS_DIR / f"{name.lower()}_model.pkl"
            joblib.dump(model, path)
            print(f"  Saved model → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────
def train_tabular_models(
    feature_csv: str | Path,
    label_col:   str  = "fire_risk",
    region_col:  str  = "region",
    n_splits:    int  = 5,
    save:        bool = True,
) -> dict:
    """
    End-to-end convenience function.

    Parameters
    ----------
    feature_csv : path to the engineered feature CSV
    label_col   : target column
    region_col  : grouping column (optional, for reporting)
    n_splits    : CV folds
    save        : whether to save models and plots

    Returns
    -------
    dict with keys 'results' and 'trainer'
    """
    df      = pd.read_csv(feature_csv)
    trainer = TabularModelTrainer(df, label_col=label_col,
                                  region_col=region_col, n_splits=n_splits)
    results = trainer.train_all(save=save)
    # Feature importance for every model
    for model_name in trainer.trained_models_:
        trainer.plot_feature_importance(model_name=model_name, save=save)
    return {"results": results, "trainer": trainer}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train tabular models")
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--label_col", default="fire_risk")
    parser.add_argument("--region_col", default="region")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    train_tabular_models(
        feature_csv=args.feature_csv,
        label_col=args.label_col,
        region_col=args.region_col,
        n_splits=args.n_splits,
    )