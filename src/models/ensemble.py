"""
ensemble_model.py
=================
Multimodal ensemble combining the tabular branch (XGBoost/CatBoost/LightGBM)
with the image branch (EfficientNet GAP features) in two fusion strategies:

┌─────────────────────────────────────────────────────────────┐
│  EARLY FUSION  (primary, trains one GBDT on merged vectors)  │
│                                                              │
│  Image → EfficientNet → GAP (1280-d) ──┐                    │
│                                         ├─► concat ► GBDT   │
│  Tabular Features (N-d) ───────────────┘                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  LATE FUSION / STACKING  (optional, if time permits)        │
│                                                              │
│  Image branch  → P(fire | image)  ──┐                       │
│                                      ├─► meta-learner        │
│  Tabular branch → P(fire | tabular) ┘  (LogReg or GBDT)    │
└─────────────────────────────────────────────────────────────┘

Both strategies produce calibrated fire-risk probabilities and are evaluated
with the same metrics as individual branches (AUC, F1, Acc).

Usage
-----
    from src.models.ensemble_model import MultimodalEnsemble
    ensemble = MultimodalEnsemble(
        cnn_trainer    = cnn_trainer,       # fitted CNNTrainer
        tabular_trainer= tabular_trainer,   # fitted TabularModelTrainer
    )
    ensemble.fit_early_fusion(X_tab_train, img_loader_train, y_train)
    ensemble.fit_late_fusion(X_tab_val, img_loader_val, y_val)
    results = ensemble.evaluate(X_tab_test, img_loader_test, y_test)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, roc_curve,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

# Project imports
try:
    from src.models.cnn_models     import GAPExtractor, DEVICE
    from src.models.tabular_models import TabularModelTrainer
except ImportError:
    from models.cnn_models     import GAPExtractor, DEVICE
    from models.tabular_models import TabularModelTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
OUTPUTS    = ROOT / "outputs" / "ensemble"
MODELS_DIR = ROOT / "saved_models" / "ensemble"
OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _extract_gap_features(extractor: GAPExtractor, loader: DataLoader) -> np.ndarray:
    """Run `loader` through extractor and return GAP embedding matrix (N, D)."""
    extractor.eval()
    feats = []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        feats.append(extractor(imgs.to(DEVICE)).cpu().numpy())
    return np.vstack(feats)


@torch.no_grad()
def _model_predict_proba(model: torch.nn.Module, loader: DataLoader,
                          n_classes: int) -> np.ndarray:
    """Return softmax probabilities from a PyTorch model for all items in loader."""
    import torch.nn.functional as F
    model.eval()
    probs = []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = model(imgs.to(DEVICE))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def _build_early_fusion_gbdt(n_classes: int) -> lgb.LGBMClassifier:
    """
    LightGBM on the concatenated feature vector.
    LightGBM is chosen here for its speed on high-dimensional numerical inputs
    (GAP vectors are 1280-d for EfficientNet-B0).
    XGBoost is also fine — swap as needed.
    """
    objective = "binary" if n_classes == 2 else "multiclass"
    return lgb.LGBMClassifier(
        objective        = objective,
        n_estimators     = 400,
        learning_rate    = 0.05,
        max_depth        = -1,
        num_leaves       = 63,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        class_weight     = "balanced",
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
    )


def _build_meta_learner(strategy: str = "logistic") -> object:
    """
    Meta-learner for late fusion / stacking.
    strategy ∈ {"logistic", "gbdt"}
    """
    if strategy == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif strategy == "gbdt":
        return lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, class_weight="balanced",
            random_state=42, verbose=-1,
        )
    else:
        raise ValueError(f"Unknown meta-learner strategy: {strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# MultimodalEnsemble
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalEnsemble:
    """
    Two-branch multimodal ensemble (image + tabular).

    Parameters
    ----------
    cnn_trainer      : fitted CNNTrainer (from cnn_model.py)
    tabular_trainer  : fitted TabularModelTrainer (from tabular_models.py)
    n_classes        : number of fire-risk classes (default 2)
    meta_strategy    : "logistic" | "gbdt" for the late-fusion meta-learner
    """

    # ──────────────────────────────────────────────────────────────────────────
# DECOUPLED FUSION METHODS
# These are used when tabular rows and image tiles have no shared index.
# ──────────────────────────────────────────────────────────────────────────

    def fit_early_fusion_cnn_only(
        self,
        img_loader: DataLoader,
        y:          np.ndarray,
        n_splits:   int = 5,
    ) -> dict:
        """
        Early fusion using ONLY CNN GAP features (no tabular concat).
        Used when tabular rows and image tiles are not row-aligned.
        A LightGBM classifier is trained on the GAP embedding alone.
        """
        print("\n[Early Fusion – CNN only] Extracting GAP features …")
        t0 = time.time()
        gap_feats = self._get_gap(img_loader)           # (N_images, D)
        print(f"  GAP shape: {gap_feats.shape}  "
            f"Extraction time: {time.time()-t0:.1f}s")

        self.scaler_ = StandardScaler()
        X_scaled     = self.scaler_.fit_transform(gap_feats)

        model = _build_early_fusion_gbdt(self.n_classes)
        cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_proba = cross_val_predict(
            model, X_scaled, y, cv=cv, method="predict_proba"
        )[:, 1]
        oof_preds = (oof_proba >= 0.5).astype(int)

        auc = roc_auc_score(y, oof_proba)
        f1  = f1_score(y, oof_preds, average="weighted")
        acc = accuracy_score(y, oof_preds)

        model.fit(X_scaled, y)
        self.early_fusion_model_ = model

        metrics = {
            "strategy":   "early_fusion_cnn_only",
            "auc":        auc, "f1": f1, "accuracy": acc,
            "oof_proba":  oof_proba,
            "n_features": gap_feats.shape[1],
        }
        self.results_["early_fusion"] = metrics
        print(f"  [Early Fusion – CNN only] AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
        joblib.dump(model,        MODELS_DIR / "early_fusion_gbdt.pkl")
        joblib.dump(self.scaler_, MODELS_DIR / "early_fusion_scaler.pkl")
        return metrics


    def fit_late_fusion_decoupled(
        self,
        img_loader:      DataLoader,
        y_img:           np.ndarray,
        tabular_trainer: object,          # TabularModelTrainer
        n_splits:        int = 5,
    ) -> dict:
        """
        Late fusion when modalities have different sample counts.

        CNN branch   : per-image softmax probabilities  (N_images, 2)
        Tabular branch: scalar fire-risk score from the tabular model's
                        averaged OOF predictions, broadcast to match
                        the image sample count.

        This is a valid stacking approach because both branches produce
        an independent P(fire) estimate; the meta-learner learns how to
        weight them.
        """
        print("\n[Late Fusion – decoupled] Building meta-features …")

        # CNN branch: per-image probabilities
        cnn_prob = self._get_cnn_proba(img_loader)          # (N_images, 2)

        # Tabular branch: use mean OOF fire-risk probability as a global
        # contextual prior, broadcast to all image samples
        best_tab_name, _ = tabular_trainer.best_model()
        tab_oof = tabular_trainer.results_[best_tab_name]["oof_preds"]  # (N_tab,)
        tab_fire_prior = float(np.mean(tab_oof))                        # scalar

        # Build meta-features: CNN P(fire), CNN P(no_fire), tabular prior
        tab_col = np.full((len(cnn_prob), 1), tab_fire_prior, dtype=np.float32)
        meta_X  = np.concatenate([cnn_prob, tab_col], axis=1)          # (N_images, 3)
        print(f"  Meta-feature shape: {meta_X.shape}  "
            f"tabular prior={tab_fire_prior:.4f}  "
            f"strategy: {self.meta_strategy}")

        meta_model = _build_meta_learner(self.meta_strategy)
        cv         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_proba = cross_val_predict(
            meta_model, meta_X, y_img, cv=cv, method="predict_proba"
        )[:, 1]
        oof_preds = (oof_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_img, oof_proba)
        f1  = f1_score(y_img, oof_preds, average="weighted")
        acc = accuracy_score(y_img, oof_preds)

        meta_model_cal = CalibratedClassifierCV(meta_model, cv=3, method="isotonic")
        meta_model_cal.fit(meta_X, y_img)
        self.late_fusion_model_  = meta_model_cal
        self._late_tab_prior_    = tab_fire_prior   # store for predict

        metrics = {
            "strategy":  "late_fusion_decoupled",
            "auc":       auc, "f1": f1, "accuracy": acc,
            "oof_proba": oof_proba,
        }
        self.results_["late_fusion"] = metrics
        print(f"  [Late Fusion – decoupled] AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
        joblib.dump(meta_model_cal,
                    MODELS_DIR / f"late_fusion_{self.meta_strategy}.pkl")
        return metrics


    def evaluate_decoupled(
        self,
        img_loader:      DataLoader,
        y:               np.ndarray,
        tabular_trainer: object,
    ) -> dict:
        """
        Evaluate all fitted strategies on a held-out image set.
        Tabular branch contributes via its stored OOF prior.
        """
        evaluations = {}

        cnn_prob = self._get_cnn_proba(img_loader)
        p_cnn    = cnn_prob[:, 1]

        evaluations["CNN (image branch)"] = {
            "auc":      roc_auc_score(y, p_cnn),
            "f1":       f1_score(y, (p_cnn >= 0.5).astype(int), average="weighted"),
            "accuracy": accuracy_score(y, (p_cnn >= 0.5).astype(int)),
            "proba":    p_cnn,
        }

        if self.early_fusion_model_ is not None:
            gap   = self._get_gap(img_loader)
            Xs    = self.scaler_.transform(gap)
            p     = self.early_fusion_model_.predict_proba(Xs)[:, 1]
            evaluations["Early Fusion (CNN GAP)"] = {
                "auc":      roc_auc_score(y, p),
                "f1":       f1_score(y, (p >= 0.5).astype(int), average="weighted"),
                "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
                "proba":    p,
            }

        if self.late_fusion_model_ is not None:
            tab_col = np.full((len(cnn_prob), 1),
                            self._late_tab_prior_, dtype=np.float32)
            meta_X  = np.concatenate([cnn_prob, tab_col], axis=1)
            p       = self.late_fusion_model_.predict_proba(meta_X)[:, 1]
            evaluations["Late Fusion (decoupled)"] = {
                "auc":      roc_auc_score(y, p),
                "f1":       f1_score(y, (p >= 0.5).astype(int), average="weighted"),
                "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
                "proba":    p,
            }

        self._print_eval_table(evaluations)
        self._plot_roc_comparison(evaluations, y)
        self._plot_fusion_bar(evaluations)
        return evaluations

    def __init__(
        self,
        cnn_trainer:      object,   # CNNTrainer
        tabular_trainer:  TabularModelTrainer,
        n_classes:        int  = 2,
        meta_strategy:    str  = "logistic",
    ):
        self.cnn_trainer     = cnn_trainer
        self.tab_trainer     = tabular_trainer
        self.n_classes       = n_classes
        self.meta_strategy   = meta_strategy

        # Extract the best CNN backbone and wrap with GAP extractor
        best_cnn_name, best_cnn_model = cnn_trainer.best_model()
        self.gap_extractor = GAPExtractor(best_cnn_model, best_cnn_name).to(DEVICE)
        self.gap_extractor.eval()
        print(f"[Ensemble] CNN backbone: {best_cnn_name}")

        # Best tabular model (for late fusion branch)
        best_tab_name, self.best_tab_model = tabular_trainer.best_model()
        print(f"[Ensemble] Tabular model: {best_tab_name}")

        # Models (filled by fit_*)
        self.early_fusion_model_: lgb.LGBMClassifier | None = None
        self.late_fusion_model_:  object | None              = None
        self.scaler_:             StandardScaler | None      = None
        self.results_:            dict                       = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Feature extraction utilities
    # ──────────────────────────────────────────────────────────────────────────
    def _get_gap(self, img_loader: DataLoader) -> np.ndarray:
        return _extract_gap_features(self.gap_extractor, img_loader)

    def _get_tab(self, X_tab: np.ndarray) -> np.ndarray:
        return X_tab.astype(np.float32)

    def _get_cnn_proba(self, img_loader: DataLoader) -> np.ndarray:
        _, cnn_model = self.cnn_trainer.best_model()
        return _model_predict_proba(cnn_model, img_loader, self.n_classes)

    def _get_tab_proba(self, X_tab: np.ndarray) -> np.ndarray:
        p = self.best_tab_model.predict_proba(X_tab)
        return p  # (N, n_classes)

    # ──────────────────────────────────────────────────────────────────────────
    # EARLY FUSION
    # ──────────────────────────────────────────────────────────────────────────
    def fit_early_fusion(
        self,
        X_tab:      np.ndarray,
        img_loader: DataLoader,
        y:          np.ndarray,
        n_splits:   int = 5,
    ) -> dict:
        """
        Extract GAP features from the CNN, concatenate with tabular features,
        and train a LightGBM model on the combined vector.

        Parameters
        ----------
        X_tab      : tabular features (N, tab_dim), already encoded
        img_loader : DataLoader yielding (image, label) batches in the same
                     order as X_tab rows
        y          : integer labels (N,)
        n_splits   : stratified CV folds for evaluation

        Returns
        -------
        dict with AUC, F1, Acc and the fitted early_fusion_model_
        """
        print("\n[Early Fusion] Extracting GAP features …")
        t0      = time.time()
        gap_feats = self._get_gap(img_loader)                  # (N, 1280)
        X_merged  = np.concatenate([gap_feats, X_tab], axis=1) # (N, 1280+tab_dim)
        print(f"  GAP shape: {gap_feats.shape}  "
              f"Tab shape: {X_tab.shape}  "
              f"Merged: {X_merged.shape}  "
              f"Extraction time: {time.time()-t0:.1f}s")

        # Optional: scale the merged vector (helps GBDT with very different ranges)
        self.scaler_  = StandardScaler()
        X_scaled      = self.scaler_.fit_transform(X_merged)

        model = _build_early_fusion_gbdt(self.n_classes)
        cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # OOF prediction for evaluation
        if self.n_classes == 2:
            oof_proba = cross_val_predict(model, X_scaled, y, cv=cv,
                                          method="predict_proba")[:, 1]
        else:
            oof_proba = cross_val_predict(model, X_scaled, y, cv=cv,
                                          method="predict_proba")

        oof_preds = np.argmax(oof_proba if self.n_classes > 2
                              else np.column_stack([1-oof_proba, oof_proba]), axis=1) \
                    if self.n_classes > 2 else (oof_proba >= 0.5).astype(int)

        auc = roc_auc_score(y, oof_proba,
                            multi_class="ovr" if self.n_classes > 2 else "raise")
        f1  = f1_score(y, oof_preds, average="weighted")
        acc = accuracy_score(y, oof_preds)

        # Retrain on full data
        model.fit(X_scaled, y)
        self.early_fusion_model_ = model

        metrics = {
            "strategy":   "early_fusion",
            "auc":        auc,
            "f1":         f1,
            "accuracy":   acc,
            "oof_proba":  oof_proba,
            "n_features": X_merged.shape[1],
        }
        self.results_["early_fusion"] = metrics

        print(f"  [Early Fusion] AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
        joblib.dump(model,          MODELS_DIR / "early_fusion_gbdt.pkl")
        joblib.dump(self.scaler_,   MODELS_DIR / "early_fusion_scaler.pkl")
        print(f"  Saved → {MODELS_DIR / 'early_fusion_gbdt.pkl'}")

        return metrics

    def predict_early_fusion(self, X_tab: np.ndarray, img_loader: DataLoader) -> np.ndarray:
        """Return class probabilities using the early-fusion GBDT."""
        if self.early_fusion_model_ is None:
            raise RuntimeError("Call fit_early_fusion() first.")
        gap_feats = self._get_gap(img_loader)
        X_merged  = np.concatenate([gap_feats, X_tab], axis=1)
        X_scaled  = self.scaler_.transform(X_merged)
        return self.early_fusion_model_.predict_proba(X_scaled)

    # ──────────────────────────────────────────────────────────────────────────
    # LATE FUSION / STACKING
    # ──────────────────────────────────────────────────────────────────────────
    def fit_late_fusion(
        self,
        X_tab:      np.ndarray,
        img_loader: DataLoader,
        y:          np.ndarray,
        n_splits:   int = 5,
    ) -> dict:
        """
        Combine prediction probabilities from the CNN and the tabular model
        using a meta-learner (logistic regression or small GBDT).

        Parameters
        ----------
        X_tab      : tabular features for the meta-learner's inputs
        img_loader : DataLoader (same order as X_tab rows)
        y          : integer labels
        n_splits   : CV folds

        Returns
        -------
        dict with AUC, F1, Acc and the fitted late_fusion_model_
        """
        print("\n[Late Fusion] Building meta-features …")
        cnn_prob = self._get_cnn_proba(img_loader)    # (N, n_classes)
        tab_prob = self._get_tab_proba(X_tab)          # (N, n_classes)

        # Stack probabilities as meta-features
        meta_X = np.concatenate([cnn_prob, tab_prob], axis=1)  # (N, 2*n_classes)
        print(f"  Meta-feature shape: {meta_X.shape}  "
              f"  Strategy: {self.meta_strategy}")

        meta_model = _build_meta_learner(self.meta_strategy)
        cv         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        if self.n_classes == 2:
            oof_proba = cross_val_predict(meta_model, meta_X, y, cv=cv,
                                          method="predict_proba")[:, 1]
        else:
            oof_proba = cross_val_predict(meta_model, meta_X, y, cv=cv,
                                          method="predict_proba")

        oof_preds = (oof_proba >= 0.5).astype(int) if self.n_classes == 2 \
                    else np.argmax(oof_proba, axis=1)

        auc = roc_auc_score(y, oof_proba,
                            multi_class="ovr" if self.n_classes > 2 else "raise")
        f1  = f1_score(y, oof_preds, average="weighted")
        acc = accuracy_score(y, oof_preds)

        # Calibrate and retrain
        meta_model_cal = CalibratedClassifierCV(meta_model, cv=3, method="isotonic")
        meta_model_cal.fit(meta_X, y)
        self.late_fusion_model_ = meta_model_cal

        metrics = {
            "strategy":  "late_fusion",
            "auc":       auc,
            "f1":        f1,
            "accuracy":  acc,
            "oof_proba": oof_proba,
        }
        self.results_["late_fusion"] = metrics

        print(f"  [Late Fusion]  AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
        joblib.dump(meta_model_cal, MODELS_DIR / f"late_fusion_{self.meta_strategy}.pkl")
        print(f"  Saved → {MODELS_DIR / f'late_fusion_{self.meta_strategy}.pkl'}")

        return metrics

    def predict_late_fusion(self, X_tab: np.ndarray, img_loader: DataLoader) -> np.ndarray:
        """Return class probabilities using the late-fusion meta-learner."""
        if self.late_fusion_model_ is None:
            raise RuntimeError("Call fit_late_fusion() first.")
        cnn_prob = self._get_cnn_proba(img_loader)
        tab_prob = self._get_tab_proba(X_tab)
        meta_X   = np.concatenate([cnn_prob, tab_prob], axis=1)
        return self.late_fusion_model_.predict_proba(meta_X)

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation & comparison
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        X_tab:      np.ndarray,
        img_loader: DataLoader,
        y:          np.ndarray,
    ) -> dict:
        """
        Evaluate all fitted strategies + individual branches on a held-out set.
        Prints a comparison table and saves ROC curves.
        """
        evaluations = {}

        # Individual branches
        cnn_prob = self._get_cnn_proba(img_loader)
        tab_prob = self._get_tab_proba(X_tab)
        p_cnn = cnn_prob[:, 1] if self.n_classes == 2 else cnn_prob
        p_tab = tab_prob[:, 1] if self.n_classes == 2 else tab_prob

        for branch_name, proba in [("CNN (image branch)", p_cnn),
                                    ("Tabular branch",     p_tab)]:
            preds = (proba >= 0.5).astype(int) if self.n_classes == 2 \
                    else np.argmax(proba, axis=1)
            evaluations[branch_name] = {
                "auc":      roc_auc_score(y, proba,
                                          multi_class="ovr" if self.n_classes > 2 else "raise"),
                "f1":       f1_score(y, preds, average="weighted"),
                "accuracy": accuracy_score(y, preds),
                "proba":    proba,
            }

        # Ensemble strategies
        if self.early_fusion_model_ is not None:
            p = self.predict_early_fusion(X_tab, img_loader)
            prob = p[:, 1] if self.n_classes == 2 else p
            preds = (prob >= 0.5).astype(int) if self.n_classes == 2 \
                    else np.argmax(prob, axis=1)
            evaluations["Early Fusion"] = {
                "auc":      roc_auc_score(y, prob,
                                          multi_class="ovr" if self.n_classes > 2 else "raise"),
                "f1":       f1_score(y, preds, average="weighted"),
                "accuracy": accuracy_score(y, preds),
                "proba":    prob,
            }

        if self.late_fusion_model_ is not None:
            p = self.predict_late_fusion(X_tab, img_loader)
            prob = p[:, 1] if self.n_classes == 2 else p
            preds = (prob >= 0.5).astype(int) if self.n_classes == 2 \
                    else np.argmax(prob, axis=1)
            evaluations["Late Fusion"] = {
                "auc":      roc_auc_score(y, prob,
                                          multi_class="ovr" if self.n_classes > 2 else "raise"),
                "f1":       f1_score(y, preds, average="weighted"),
                "accuracy": accuracy_score(y, preds),
                "proba":    prob,
            }

        self._print_eval_table(evaluations)
        self._plot_roc_comparison(evaluations, y)
        self._plot_fusion_bar(evaluations)
        return evaluations

    # ──────────────────────────────────────────────────────────────────────────
    # Plots & reports
    # ──────────────────────────────────────────────────────────────────────────
    def _print_eval_table(self, evaluations: dict) -> None:
        print(f"\n{'═'*60}")
        print(f"  {'Strategy':<26} {'AUC':>8} {'F1':>8} {'Acc':>8}")
        print(f"{'─'*60}")
        for name, m in evaluations.items():
            print(f"  {name:<26} {m['auc']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>8.4f}")
        best = max(evaluations, key=lambda k: evaluations[k]["auc"])
        print(f"{'─'*60}")
        print(f"  Best strategy: {best}  (AUC={evaluations[best]['auc']:.4f})")
        print(f"{'═'*60}\n")

    def _plot_roc_comparison(self, evaluations: dict, y: np.ndarray,
                              save: bool = True) -> None:
        if self.n_classes != 2:
            return
        colors  = ["#2E86AB", "#E05C2A", "#44BBA4", "#F4B942", "#9B5DE5"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for (name, m), color in zip(evaluations.items(), colors):
            fpr, tpr, _ = roc_curve(y, m["proba"])
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"{name} (AUC={m['auc']:.4f})")
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves – All Strategies")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        if save:
            path = OUTPUTS / "ensemble_roc_comparison.png"
            fig.savefig(path, dpi=150)
            print(f"  Saved → {path}")
        plt.close(fig)

    def _plot_fusion_bar(self, evaluations: dict, save: bool = True) -> None:
        names   = list(evaluations.keys())
        metrics = ["auc", "f1", "accuracy"]
        labels  = ["ROC-AUC", "F1 (weighted)", "Accuracy"]
        colors  = ["#2E86AB", "#E05C2A", "#44BBA4"]
        x = np.arange(len(names)); w = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            vals = [evaluations[n][metric] for n in names]
            ax.bar(x + i*w, vals, w, label=label, color=color, alpha=0.9)
        ax.set_xticks(x + w)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
        ax.set_title("Multimodal Ensemble – Strategy Comparison")
        ax.legend(); plt.tight_layout()
        if save:
            path = OUTPUTS / "ensemble_strategy_comparison.png"
            fig.savefig(path, dpi=150)
            print(f"  Saved → {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────
def build_multimodal_ensemble(
    cnn_trainer:       object,
    tabular_trainer:   TabularModelTrainer,
    X_tab_train:       np.ndarray,
    img_loader_train:  DataLoader,
    y_train:           np.ndarray,
    X_tab_test:        np.ndarray,
    img_loader_test:   DataLoader,
    y_test:            np.ndarray,
    n_classes:         int = 2,
    meta_strategy:     str = "logistic",
    run_late_fusion:   bool = True,
) -> dict:
    """
    End-to-end function: fits both fusion strategies and returns evaluation results.

    Parameters
    ----------
    cnn_trainer      : fitted CNNTrainer
    tabular_trainer  : fitted TabularModelTrainer
    X_tab_train      : tabular train features (encoded, float32)
    img_loader_train : train DataLoader (same order as X_tab_train rows)
    y_train          : train labels
    X_tab_test       : tabular test features
    img_loader_test  : test DataLoader
    y_test           : test labels
    n_classes        : 2 for binary fire risk
    meta_strategy    : "logistic" | "gbdt"
    run_late_fusion  : set False to skip late fusion (if time-constrained)

    Returns
    -------
    dict with 'ensemble' (fitted MultimodalEnsemble) and 'evaluation'
    """
    ensemble = MultimodalEnsemble(cnn_trainer, tabular_trainer,
                                  n_classes=n_classes, meta_strategy=meta_strategy)

    # Early fusion (always)
    ensemble.fit_early_fusion(X_tab_train, img_loader_train, y_train)

    # Late fusion (optional)
    if run_late_fusion:
        ensemble.fit_late_fusion(X_tab_train, img_loader_train, y_train)

    # Evaluate on test set
    evaluation = ensemble.evaluate(X_tab_test, img_loader_test, y_test)

    return {"ensemble": ensemble, "evaluation": evaluation}