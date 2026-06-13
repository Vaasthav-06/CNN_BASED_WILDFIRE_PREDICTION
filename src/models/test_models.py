"""
test_models.py
==============
Smoke-test: generates synthetic data and verifies all three modules run
end-to-end without errors. Run this before training on real data.

Usage
-----
    python src/models/test_models.py
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

print("=" * 70)
print("WILDFIRE PREDICTION MODELS – QUICK TEST")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Tabular Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 1/3] Tabular Models (XGBoost + CatBoost + LightGBM)")
print("─" * 70)
try:
    from src.models.tabular_models import TabularModelTrainer

    np.random.seed(42)
    n_samples = 200
    X_num = np.random.randn(n_samples, 10)
    y = (X_num[:, 0] + X_num[:, 1] > 0).astype(int)

    df = pd.DataFrame({f"num_{i}": X_num[:, i] for i in range(10)})
    # Use "fire" — the actual label column name in all feature CSVs
    df["fire"]   = y
    df["region"] = np.random.choice(["Region_A", "Region_B"], n_samples)

    print(f"  Generated {df.shape[0]} samples × {df.shape[1]} features")

    trainer = TabularModelTrainer(df, label_col="fire", n_splits=3, region_col="region")
    results = trainer.train_all(save=False)

    print(f"  ✓ XGBoost   AUC={results['XGBoost']['auc']:.4f}")
    print(f"  ✓ CatBoost  AUC={results['CatBoost']['auc']:.4f}")
    print(f"  ✓ LightGBM  AUC={results['LightGBM']['auc']:.4f}")
    print("  Status: PASS ✓")

except Exception as e:
    print(f"  Status: FAIL ✗  —  {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: CNN Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 2/3] CNN Models (ResNet50 + EfficientNet-B0)")
print("─" * 70)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from src.models.cnn_models import CNNTrainer

    class DummyImageDataset(Dataset):
        def __init__(self, n=50):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            # Correctly-normalised 3×256×256 tensor
            img   = torch.randn(3, 256, 256)
            label = idx % 2
            return img, label

    train_loader = DataLoader(DummyImageDataset(50), batch_size=8, shuffle=True)
    val_loader   = DataLoader(DummyImageDataset(10), batch_size=8)

    print(f"  Generated 50 training + 10 validation images")

    # Use 2 epochs only for the smoke-test — real training uses 50
    trainer_cnn = CNNTrainer(train_loader, val_loader,
                             num_classes=2, epochs=2, patience=5)
    trainer_cnn.run_baseline_comparison()
    trainer_cnn.scale_if_needed()

    best_name, _ = trainer_cnn.best_model()
    print(f"  ✓ Best CNN: {best_name}")
    print(f"    Val Acc: {trainer_cnn.results_[best_name]['best_val_acc']:.4f}")
    print("  Status: PASS ✓")

except Exception as e:
    print(f"  Status: FAIL ✗  —  {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Ensemble (Early + Late Fusion)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 3/3] Multimodal Ensemble (Early + Late Fusion)")
print("─" * 70)
try:
    from src.models.ensemble import MultimodalEnsemble

    # 50 samples; 10 tabular features matching Test 1
    X_tab  = np.random.randn(50, 10).astype(np.float32)
    y_ens  = np.array([i % 2 for i in range(50)])

    # Re-use the same dummy loaders so order is consistent
    ens_loader = DataLoader(DummyImageDataset(50), batch_size=8, shuffle=False)

    print(f"  Generated {len(X_tab)} multimodal samples (tabular + images)")

    ensemble = MultimodalEnsemble(trainer_cnn, trainer, n_classes=2)

    # Both fusion strategies must actually be called and fitted
    early_res = ensemble.fit_early_fusion(X_tab, ens_loader, y_ens, n_splits=3)
    print(f"  ✓ Early Fusion  AUC={early_res['auc']:.4f}")

    late_res = ensemble.fit_late_fusion(X_tab, ens_loader, y_ens, n_splits=3)
    print(f"  ✓ Late Fusion   AUC={late_res['auc']:.4f}")

    print("  Status: PASS ✓")

except Exception as e:
    print(f"  Status: FAIL ✗  —  {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nPipeline verified. Ready for real data.")
print("\nNext steps:")
print("  Train:    python train_all_models.py --feature_csv <path>")
print("  Evaluate: python src/models/test_models.py  (this script, already done)")