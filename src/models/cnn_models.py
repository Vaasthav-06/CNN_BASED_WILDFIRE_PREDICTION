"""
cnn_model.py
============
CNN backbone trainer for wildfire UAV/satellite image patches (256×256).

Professor's prescribed progression
------------------------------------
Stage 1 – Baselines
  • ResNet50     : reliable baseline, well-understood
  • EfficientNet-B0 : lightweight, fast to train

Stage 2 – Compare on
  • Accuracy (val top-1)
  • Training time per epoch
  • Overfitting gap (train_acc − val_acc)

Stage 3 – Scale if justified
  • EfficientNet-B2 or B3 if B0 underfits or headroom exists

The class also exposes extract_features() which returns the global-average-pool
(GAP) embedding vector — used by the early-fusion ensemble.

Usage
-----
    from src.models.cnn_model import CNNTrainer
    trainer = CNNTrainer(train_loader, val_loader, num_classes=2)
    trainer.run_baseline_comparison()        # Stage 1+2
    trainer.scale_if_needed(threshold=0.02) # Stage 3
    feat_vec = trainer.extract_features(image_tensor)
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
OUTPUTS    = ROOT / "outputs" / "cnn_models"
MODELS_DIR = ROOT / "saved_models" / "cnn"
OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CNNTrainer] Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Standard transforms for 256×256 patches
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def _build_resnet50(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """ResNet50 – Stage 1 baseline."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    # Replace final FC
    in_feats = model.fc.in_features                  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feats, num_classes),
    )
    return model


def _build_efficientnet(variant: str, num_classes: int,
                         freeze_backbone: bool = False) -> nn.Module:
    """EfficientNet-B0 / B2 / B3 factory."""
    builders = {
        "b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
        "b2": (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
        "b3": (models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    }
    if variant not in builders:
        raise ValueError(f"variant must be one of {list(builders)}; got '{variant}'")

    builder, weights = builders[variant]
    model = builder(weights=weights)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    # Replace classifier
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feats, num_classes),
    )
    return model


def build_model(name: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """
    Unified factory.
    name ∈ {"resnet50", "efficientnet_b0", "efficientnet_b2", "efficientnet_b3"}
    """
    name = name.lower()
    if name == "resnet50":
        return _build_resnet50(num_classes, freeze_backbone)
    elif name.startswith("efficientnet_"):
        variant = name.split("_")[-1]
        return _build_efficientnet(variant, num_classes, freeze_backbone)
    else:
        raise ValueError(f"Unknown model name: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# GAP feature extractor hook (for early fusion)
# ─────────────────────────────────────────────────────────────────────────────
class GAPExtractor(nn.Module):
    """
    Wraps any torchvision model and returns the Global Average Pool
    embedding (before the final classifier) for multimodal fusion.
    """
    def __init__(self, backbone: nn.Module, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        if "resnet" in self.backbone_name:
            # ResNet: everything except the FC layer
            self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B,2048,1,1)
        else:
            # EfficientNet: backbone.features + avgpool
            self.features = nn.Sequential(backbone.features, backbone.avgpool)  # → (B,C,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return feat.flatten(1)     # (B, embedding_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Single-model training loop
# ─────────────────────────────────────────────────────────────────────────────
def _train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def train_model(
    model_name:    str,
    num_classes:   int,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    epochs:        int   = 30,
    lr:            float = 1e-3,
    patience:      int   = 7,
    min_delta:     float = 1e-3,   # minimum improvement to reset patience counter
    freeze_epochs: int   = 5,      # freeze backbone for first N epochs
    save:          bool  = True,
) -> dict:
    """
    Full training loop for one model variant.

    Returns
    -------
    dict with history, best val_acc, avg_epoch_time, overfit_gap

    Early stopping
    --------------
    Patience counter only resets when val_acc improves by more than
    min_delta. This prevents noise-level fluctuations (e.g. 0.8201 →
    0.8203) from resetting the counter and burning extra epochs.
    """
    model = build_model(model_name, num_classes, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Optimizer: only train the new head initially
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "epoch_time"]}
    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    epoch_times  = []

    print(f"\n  ── {model_name} ──  epochs={epochs}  lr={lr}  "
          f"patience={patience}  min_delta={min_delta}  device={DEVICE}")

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=epochs - freeze_epochs)
            print(f"    [Epoch {epoch}] Backbone unfrozen – fine-tuning all layers")

        t0 = time.time()
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = _evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["epoch_time"].append(elapsed)

        print(f"    Epoch {epoch:03d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}  "
              f"time={elapsed:.1f}s")

        # Early stopping: only reset patience on a meaningful improvement
        if vl_acc > best_val_acc + min_delta:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            print(f"    ✓ val_acc improved to {best_val_acc:.4f} — checkpoint saved")
        else:
            no_improve += 1
            print(f"    No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch} — best val_acc={best_val_acc:.4f}")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    avg_epoch_time = float(np.mean(epoch_times))
    overfit_gap    = float(np.mean(history["train_acc"][-5:]) -
                           np.mean(history["val_acc"][-5:]))

    result = {
        "model":          model,
        "history":        history,
        "best_val_acc":   best_val_acc,
        "avg_epoch_time": avg_epoch_time,
        "overfit_gap":    overfit_gap,
        "epochs_run":     len(history["train_acc"]),
    }

    if save:
        ckpt_path = MODELS_DIR / f"{model_name}_best.pt"
        torch.save({"state_dict": best_state, "val_acc": best_val_acc,
                    "overfit_gap": overfit_gap}, ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CNNTrainer: orchestrates the full professor-prescribed progression
# ─────────────────────────────────────────────────────────────────────────────
class CNNTrainer:
    """
    Orchestrates the full CNN progression:
      1. Baseline comparison (ResNet50 vs EfficientNet-B0)
      2. Benchmark table (accuracy / training time / overfitting gap)
      3. Conditional scaling to EfficientNet-B2 or B3

    Parameters
    ----------
    train_loader : DataLoader for training patches
    val_loader   : DataLoader for validation patches
    num_classes  : number of output classes
    epochs       : max epochs per model
    lr           : initial learning rate
    patience     : early-stopping patience
    min_delta    : minimum val_acc improvement to reset patience counter
    """

    # Overfitting and accuracy thresholds that trigger scaling
    OVERFITTING_THRESHOLD = 0.05   # train_acc − val_acc > 5% = overfitting
    ACCURACY_HEADROOM     = 0.85   # val_acc < 85% = scaling might help

    def __init__(
        self,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        num_classes:   int   = 2,
        epochs:        int   = 30,
        lr:            float = 1e-3,
        patience:      int   = 7,
        min_delta:     float = 1e-3,
    ):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.num_classes  = num_classes
        self.epochs       = epochs
        self.lr           = lr
        self.patience     = patience
        self.min_delta    = min_delta
        self.results_: dict[str, dict] = {}

    def run_baseline_comparison(self) -> dict:
        """Stage 1+2: train ResNet50 and EfficientNet-B0, print comparison."""
        stage1_models = ["resnet50", "efficientnet_b0"]
        for name in stage1_models:
            res = train_model(
                name, self.num_classes,
                self.train_loader, self.val_loader,
                epochs=self.epochs, lr=self.lr,
                patience=self.patience, min_delta=self.min_delta,
            )
            self.results_[name] = res

        self._print_comparison_table(list(self.results_.keys()))
        self._plot_training_curves(list(self.results_.keys()))
        return self.results_

    def scale_if_needed(self, scale_target: str = "auto") -> dict | None:
        """
        Stage 3: decide whether scaling is warranted based on baseline results.

        Scaling is triggered when:
          • val_acc of best baseline < ACCURACY_HEADROOM, OR
          • overfit_gap > OVERFITTING_THRESHOLD

        scale_target : "b2" | "b3" | "auto"
            auto → use B2 if moderate underfit, B3 if severe underfit
        """
        if not self.results_:
            raise RuntimeError("Call run_baseline_comparison() first.")

        best_name = max(self.results_, key=lambda k: self.results_[k]["best_val_acc"])
        best_acc  = self.results_[best_name]["best_val_acc"]
        best_gap  = self.results_[best_name]["overfit_gap"]

        print(f"\n{'─'*60}")
        print(f"  Scale decision:  best_val_acc={best_acc:.4f}  "
              f"overfit_gap={best_gap:.4f}")

        # Fixed: gap > threshold (not <) means overfitting → scaling may help
        should_scale = (best_acc < self.ACCURACY_HEADROOM) or \
                       (best_gap > self.OVERFITTING_THRESHOLD)

        if not should_scale:
            print(f"  → No scaling needed. {best_name} meets accuracy target.")
            return None

        if scale_target == "auto":
            scale_target = "b2" if best_acc > 0.75 else "b3"

        scaled_name = f"efficientnet_{scale_target}"
        print(f"  → Scaling to {scaled_name} …")
        res = train_model(
            scaled_name, self.num_classes,
            self.train_loader, self.val_loader,
            epochs=self.epochs, lr=self.lr * 0.5,
            patience=self.patience, min_delta=self.min_delta,
        )
        self.results_[scaled_name] = res
        self._print_comparison_table(list(self.results_.keys()))
        self._plot_training_curves(list(self.results_.keys()))
        return self.results_

    def best_model(self) -> tuple[str, nn.Module]:
        """Return (name, model) of the best-performing trained variant."""
        if not self.results_:
            raise RuntimeError("No trained models yet.")
        name = max(self.results_, key=lambda k: self.results_[k]["best_val_acc"])
        return name, self.results_[name]["model"]

    # ──────────────────────────────────────────────────────
    # Feature extraction for early fusion
    # ──────────────────────────────────────────────────────
    def get_gap_extractor(self, model_name: str | None = None) -> GAPExtractor:
        """
        Returns a GAPExtractor wrapping the chosen (or best) backbone.
        The extractor's output is concatenated with tabular features in
        the early-fusion ensemble.
        """
        if model_name is None:
            model_name, _ = self.best_model()
        model = self.results_[model_name]["model"]
        extractor = GAPExtractor(model, model_name).to(DEVICE)
        extractor.eval()
        return extractor

    @torch.no_grad()
    def extract_features(
        self,
        loader:       DataLoader,
        model_name:   str | None = None,
    ) -> np.ndarray:
        """
        Extract GAP feature vectors for every image in `loader`.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim)
        """
        extractor = self.get_gap_extractor(model_name)
        all_feats = []
        for imgs, _ in loader:
            feats = extractor(imgs.to(DEVICE))
            all_feats.append(feats.cpu().numpy())
        return np.vstack(all_feats)

    # ──────────────────────────────────────────────────────
    # Reporting helpers
    # ──────────────────────────────────────────────────────
    def _print_comparison_table(self, names: list[str]) -> None:
        print(f"\n{'═'*65}")
        print(f"  {'Model':<22} {'Val Acc':>8} {'Time/Ep(s)':>11} {'Overfit Gap':>12}")
        print(f"{'─'*65}")
        for n in names:
            r = self.results_[n]
            print(f"  {n:<22} {r['best_val_acc']:>8.4f} "
                  f"{r['avg_epoch_time']:>11.1f} {r['overfit_gap']:>12.4f}")
        best = max(names, key=lambda k: self.results_[k]["best_val_acc"])
        print(f"{'─'*65}")
        print(f"  Best: {best}  (val_acc={self.results_[best]['best_val_acc']:.4f})")
        print(f"{'═'*65}\n")

    def _plot_training_curves(self, names: list[str], save: bool = True) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ["#2E86AB", "#E05C2A", "#44BBA4", "#F4B942"]

        for i, name in enumerate(names):
            h = self.results_[name]["history"]
            color = colors[i % len(colors)]
            ep = range(1, len(h["train_acc"]) + 1)
            axes[0].plot(ep, h["train_acc"],  "--", color=color, alpha=0.5)
            axes[0].plot(ep, h["val_acc"],          color=color, label=name)
            axes[1].plot(ep, h["train_loss"], "--", color=color, alpha=0.5)
            axes[1].plot(ep, h["val_loss"],         color=color, label=name)

        for ax, title in zip(axes[:2], ["Accuracy (solid=val)", "Loss (solid=val)"]):
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()

        # Bar chart: overfit gap comparison
        gap_vals  = [self.results_[n]["overfit_gap"] for n in names]
        time_vals = [self.results_[n]["avg_epoch_time"] for n in names]
        ax2 = axes[2]
        x   = np.arange(len(names))
        ax2.bar(x - 0.2, gap_vals,  0.35, label="Overfit Gap", color="#E05C2A")
        ax2b = ax2.twinx()
        ax2b.bar(x + 0.2, time_vals, 0.35, label="Time/Epoch (s)", color="#2E86AB", alpha=0.7)
        ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15)
        ax2.set_ylabel("Overfit Gap (train−val acc)")
        ax2b.set_ylabel("Avg Epoch Time (s)")
        ax2.set_title("Overfitting & Speed Comparison")
        ax2.axhline(self.OVERFITTING_THRESHOLD, color="red", lw=1, ls="--",
                    label=f"Threshold ({self.OVERFITTING_THRESHOLD})")
        ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")

        plt.tight_layout()
        if save:
            path = OUTPUTS / "cnn_training_comparison.png"
            fig.savefig(path, dpi=150)
            print(f"  Saved → {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────
def train_cnn_models(
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    num_classes:   int   = 2,
    epochs:        int   = 30,
    lr:            float = 1e-3,
    patience:      int   = 7,
    min_delta:     float = 1e-3,
) -> CNNTrainer:
    """
    Full professor-prescribed CNN progression:
      1. Baseline: ResNet50 + EfficientNet-B0
      2. Benchmark on accuracy / speed / overfitting
      3. Scale to B2/B3 if needed

    Returns the fitted CNNTrainer object.
    """
    trainer = CNNTrainer(train_loader, val_loader,
                         num_classes=num_classes, epochs=epochs,
                         lr=lr, patience=patience, min_delta=min_delta)
    trainer.run_baseline_comparison()
    trainer.scale_if_needed(scale_target="auto")
    return trainer