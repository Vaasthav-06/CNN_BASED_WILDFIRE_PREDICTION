#!/usr/bin/env python3
"""
prepare_combined_cnn_dataset.py
================================
Symlinks (or copies) image tiles from all four region cnn_datasets into a
single combined ImageFolder at  data/combined/cnn_dataset/{fire,no_fire}/.

Using symlinks avoids duplicating hundreds of MB of images; pass --copy to
physically copy if symlinks cause issues on Windows.

Usage
-----
    python prepare_combined_cnn_dataset.py
    python prepare_combined_cnn_dataset.py --output_dir data/combined/cnn_dataset
    python prepare_combined_cnn_dataset.py --copy        # hard copies instead of symlinks
    python prepare_combined_cnn_dataset.py --regions corbett similipal
"""

import sys
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REGION_IMG_DIRS = {
    "corbett":    ROOT / "data/case_studies/corbett/processed/cnn_dataset",
    "jyotikuchi": ROOT / "data/case_studies/jyotikuchi/processed/cnn_dataset",
    "laisong":    ROOT / "data/case_studies/laisong/processed/cnn_dataset",
    "similipal":  ROOT / "data/case_studies/similipal/processed/cnn_dataset",
}

DEFAULT_OUTPUT = ROOT / "data/combined/cnn_dataset"
CLASS_DIRS     = ["fire", "no_fire"]


def _link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return  # already done
    if use_copy:
        shutil.copy2(src, dst)
    else:
        # On Windows this requires Developer Mode or admin rights
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)   # fallback to copy


def prepare_combined_cnn(
    regions:    list[str] | None = None,
    output_dir: Path | str       = DEFAULT_OUTPUT,
    use_copy:   bool             = False,
) -> Path:
    regions    = regions or list(REGION_IMG_DIRS.keys())
    output_dir = Path(output_dir)

    for cls in CLASS_DIRS:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    totals = {cls: 0 for cls in CLASS_DIRS}

    for region in regions:
        if region not in REGION_IMG_DIRS:
            print(f"  ⚠  Unknown region '{region}', skipping.")
            continue
        src_base = REGION_IMG_DIRS[region]
        if not src_base.exists():
            print(f"  ⚠  Image dir not found for '{region}': {src_base}, skipping.")
            continue

        for cls in CLASS_DIRS:
            src_cls  = src_base / cls
            dst_cls  = output_dir / cls
            if not src_cls.exists():
                print(f"  ⚠  {region}/{cls} does not exist, skipping.")
                continue

            imgs = list(src_cls.glob("*"))
            for img_path in imgs:
                if not img_path.is_file():
                    continue
                # Prefix filename with region to avoid collisions
                dst_name = f"{region}_{img_path.name}"
                _link_or_copy(img_path, dst_cls / dst_name, use_copy)
                totals[cls] += 1

        print(f"  ✓  {region}")

    print(f"\n  Combined image totals:")
    for cls in CLASS_DIRS:
        print(f"    {cls:<10}: {totals[cls]:,}")
    print(f"\n  ✅  Combined CNN dataset → {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-region CNN image datasets into one combined ImageFolder."
    )
    parser.add_argument("--regions",    nargs="+", default=list(REGION_IMG_DIRS.keys()))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--copy",       action="store_true",
                        help="Physically copy images instead of symlinking")
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPARING COMBINED CNN DATASET")
    print("=" * 60)
    prepare_combined_cnn(
        regions    = args.regions,
        output_dir = args.output_dir,
        use_copy   = args.copy,
    )


if __name__ == "__main__":
    main()