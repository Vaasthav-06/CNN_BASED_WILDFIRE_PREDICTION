#!/usr/bin/env python3
"""
prepare_combined_dataset.py
============================
Merges all four region feature CSVs into a single combined CSV with a
`region` column added, so `train_global_model.py` can train one model
across all regions.

Usage
-----
    python prepare_combined_dataset.py
    python prepare_combined_dataset.py --output_csv data/combined/features_all_regions.csv
    python prepare_combined_dataset.py --regions corbett similipal   # subset only
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REGION_CSVS = {
    "corbett":    ROOT / "data/case_studies/corbett/processed/tabular/features_corbett.csv",
    "jyotikuchi": ROOT / "data/case_studies/jyotikuchi/processed/tabular/features_jyotikuchi.csv",
    "laisong":    ROOT / "data/case_studies/laisong/processed/tabular/features_laisong.csv",
    "similipal":  ROOT / "data/case_studies/similipal/processed/tabular/features_similipal.csv",
}

DEFAULT_OUTPUT = ROOT / "data/combined/features_all_regions.csv"


def prepare_combined(
    regions:    list[str] | None = None,
    output_csv: Path | str       = DEFAULT_OUTPUT,
    label_col:  str              = "fire",
) -> Path:
    """
    Load each region's feature CSV, stamp a `region` column, concatenate,
    align columns, and write to output_csv.

    Returns the output path.
    """
    regions = regions or list(REGION_CSVS.keys())
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for region in regions:
        if region not in REGION_CSVS:
            print(f"  ⚠  Unknown region '{region}', skipping.")
            continue
        csv_path = REGION_CSVS[region]
        if not csv_path.exists():
            print(f"  ⚠  CSV not found for '{region}': {csv_path}, skipping.")
            continue

        df = pd.read_csv(csv_path)
        # Stamp region identifier (used as a feature + for stratification)
        df["region"] = region
        frames.append(df)
        print(f"  ✓  {region:<14}  {len(df):>6,} rows  ×  {len(df.columns)} cols")

    if not frames:
        sys.exit("❌  No region CSVs loaded. Check paths in REGION_CSVS.")

    # ── Align columns across regions ──────────────────────────────────────────
    # Some regions may have slightly different feature sets; we take the
    # intersection so all models receive the same feature space. The label
    # column and `region` are always kept even if missing from some frames.
    all_cols   = [set(f.columns) for f in frames]
    common     = set.intersection(*all_cols)
    # Always keep label + region
    keep = sorted(common | {label_col, "region"})

    aligned = []
    for df in frames:
        missing = [c for c in keep if c not in df.columns]
        if missing:
            print(f"  ⚠  Region '{df['region'].iloc[0]}' missing cols: {missing} — filling with NaN")
        aligned.append(df.reindex(columns=keep))

    combined = pd.concat(aligned, ignore_index=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Combined shape : {combined.shape[0]:,} rows × {combined.shape[1]} cols")
    if label_col in combined.columns:
        dist = combined[label_col].value_counts().to_dict()
        print(f"  Label dist     : {dist}")
    print(f"  Region counts  : {combined['region'].value_counts().to_dict()}")

    combined.to_csv(output_csv, index=False)
    print(f"\n  ✅  Saved → {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-region feature CSVs into one combined dataset."
    )
    parser.add_argument(
        "--regions", nargs="+",
        default=list(REGION_CSVS.keys()),
        help="Which regions to include (default: all four)",
    )
    parser.add_argument(
        "--output_csv", default=str(DEFAULT_OUTPUT),
        help="Output path for the combined CSV",
    )
    parser.add_argument(
        "--label_col", default="fire",
        help="Target/label column name (default: fire)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPARING COMBINED MULTI-REGION DATASET")
    print("=" * 60)
    prepare_combined(
        regions    = args.regions,
        output_csv = args.output_csv,
        label_col  = args.label_col,
    )


if __name__ == "__main__":
    main()