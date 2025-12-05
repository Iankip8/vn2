#!/usr/bin/env python3
"""
Apply conformal calibration to all forecast models.

This script:
1. Loads existing model checkpoints
2. Fits calibration from holdout actuals
3. Creates calibrated versions of all checkpoints

Usage:
    python scripts/apply_conformal_calibration.py \
        --checkpoints-dir models/checkpoints_h3 \
        --actuals-path data/processed/demand_long.parquet \
        --output-dir models/checkpoints_h3_conformal \
        --models zinb lightgbm_quantile slurp_bootstrap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from vn2.forecast.calibration.conformal import calibrate_model_quantiles

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])


def load_actuals(path: Path) -> pd.DataFrame:
    """Load actual demand data."""
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Rename columns if needed
    if 'demand' not in df.columns and 'sales' in df.columns:
        df['demand'] = df['sales']
    
    # Convert week to integer index
    if 'week' in df.columns:
        if df['week'].dtype == 'datetime64[ns]':
            weeks_sorted = sorted(df['week'].unique())
            week_map = {w: i for i, w in enumerate(weeks_sorted)}
            df['week'] = df['week'].map(week_map)
    
    df = df.rename(columns={'demand': 'actual'})
    return df[['store', 'product', 'week', 'actual']]


def main():
    parser = argparse.ArgumentParser(description="Apply conformal calibration to forecast models")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--actuals-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/checkpoints_h3_conformal"))
    parser.add_argument("--models", nargs="+", default=None, help="Models to calibrate (default: all)")
    parser.add_argument("--method", choices=['isotonic', 'scaling', 'additive'], default='isotonic')
    parser.add_argument("--holdout-folds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()
    
    print(f"Loading actuals from {args.actuals_path}")
    actuals = load_actuals(args.actuals_path)
    print(f"  {len(actuals)} demand observations")
    
    # Get list of models to calibrate
    if args.models:
        models = args.models
    else:
        models = [d.name for d in args.checkpoints_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\nCalibrating {len(models)} models:")
    for model in models:
        print(f"  - {model}")
    
    results = []
    for model in models:
        print(f"\nCalibrating {model}...")
        try:
            result = calibrate_model_quantiles(
                model_name=model,
                checkpoints_dir=args.checkpoints_dir,
                actuals_df=actuals,
                output_dir=args.output_dir,
                quantile_levels=QUANTILE_LEVELS,
                method=args.method,
                holdout_folds=args.holdout_folds
            )
            results.append(result)
            
            if 'error' in result:
                print(f"  Warning: {result['error']}")
            else:
                print(f"  Calibrated {result['n_checkpoints_calibrated']} checkpoints")
                print(f"  Output: {result['output_dir']}")
                
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'model': model, 'error': str(e)})
    
    # Summary
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nOutput directory: {args.output_dir}")
        print("Calibrated models:")
        for r in successful:
            print(f"  - conformal_{r['model']} ({r['n_checkpoints_calibrated']} checkpoints)")
    
    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {r['model']}: {r.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()

