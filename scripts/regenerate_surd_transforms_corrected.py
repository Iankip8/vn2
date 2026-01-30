#!/usr/bin/env python3
"""
Regenerate surd_transforms.parquet with corrected CV calculations.

This script fixes the CV calculation bug where negative means in log space
caused invalid negative CV values. It also only selects transforms that
actually reduce variance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import argparse

# Transform functions
TRANSFORMS = {
    'identity': {
        'forward': lambda x: x,
    },
    'log': {
        'forward': lambda x: np.log(np.maximum(x, 1e-10)),
    },
    'log1p': {
        'forward': lambda x: np.log1p(x),
    },
    'sqrt': {
        'forward': lambda x: np.sqrt(np.maximum(x, 0)),
    },
    'cbrt': {
        'forward': lambda x: np.cbrt(x),
    },
}


def calculate_cv(series: np.ndarray) -> float:
    """
    Calculate coefficient of variation properly.
    
    Uses absolute value of mean to handle negative means in transformed space.
    """
    mean_val = series.mean()
    std_val = series.std()
    
    if abs(mean_val) < 1e-9:
        return np.inf if std_val > 0 else 0.0
    
    return std_val / abs(mean_val)


def evaluate_transform(series: np.ndarray, transform_name: str) -> Tuple[float, np.ndarray]:
    """
    Evaluate a transform and return (cv, transformed_series).
    
    Returns (np.inf, None) if transform is invalid.
    """
    try:
        transform_func = TRANSFORMS[transform_name]['forward']
        transformed = transform_func(series)
        
        # Check for invalid values
        if not np.all(np.isfinite(transformed)):
            return np.inf, None
        
        cv = calculate_cv(transformed)
        return cv, transformed
    except Exception:
        return np.inf, None


def select_best_transform(series: np.ndarray, min_improvement: float = 0.0) -> Tuple[str, float, float, float]:
    """
    Select best variance-stabilizing transform.
    
    Args:
        series: Time series values
        min_improvement: Minimum CV reduction required (as fraction, e.g., 0.05 = 5%)
    
    Returns:
        (best_transform, original_cv, transformed_cv, cv_reduction_pct)
    """
    # Filter to positive values for transforms that require it
    y_positive = series[series > 0] if np.any(series > 0) else series
    
    if len(y_positive) < 2:
        return 'identity', calculate_cv(series), calculate_cv(series), 0.0
    
    # Calculate original CV
    original_cv = calculate_cv(series)
    
    # Try each transform
    best_transform = 'identity'
    best_transformed_cv = original_cv
    best_cv_reduction = 0.0
    
    for transform_name in TRANSFORMS.keys():
        transformed_cv, transformed_series = evaluate_transform(y_positive, transform_name)
        
        if transformed_cv == np.inf:
            continue
        
        # Calculate CV reduction
        if original_cv > 0:
            cv_reduction_pct = (original_cv - transformed_cv) / original_cv
        else:
            cv_reduction_pct = 0.0
        
        # Only consider if it improves CV by at least min_improvement
        if cv_reduction_pct >= min_improvement:
            if transformed_cv < best_transformed_cv:
                best_transform = transform_name
                best_transformed_cv = transformed_cv
                best_cv_reduction = cv_reduction_pct
    
    return best_transform, original_cv, best_transformed_cv, best_cv_reduction


def regenerate_transforms(
    demand_path: Path,
    output_path: Path,
    min_observations: int = 12,
    min_improvement: float = 0.0,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Regenerate SURD transforms for all SKUs.
    
    Args:
        demand_path: Path to demand data parquet
        output_path: Path to save results
        min_observations: Minimum observations required
        min_improvement: Minimum CV reduction required (0.0 = any improvement, 0.05 = 5%)
        n_jobs: Number of parallel jobs (1 = sequential)
    """
    print(f"Loading demand data from {demand_path}...")
    df = pd.read_parquet(demand_path)
    
    # Standardize column names
    if 'Sales' in df.columns:
        df = df.rename(columns={'Sales': 'sales'})
    if 'demand' in df.columns:
        df = df.rename(columns={'demand': 'sales'})
    
    store_col = 'Store' if 'Store' in df.columns else 'store'
    product_col = 'Product' if 'Product' in df.columns else 'product'
    
    print(f"Found {len(df)} observations")
    print(f"Unique SKUs: {len(df[[store_col, product_col]].drop_duplicates())}")
    
    results = []
    
    # Group by SKU
    sku_groups = list(df.groupby([store_col, product_col]))
    print(f"\nEvaluating transforms for {len(sku_groups)} SKUs...")
    
    for idx, ((store, product), sku_data) in enumerate(sku_groups):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(sku_groups)} SKUs...")
        
        sales = sku_data['sales'].values
        
        if len(sales) < min_observations:
            # Not enough data, use identity
            original_cv = calculate_cv(sales)
            results.append({
                'Store': store,
                'Product': product,
                'best_transform': 'identity',
                'original_cv': original_cv,
                'transformed_cv': original_cv,
                'cv_reduction': 0.0,
                'n_observations': len(sales),
            })
            continue
        
        # Select best transform
        best_transform, orig_cv, trans_cv, cv_reduction = select_best_transform(
            sales, min_improvement=min_improvement
        )
        
        results.append({
            'Store': store,
            'Product': product,
            'best_transform': best_transform,
            'original_cv': orig_cv,
            'transformed_cv': trans_cv,
            'cv_reduction': cv_reduction,
            'n_observations': len(sales),
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("Transform Selection Summary")
    print(f"{'='*80}")
    print(f"\nTotal SKUs: {len(results_df)}")
    print(f"\nTransform distribution:")
    print(results_df['best_transform'].value_counts())
    
    print(f"\nCV Reduction statistics:")
    print(f"  Mean: {results_df['cv_reduction'].mean():.4f} ({results_df['cv_reduction'].mean()*100:.2f}%)")
    print(f"  Median: {results_df['cv_reduction'].median():.4f} ({results_df['cv_reduction'].median()*100:.2f}%)")
    print(f"  Min: {results_df['cv_reduction'].min():.4f} ({results_df['cv_reduction'].min()*100:.2f}%)")
    print(f"  Max: {results_df['cv_reduction'].max():.4f} ({results_df['cv_reduction'].max()*100:.2f}%)")
    
    improved = (results_df['cv_reduction'] > 0).sum()
    worsened = (results_df['cv_reduction'] < 0).sum()
    no_change = (results_df['cv_reduction'] == 0).sum()
    
    print(f"\nEffectiveness:")
    print(f"  ✅ Improved: {improved} ({improved/len(results_df)*100:.1f}%)")
    print(f"  ❌ Worsened: {worsened} ({worsened/len(results_df)*100:.1f}%)")
    print(f"  ➖ No change: {no_change} ({no_change/len(results_df)*100:.1f}%)")
    
    print(f"\nBy transform type:")
    for transform in results_df['best_transform'].unique():
        subset = results_df[results_df['best_transform'] == transform]
        improved_pct = (subset['cv_reduction'] > 0).mean() * 100
        mean_reduction = subset['cv_reduction'].mean() * 100
        print(f"  {transform:8s}: {len(subset):3d} SKUs, {improved_pct:5.1f}% improved, mean reduction: {mean_reduction:6.2f}%")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    print(f"\n💾 Saved corrected transforms to {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Regenerate SURD transforms with corrected CV calculation')
    parser.add_argument('--input', type=str, default='data/processed/demand_long.parquet',
                       help='Path to demand data parquet file')
    parser.add_argument('--output', type=str, default='data/processed/surd_transforms_corrected.parquet',
                       help='Path to save corrected transforms')
    parser.add_argument('--min-observations', type=int, default=12,
                       help='Minimum observations required for transform selection')
    parser.add_argument('--min-improvement', type=float, default=0.0,
                       help='Minimum CV reduction required (0.0 = any improvement, 0.05 = 5%%)')
    parser.add_argument('--backup-original', action='store_true',
                       help='Backup original surd_transforms.parquet before overwriting')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        # Try alternatives
        alt_paths = [
            Path('data/processed/demand_imputed.parquet'),
            Path('data/processed/demand_imputed_capped.parquet'),
        ]
        for p in alt_paths:
            if p.exists():
                input_path = p
                print(f"Using alternative path: {input_path}")
                break
        
        if not input_path.exists():
            print(f"Error: Could not find demand data at {args.input} or alternatives")
            return 1
    
    # Backup original if requested
    original_path = Path('data/processed/surd_transforms.parquet')
    if args.backup_original and original_path.exists():
        backup_path = Path('data/processed/surd_transforms.parquet.backup')
        import shutil
        shutil.copy2(original_path, backup_path)
        print(f"📦 Backed up original to {backup_path}")
    
    # Regenerate
    results_df = regenerate_transforms(
        input_path,
        output_path,
        min_observations=args.min_observations,
        min_improvement=args.min_improvement
    )
    
    # Optionally replace original
    if output_path.name == 'surd_transforms_corrected.parquet':
        print(f"\n💡 To use corrected transforms, either:")
        print(f"   1. Update code to use 'surd_transforms_corrected.parquet'")
        print(f"   2. Replace original: cp {output_path} {original_path}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

