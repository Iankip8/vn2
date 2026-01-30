#!/usr/bin/env python3
"""
Compare SURD vs Non-SURD Model Performance.

This script compares forecasting models that use SURD transforms vs those that don't,
across multiple metrics:
- Pinball loss at critical fractile
- Interval coverage
- Interval width
- CRPS (Continuous Ranked Probability Score)
- Realized costs (if available)

Usage:
    python scripts/compare_surd_vs_non_surd.py \
        --checkpoints-dir models/checkpoints \
        --demand-path data/processed/demand_long.parquet \
        --output-dir reports/surd_comparison
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
CRITICAL_FRACTILE = 0.8333


def load_demand(path: Path) -> pd.DataFrame:
    """Load demand data."""
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'demand' not in df.columns and 'sales' in df.columns:
        df['demand'] = df['sales']
    return df


def crps(actual: float, quantiles: np.ndarray, q_levels: np.ndarray) -> float:
    """Compute Continuous Ranked Probability Score."""
    # CRPS = integral of (F(x) - 1(x >= actual))^2 dx
    # Approximate using quantiles
    q_sorted = np.sort(quantiles)
    q_levels_sorted = np.sort(q_levels)
    
    # Find where actual falls
    if actual <= q_sorted[0]:
        return np.mean((q_sorted - actual)**2)
    elif actual >= q_sorted[-1]:
        return np.mean((actual - q_sorted)**2)
    else:
        # Interpolate
        idx = np.searchsorted(q_sorted, actual)
        w = (actual - q_sorted[idx-1]) / (q_sorted[idx] - q_sorted[idx-1])
        crps_val = 0.0
        for i in range(len(q_sorted)):
            if i < idx - 1:
                crps_val += (q_levels_sorted[i])**2 * (q_sorted[i+1] - q_sorted[i])
            elif i == idx - 1:
                crps_val += (q_levels_sorted[i] - w)**2 * (actual - q_sorted[i])
                crps_val += (1 - q_levels_sorted[i] - (1-w))**2 * (q_sorted[i+1] - actual)
            else:
                crps_val += (1 - q_levels_sorted[i])**2 * (q_sorted[i+1] - q_sorted[i])
        return crps_val


def pinball_loss(actual: float, forecast: float, tau: float) -> float:
    """Compute pinball (quantile) loss."""
    error = actual - forecast
    if error >= 0:
        return tau * error
    else:
        return -(1 - tau) * error


def compute_model_metrics(
    model_name: str,
    checkpoints_dir: Path,
    demand_df: pd.DataFrame,
    max_skus: Optional[int] = None
) -> pd.DataFrame:
    """Compute comprehensive metrics for a model."""
    model_dir = checkpoints_dir / model_name
    if not model_dir.exists():
        return pd.DataFrame()
    
    records = []
    skus = demand_df.groupby(['store', 'product']).size().index.tolist()
    if max_skus:
        skus = skus[:max_skus]
    
    for store, product in skus:
        sku_dir = model_dir / f'{store}_{product}'
        if not sku_dir.exists():
            continue
        
        sku_demand = demand_df[
            (demand_df['store'] == store) & 
            (demand_df['product'] == product)
        ].sort_values('week')
        
        for fold_idx in range(min(6, len(sku_demand))):
            fold_path = sku_dir / f'fold_{fold_idx}.pkl'
            if not fold_path.exists():
                continue
            
            try:
                with open(fold_path, 'rb') as f:
                    data = pickle.load(f)
                qdf = data.get('quantiles')
                if qdf is None or qdf.empty or 1 not in qdf.index:
                    continue
                
                quantiles = qdf.loc[1].values
                
                # Get actual
                actual_row = sku_demand.iloc[fold_idx] if fold_idx < len(sku_demand) else None
                if actual_row is None:
                    continue
                actual = actual_row['demand']
                
                # Compute metrics
                q_cf = np.interp(CRITICAL_FRACTILE, QUANTILE_LEVELS, quantiles)
                q20 = np.interp(0.20, QUANTILE_LEVELS, quantiles)
                q80 = np.interp(0.80, QUANTILE_LEVELS, quantiles)
                q50 = np.interp(0.50, QUANTILE_LEVELS, quantiles)
                
                pl_cf = pinball_loss(actual, q_cf, CRITICAL_FRACTILE)
                interval_width = q80 - q20
                coverage_80 = 1.0 if (q20 <= actual <= q80) else 0.0
                coverage_90 = 1.0 if (np.interp(0.05, QUANTILE_LEVELS, quantiles) <= actual <= 
                                      np.interp(0.95, QUANTILE_LEVELS, quantiles)) else 0.0
                crps_val = crps(actual, quantiles, QUANTILE_LEVELS)
                mae = abs(actual - q50)
                mse = (actual - q50)**2
                
                records.append({
                    'store': store,
                    'product': product,
                    'week': fold_idx + 1,
                    'model': model_name,
                    'actual': actual,
                    'forecast_median': q50,
                    'forecast_q_cf': q_cf,
                    'pinball_cf': pl_cf,
                    'interval_width_80': interval_width,
                    'coverage_80': coverage_80,
                    'coverage_90': coverage_90,
                    'crps': crps_val,
                    'mae': mae,
                    'mse': mse,
                })
            except Exception as e:
                continue
    
    return pd.DataFrame(records)


def compare_models(
    surd_metrics: pd.DataFrame,
    non_surd_metrics: pd.DataFrame,
    surd_model_name: str,
    non_surd_model_name: str
) -> Dict:
    """Compare SURD vs non-SURD models."""
    
    # Merge on common SKU-weeks
    merged = surd_metrics.merge(
        non_surd_metrics,
        on=['store', 'product', 'week', 'actual'],
        suffixes=('_surd', '_non_surd')
    )
    
    if len(merged) < 30:
        return {'error': f'Insufficient data: {len(merged)} samples'}
    
    results = {
        'n_samples': len(merged),
        'surd_model': surd_model_name,
        'non_surd_model': non_surd_model_name,
    }
    
    # Pinball loss at critical fractile
    pl_surd = merged['pinball_cf_surd'].values
    pl_non_surd = merged['pinball_cf_non_surd'].values
    pl_delta = pl_non_surd - pl_surd  # Positive = SURD better
    t_stat_pl, p_value_pl = stats.ttest_rel(pl_non_surd, pl_surd)
    results.update({
        'pinball_cf_surd_mean': float(np.mean(pl_surd)),
        'pinball_cf_non_surd_mean': float(np.mean(pl_non_surd)),
        'pinball_cf_delta': float(np.mean(pl_delta)),
        'pinball_cf_t_stat': float(t_stat_pl),
        'pinball_cf_p_value': float(p_value_pl),
        'pinball_cf_surd_better_pct': float((pl_delta > 0).mean() * 100),
    })
    
    # Interval width
    iw_surd = merged['interval_width_80_surd'].values
    iw_non_surd = merged['interval_width_80_non_surd'].values
    iw_delta = iw_non_surd - iw_surd  # Positive = SURD sharper
    results.update({
        'interval_width_surd_mean': float(np.mean(iw_surd)),
        'interval_width_non_surd_mean': float(np.mean(iw_non_surd)),
        'interval_width_delta': float(np.mean(iw_delta)),
        'interval_width_surd_sharper_pct': float((iw_delta > 0).mean() * 100),
    })
    
    # Coverage
    cov80_surd = merged['coverage_80_surd'].mean()
    cov80_non_surd = merged['coverage_80_non_surd'].mean()
    cov90_surd = merged['coverage_90_surd'].mean()
    cov90_non_surd = merged['coverage_90_non_surd'].mean()
    results.update({
        'coverage_80_surd': float(cov80_surd),
        'coverage_80_non_surd': float(cov80_non_surd),
        'coverage_90_surd': float(cov90_surd),
        'coverage_90_non_surd': float(cov90_non_surd),
    })
    
    # CRPS
    crps_surd = merged['crps_surd'].values
    crps_non_surd = merged['crps_non_surd'].values
    crps_delta = crps_non_surd - crps_surd  # Positive = SURD better
    t_stat_crps, p_value_crps = stats.ttest_rel(crps_non_surd, crps_surd)
    results.update({
        'crps_surd_mean': float(np.mean(crps_surd)),
        'crps_non_surd_mean': float(np.mean(crps_non_surd)),
        'crps_delta': float(np.mean(crps_delta)),
        'crps_t_stat': float(t_stat_crps),
        'crps_p_value': float(p_value_crps),
        'crps_surd_better_pct': float((crps_delta > 0).mean() * 100),
    })
    
    # MAE
    mae_surd = merged['mae_surd'].values
    mae_non_surd = merged['mae_non_surd'].values
    mae_delta = mae_non_surd - mae_surd  # Positive = SURD better
    results.update({
        'mae_surd_mean': float(np.mean(mae_surd)),
        'mae_non_surd_mean': float(np.mean(mae_non_surd)),
        'mae_delta': float(np.mean(mae_delta)),
    })
    
    # Overall verdict
    results['surd_better'] = (
        np.mean(pl_delta) > 0 and p_value_pl < 0.05
    ) or (
        np.mean(crps_delta) > 0 and p_value_crps < 0.05
    )
    
    return results


def create_visualizations(
    surd_metrics: pd.DataFrame,
    non_surd_metrics: pd.DataFrame,
    output_dir: Path
):
    """Create comparison visualizations."""
    # Merge for plotting
    merged = surd_metrics.merge(
        non_surd_metrics,
        on=['store', 'product', 'week', 'actual'],
        suffixes=('_surd', '_non_surd')
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Pinball loss comparison
    ax = axes[0, 0]
    ax.scatter(merged['pinball_cf_non_surd'], merged['pinball_cf_surd'], 
               alpha=0.3, s=10)
    max_pl = max(merged['pinball_cf_non_surd'].max(), merged['pinball_cf_surd'].max())
    ax.plot([0, max_pl], [0, max_pl], 'r--', linewidth=2, label='Equal')
    ax.set_xlabel('Non-SURD Pinball Loss')
    ax.set_ylabel('SURD Pinball Loss')
    ax.set_title('Pinball Loss at Critical Fractile')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Interval width comparison
    ax = axes[0, 1]
    ax.scatter(merged['interval_width_80_non_surd'], merged['interval_width_80_surd'],
               alpha=0.3, s=10)
    max_iw = max(merged['interval_width_80_non_surd'].max(), 
                 merged['interval_width_80_surd'].max())
    ax.plot([0, max_iw], [0, max_iw], 'r--', linewidth=2, label='Equal')
    ax.set_xlabel('Non-SURD Interval Width')
    ax.set_ylabel('SURD Interval Width')
    ax.set_title('80% Prediction Interval Width')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. CRPS comparison
    ax = axes[1, 0]
    ax.scatter(merged['crps_non_surd'], merged['crps_surd'],
               alpha=0.3, s=10)
    max_crps = max(merged['crps_non_surd'].max(), merged['crps_surd'].max())
    ax.plot([0, max_crps], [0, max_crps], 'r--', linewidth=2, label='Equal')
    ax.set_xlabel('Non-SURD CRPS')
    ax.set_ylabel('SURD CRPS')
    ax.set_title('Continuous Ranked Probability Score')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Delta distribution
    ax = axes[1, 1]
    pl_delta = merged['pinball_cf_non_surd'] - merged['pinball_cf_surd']
    ax.hist(pl_delta, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.axvline(pl_delta.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {pl_delta.mean():.3f}')
    ax.set_xlabel('Pinball Loss Delta (Non-SURD - SURD)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Pinball Loss Improvement')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'surd_vs_non_surd_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare SURD vs Non-SURD Models")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/surd_comparison"))
    parser.add_argument("--surd-model", default=None,
                        help="SURD model name (auto-detect if not specified)")
    parser.add_argument("--non-surd-model", default=None,
                        help="Non-SURD model name (auto-detect if not specified)")
    parser.add_argument("--max-skus", type=int, default=300)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SURD vs Non-SURD Model Comparison")
    print("="*80)
    
    # Load demand
    print("\nLoading demand data...")
    demand_df = load_demand(args.demand_path)
    
    # Find available models
    available = [d.name for d in args.checkpoints_dir.iterdir() 
                 if d.is_dir() and not d.name.startswith('.')]
    
    # Identify SURD vs non-SURD models
    surd_models = [m for m in available if 'surd' in m.lower()]
    non_surd_models = [m for m in available if 'surd' not in m.lower() and 
                       'slurp' in m.lower() or 'seasonal' in m.lower()]
    
    if args.surd_model:
        surd_model = args.surd_model
    elif surd_models:
        surd_model = surd_models[0]
    else:
        print("ERROR: No SURD model found")
        return 1
    
    if args.non_surd_model:
        non_surd_model = args.non_surd_model
    elif non_surd_models:
        non_surd_model = non_surd_models[0]
    else:
        print("ERROR: No non-SURD model found")
        return 1
    
    print(f"\nComparing models:")
    print(f"  SURD: {surd_model}")
    print(f"  Non-SURD: {non_surd_model}")
    
    # Load metrics
    print(f"\nLoading {surd_model} metrics...")
    surd_metrics = compute_model_metrics(surd_model, args.checkpoints_dir, demand_df, args.max_skus)
    print(f"  {len(surd_metrics)} records")
    
    print(f"\nLoading {non_surd_model} metrics...")
    non_surd_metrics = compute_model_metrics(non_surd_model, args.checkpoints_dir, demand_df, args.max_skus)
    print(f"  {len(non_surd_metrics)} records")
    
    if len(surd_metrics) < 50 or len(non_surd_metrics) < 50:
        print("\nERROR: Insufficient data for comparison")
        return 1
    
    # Compare
    print("\nRunning comparison...")
    results = compare_models(surd_metrics, non_surd_metrics, surd_model, non_surd_model)
    
    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        return 1
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nSamples: {results['n_samples']}")
    
    print(f"\n📊 Pinball Loss at τ* = {CRITICAL_FRACTILE}:")
    print(f"  Non-SURD: {results['pinball_cf_non_surd_mean']:.3f}")
    print(f"  SURD:     {results['pinball_cf_surd_mean']:.3f}")
    print(f"  Delta:    {results['pinball_cf_delta']:.3f} (positive = SURD better)")
    print(f"  p-value:  {results['pinball_cf_p_value']:.4f}")
    print(f"  SURD better: {results['pinball_cf_surd_better_pct']:.1f}% of samples")
    
    print(f"\n📊 Interval Width (80%):")
    print(f"  Non-SURD: {results['interval_width_non_surd_mean']:.2f}")
    print(f"  SURD:     {results['interval_width_surd_mean']:.2f}")
    print(f"  SURD sharper: {results['interval_width_surd_sharper_pct']:.1f}% of samples")
    
    print(f"\n📊 Coverage:")
    print(f"  80% interval - Non-SURD: {results['coverage_80_non_surd']:.3f}, SURD: {results['coverage_80_surd']:.3f}")
    print(f"  90% interval - Non-SURD: {results['coverage_90_non_surd']:.3f}, SURD: {results['coverage_90_surd']:.3f}")
    
    print(f"\n📊 CRPS:")
    print(f"  Non-SURD: {results['crps_non_surd_mean']:.3f}")
    print(f"  SURD:     {results['crps_surd_mean']:.3f}")
    print(f"  Delta:    {results['crps_delta']:.3f} (positive = SURD better)")
    print(f"  p-value:  {results['crps_p_value']:.4f}")
    
    print(f"\n📊 MAE:")
    print(f"  Non-SURD: {results['mae_non_surd_mean']:.3f}")
    print(f"  SURD:     {results['mae_surd_mean']:.3f}")
    
    # Verdict
    print("\n" + "="*80)
    if results['surd_better']:
        print("✓ SURD models show statistically significant improvement")
    else:
        print("✗ SURD models do not show statistically significant improvement")
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(args.output_dir / 'surd_comparison_results.csv', index=False)
    
    # Create visualizations
    create_visualizations(surd_metrics, non_surd_metrics, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

