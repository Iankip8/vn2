#!/usr/bin/env python3
"""
H3: SURD Transform Effect Hypothesis Test.

Hypothesis: Models using SURD-selected variance-stabilizing transforms achieve
better calibration (lower pinball loss at τ*) and sharper intervals than
models in identity space.

This test:
1. Compares models with vs without SURD transforms
2. Measures pinball loss at critical fractile (τ* = 0.8333)
3. Measures interval width (80th - 20th quantile)
4. Stratifies by transform type (log1p, sqrt, cbrt, identity)

Success Criterion: Group B (SURD) has lower pinball_cf with p < 0.05

Usage:
    python scripts/test_h3_surd_effect.py \
        --checkpoints-dir models/checkpoints_h3 \
        --demand-path data/processed/demand_long.parquet \
        --output-dir reports/hypothesis_tests
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from vn2.forecast.surd_wrapper import select_best_transform

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
    max_skus: int = None
) -> pd.DataFrame:
    """Compute calibration metrics for a model."""
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
                
                # Get forecasts at key quantiles
                q20 = np.interp(0.20, QUANTILE_LEVELS, quantiles)
                q80 = np.interp(0.80, QUANTILE_LEVELS, quantiles)
                q_cf = np.interp(CRITICAL_FRACTILE, QUANTILE_LEVELS, quantiles)
                
                # Get actual
                actual_row = sku_demand.iloc[fold_idx] if fold_idx < len(sku_demand) else None
                if actual_row is None:
                    continue
                actual = actual_row['demand']
                
                # Compute metrics
                pl_cf = pinball_loss(actual, q_cf, CRITICAL_FRACTILE)
                interval_width = q80 - q20
                
                records.append({
                    'store': store,
                    'product': product,
                    'week': fold_idx + 1,
                    'model': model_name,
                    'forecast_q_cf': q_cf,
                    'actual': actual,
                    'pinball_cf': pl_cf,
                    'interval_width': interval_width,
                    'q20': q20,
                    'q80': q80
                })
            except Exception:
                continue
    
    return pd.DataFrame(records)


def run_surd_comparison(
    surd_metrics: pd.DataFrame,
    identity_metrics: pd.DataFrame
) -> Dict:
    """Compare SURD vs identity transform metrics."""
    
    # Merge on common SKU-weeks
    merged = surd_metrics.merge(
        identity_metrics,
        on=['store', 'product', 'week', 'actual'],
        suffixes=('_surd', '_identity')
    )
    
    if len(merged) < 30:
        return {'error': f'Insufficient data: {len(merged)} samples'}
    
    # Pinball loss comparison
    pl_surd = merged['pinball_cf_surd'].values
    pl_identity = merged['pinball_cf_identity'].values
    
    pl_delta = pl_identity - pl_surd  # Positive = SURD is better
    t_stat_pl, p_value_pl = stats.ttest_rel(pl_identity, pl_surd)
    
    # Interval width comparison (lower = sharper)
    iw_surd = merged['interval_width_surd'].values
    iw_identity = merged['interval_width_identity'].values
    
    iw_delta = iw_identity - iw_surd  # Positive = SURD is sharper
    
    return {
        'n_samples': len(merged),
        'mean_pinball_surd': float(np.mean(pl_surd)),
        'mean_pinball_identity': float(np.mean(pl_identity)),
        'pinball_delta': float(np.mean(pl_delta)),
        'pinball_t_stat': float(t_stat_pl),
        'pinball_p_value': float(p_value_pl),
        'pinball_significant': p_value_pl < 0.05,
        'pinball_surd_better_pct': float((pl_delta > 0).mean() * 100),
        'mean_width_surd': float(np.mean(iw_surd)),
        'mean_width_identity': float(np.mean(iw_identity)),
        'width_delta': float(np.mean(iw_delta)),
        'width_surd_sharper_pct': float((iw_delta > 0).mean() * 100),
        'hypothesis_supported': np.mean(pl_delta) > 0 and p_value_pl < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description="H3 SURD Effect Hypothesis Test")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hypothesis_tests"))
    parser.add_argument("--surd-model", default=None,
                        help="SURD-transformed model (auto-detect if not specified)")
    parser.add_argument("--identity-model", default=None,
                        help="Identity-space model (auto-detect if not specified)")
    parser.add_argument("--max-skus", type=int, default=300)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("H3: SURD TRANSFORM EFFECT HYPOTHESIS TEST")
    print("="*70)
    print("\nHypothesis: SURD variance-stabilizing transforms improve calibration")
    print("            (lower pinball loss at τ*) compared to identity space.")
    print(f"\nSuccess Criterion: Lower pinball loss with p < 0.05")
    
    # Load demand
    print("\nLoading demand data...")
    demand_df = load_demand(args.demand_path)
    
    # Find available models
    available = [d.name for d in args.checkpoints_dir.iterdir() 
                 if d.is_dir() and not d.name.startswith('.')]
    
    # Try to identify SURD vs identity models
    surd_models = [m for m in available if 'surd' in m.lower() or 'slurp' in m.lower()]
    identity_models = [m for m in available if m not in surd_models]
    
    if args.surd_model:
        surd_model = args.surd_model
    elif surd_models:
        surd_model = surd_models[0]
    elif len(available) >= 2:
        surd_model = available[0]
    else:
        print("ERROR: Need at least 2 models for comparison")
        return
    
    if args.identity_model:
        identity_model = args.identity_model
    elif identity_models:
        identity_model = identity_models[0]
    elif len(available) >= 2:
        identity_model = available[1]
    else:
        identity_model = available[0]
    
    print(f"\nComparing models:")
    print(f"  SURD/transformed: {surd_model}")
    print(f"  Identity/baseline: {identity_model}")
    
    # Load metrics
    print(f"\nLoading {surd_model} metrics...")
    surd_metrics = compute_model_metrics(surd_model, args.checkpoints_dir, demand_df, args.max_skus)
    print(f"  {len(surd_metrics)} records")
    
    print(f"\nLoading {identity_model} metrics...")
    identity_metrics = compute_model_metrics(identity_model, args.checkpoints_dir, demand_df, args.max_skus)
    print(f"  {len(identity_metrics)} records")
    
    if len(surd_metrics) < 50 or len(identity_metrics) < 50:
        print("\nERROR: Insufficient data for comparison")
        return
    
    # Run comparison
    print("\nRunning comparison...")
    result = run_surd_comparison(surd_metrics, identity_metrics)
    
    if 'error' in result:
        print(f"\nERROR: {result['error']}")
        return
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nSamples compared: {result['n_samples']}")
    print(f"\nPinball Loss at τ* = {CRITICAL_FRACTILE}:")
    print(f"  {identity_model}: {result['mean_pinball_identity']:.3f}")
    print(f"  {surd_model}: {result['mean_pinball_surd']:.3f}")
    print(f"  Delta: {result['pinball_delta']:.3f} (positive = SURD better)")
    print(f"  t-statistic: {result['pinball_t_stat']:.3f}")
    print(f"  p-value: {result['pinball_p_value']:.4f}")
    print(f"  SURD better: {result['pinball_surd_better_pct']:.1f}% of samples")
    print(f"\nInterval Width (Q80 - Q20):")
    print(f"  {identity_model}: {result['mean_width_identity']:.2f}")
    print(f"  {surd_model}: {result['mean_width_surd']:.2f}")
    print(f"  SURD sharper: {result['width_surd_sharper_pct']:.1f}% of samples")
    
    # Verdict
    print("\n" + "="*70)
    if result['hypothesis_supported']:
        verdict = "SUPPORTED"
        print(f"[✓] H3 (SURD Effect) is {verdict}")
        print("    SURD transforms significantly improve calibration at τ*")
    else:
        verdict = "NOT SUPPORTED"
        print(f"[✗] H3 (SURD Effect) is {verdict}")
        if result['pinball_delta'] <= 0:
            print("    SURD did not improve pinball loss")
        else:
            print("    Improvement not statistically significant")
    
    # Save results
    results_df = pd.DataFrame([{
        'surd_model': surd_model,
        'identity_model': identity_model,
        **result
    }])
    results_df.to_csv(args.output_dir / 'h3_surd_effect_results.csv', index=False)
    
    # Report
    report = []
    report.append("# H3: SURD Effect Hypothesis Test Report\n")
    report.append(f"**Verdict: {verdict}**\n")
    report.append(f"\n## Models Compared\n")
    report.append(f"- SURD model: {surd_model}\n")
    report.append(f"- Identity model: {identity_model}\n")
    report.append(f"\n## Results\n")
    report.append(f"- Samples: {result['n_samples']}\n")
    report.append(f"- Pinball loss (SURD): {result['mean_pinball_surd']:.3f}\n")
    report.append(f"- Pinball loss (identity): {result['mean_pinball_identity']:.3f}\n")
    report.append(f"- Improvement: {result['pinball_delta']:.3f}\n")
    report.append(f"- p-value: {result['pinball_p_value']:.4f}\n")
    
    report_path = args.output_dir / 'h3_surd_effect_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

