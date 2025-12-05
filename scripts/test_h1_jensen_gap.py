#!/usr/bin/env python3
"""
H1: Jensen Gap Hypothesis Test.

Hypothesis: For the same forecast information set, density-aware SIP optimization
achieves strictly lower realized cost than point+service-level policies.

This test:
1. For each model, computes cost using density-aware SIP (PMF -> optimize)
2. Computes cost using point + service level (median -> CF formula)
3. Computes Jensen Delta = cost(point) - cost(SIP)
4. Statistical test: paired t-test across SKUs
5. Reports per-model delta, confidence interval, p-value

Success Criterion: Jensen Delta > 0 with p < 0.05 for majority of models

Usage:
    python scripts/test_h1_jensen_gap.py \
        --checkpoints-dir models/checkpoints_h3 \
        --demand-path data/processed/demand_long.parquet \
        --output-dir reports/hypothesis_tests
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import optimization functions
from vn2.analyze.sequential_planner import Costs
from vn2.analyze.sequential_backtest import quantiles_to_pmf

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])

# Cost parameters
CU = 1.0  # Shortage cost
CO = 0.2  # Holding cost
CRITICAL_FRACTILE = CU / (CU + CO)  # 0.8333


def load_demand(path: Path) -> pd.DataFrame:
    """Load demand data."""
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'demand' not in df.columns and 'sales' in df.columns:
        df['demand'] = df['sales']
    return df


def compute_sip_order(pmf: np.ndarray, inventory: int, tau: float = CRITICAL_FRACTILE) -> int:
    """Compute optimal order using SIP (density-aware) approach.
    
    Uses newsvendor fractile on the full PMF.
    """
    support = np.arange(len(pmf))
    cdf = np.cumsum(pmf)
    
    # Find net inventory position after expected sales
    # Optimal order targets the tau-quantile of demand
    target_position = support[np.searchsorted(cdf, tau)]
    
    order = max(0, int(target_position - inventory))
    return order


def compute_point_order(
    median_forecast: float,
    std_forecast: float,
    inventory: int,
    tau: float = CRITICAL_FRACTILE
) -> int:
    """Compute order using point + service level approach.
    
    Uses normal approximation: Q = median + z*std - inventory
    where z is chosen to achieve tau service level.
    """
    z = stats.norm.ppf(tau)
    target_position = median_forecast + z * std_forecast
    order = max(0, int(np.ceil(target_position - inventory)))
    return order


def compute_realized_cost(
    order: int,
    inventory: int,
    actual_demand: int,
    cu: float = CU,
    co: float = CO
) -> float:
    """Compute realized cost given order and actual demand."""
    available = inventory + order
    sold = min(available, actual_demand)
    shortage = max(0, actual_demand - available)
    leftover = max(0, available - actual_demand)
    
    return cu * shortage + co * leftover


def run_jensen_test_for_model(
    model_name: str,
    checkpoints_dir: Path,
    demand_df: pd.DataFrame,
    max_skus: int = None
) -> Dict:
    """Run Jensen gap test for a single model."""
    model_dir = checkpoints_dir / model_name
    if not model_dir.exists():
        return {'model': model_name, 'error': 'Model directory not found'}
    
    sip_costs = []
    point_costs = []
    
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
                
                # Get forecast quantiles
                quantiles = qdf.loc[1].values
                
                # Create PMF for SIP approach
                pmf = quantiles_to_pmf(quantiles, QUANTILE_LEVELS, grain=500)
                
                # Get point statistics for point approach
                median = np.interp(0.50, QUANTILE_LEVELS, quantiles)
                q25 = np.interp(0.25, QUANTILE_LEVELS, quantiles)
                q75 = np.interp(0.75, QUANTILE_LEVELS, quantiles)
                std_proxy = (q75 - q25) / 1.35  # IQR to std approximation
                
                # Get actual demand
                actual_row = sku_demand.iloc[fold_idx] if fold_idx < len(sku_demand) else None
                if actual_row is None:
                    continue
                actual = int(actual_row['demand'])
                
                # Assume starting inventory of 0 for fair comparison
                inventory = 0
                
                # SIP approach
                sip_order = compute_sip_order(pmf, inventory)
                sip_cost = compute_realized_cost(sip_order, inventory, actual)
                
                # Point approach
                point_order = compute_point_order(median, std_proxy, inventory)
                point_cost = compute_realized_cost(point_order, inventory, actual)
                
                sip_costs.append(sip_cost)
                point_costs.append(point_cost)
                
            except Exception:
                continue
    
    if len(sip_costs) < 50:
        return {
            'model': model_name,
            'error': f'Insufficient data: {len(sip_costs)} samples'
        }
    
    sip_costs = np.array(sip_costs)
    point_costs = np.array(point_costs)
    deltas = point_costs - sip_costs  # Positive = SIP is better
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(point_costs, sip_costs)
    
    # Confidence interval for mean delta
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas, ddof=1)
    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=mean_delta, scale=std_delta/np.sqrt(n))
    
    return {
        'model': model_name,
        'n_samples': n,
        'mean_sip_cost': float(np.mean(sip_costs)),
        'mean_point_cost': float(np.mean(point_costs)),
        'mean_delta': float(mean_delta),
        'std_delta': float(std_delta),
        'ci_lower': float(ci_95[0]),
        'ci_upper': float(ci_95[1]),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'sip_wins_pct': float((deltas > 0).mean() * 100),
        'hypothesis_supported': mean_delta > 0 and p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description="H1 Jensen Gap Hypothesis Test")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hypothesis_tests"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-skus", type=int, default=300, help="Limit SKUs for faster testing")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("H1: JENSEN GAP HYPOTHESIS TEST")
    print("="*70)
    print("\nHypothesis: Density-aware SIP optimization achieves lower realized cost")
    print("            than point+service-level policies.")
    print(f"\nSuccess Criterion: Jensen Delta > 0 with p < 0.05")
    print(f"Critical Fractile: τ* = {CRITICAL_FRACTILE:.4f}")
    
    # Load demand
    print("\nLoading demand data...")
    demand_df = load_demand(args.demand_path)
    
    # Get models
    if args.models:
        models = args.models
    else:
        models = [d.name for d in args.checkpoints_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\nTesting {len(models)} models...")
    
    # Run tests
    results = []
    for model in models:
        print(f"\n  Testing {model}...")
        result = run_jensen_test_for_model(
            model, args.checkpoints_dir, demand_df, args.max_skus
        )
        results.append(result)
        
        if 'error' not in result:
            sig = "***" if result['hypothesis_supported'] else ""
            print(f"    Delta: {result['mean_delta']:.2f} ± {result['std_delta']:.2f}")
            print(f"    p-value: {result['p_value']:.4f} {sig}")
            print(f"    SIP wins: {result['sip_wins_pct']:.1f}%")
    
    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / 'h1_jensen_gap_results.csv', index=False)
    
    valid_results = results_df[~results_df['p_value'].isna()]
    supported = valid_results['hypothesis_supported'].sum()
    total = len(valid_results)
    
    print("\n" + "="*70)
    print("HYPOTHESIS TEST SUMMARY")
    print("="*70)
    print(f"\nModels tested: {total}")
    print(f"H1 Supported: {supported} ({supported/total*100:.1f}%)")
    
    # Overall verdict
    if supported >= total * 0.5:
        verdict = "SUPPORTED"
        print(f"\n[✓] H1 (Jensen Gap) is {verdict}")
        print("    Density-aware SIP optimization provides significant cost savings")
        print("    over point+service-level policies for majority of models.")
    else:
        verdict = "NOT SUPPORTED"
        print(f"\n[✗] H1 (Jensen Gap) is {verdict}")
        print("    The hypothesized advantage of density-aware optimization")
        print("    was not observed across the majority of models.")
    
    # Generate report
    report = []
    report.append("# H1: Jensen Gap Hypothesis Test Report\n")
    report.append(f"**Verdict: {verdict}**\n")
    report.append(f"\n## Summary\n")
    report.append(f"- Models tested: {total}\n")
    report.append(f"- H1 supported: {supported} ({supported/total*100:.1f}%)\n")
    report.append(f"- Critical fractile: τ* = {CRITICAL_FRACTILE:.4f}\n")
    report.append(f"\n## Results by Model\n")
    report.append(results_df.to_markdown(index=False))
    report.append(f"\n\n## Interpretation\n")
    if supported >= total * 0.5:
        report.append("The Jensen Gap hypothesis is supported. Using density forecasts\n")
        report.append("with SIP optimization consistently outperforms point forecasts\n")
        report.append("with service-level targeting. This validates the approach of\n")
        report.append("optimizing on the full predictive distribution rather than\n")
        report.append("chaining point forecasts to deterministic rules.\n")
    else:
        report.append("The Jensen Gap hypothesis is not strongly supported.\n")
        report.append("This may indicate that forecast quality issues dominate\n")
        report.append("the optimization approach differences.\n")
    
    report_path = args.output_dir / 'h1_jensen_gap_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

