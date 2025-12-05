#!/usr/bin/env python3
"""
SURD Information-Theoretic Decomposition Analysis.

This script performs three types of analysis:

1. Forecast Error Decomposition
   - Decomposes forecast error variance into systematic (bias) vs random components
   - Identifies what portion of error is predictable vs irreducible

2. Model Information Comparison
   - Computes unique vs redundant information across models
   - Identifies which models capture different aspects of demand

3. Feature Attribution
   - Analyzes which features drive forecast uncertainty
   - Maps feature importance to interval width and calibration

Usage:
    python scripts/surd_decomposition_analysis.py \
        --checkpoints-dir models/checkpoints_h3 \
        --demand-path data/processed/demand_long.parquet \
        --output-dir reports/surd_analysis
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


def load_demand(path: Path) -> pd.DataFrame:
    """Load demand data."""
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'demand' not in df.columns and 'sales' in df.columns:
        df['demand'] = df['sales']
    return df


def load_forecasts_for_model(
    model_name: str,
    checkpoints_dir: Path,
    demand_df: pd.DataFrame,
    max_skus: int = None
) -> pd.DataFrame:
    """Load forecasts for a model and match with actuals."""
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
                
                # Get forecast median and Q83
                q50 = np.interp(0.50, qdf.columns, qdf.loc[1].values)
                q83 = np.interp(0.8333, qdf.columns, qdf.loc[1].values)
                q10 = np.interp(0.10, qdf.columns, qdf.loc[1].values)
                q90 = np.interp(0.90, qdf.columns, qdf.loc[1].values)
                
                # Get actual
                actual_row = sku_demand.iloc[fold_idx] if fold_idx < len(sku_demand) else None
                if actual_row is None:
                    continue
                
                actual = actual_row['demand']
                
                records.append({
                    'store': store,
                    'product': product,
                    'week': fold_idx + 1,
                    'model': model_name,
                    'forecast_q50': q50,
                    'forecast_q83': q83,
                    'forecast_q10': q10,
                    'forecast_q90': q90,
                    'interval_width': q90 - q10,
                    'actual': actual,
                    'error_q50': actual - q50,
                    'error_q83': actual - q83,
                })
            except Exception:
                continue
    
    return pd.DataFrame(records)


def decompose_forecast_error(forecasts_df: pd.DataFrame) -> Dict:
    """Decompose forecast error into components.
    
    Error = Bias (systematic) + Variance (random)
    
    Returns dict with:
    - total_mse: Total mean squared error
    - bias_squared: Squared bias component
    - variance: Variance component
    - explained_ratio: Proportion of error that is bias (predictable)
    """
    if len(forecasts_df) < 10:
        return {'error': 'Insufficient data'}
    
    errors = forecasts_df['error_q83'].values
    
    total_mse = np.mean(errors ** 2)
    bias = np.mean(errors)
    bias_squared = bias ** 2
    variance = np.var(errors)
    
    # R-squared: how much variance in errors is explained by features
    # (This would require features; for now we use bias ratio)
    explained_ratio = bias_squared / total_mse if total_mse > 0 else 0
    
    return {
        'total_mse': float(total_mse),
        'rmse': float(np.sqrt(total_mse)),
        'bias': float(bias),
        'bias_squared': float(bias_squared),
        'variance': float(variance),
        'std': float(np.std(errors)),
        'explained_by_bias': float(explained_ratio),
        'unexplained': float(1 - explained_ratio)
    }


def compute_model_information_overlap(
    model_a_forecasts: pd.DataFrame,
    model_b_forecasts: pd.DataFrame,
    n_bins: int = 10
) -> Dict:
    """Compute information overlap between two models.
    
    Uses mutual information to measure:
    - Redundant information: shared by both models
    - Unique information: captured by one model but not the other
    
    Returns dict with mutual information metrics.
    """
    # Merge on common SKU-weeks
    merged = model_a_forecasts.merge(
        model_b_forecasts,
        on=['store', 'product', 'week', 'actual'],
        suffixes=('_a', '_b')
    )
    
    if len(merged) < 50:
        return {'error': 'Insufficient overlapping data'}
    
    # Discretize for mutual information
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    
    errors_a = merged['error_q83_a'].values.reshape(-1, 1)
    errors_b = merged['error_q83_b'].values.reshape(-1, 1)
    actuals = merged['actual'].values.reshape(-1, 1)
    
    try:
        errors_a_binned = discretizer.fit_transform(errors_a).flatten().astype(int)
        errors_b_binned = discretizer.fit_transform(errors_b).flatten().astype(int)
        actuals_binned = discretizer.fit_transform(actuals).flatten().astype(int)
    except Exception:
        return {'error': 'Discretization failed'}
    
    # Mutual information between errors and actuals
    mi_a_actual = mutual_info_score(errors_a_binned, actuals_binned)
    mi_b_actual = mutual_info_score(errors_b_binned, actuals_binned)
    mi_a_b = mutual_info_score(errors_a_binned, errors_b_binned)
    
    # Entropy of each error distribution
    h_a = stats.entropy(np.bincount(errors_a_binned) / len(errors_a_binned))
    h_b = stats.entropy(np.bincount(errors_b_binned) / len(errors_b_binned))
    
    return {
        'mi_a_actual': float(mi_a_actual),
        'mi_b_actual': float(mi_b_actual),
        'mi_a_b': float(mi_a_b),  # Redundant information
        'h_a': float(h_a),
        'h_b': float(h_b),
        'unique_a': float(max(0, h_a - mi_a_b)),  # Unique to A
        'unique_b': float(max(0, h_b - mi_a_b)),  # Unique to B
        'redundant': float(mi_a_b),
        'n_samples': len(merged)
    }


def analyze_interval_drivers(
    forecasts_df: pd.DataFrame,
    demand_df: pd.DataFrame
) -> pd.DataFrame:
    """Analyze which SKU features drive forecast interval width.
    
    Returns DataFrame with correlation between features and interval width.
    """
    # Compute SKU features
    sku_features = []
    for (store, product), group in demand_df.groupby(['store', 'product']):
        y = group['demand'].values
        if len(y) < 2:
            continue
        
        sku_features.append({
            'store': store,
            'product': product,
            'mean_demand': np.mean(y),
            'std_demand': np.std(y),
            'cv': np.std(y) / max(np.mean(y), 0.01),
            'zero_rate': np.mean(y == 0),
            'max_demand': np.max(y)
        })
    
    features_df = pd.DataFrame(sku_features)
    
    # Aggregate interval width per SKU
    width_by_sku = forecasts_df.groupby(['store', 'product'])['interval_width'].mean().reset_index()
    
    # Merge
    merged = width_by_sku.merge(features_df, on=['store', 'product'])
    
    if len(merged) < 10:
        return pd.DataFrame()
    
    # Compute correlations
    feature_cols = ['mean_demand', 'std_demand', 'cv', 'zero_rate', 'max_demand']
    correlations = []
    
    for col in feature_cols:
        corr, pvalue = stats.pearsonr(merged[col], merged['interval_width'])
        correlations.append({
            'feature': col,
            'correlation': float(corr),
            'p_value': float(pvalue),
            'significant': pvalue < 0.05
        })
    
    return pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)


def main():
    parser = argparse.ArgumentParser(description="SURD Information-Theoretic Analysis")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/surd_analysis"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-skus", type=int, default=200, help="Limit SKUs for faster analysis")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("SURD Information-Theoretic Analysis")
    print("="*60)
    
    # Load demand data
    print("\nLoading demand data...")
    demand_df = load_demand(args.demand_path)
    print(f"  {len(demand_df)} observations")
    
    # Get models
    if args.models:
        models = args.models
    else:
        models = [d.name for d in args.checkpoints_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\nAnalyzing {len(models)} models: {', '.join(models[:5])}...")
    
    # Load forecasts for each model
    model_forecasts = {}
    for model in models:
        print(f"  Loading {model}...")
        df = load_forecasts_for_model(model, args.checkpoints_dir, demand_df, args.max_skus)
        if len(df) > 0:
            model_forecasts[model] = df
            print(f"    {len(df)} forecast records")
    
    # ===== Analysis 1: Error Decomposition =====
    print("\n" + "="*60)
    print("1. FORECAST ERROR DECOMPOSITION")
    print("="*60)
    
    decomposition_results = []
    for model, forecasts in model_forecasts.items():
        decomp = decompose_forecast_error(forecasts)
        decomp['model'] = model
        decomposition_results.append(decomp)
        
        if 'error' not in decomp:
            print(f"\n{model}:")
            print(f"  RMSE: {decomp['rmse']:.2f}")
            print(f"  Bias: {decomp['bias']:.2f}")
            print(f"  Std:  {decomp['std']:.2f}")
            print(f"  Explained by bias: {decomp['explained_by_bias']*100:.1f}%")
    
    decomp_df = pd.DataFrame(decomposition_results)
    decomp_df.to_csv(args.output_dir / 'error_decomposition.csv', index=False)
    
    # ===== Analysis 2: Model Information Overlap =====
    print("\n" + "="*60)
    print("2. MODEL INFORMATION OVERLAP")
    print("="*60)
    
    overlap_results = []
    model_list = list(model_forecasts.keys())
    
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i+1:]:
            overlap = compute_model_information_overlap(
                model_forecasts[model_a],
                model_forecasts[model_b]
            )
            overlap['model_a'] = model_a
            overlap['model_b'] = model_b
            overlap_results.append(overlap)
            
            if 'error' not in overlap:
                print(f"\n{model_a} vs {model_b}:")
                print(f"  Redundant: {overlap['redundant']:.3f}")
                print(f"  Unique to {model_a}: {overlap['unique_a']:.3f}")
                print(f"  Unique to {model_b}: {overlap['unique_b']:.3f}")
    
    overlap_df = pd.DataFrame(overlap_results)
    overlap_df.to_csv(args.output_dir / 'model_information_overlap.csv', index=False)
    
    # ===== Analysis 3: Interval Width Drivers =====
    print("\n" + "="*60)
    print("3. INTERVAL WIDTH DRIVERS")
    print("="*60)
    
    driver_results = []
    for model, forecasts in model_forecasts.items():
        drivers = analyze_interval_drivers(forecasts, demand_df)
        if len(drivers) > 0:
            drivers['model'] = model
            driver_results.append(drivers)
            
            print(f"\n{model} - Top interval width drivers:")
            for _, row in drivers.head(3).iterrows():
                sig = "*" if row['significant'] else ""
                print(f"  {row['feature']}: r={row['correlation']:.3f}{sig}")
    
    if driver_results:
        all_drivers = pd.concat(driver_results, ignore_index=True)
        all_drivers.to_csv(args.output_dir / 'interval_width_drivers.csv', index=False)
    
    # ===== Summary Report =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    report = []
    report.append("# SURD Information-Theoretic Analysis Report\n")
    report.append("## 1. Error Decomposition\n")
    report.append("Shows how much of each model's error is systematic (bias) vs random.\n")
    report.append(decomp_df.to_markdown(index=False))
    report.append("\n\n## 2. Model Information Overlap\n")
    report.append("Shows redundant vs unique information captured by different models.\n")
    report.append("High redundancy suggests models capture similar patterns.\n")
    report.append(overlap_df.to_markdown(index=False) if len(overlap_df) > 0 else "Not enough data")
    report.append("\n\n## 3. Interval Width Drivers\n")
    report.append("Shows which SKU features are correlated with forecast uncertainty.\n")
    if driver_results:
        report.append(all_drivers.to_markdown(index=False))
    
    report_path = args.output_dir / 'surd_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

