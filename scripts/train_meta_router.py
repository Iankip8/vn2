#!/usr/bin/env python3
"""
Train meta-router for dynamic model selection.

The meta-router learns which forecasting model works best for each SKU based
on historical cost outcomes. It outputs a selector override file that can be
used by the backtest harness.

Usage:
    python scripts/train_meta_router.py \
        --cost-history reports/model_costs_history.csv \
        --demand-path data/processed/demand_long.parquet \
        --output-dir models/results \
        --models zinb slurp_bootstrap lightgbm_quantile ets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from vn2.forecast.meta_router import MetaRouter, RouterConfig, compute_sku_features


def load_cost_history(path: Path) -> pd.DataFrame:
    """Load historical cost data per model per SKU."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    required = {'store', 'product', 'week', 'model', 'realized_cost'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cost history missing columns: {missing}")
    
    return df


def load_demand(path: Path) -> pd.DataFrame:
    """Load demand data."""
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'demand' not in df.columns and 'sales' in df.columns:
        df['demand'] = df['sales']
    
    # Convert week to integer if datetime
    if df['week'].dtype == 'datetime64[ns]':
        weeks_sorted = sorted(df['week'].unique())
        week_map = {w: i for i, w in enumerate(weeks_sorted)}
        df['week'] = df['week'].map(week_map)
    
    return df[['store', 'product', 'week', 'demand']]


def generate_cost_history_from_checkpoints(
    checkpoints_dir: Path,
    actuals_df: pd.DataFrame,
    models: List[str],
    costs: dict = {'holding': 0.2, 'shortage': 1.0}
) -> pd.DataFrame:
    """Generate cost history by evaluating all models on actual demand.
    
    This is used when we don't have pre-computed cost history.
    """
    import pickle
    from vn2.analyze.sequential_backtest import quantiles_to_pmf
    from vn2.analyze.sequential_planner import Costs, choose_order_L3
    
    QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                                0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    
    cost_obj = Costs(**costs)
    records = []
    
    skus = actuals_df.groupby(['store', 'product']).size().index.tolist()
    
    for model in models:
        model_dir = checkpoints_dir / model
        if not model_dir.exists():
            print(f"  Skipping {model} - directory not found")
            continue
        
        print(f"  Evaluating {model}...")
        
        for store, product in skus:
            sku_actuals = actuals_df[
                (actuals_df['store'] == store) & 
                (actuals_df['product'] == product)
            ].sort_values('week')
            
            sku_dir = model_dir / f'{store}_{product}'
            if not sku_dir.exists():
                continue
            
            for fold_idx in range(min(6, len(sku_actuals))):
                fold_path = sku_dir / f'fold_{fold_idx}.pkl'
                if not fold_path.exists():
                    continue
                
                try:
                    with open(fold_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    qdf = data.get('quantiles')
                    if qdf is None or qdf.empty:
                        continue
                    
                    # Get actual demand for this week
                    week = fold_idx + 1
                    actual_row = sku_actuals[sku_actuals['week'] == week]
                    if actual_row.empty:
                        continue
                    
                    actual = actual_row['demand'].iloc[0]
                    
                    # Get forecast at critical fractile (Q83.33)
                    if 1 in qdf.index:
                        q83 = np.interp(0.8333, qdf.columns, qdf.loc[1].values)
                        
                        # Simple cost calculation: compare to actual
                        if q83 < actual:
                            cost = (actual - q83) * costs['shortage']
                        else:
                            cost = (q83 - actual) * costs['holding']
                        
                        records.append({
                            'store': store,
                            'product': product,
                            'week': week,
                            'model': model,
                            'realized_cost': cost,
                            'forecast_q83': q83,
                            'actual': actual
                        })
                
                except Exception:
                    continue
    
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Train meta-router for model selection")
    parser.add_argument("--cost-history", type=Path, default=None,
                        help="Path to cost history CSV (optional - will generate if missing)")
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/results"))
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to include (default: auto-detect from checkpoints)")
    parser.add_argument("--classifier", choices=['random_forest', 'gradient_boosting'],
                        default='random_forest')
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    
    # Determine models to include
    if args.models:
        models = args.models
    else:
        models = [d.name for d in args.checkpoints_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Meta-Router Training")
    print(f"  Models: {', '.join(models)}")
    print(f"  Classifier: {args.classifier}")
    
    # Load demand data for feature computation
    print(f"\nLoading demand data...")
    demand_df = load_demand(args.demand_path)
    print(f"  {len(demand_df)} demand observations")
    
    # Compute SKU features
    print("\nComputing SKU features...")
    features = compute_sku_features(demand_df)
    print(f"  {len(features)} SKUs with features")
    
    # Get or generate cost history
    if args.cost_history and args.cost_history.exists():
        print(f"\nLoading cost history from {args.cost_history}")
        cost_history = load_cost_history(args.cost_history)
    else:
        print("\nGenerating cost history from checkpoints...")
        cost_history = generate_cost_history_from_checkpoints(
            checkpoints_dir=args.checkpoints_dir,
            actuals_df=demand_df.rename(columns={'demand': 'demand'}),
            models=models
        )
        
        # Save generated cost history
        cost_history_path = args.output_dir / 'generated_cost_history.csv'
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cost_history.to_csv(cost_history_path, index=False)
        print(f"  Saved cost history to {cost_history_path}")
    
    print(f"  {len(cost_history)} cost records")
    
    # Add week to features (for merging)
    # We'll use the features computed from full history, expanded per week
    weeks = cost_history['week'].unique()
    features_expanded = []
    for week in weeks:
        week_features = features.copy()
        week_features['week'] = week
        features_expanded.append(week_features)
    
    features_df = pd.concat(features_expanded, ignore_index=True)
    
    # Create and train router
    print("\nTraining meta-router...")
    config = RouterConfig(
        model_names=models,
        classifier_type=args.classifier,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    router = MetaRouter(config)
    
    # Cross-validate first
    print(f"\nCross-validation ({args.cv_folds} folds)...")
    cv_results = router.cross_validate(cost_history, features_df, cv=args.cv_folds)
    print(f"  Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    
    # Fit final router
    router.fit(cost_history, features_df)
    
    # Feature importance
    print("\nFeature importance:")
    importance_df = router.get_feature_importance_summary()
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Generate selector overrides for current data
    print("\nGenerating selector overrides...")
    current_features = features.copy()  # Use most recent features
    selections = router.predict(current_features)
    
    # Format for harness
    selector_overrides = selections[['store', 'product', 'selected_model', 'confidence']].copy()
    selector_overrides = selector_overrides.rename(columns={'selected_model': 'best_challenger'})
    selector_overrides['recommend_switch'] = True
    
    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    router_path = args.output_dir / 'meta_router.pkl'
    router.save(router_path)
    print(f"\nRouter saved to: {router_path}")
    
    overrides_path = args.output_dir / 'selector_overrides_meta_router.csv'
    selector_overrides.to_csv(overrides_path, index=False)
    print(f"Selector overrides saved to: {overrides_path}")
    
    importance_path = args.output_dir / 'meta_router_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("META-ROUTER SUMMARY")
    print("="*60)
    print(f"Models: {len(models)}")
    print(f"SKUs: {len(selections)}")
    print(f"CV Accuracy: {cv_results['mean_accuracy']:.1%}")
    
    print("\nModel selection distribution:")
    for model, count in selections['selected_model'].value_counts().items():
        pct = count / len(selections) * 100
        print(f"  {model}: {count} ({pct:.1f}%)")
    
    low_confidence = (selections['confidence'] < config.confidence_threshold).sum()
    print(f"\nLow-confidence predictions: {low_confidence} ({low_confidence/len(selections)*100:.1f}%)")


if __name__ == "__main__":
    main()

