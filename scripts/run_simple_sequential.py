"""
Simple sequential evaluation using corrected_policy.py 
Matches VN2.py structure exactly.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from vn2.policy.corrected_policy import compute_order_quantity_corrected
from vn2.sim import Simulator, Costs, LeadTime


def load_checkpoint_quantiles(checkpoint_path: Path, fold_idx: int):
    """Load quantiles from checkpoint."""
    fold_file = checkpoint_path / f"fold_{fold_idx}.pkl"
    if not fold_file.exists():
        return None
    
    with open(fold_file, 'rb') as f:
        data = pickle.load(f)
    
    return data.get('quantiles')


def sequential_eval_simple(
    model_name: str,
    store: int,
    product: int,
    checkpoints_dir: Path,
    demand_df: pd.DataFrame,
    initial_state_df: pd.DataFrame,
    costs: Costs,
    lt: LeadTime,
    n_folds: int = 12
):
    """
    Simple sequential evaluation using corrected_policy.
    
    For each of 12 folds:
    - Place order using corrected_policy (h=3,4,5 aggregation)
    - Simulate 12 weeks with actual demand
    - Track cumulative costs
    """
    checkpoint_path = checkpoints_dir / model_name / f"{store}_{product}"
    
    if not checkpoint_path.exists():
        return None
    
    # Get initial state
    mask = (initial_state_df['Store'] == store) & (initial_state_df['Product'] == product)
    if not mask.any():
        return None
    
    initial_row = initial_state_df[mask].iloc[0]
    
    # Get actual demand for this SKU
    sku_mask = (demand_df['Store'] == store) & (demand_df['Product'] == product)
    if not sku_mask.any():
        return None
    
    sku_demand = demand_df[sku_mask].iloc[0]
    week_cols = [c for c in demand_df.columns if c.startswith('202')]
    actuals = sku_demand[week_cols].values
    
    if len(actuals) < 52 + 12:  # Need history + 12 week horizon
        return None
    
    # Run 12-fold sequential simulation
    total_costs = []
    
    for fold_idx in range(n_folds):
        # Get forecast for this fold
        quantiles_df = load_checkpoint_quantiles(checkpoint_path, fold_idx)
        
        if quantiles_df is None or quantiles_df.empty:
            continue
        
        # Initialize state for this fold
        state = pd.DataFrame({
            'on_hand': [int(initial_row['End Inventory'])],
            'intransit_1': [int(initial_row['In Transit W+1'])],
            'intransit_2': [int(initial_row['In Transit W+2'])]
        }, index=[(store, product)])
        
        # Get actuals for this fold (12 weeks)
        fold_start = 52 + (fold_idx * 12)
        fold_actuals = actuals[fold_start:fold_start+12]
        
        if len(fold_actuals) < 12:
            continue
        
        # Simulate 12 weeks
        sim = Simulator(costs, lt)
        fold_cost = 0.0
        
        for week in range(12):
            # Place order using corrected policy (weeks 1-10 only)
            if week < 10:
                try:
                    order_qty = compute_order_quantity_corrected(
                        quantiles_df=quantiles_df,
                        inventory_position=state['on_hand'].iloc[0] + state['intransit_1'].iloc[0] + state['intransit_2'].iloc[0],
                        costs=costs,
                        lt=lt,
                        protection_weeks=3,
                        seed=42 + week
                    )
                except:
                    order_qty = 0
            else:
                order_qty = 0  # No new orders in weeks 11-12
            
            # Simulate this week with actual demand
            demand_t = pd.Series([float(fold_actuals[week])], index=state.index)
            order_t = pd.Series([float(order_qty)], index=state.index)
            
            state, cost_dict = sim.step(state, demand_t, order_t)
            fold_cost += cost_dict['total']
        
        total_costs.append(fold_cost)
    
    if not total_costs:
        return None
    
    return {
        'model_name': model_name,
        'store': store,
        'product': product,
        'mean_cost': np.mean(total_costs),
        'total_cost': np.sum(total_costs),
        'n_folds': len(total_costs)
    }


def run_simple_sequential(
    checkpoints_dir: Path,
    demand_path: Path,
    state_path: Path,
    output_dir: Path,
    n_jobs: int = 4
):
    """Run simple sequential evaluation for all models and SKUs."""
    
    print("="*80)
    print("🔄 Simple Sequential Evaluation (Corrected Policy)")
    print("="*80)
    
    # Load data
    print("📊 Loading data...")
    demand_df = pd.read_parquet(demand_path)
    initial_state_df = pd.read_parquet(state_path)
    
    # Get models and SKUs
    models = [d.name for d in checkpoints_dir.iterdir() if d.is_dir()]
    print(f"🤖 Models: {len(models)} - {models}")
    
    # Get SKUs from one model
    sku_dirs = list((checkpoints_dir / models[0]).iterdir())
    skus = []
    for sku_dir in sku_dirs:
        try:
            store, product = sku_dir.name.split('_')
            skus.append((int(store), int(product)))
        except:
            continue
    
    print(f"📦 SKUs: {len(skus)}")
    
    # Create tasks
    tasks = []
    for model in models:
        for store, product in skus:
            tasks.append((model, store, product))
    
    print(f"✅ Total tasks: {len(tasks)}")
    
    # Run evaluation
    costs = Costs(holding=0.2, shortage=1.0)
    lt = LeadTime(lead_weeks=2, review_weeks=1)
    
    print("🚀 Running evaluations...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(sequential_eval_simple)(
            model_name, store, product,
            checkpoints_dir, demand_df, initial_state_df,
            costs, lt, n_folds=8  # 8 folds for CV
        )
        for model_name, store, product in tasks
    )
    
    # Filter and aggregate
    results = [r for r in results if r is not None]
    print(f"\n✅ Completed: {len(results)} evaluations")
    
    results_df = pd.DataFrame(results)
    
    # Aggregate by model
    model_summary = results_df.groupby('model_name').agg({
        'total_cost': 'sum',
        'mean_cost': 'mean',
        'store': 'count'
    }).rename(columns={'store': 'n_skus'}).sort_values('total_cost')
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "simple_sequential_results.parquet"
    results_df.to_parquet(results_path)
    print(f"\n💾 Saved: {results_path}")
    
    summary_path = output_dir / "simple_sequential_summary.parquet"
    model_summary.to_parquet(summary_path)
    print(f"💾 Saved: {summary_path}")
    
    # Print leaderboard
    print("\n" + "="*80)
    print("🏆 Simple Sequential Leaderboard (Corrected Policy)")
    print("="*80)
    print(model_summary)
    print("="*80)
    
    return results_df, model_summary


if __name__ == '__main__':
    run_simple_sequential(
        checkpoints_dir=Path("models/checkpoints"),
        demand_path=Path("data/processed/demand_imputed.parquet"),
        state_path=Path("data/processed/initial_state.parquet"),
        output_dir=Path("models/results"),
        n_jobs=4
    )
