"""
Sequential 12-week evaluation using PATRICK'S CORRECTED POLICY.

This version implements Patrick's 3 critical fixes:
1. Protection period = 3 weeks (h=3,4,5)
2. Explicit critical fractile = 0.833
3. Monte Carlo aggregation of 3-week demand distribution

Expected to significantly improve over €9,193 from VN2-style approach.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List
import scipy.stats
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.analyze.sequential_backtest import (
    BacktestState, BacktestResult, quantiles_to_pmf,
    reconstruct_initial_state, load_actual_demand
)
from vn2.policy.corrected_policy import compute_order_quantity_corrected, LeadTime, Costs


@dataclass
class WeekResult:
    """Result for a single week."""
    week: int
    order_placed: int
    demand_actual: int
    expected_cost: float
    realized_cost: float
    state_before: BacktestState
    state_after: BacktestState


def run_12week_backtest_patrick(
    store: int,
    product: int,
    model_name: str,
    quantiles_h3: List,  # 12 weeks of (13,) quantiles for h=3
    quantiles_h4: List,  # 12 weeks of (13,) quantiles for h=4
    quantiles_h5: List,  # 12 weeks of (13,) quantiles for h=5
    quantile_levels: np.ndarray,  # [0.01, 0.05, ..., 0.99]
    actuals: List[int],  # 12 weeks of actual demand
    initial_state: BacktestState,
    costs: Costs,
    lead_time: LeadTime,
    protection_weeks: int = 3
) -> BacktestResult:
    """
    Run 12-week backtest using Patrick's corrected policy.
    
    Key differences from VN2-style approach:
    - Uses compute_order_quantity_corrected() from corrected_policy.py
    - Aggregates h=[3,4,5] via Monte Carlo (10,000 samples)
    - Uses explicit critical fractile τ=0.833
    - Protection period = lead_weeks + review_weeks = 3
    """
    assert len(quantiles_h3) == 12
    assert len(quantiles_h4) == 12
    assert len(quantiles_h5) == 12
    assert len(actuals) == 12
    
    state = initial_state.copy()
    weeks = []
    n_missing = 0
    
    for t in range(1, 13):  # Weeks 1-12
        week_idx = t - 1
        q_h3 = quantiles_h3[week_idx]
        q_h4 = quantiles_h4[week_idx]
        q_h5 = quantiles_h5[week_idx]
        demand_actual = actuals[week_idx]
        
        state_before = state.copy()
        
        # Check for missing forecasts
        if q_h3 is None or q_h4 is None or q_h5 is None:
            q_t = 0
            expected_cost = 0.0
            n_missing += 1
        elif t >= 11:
            # Weeks 11-12: No new orders (would arrive after horizon)
            q_t = 0
            expected_cost = 0.0
        else:
            # Weeks 1-10: Use Patrick's corrected policy
            # Build quantiles DataFrame (3 horizons x 13 quantiles)
            quantiles_df = pd.DataFrame({
                3: q_h3,
                4: q_h4,
                5: q_h5
            })
            
            # Current inventory position
            current_position = state.on_hand + state.intransit_1 + state.intransit_2
            
            # Use Patrick's aggregate + critical fractile approach
            from vn2.policy.corrected_policy import aggregate_weekly_distributions_mc
            import scipy.stats as stats
            
            # Aggregate 3-week demand distribution via MC
            mu, sigma = aggregate_weekly_distributions_mc(
                quantiles_df=quantiles_df,
                quantile_levels=quantile_levels,
                protection_weeks=protection_weeks,
                lead_weeks=lead_time.lead_weeks,
                n_samples=10000
            )
            
            # Critical fractile τ=0.833
            tau = costs.shortage / (costs.holding + costs.shortage)
            z = stats.norm.ppf(tau)
            
            # Base stock level S = mu + z*sigma
            S = mu + z * sigma
            
            # Order quantity
            q_t = max(0, int(np.ceil(S - current_position)))
            
            # Estimate expected cost (simplified)
            # Use base stock level S and current position
            expected_holding = max(0, current_position + q_t - mu)
            expected_shortage = max(0, mu - (current_position + q_t))
            expected_cost = costs.holding * expected_holding + costs.shortage * expected_shortage
        
        # Compute realized cost for current week
        S_t = state.on_hand + state.intransit_1
        leftover_t = max(0, S_t - demand_actual)
        shortage_t = max(0, demand_actual - S_t)
        realized_cost = costs.holding * leftover_t + costs.shortage * shortage_t
        
        # Update state for next week
        state_after = BacktestState(
            week=t + 1,
            on_hand=leftover_t,
            intransit_1=state.intransit_2,
            intransit_2=q_t
        )
        
        weeks.append(WeekResult(
            week=t,
            order_placed=q_t,
            demand_actual=demand_actual,
            expected_cost=expected_cost,
            realized_cost=realized_cost,
            state_before=state_before,
            state_after=state_after
        ))
        
        state = state_after
    
    # Aggregate costs
    total_expected = sum(w.expected_cost for w in weeks)
    total_realized = sum(w.realized_cost for w in weeks)
    total_expected_excl_w1 = sum(w.expected_cost for w in weeks[1:])
    total_realized_excl_w1 = sum(w.realized_cost for w in weeks[1:])
    
    return BacktestResult(
        store=store,
        product=product,
        model_name=model_name,
        weeks=[],  # Not storing individual weeks for now
        total_expected_cost=total_expected,
        total_realized_cost=total_realized,
        total_expected_cost_excl_w1=total_expected_excl_w1,
        total_realized_cost_excl_w1=total_realized_excl_w1,
        n_weeks=12,
        n_missing_forecasts=n_missing,
        diagnostics={'method': 'patrick_corrected_policy'}
    )


def main():
    """Run sequential evaluation using Patrick's corrected policy."""
    print("="*80)
    print("Sequential Evaluation: PATRICK'S CORRECTED POLICY")
    print("="*80)
    print()
    print("Implementing Patrick's 3 critical fixes:")
    print("  1. Protection period = 3 weeks (h=3,4,5)")
    print("  2. Critical fractile = 0.833 explicit")
    print("  3. Monte Carlo aggregation (10,000 samples)")
    print()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    # Load data
    print("Loading data...")
    initial_state_df = pd.read_csv(data_dir / 'raw' / 'Week 0 - 2024-04-08 - Initial State.csv')
    sales_df = pd.read_csv(data_dir / 'raw' / 'Week 0 - 2024-04-08 - Sales.csv')
    
    # Load SLURP forecasts (our best models)
    print("Loading forecasts from sequential_results...")
    forecasts_file = base_dir / 'models' / 'results' / 'sequential_results_seq12_h12.parquet'
    if not forecasts_file.exists():
        print(f"ERROR: Sequential results not found at {forecasts_file}")
        return
    
    # We'll extract quantiles from the model evaluation results
    # Use lightgbm_quantile model forecasts
    print("Note: Using forecasts from existing sequential evaluation")
    print("Will need to regenerate with SLURP forecasts when available")
    
    # Generate simple forecasts using recent 13-week average
    # This is similar to VN2 benchmark but we'll use Patrick's policy
    print("Using simple 13-week moving average forecasts...")
    print()
    
    # Configuration
    costs = Costs(holding=0.2, shortage=1.0)
    lead_time = LeadTime(lead_weeks=2, review_weeks=1)
    protection_weeks = 3  # lead + review
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    print(f"Costs: holding={costs.holding}, shortage={costs.shortage}")
    print(f"Critical fractile: {costs.shortage/(costs.holding + costs.shortage):.3f}")
    print(f"Lead time: L={lead_time.lead_weeks}, R={lead_time.review_weeks}, T={protection_weeks}")
    print()
    
    # Get unique SKUs
    unique_skus = initial_state_df[['Store', 'Product']].drop_duplicates()
    n_skus = len(unique_skus)
    print(f"Processing {n_skus} SKUs...")
    print()
    
    # Run backtests
    results = []
    for idx, (_, row) in enumerate(unique_skus.iterrows()):
        if (idx + 1) % 50 == 0:
            print(f"Processing SKU {idx + 1}/{n_skus}...")
        
        store = int(row['Store'])
        product = int(row['Product'])
        
        try:
            # Get initial state
            initial_state = reconstruct_initial_state(
                store, product, initial_state_df, sales_df
            )
            
            # Get actual demand
            actuals = load_actual_demand(store, product, sales_df, n_weeks=12)
            
            # Get SKU recent demand (last 13 weeks before competition)
            date_cols = [c for c in sales_df.columns if c not in ['Store', 'Product']]
            sku_mask = (sales_df['Store'] == store) & (sales_df['Product'] == product)
            sku_row = sales_df[sku_mask].iloc[0]
            recent_demand = np.array([sku_row[col] for col in date_cols[-13:]])
            recent_mean = recent_demand.mean()
            recent_std = recent_demand.std() if len(recent_demand) > 1 else recent_mean * 0.5
            
            # Generate simple quantile forecasts (normal approximation)
            # For all horizons h=3,4,5, use same forecast (simplification)
            quantiles_h3 = []
            quantiles_h4 = []
            quantiles_h5 = []
            
            for week in range(1, 13):
                # Generate quantiles from normal distribution
                q_forecast = np.array([
                    max(0, recent_mean + recent_std * np.sqrt(1) * scipy.stats.norm.ppf(q))
                    for q in quantile_levels
                ])
                imple_ma',
                quantiles_h3=quantiles_h3,
                quantiles_h4=quantiles_h4,
                quantiles_h5=quantiles_h5,
                quantile_levels=quantile_levels,
                actuals=actuals,
                initial_state=initial_state,
                costs=costs,
                lead_time=lead_time,
                protection_weeks=protection_weeks
                model_name='slurp',
                quantiles_h3=quantiles_h3,
                quantiles_h4=quantiles_h4,
                quantiles_h5=quantiles_h5,
                quantile_levels=quantile_levels,
                actuals=actuals,
                initial_state=initial_state,
                costs=costs,
                lead_time=lead_time
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"\nERROR on SKU ({store}, {product}): {e}")
            continue
    
    # Aggregate results
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    if len(results) == 0:
        print("ERROR: No results generated")
        return
    
    # Compute totals
    total_realized = sum(r.total_realized_cost for r in results)
    total_realized_excl_w1 = sum(r.total_realized_cost_excl_w1 for r in results)
    total_expected = sum(r.total_expected_cost for r in results)
    n_skus_processed = len(results)
    
    print(f"SKUs processed: {n_skus_processed}/{n_skus}")
    print(f"Model: VN2 Forecasts + Patrick's Corrected Policy")
    print()
    print(f"Total Realized Cost (weeks 1-12):    €{total_realized:,.0f}")
    print(f"Total Realized Cost (weeks 2-12):    €{total_realized_excl_w1:,.0f}")
    print(f"Total Expected Cost (weeks 2-12):    €{total_expected:,.0f}")
    print()
    print(f"VN2 Benchmark (weeks 2-12):          €5,248")
    print(f"VN2-style Simple (weeks 2-12):       €9,193")
    print()
    gap = (total_realized_excl_w1 / 5248 - 1) * 100
    improvement_vs_vn2style = (9193 - total_realized_excl_w1) / 9193 * 100
    print(f"Gap vs VN2 Benchmark:                 {gap:+.1f}%")
    print(f"Change vs VN2-style Simple:           {improvement_vs_vn2style:+.1f}%")
    print()
    print("Patrick's 3-week policy + VN2 forecasts result shown above")
    
    # Save results
    output_file = base_dir / 'models' / 'results' / 'sequential_eval_patrick.csv'
    output_df = pd.DataFrame([
        {
            'store': r.store,
            'product': r.product,
            'model': r.model_name,
            'total_realized_cost': r.total_realized_cost,
            'total_realized_cost_excl_w1': r.total_realized_cost_excl_w1,
            'n_missing_forecasts': r.n_missing_forecasts
        }
        for r in results
    ])
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
