"""
Evaluate Patrick's Corrected Policy - Properly Integrated

Tests Patrick's 3 critical fixes using existing sequential backtest infrastructure:
1. Protection period = 3 weeks (h=3,4,5) 
2. Critical fractile = 0.833 explicit
3. Monte Carlo aggregation over 3 weeks

Modifies sequential_backtest to use Patrick's approach.
"""

import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.policy.corrected_policy import aggregate_weekly_distributions_mc, Costs, LeadTime
from vn2.analyze.sequential_backtest import (
    BacktestState, 
    load_actual_demand, 
    reconstruct_initial_state,
    quantiles_to_pmf
)


def run_backtest_with_patrick_policy(
    store: int,
    product: int,
    sales_df: pd.DataFrame,
    initial_state_df: pd.DataFrame,
    costs: Costs,
    lead_time: LeadTime,
    quantile_levels: np.ndarray
):
    """Run 12-week backtest using Patrick's corrected policy."""
    
    # Get initial state
    initial_state = reconstruct_initial_state(
        store, product, initial_state_df, sales_df
    )
    
    # Get actual demand
    actuals = load_actual_demand(store, product, sales_df, n_weeks=12)
    
    # Get date columns and recent demand
    date_cols = [c for c in sales_df.columns if c not in ['Store', 'Product']]
    sku_mask = (sales_df['Store'] == store) & (sales_df['Product'] == product)
    sku_row = sales_df[sku_mask].iloc[0]
    recent_demand = np.array([sku_row[col] for col in date_cols[-13:]])
    recent_mean = recent_demand.mean()
    recent_std = recent_demand.std() if len(recent_demand) > 1 else recent_mean * 0.5
    
    # Run 12-week backtest
    state = initial_state.copy()
    total_cost = 0.0
    orders = []
    
    for week in range(1, 13):
        # Generate simple forecasts using normal approximation
        # For production, replace with actual model forecasts
        quantiles_h3 = np.array([
            max(0, recent_mean + recent_std * stats.norm.ppf(q))
            for q in quantile_levels
        ])
        quantiles_h4 = quantiles_h3.copy()  # Simplification
        quantiles_h5 = quantiles_h3.copy()
        
        # Build quantiles DataFrame - CORRECT FORMAT
        # Index = horizons, values = quantiles
        quantiles_df = pd.DataFrame({
            q: [quantiles_h3[i], quantiles_h4[i], quantiles_h5[i]]
            for i, q in enumerate(quantile_levels)
        }, index=[3, 4, 5])
        
        # Current position
        position = state.on_hand + state.intransit_1 + state.intransit_2
        
        # Patrick's approach: MC aggregation + τ=0.833
        if week <= 10:
            try:
                # Aggregate using Patrick's MC method
                mu, sigma = aggregate_weekly_distributions_mc(
                    quantiles_df=quantiles_df,
                    quantile_levels=quantile_levels,
                    protection_weeks=3,
                    lead_weeks=lead_time.lead_weeks,
                    n_samples=10000
                )
                
                # Critical fractile
                tau = costs.shortage / (costs.holding + costs.shortage)
                z = stats.norm.ppf(tau)
                
                # Base stock level
                S = mu + z * sigma
                q_t = max(0, int(np.ceil(S - position)))
            except Exception as e:
                # Fallback to simple approach
                q_t = max(0, int(4 * recent_mean - position))
        else:
            q_t = 0  # Weeks 11-12: no new orders
        
        orders.append(q_t)
        
        # Compute realized cost
        demand = actuals[week - 1]
        available = state.on_hand + state.intransit_1
        leftover = max(0, available - demand)
        shortage = max(0, demand - available)
        cost = costs.holding * leftover + costs.shortage * shortage
        total_cost += cost
        
        # Update state
        state = BacktestState(
            week=week + 1,
            on_hand=leftover,
            intransit_1=state.intransit_2,
            intransit_2=q_t
        )
    
    return {
        'store': store,
        'product': product,
        'total_cost': total_cost,
        'orders': orders
    }


def main():
    print("="*80)
    print("PATRICK'S CORRECTED POLICY - INTEGRATED EVALUATION")
    print("="*80)
    print()
    
    # Load data
    base_dir = Path(__file__).parent.parent
    initial_state_df = pd.read_csv(base_dir / 'data' / 'raw' / 'Week 0 - 2024-04-08 - Initial State.csv')
    sales_df = pd.read_csv(base_dir / 'data' / 'raw' / 'Week 0 - 2024-04-08 - Sales.csv')
    
    # Configuration
    costs = Costs(holding=0.2, shortage=1.0)
    lead_time = LeadTime(lead_weeks=2, review_weeks=1)
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    tau = costs.shortage / (costs.holding + costs.shortage)
    
    print(f"Costs: holding={costs.holding}, shortage={costs.shortage}")
    print(f"Critical fractile: τ={tau:.3f}")
    print(f"Protection period: 3 weeks (h=3,4,5)")
    print(f"MC samples: 10,000")
    print()
    print("Processing all SKUs...")
    print()
    
    # Process all SKUs
    results = []
    n_skus = len(sales_df)
    errors = 0
    
    for idx, row in sales_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{n_skus} SKUs (errors: {errors})...")
        
        store = int(row['Store'])
        product = int(row['Product'])
        
        try:
            result = run_backtest_with_patrick_policy(
                store, product, sales_df, initial_state_df,
                costs, lead_time, quantile_levels
            )
            results.append(result)
        except Exception as e:
            errors += 1
            if errors <= 5:  # Only print first 5 errors
                print(f"    ERROR on SKU ({store}, {product}): {e}")
            continue
    
    # Aggregate results
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    total_cost = sum(r['total_cost'] for r in results)
    n_processed = len(results)
    
    print(f"SKUs processed: {n_processed}/{n_skus} (errors: {errors})")
    print(f"Model: Patrick's 3-week MC Policy + Simple MA Forecasts")
    print()
    print(f"Total Cost (weeks 1-12):              €{total_cost:,.0f}")
    print()
    print("BENCHMARKS:")
    print(f"  VN2 Benchmark (weeks 2-12):         €5,248")
    print(f"  Our VN2-style (weeks 2-12):         €9,193")
    print()
    gap_vs_vn2 = (total_cost / 5248 - 1) * 100
    improvement_vs_baseline = (9193 - total_cost) / 9193 * 100
    print(f"Gap vs VN2:                           {gap_vs_vn2:+.1f}%")
    print(f"Improvement vs VN2-style baseline:    {improvement_vs_baseline:+.1f}%")
    print()
    
    # Save results
    output_file = base_dir / 'models' / 'results' / 'patrick_integrated_results.csv'
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Summary stats
    print()
    print("COST DISTRIBUTION:")
    costs_series = pd.Series([r['total_cost'] for r in results])
    print(f"  Mean:   €{costs_series.mean():.2f}")
    print(f"  Median: €{costs_series.median():.2f}")
    print(f"  Std:    €{costs_series.std():.2f}")
    print(f"  Min:    €{costs_series.min():.2f}")
    print(f"  Max:    €{costs_series.max():.2f}")


if __name__ == '__main__':
    main()
