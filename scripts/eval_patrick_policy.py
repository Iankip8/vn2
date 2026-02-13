"""
Evaluate Patrick's Corrected Policy

Tests Patrick's 3 critical fixes:
1. Protection period = 3 weeks (h=3,4,5) 
2. Critical fractile = 0.833 explicit
3. Monte Carlo aggregation over 3 weeks

Uses simple 13-week MA forecasts + Patrick's optimal ordering policy.
"""

import sys
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.policy.corrected_policy import aggregate_weekly_distributions_mc, Costs, LeadTime
from vn2.analyze.sequential_backtest import BacktestState, load_actual_demand, reconstruct_initial_state


def simple_quantile_forecast(recent_demand, quantile_levels):
    """Generate quantiles from recent demand using normal approximation."""
    mean = recent_demand.mean()
    std = recent_demand.std() if len(recent_demand) > 1 else mean * 0.5
    
    quantiles = np.array([
        max(0, mean + std * scipy.stats.norm.ppf(q))
        for q in quantile_levels
    ])
    return quantiles


def main():
    print("="*80)
    print("PATRICK'S CORRECTED POLICY EVALUATION")
    print("="*80)
    print()
    
    # Load data
    base_dir = Path(__file__).parent.parent
    initial_state_df = pd.read_csv(base_dir / 'data' / 'raw' / 'Week 0 - 2024-04-08 - Initial State.csv')
    sales_df = pd.read_csv(base_dir / 'data' / 'raw' / 'Week 0 - 2024-04-08 - Sales.csv')
    
    # Configuration
    costs = Costs(holding=0.2, shortage=1.0)
    lead_time = LeadTime(lead_weeks=2, review_weeks=1)
    protection_weeks = 3
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    tau = costs.shortage / (costs.holding + costs.shortage)
    
    print(f"Costs: holding={costs.holding}, shortage={costs.shortage}")
    print(f"Critical fractile: τ={tau:.3f}")
    print(f"Protection period: {protection_weeks} weeks (h=3,4,5)")
    print(f"MC samples: 10,000")
    print()
    
    # Get date columns
    date_cols = [c for c in sales_df.columns if c not in ['Store', 'Product']]
    
    # Process all SKUs
    results = []
    n_skus = len(sales_df)
    
    for idx, row in sales_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Processing SKU {idx + 1}/{n_skus}...")
        
        store = int(row['Store'])
        product = int(row['Product'])
        
        try:
            # Get initial state
            initial_state = reconstruct_initial_state(
                store, product, initial_state_df, sales_df
            )
            
            # Get actual demand for 12 weeks
            actuals = load_actual_demand(store, product, sales_df, n_weeks=12)
            
            # Get recent demand (last 13 weeks before competition)
            recent_demand = np.array([row[col] for col in date_cols[-13:]])
            
            # Run 12-week backtest
            state = initial_state.copy()
            total_cost = 0.0
            orders = []
            
            for week in range(1, 13):
                # Generate forecasts for h=3,4,5 (simple: all same)
                q_h3 = simple_quantile_forecast(recent_demand, quantile_levels)
                q_h4 = simple_quantile_forecast(recent_demand, quantile_levels)
                q_h5 = simple_quantile_forecast(recent_demand, quantile_levels)
                
                # Build quantiles DataFrame
                quantiles_df = pd.DataFrame({
                    3: q_h3,
                    4: q_h4,
                    5: q_h5
                })
                
                # Current position
                position = state.on_hand + state.intransit_1 + state.intransit_2
                
                # Patrick's approach: MC aggregation + τ=0.833
                if week <= 10:
                    mu, sigma = aggregate_weekly_distributions_mc(
                        quantiles_df=quantiles_df,
                        quantile_levels=quantile_levels,
                        protection_weeks=protection_weeks,
                        lead_weeks=lead_time.lead_weeks,
                        n_samples=10000
                    )
                    
                    z = scipy.stats.norm.ppf(tau)
                    S = mu + z * sigma
                    q_t = max(0, int(np.ceil(S - position)))
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
            
            results.append({
                'store': store,
                'product': product,
                'total_cost': total_cost,
                'orders': orders
            })
            
        except Exception as e:
            print(f"ERROR on SKU ({store}, {product}): {e}")
            continue
    
    # Aggregate results
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    total_cost = sum(r['total_cost'] for r in results)
    n_processed = len(results)
    
    print(f"SKUs processed: {n_processed}/{n_skus}")
    print(f"Model: Simple 13-week MA + Patrick's Policy")
    print()
    print(f"Total Cost (weeks 1-12):              €{total_cost:,.0f}")
    print()
    print(f"VN2 Benchmark (weeks 2-12):           €5,248")
    print(f"Our VN2-style (weeks 2-12):           €9,193")
    print()
    gap_vs_vn2 = (total_cost / 5248 - 1) * 100
    improvement_vs_baseline = (9193 - total_cost) / 9193 * 100
    print(f"Gap vs VN2:                           {gap_vs_vn2:+.1f}%")
    print(f"Improvement vs VN2-style baseline:    {improvement_vs_baseline:+.1f}%")
    print()
    
    # Save results
    output_file = base_dir / 'models' / 'results' / 'patrick_policy_results.csv'
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
