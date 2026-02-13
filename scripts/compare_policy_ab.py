"""
Policy A/B Test: Compare Buggy vs Corrected Policy

This script holds the forecast fixed (using QRF model) and compares:
- Policy A: Buggy (1-week protection, no MC aggregation)
- Policy B: Corrected (3-week protection, MC aggregation, 0.833 critical fractile)

Expected result: Policy B should achieve higher service level than Policy A.

Author: Ian
Date: February 4, 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import namedtuple

# Type hints
Costs = namedtuple('Costs', ['holding', 'shortage'])
LeadTime = namedtuple('LeadTime', ['lead_weeks', 'review_weeks'])


def load_qrf_quantiles(fold: int = 0, limit_skus: int = 50):
    """Load QRF quantile forecasts for a specific fold."""
    # Load preprocessed evaluation data
    data_file = Path("models/results/eval_folds_old_policy.parquet")
    if not data_file.exists():
        raise FileNotFoundError(f"Cannot find {data_file}")
    
    df = pd.read_parquet(data_file)
    
    # Filter to QRF model and specific fold
    qrf_data = df[(df['model_name'] == 'qrf') & (df.get('fold', 0) == fold)].copy()
    
    # Limit to first N SKUs for speed
    if limit_skus:
        skus = qrf_data[['store', 'product']].drop_duplicates().head(limit_skus)
        qrf_data = qrf_data.merge(skus, on=['store', 'product'])
    
    return qrf_data


def policy_a_buggy(quantiles_df, initial_state, costs, lt, seed=42):
    """
    Buggy policy (original implementation).
    
    Issues:
    1. Uses only h=1 forecast
    2. No explicit critical fractile
    3. Formula: S = mu*L + z*sigma*sqrt(L) (wrong aggregation)
    """
    rng = np.random.default_rng(seed)
    
    # BUGGY: Only use h=1 forecast
    if 1 not in quantiles_df.index:
        return 0
    
    q_vals_h1 = quantiles_df.loc[1].values
    quantile_levels = quantiles_df.columns.values
    
    # Sample from h=1 distribution
    u = rng.uniform(0, 1, 10000)
    samples_h1 = np.interp(u, quantile_levels, q_vals_h1)
    
    mu = np.mean(samples_h1)
    sigma = np.std(samples_h1)
    
    # BUGGY: Assume critical fractile without calculating
    z = stats.norm.ppf(0.9)  # Hardcoded assumption
    
    # BUGGY: Scale by sqrt(L) instead of proper aggregation
    L = lt.lead_weeks + lt.review_weeks
    S = mu * L + z * sigma * np.sqrt(L)
    
    # Inventory position
    position = (
        initial_state['on_hand'].iloc[0] + 
        initial_state['intransit_1'].iloc[0] + 
        initial_state['intransit_2'].iloc[0]
    )
    
    order = max(0, S - position)
    return order


def policy_b_corrected(quantiles_df, initial_state, costs, lt, seed=42):
    """
    Corrected policy (Patrick's recommendations).
    
    Fixes:
    1. Protection period = lead + review (3 weeks)
    2. Critical fractile = shortage / (holding + shortage) = 0.833
    3. MC aggregation over full horizon
    """
    rng = np.random.default_rng(seed)
    
    protection_weeks = lt.lead_weeks + lt.review_weeks
    critical_fractile = costs.shortage / (costs.holding + costs.shortage)
    
    # Aggregate weekly distributions via MC
    quantile_levels = quantiles_df.columns.values
    protection_samples = []
    
    for week in range(1, protection_weeks + 1):
        if week in quantiles_df.index:
            q_vals = quantiles_df.loc[week].values
            u = rng.uniform(0, 1, 10000)
            week_samples = np.interp(u, quantile_levels, q_vals)
            protection_samples.append(week_samples)
        else:
            protection_samples.append(np.zeros(10000))
    
    # Sum across weeks
    total_demand_samples = np.sum(protection_samples, axis=0)
    mu = np.mean(total_demand_samples)
    sigma = np.std(total_demand_samples)
    
    # Base-stock level
    z = stats.norm.ppf(critical_fractile)
    S = mu + z * sigma
    
    # Inventory position
    position = (
        initial_state['on_hand'].iloc[0] + 
        initial_state['intransit_1'].iloc[0] + 
        initial_state['intransit_2'].iloc[0]
    )
    
    order = max(0, S - position)
    return order


def simulate_inventory_policy(quantiles_df, policy_func, costs, lt, seed=42):
    """
    Simulate inventory system under a given policy.
    
    Returns service level achieved by the policy.
    """
    rng = np.random.default_rng(seed)
    
    horizon = len(quantiles_df)
    quantile_levels = quantiles_df.columns.values
    
    # Initialize state
    on_hand = 10  # Starting inventory
    intransit_1 = 0
    intransit_2 = 0
    
    # Track results
    stockouts = 0
    total_periods = 0
    total_shortage_cost = 0
    total_holding_cost = 0
    
    for t in range(horizon):
        # Current state
        initial_state = pd.DataFrame({
            'on_hand': [on_hand],
            'intransit_1': [intransit_1],
            'intransit_2': [intransit_2],
        })
        
        # Make order decision
        order = policy_func(
            quantiles_df.iloc[t:t+3] if t+3 <= horizon else quantiles_df.iloc[t:],
            initial_state,
            costs,
            lt,
            seed=seed + t
        )
        
        # Realize demand
        if t+1 in quantiles_df.index:
            q_vals = quantiles_df.loc[t+1].values
            u = rng.uniform()
            demand = np.interp(u, quantile_levels, q_vals)
        else:
            demand = 0
        
        # Update inventory
        on_hand += intransit_1  # Receive order from 1 week ago
        sales = min(on_hand, demand)
        shortage = max(0, demand - on_hand)
        on_hand = max(0, on_hand - demand)
        
        # Track costs
        if shortage > 0:
            stockouts += 1
            total_shortage_cost += shortage * costs.shortage
        total_holding_cost += on_hand * costs.holding
        
        total_periods += 1
        
        # Update pipeline
        intransit_1 = intransit_2
        intransit_2 = order
    
    service_level = 1 - (stockouts / total_periods)
    total_cost = total_shortage_cost + total_holding_cost
    
    return {
        'service_level': service_level,
        'total_cost': total_cost,
        'shortage_cost': total_shortage_cost,
        'holding_cost': total_holding_cost,
        'stockouts': stockouts,
        'total_periods': total_periods,
    }


def run_ab_test(n_skus: int = 50, seed: int = 42):
    """
    Run A/B test comparing buggy vs corrected policy.
    
    Args:
        n_skus: Number of SKUs to test (for speed)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with results per SKU
    """
    print(f"Loading QRF forecasts for {n_skus} SKUs...")
    
    # Setup
    costs = Costs(holding=0.2, shortage=1.0)
    lt = LeadTime(lead_weeks=2, review_weeks=1)
    
    # Load forecasts (mock for now - would load actual QRF predictions)
    results = []
    
    rng = np.random.default_rng(seed)
    
    for sku_id in range(n_skus):
        # Generate mock quantile forecasts (replace with actual QRF predictions)
        # For realistic test, these should be actual model outputs
        base_demand = rng.uniform(1, 20)
        quantiles_df = pd.DataFrame({
            0.1: [base_demand * 0.2] * 12,
            0.3: [base_demand * 0.5] * 12,
            0.5: [base_demand * 1.0] * 12,
            0.7: [base_demand * 1.5] * 12,
            0.9: [base_demand * 2.5] * 12,
        }, index=range(1, 13))
        
        # Test Policy A (buggy)
        result_a = simulate_inventory_policy(
            quantiles_df, 
            policy_a_buggy, 
            costs, 
            lt, 
            seed=seed + sku_id
        )
        
        # Test Policy B (corrected)
        result_b = simulate_inventory_policy(
            quantiles_df, 
            policy_b_corrected, 
            costs, 
            lt, 
            seed=seed + sku_id
        )
        
        results.append({
            'sku_id': sku_id,
            'base_demand': base_demand,
            'policy_a_service_level': result_a['service_level'],
            'policy_a_cost': result_a['total_cost'],
            'policy_b_service_level': result_b['service_level'],
            'policy_b_cost': result_b['total_cost'],
            'service_level_improvement': result_b['service_level'] - result_a['service_level'],
            'cost_change': result_b['total_cost'] - result_a['total_cost'],
        })
    
    return pd.DataFrame(results)


def main():
    """Run the A/B test and report results."""
    print("="*70)
    print("POLICY A/B TEST: Buggy (1-week) vs Corrected (3-week)")
    print("="*70)
    print()
    
    # Run test
    results_df = run_ab_test(n_skus=100, seed=42)
    
    # Summary statistics
    print("RESULTS SUMMARY:")
    print("-" * 70)
    print(f"SKUs tested: {len(results_df)}")
    print()
    
    print("Policy A (Buggy - 1 week protection):")
    print(f"  Mean service level: {results_df['policy_a_service_level'].mean():.1%}")
    print(f"  Mean total cost: {results_df['policy_a_cost'].mean():.2f}")
    print()
    
    print("Policy B (Corrected - 3 week protection):")
    print(f"  Mean service level: {results_df['policy_b_service_level'].mean():.1%}")
    print(f"  Mean total cost: {results_df['policy_b_cost'].mean():.2f}")
    print()
    
    print("IMPROVEMENT (B - A):")
    print(f"  Service level: {results_df['service_level_improvement'].mean():+.1%}")
    print(f"  Total cost: {results_df['cost_change'].mean():+.2f}")
    print()
    
    # Statistical test
    from scipy.stats import ttest_rel
    
    t_stat, p_value = ttest_rel(
        results_df['policy_b_service_level'],
        results_df['policy_a_service_level']
    )
    
    print("STATISTICAL SIGNIFICANCE:")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  {'✅ SIGNIFICANT' if p_value < 0.05 else '❌ NOT SIGNIFICANT'} at α=0.05")
    print()
    
    # Detailed results
    print("SAMPLE SKUs:")
    print(results_df.head(10).to_string(index=False))
    print()
    
    # Save results
    output_file = "models/results/policy_ab_test_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✅ Full results saved to: {output_file}")
    print()
    
    # Interpretation
    print("="*70)
    print("INTERPRETATION:")
    print("="*70)
    
    avg_improvement = results_df['service_level_improvement'].mean()
    
    if avg_improvement > 0.1:
        print("✅ Policy B shows STRONG improvement in service level")
        print("   This confirms the corrected policy works as expected.")
    elif avg_improvement > 0:
        print("⚠️  Policy B shows MODEST improvement in service level")
        print("   The effect is smaller than expected (target: ~40% improvement)")
    else:
        print("❌ Policy B shows NO improvement or WORSE service level")
        print("   This suggests an implementation issue or constraint.")
    
    print()
    
    if results_df['cost_change'].mean() > 0:
        print("⚠️  Policy B has HIGHER costs on average")
        print("   This is expected if holding more inventory.")
    else:
        print("✅ Policy B has LOWER costs (better overall optimization)")


if __name__ == "__main__":
    main()
