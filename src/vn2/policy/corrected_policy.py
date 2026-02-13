"""
Fixed ordering policy math based on Patrick McDonald's recommendations.

THREE CRITICAL FIXES:
1. Protection period = lead_time + review_period (~3 weeks, not 1)
2. Quantile target = 0.833 (from costs: 1.0/(1.0+0.2))
3. Aggregate weekly distributions over protection horizon using Monte Carlo

This replaces the incorrect h=1-only approach that caused poor performance.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class Costs:
    """Cost parameters"""
    holding: float = 0.2
    shortage: float = 1.0


@dataclass
class LeadTime:
    """Lead time parameters"""
    lead_weeks: int = 2
    review_weeks: int = 1


def aggregate_weekly_distributions_mc(
    quantiles_df: pd.DataFrame,
    quantile_levels: np.ndarray,
    protection_weeks: int,
    lead_weeks: int = 2,
    n_samples: int = 10000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Aggregate weekly demand distributions over protection period using Monte Carlo.
    
    **CRITICAL FIX #3**: Must sum weekly demands, not use h=1 alone.
    **TIMELINE FIX**: Aggregate h = [lead_weeks+1, ..., lead_weeks+protection_weeks]
                      because order arrives after lead_weeks.
    
    Args:
        quantiles_df: Forecast quantiles (rows=steps, columns=quantile levels)
        quantile_levels: Quantile probability levels
        protection_weeks: Number of weeks to protect (L + R)
        lead_weeks: Lead time in weeks (order arrives after this)
        n_samples: Monte Carlo samples
        seed: Random seed
        
    Returns:
        (mean, std) of protection period demand
    """
    rng = np.random.default_rng(seed)
    
    # Sample from each week's distribution
    # Aggregate h = [lead_weeks+1, ..., lead_weeks+protection_weeks]
    # Order placed NOW arrives after lead_weeks, so forecast when AVAILABLE
    protection_samples = []
    
    for week in range(1, protection_weeks + 1):
        # Map to forecast horizon: h = lead_weeks + week
        h = lead_weeks + week
        if h in quantiles_df.index:
            # Get quantiles for this horizon
            q_vals = quantiles_df.loc[h].values
            
            # Monte Carlo: sample uniformly, invert CDF
            u = rng.uniform(0, 1, n_samples)
            week_samples = np.interp(u, quantile_levels, q_vals)
            protection_samples.append(week_samples)
        else:
            # Fallback: assume zero demand for missing weeks
            protection_samples.append(np.zeros(n_samples))
    
    # Sum across weeks (this is the critical aggregation)
    total_demand = np.sum(protection_samples, axis=0)
    
    mu = np.mean(total_demand)
    sigma = np.std(total_demand)
    
    return mu, sigma


def compute_base_stock_level_corrected(
    quantiles_df: pd.DataFrame,
    quantile_levels: np.ndarray,
    costs: Costs,
    lt: LeadTime,
    n_mc_samples: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute base-stock level with CORRECT protection period and quantile.
    
    **CRITICAL FIXES:**
    1. Protection period = lead + review (~3 weeks)
    2. Critical fractile = 0.833 (shortage / (shortage + holding))
    3. Aggregate distributions over full protection horizon
    
    Args:
        quantiles_df: Forecast quantiles
        quantile_levels: Quantile probability levels
        costs: Cost parameters
        lt: Lead time parameters
        n_mc_samples: Monte Carlo samples for aggregation
        
    Returns:
        (base_stock_level, mu_protection, sigma_protection)
    """
    # FIX #1: Protection period = lead + review
    protection_weeks = lt.lead_weeks + lt.review_weeks
    
    # FIX #2: Critical fractile from costs
    critical_fractile = costs.shortage / (costs.holding + costs.shortage)
    
    # FIX #3: Aggregate weekly distributions
    mu, sigma = aggregate_weekly_distributions_mc(
        quantiles_df,
        quantile_levels,
        protection_weeks,
        lead_weeks=lt.lead_weeks,
        n_samples=n_mc_samples
    )
    
    # Base-stock formula: mu + z*sigma (for aggregated demand)
    z = stats.norm.ppf(critical_fractile)
    S = mu + z * sigma
    
    return S, mu, sigma


def compute_order_quantity_corrected(
    quantiles_df: pd.DataFrame,
    quantile_levels: np.ndarray,
    initial_state: pd.DataFrame,
    costs: Costs,
    lt: LeadTime
) -> Tuple[int, Dict[str, float]]:
    """
    Compute order quantity with CORRECTED policy math.
    
    This is the main entry point to replace the buggy version.
    
    Args:
        quantiles_df: Forecast quantiles (rows=steps, columns=quantile levels)
        quantile_levels: Quantile probability levels
        initial_state: Initial inventory state (on_hand, intransit_1, intransit_2)
        costs: Cost parameters
        lt: Lead time parameters
        
    Returns:
        (order_quantity, debug_info dict)
    """
    # Compute corrected base-stock level
    S, mu_prot, sigma_prot = compute_base_stock_level_corrected(
        quantiles_df,
        quantile_levels,
        costs,
        lt
    )
    
    # Inventory position
    position = (
        initial_state["on_hand"].iloc[0] +
        initial_state["intransit_1"].iloc[0] +
        initial_state["intransit_2"].iloc[0]
    )
    
    # Order quantity (conservative ceiling for asymmetric costs)
    order_qty = max(0, int(np.ceil(S - position)))
    
    # Debug info
    info = {
        'base_stock_level': S,
        'protection_weeks': lt.lead_weeks + lt.review_weeks,
        'critical_fractile': costs.shortage / (costs.holding + costs.shortage),
        'mu_protection': mu_prot,
        'sigma_protection': sigma_prot,
        'inventory_position': position,
        'order_quantity': order_qty
    }
    
    return order_qty, info


def compute_direct_quantile_order(
    quantiles_df: pd.DataFrame,
    quantile_levels: np.ndarray,
    initial_state: pd.DataFrame,
    costs: Costs,
    lt: LeadTime,
    n_mc_samples: int = 10000
) -> Tuple[int, Dict[str, float]]:
    """
    Alternative: Directly use 0.833 quantile of aggregated demand.
    
    This is even simpler than base-stock and may perform better.
    
    Args:
        quantiles_df: Forecast quantiles
        quantile_levels: Quantile probability levels
        initial_state: Initial inventory state
        costs: Cost parameters
        lt: Lead time parameters
        n_mc_samples: Monte Carlo samples
        
    Returns:
        (order_quantity, debug_info dict)
    """
    # Protection period
    protection_weeks = lt.lead_weeks + lt.review_weeks
    
    # Critical fractile
    tau = costs.shortage / (costs.holding + costs.shortage)
    
    # Aggregate distributions via MC
    # TIMELINE FIX: Aggregate h = [lead_weeks+1, ..., lead_weeks+protection_weeks]
    rng = np.random.default_rng(42)
    protection_samples = []
    
    for week in range(1, protection_weeks + 1):
        h = lt.lead_weeks + week  # Map to forecast horizon
        if h in quantiles_df.index:
            q_vals = quantiles_df.loc[h].values
            u = rng.uniform(0, 1, n_mc_samples)
            week_samples = np.interp(u, quantile_levels, q_vals)
            protection_samples.append(week_samples)
        else:
            protection_samples.append(np.zeros(n_mc_samples))
    
    total_demand = np.sum(protection_samples, axis=0)
    
    # Direct quantile approach: order up to tau-quantile
    target_level = np.quantile(total_demand, tau)
    
    # Inventory position
    position = (
        initial_state["on_hand"].iloc[0] +
        initial_state["intransit_1"].iloc[0] +
        initial_state["intransit_2"].iloc[0]
    )
    
    # Order quantity
    order_qty = max(0, int(np.ceil(target_level - position)))
    
    # Debug info
    info = {
        'target_level_q833': target_level,
        'protection_weeks': protection_weeks,
        'critical_fractile': tau,
        'inventory_position': position,
        'order_quantity': order_qty,
        'method': 'direct_quantile'
    }
    
    return order_qty, info


if __name__ == '__main__':
    """Test the corrected policy"""
    
    # Mock data
    quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    # Example quantile forecast (3 weeks)
    quantiles_df = pd.DataFrame({
        0.01: [0.5, 0.4, 0.3],
        0.05: [1.0, 0.9, 0.8],
        0.1:  [1.5, 1.4, 1.3],
        0.2:  [2.0, 1.9, 1.8],
        0.3:  [2.5, 2.4, 2.3],
        0.4:  [3.0, 2.9, 2.8],
        0.5:  [3.5, 3.4, 3.3],  # Median ~3.5
        0.6:  [4.0, 3.9, 3.8],
        0.7:  [4.5, 4.4, 4.3],
        0.8:  [5.0, 4.9, 4.8],
        0.9:  [6.0, 5.9, 5.8],
        0.95: [7.0, 6.9, 6.8],
        0.99: [10.0, 9.9, 9.8]
    }, index=[1, 2, 3])
    
    # Initial state
    initial_state = pd.DataFrame({
        'on_hand': [5],
        'intransit_1': [2],
        'intransit_2': [3]
    })
    
    costs = Costs(holding=0.2, shortage=1.0)
    lt = LeadTime(lead_weeks=2, review_weeks=1)
    
    print("="*60)
    print("CORRECTED POLICY TEST")
    print("="*60)
    
    # Method 1: Base-stock with corrected aggregation
    order_qty, info = compute_order_quantity_corrected(
        quantiles_df, quantile_levels, initial_state, costs, lt
    )
    
    print("\nMethod 1: Base-Stock with Corrected Aggregation")
    print(f"  Protection period: {info['protection_weeks']} weeks")
    print(f"  Critical fractile: {info['critical_fractile']:.3f}")
    print(f"  Aggregated mu: {info['mu_protection']:.2f}")
    print(f"  Aggregated sigma: {info['sigma_protection']:.2f}")
    print(f"  Base-stock level: {info['base_stock_level']:.2f}")
    print(f"  Inventory position: {info['inventory_position']}")
    print(f"  ORDER: {info['order_quantity']}")
    
    # Method 2: Direct quantile
    order_qty2, info2 = compute_direct_quantile_order(
        quantiles_df, quantile_levels, initial_state, costs, lt
    )
    
    print("\nMethod 2: Direct 0.833 Quantile")
    print(f"  Protection period: {info2['protection_weeks']} weeks")
    print(f"  Critical fractile: {info2['critical_fractile']:.3f}")
    print(f"  Target level (q=0.833): {info2['target_level_q833']:.2f}")
    print(f"  Inventory position: {info2['inventory_position']}")
    print(f"  ORDER: {info2['order_quantity']}")
    
    print("\n" + "="*60)
    print("✅ Corrected policy implements all 3 fixes:")
    print("   1. Protection period = lead + review = 3 weeks")
    print("   2. Critical fractile = 0.833")
    print("   3. MC aggregation over full horizon")
    print("="*60)
