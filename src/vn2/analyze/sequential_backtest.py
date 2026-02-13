"""
12-week progressive backtest with L=2 lead time and PMF-based optimization.

Implements a realistic backtest where:
- Order placed at start of week t arrives at start of week t+2
- State is updated with actual demand as it becomes available
- Expected costs use PMFs; realized costs use actual demand
- Both include-week1 and exclude-week1 totals are computed
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .sequential_planner import (
    Costs,
    _safe_pmf,
    choose_order_L2,
    leftover_from_stock_and_demand,
    _shift_right,
    diff_pmf_D_minus_L,
    expected_pos_neg_from_Z,
    _conv_fft
)
from vn2.policy.corrected_policy import compute_order_quantity_corrected, LeadTime


def convolve_pmfs(pmf1: np.ndarray, pmf2: np.ndarray) -> np.ndarray:
    """
    Convolve two PMFs (distribution of sum of two random variables).
    
    Args:
        pmf1: First PMF (probabilities at 0, 1, 2, ...)
        pmf2: Second PMF (probabilities at 0, 1, 2, ...)
    
    Returns:
        PMF of the sum (longer array)
    """
    return np.convolve(pmf1, pmf2)


def aggregate_3week_pmf(
    h3_pmf: np.ndarray,
    h4_pmf: np.ndarray, 
    h5_pmf: np.ndarray,
    max_support: int = 1500
) -> np.ndarray:
    """
    Aggregate 3 weekly PMFs to get distribution of total demand over 3 weeks.
    Uses convolution to properly handle dependencies.
    
    Args:
        h3_pmf: PMF for horizon 3 (week order arrives)
        h4_pmf: PMF for horizon 4 (next week)
        h5_pmf: PMF for horizon 5 (third week)
        max_support: Maximum support size to prevent explosion
    
    Returns:
        PMF of 3-week total demand
    """
    # Convolve h3 and h4
    agg_pmf = convolve_pmfs(h3_pmf, h4_pmf)
    
    # Convolve with h5
    agg_pmf = convolve_pmfs(agg_pmf, h5_pmf)
    
    # Truncate if too long
    if len(agg_pmf) > max_support:
        agg_pmf = agg_pmf[:max_support]
        agg_pmf = agg_pmf / agg_pmf.sum()  # Re-normalize
    
    return agg_pmf


def choose_order_simple_vn2(
    h3_pmf: np.ndarray,
    h4_pmf: np.ndarray,
    h5_pmf: Optional[np.ndarray],
    current_position: int,
    costs: Costs,
    coverage_weeks: float = 3.0
) -> Tuple[int, float]:
    """
    VN2-style simple order-up-to policy using forecast medians.
    
    VN2 benchmark: order_up_to = 4 weeks × recent average demand
    We use: order_up_to = coverage_weeks × median(h3, h4, h5)
    
    This is simpler than probabilistic newsvendor and matches VN2's approach.
    
    Args:
        h3_pmf: PMF for h=3
        h4_pmf: PMF for h=4  
        h5_pmf: PMF for h=5 (optional)
        current_position: Current inventory position
        costs: Cost parameters
        coverage_weeks: Number of weeks to cover (default 3.0)
    
    Returns:
        (order_quantity, expected_cost)
    """
    # Extract medians from PMFs
    cdf_h3 = np.cumsum(h3_pmf)
    median_h3 = np.searchsorted(cdf_h3, 0.5)
    
    cdf_h4 = np.cumsum(h4_pmf)
    median_h4 = np.searchsorted(cdf_h4, 0.5)
    
    if h5_pmf is not None and len(h5_pmf) > 0:
        cdf_h5 = np.cumsum(h5_pmf)
        median_h5 = np.searchsorted(cdf_h5, 0.5)
        avg_weekly_demand = (median_h3 + median_h4 + median_h5) / 3.0
    else:
        avg_weekly_demand = (median_h3 + median_h4) / 2.0
    
    # Order up to coverage_weeks of average demand
    order_up_to = int(coverage_weeks * avg_weekly_demand)
    
    # Order quantity
    order_qty = max(0, order_up_to - current_position)
    
    # Simple expected cost estimate
    expected_cost = 0.0  # Placeholder
    
    return int(order_qty), expected_cost


def choose_order_3week(
    h3_pmf: np.ndarray,
    h4_pmf: np.ndarray,
    h5_pmf: np.ndarray,
    current_position: int,
    costs: Costs,
    critical_fractile: float = 0.833
) -> Tuple[int, float]:
    """
    Compute optimal order for 3-week protection period using aggregated PMF.
    
    Simple order-up-to policy:
    - Aggregate demand over 3 weeks (h3+h4+h5)
    - Find order-up-to level S at critical fractile
    - Order Q = max(0, S - current_position)
    
    Args:
        h3_pmf: PMF for h=3 (week order arrives)
        h4_pmf: PMF for h=4 (next week)
        h5_pmf: PMF for h=5 (third week)
        current_position: Current inventory position (on_hand + in_transit_1 + in_transit_2)
        costs: Cost parameters
        critical_fractile: Target service level (default 0.833 for cu/(cu+co) = 1.0/1.2)
    
    Returns:
        (order_quantity, expected_cost)
    """
    # Aggregate 3-week demand distribution
    agg_pmf = aggregate_3week_pmf(h3_pmf, h4_pmf, h5_pmf)
    
    # Compute CDF
    cdf = np.cumsum(agg_pmf)
    
    # Find order-up-to level at critical fractile
    # This is the smallest S such that P(D_3week <= S) >= critical_fractile
    order_up_to = np.searchsorted(cdf, critical_fractile)
    
    # Order quantity
    order_qty = max(0, order_up_to - current_position)
    
    # Estimate expected cost (simplified)
    # Expected shortage = E[max(0, D - S)]
    # Expected holding = E[max(0, S - D)]
    expected_shortage = sum(max(0, d - (current_position + order_qty)) * agg_pmf[d] 
                           for d in range(len(agg_pmf)))
    expected_holding = sum(max(0, (current_position + order_qty) - d) * agg_pmf[d]
                          for d in range(len(agg_pmf)))
    
    expected_cost = costs.shortage * expected_shortage + costs.holding * expected_holding
    
    return int(order_qty), expected_cost


@dataclass
class BacktestState:
    """Inventory state at a point in time."""
    week: int  # Current week number (1-indexed)
    on_hand: int  # On-hand inventory at start of week (before arrivals)
    intransit_1: int  # Arriving at start of current week (placed at week-2)
    intransit_2: int  # Arriving at start of next week (placed at week-1)
    
    def copy(self) -> 'BacktestState':
        return BacktestState(
            week=self.week,
            on_hand=self.on_hand,
            intransit_1=self.intransit_1,
            intransit_2=self.intransit_2
        )


@dataclass
class WeekResult:
    """Result for a single week."""
    week: int
    order_placed: int  # Order placed at start of week (arrives week+2)
    demand_actual: Optional[int]  # Actual demand (None if not yet observed)
    expected_cost: float  # Expected cost at decision time
    realized_cost: Optional[float]  # Realized cost (None if demand not observed)
    state_before: BacktestState  # State before week starts
    state_after: Optional[BacktestState]  # State after week ends (None if demand not observed)
    pmf_residual: float  # 1 - sum(PMF) for normalization check


@dataclass
class BacktestResult:
    """Complete backtest result for one (model, SKU)."""
    store: int
    product: int
    model_name: str
    weeks: List[WeekResult]
    total_expected_cost: float
    total_realized_cost: float
    total_expected_cost_excl_w1: float
    total_realized_cost_excl_w1: float
    n_weeks: int
    n_missing_forecasts: int
    diagnostics: Dict = field(default_factory=dict)


def reconstruct_initial_state(
    store: int,
    product: int,
    initial_state_df: pd.DataFrame,
    sales_history: pd.DataFrame,
    backtest_start_week: int = 1
) -> BacktestState:
    """
    Reconstruct initial inventory state for backtest start.
    
    Args:
        store: Store ID
        product: Product ID
        initial_state_df: DataFrame with columns [Store, Product, End Inventory, In Transit W+1, In Transit W+2]
        sales_history: DataFrame with weekly sales columns
        backtest_start_week: Week number to start backtest (1-indexed)
    
    Returns:
        BacktestState for start of backtest_start_week
    """
    # Get initial state from file (this is state at end of week 0)
    mask = (initial_state_df['Store'] == store) & (initial_state_df['Product'] == product)
    if not mask.any():
        raise ValueError(f"SKU ({store}, {product}) not found in initial state")
    
    row = initial_state_df[mask].iloc[0]
    
    # At start of week 1:
    # - on_hand = End Inventory from week 0
    # - intransit_1 = In Transit W+1 (arrives at start of week 1, placed at week -1)
    # - intransit_2 = In Transit W+2 (arrives at start of week 2, placed at week 0)
    
    if backtest_start_week == 1:
        return BacktestState(
            week=1,
            on_hand=int(row['End Inventory']),
            intransit_1=int(row['In Transit W+1']),
            intransit_2=int(row['In Transit W+2'])
        )
    else:
        # For later weeks, would need to simulate forward from week 1
        # Not implemented yet
        raise NotImplementedError(f"Backtest start week {backtest_start_week} > 1 not yet supported")


def load_actual_demand(
    store: int,
    product: int,
    sales_df: pd.DataFrame,
    n_weeks: int = 12
) -> List[int]:
    """
    Load actual demand for a SKU from sales history.
    
    Args:
        store: Store ID
        product: Product ID
        sales_df: Sales DataFrame with weekly columns
        n_weeks: Number of weeks to load
    
    Returns:
        List of integer demands (length n_weeks)
    """
    mask = (sales_df['Store'] == store) & (sales_df['Product'] == product)
    if not mask.any():
        raise ValueError(f"SKU ({store}, {product}) not found in sales data")
    
    row = sales_df[mask].iloc[0]
    
    # Get last n_weeks of sales (most recent columns)
    date_cols = [c for c in sales_df.columns if c not in ['Store', 'Product']]
    if len(date_cols) < n_weeks:
        raise ValueError(f"Not enough history: {len(date_cols)} < {n_weeks}")
    
    recent_cols = date_cols[-n_weeks:]
    demands = [int(row[col]) for col in recent_cols]
    
    return demands


def run_12week_backtest(
    store: int,
    product: int,
    model_name: str,
    forecasts_h1: List[Optional[np.ndarray]],  # PMF for h=3 (week order arrives)
    forecasts_h2: List[Optional[np.ndarray]],  # PMF for h=4 (next week after arrival)
    forecasts_h3: Optional[List[Optional[np.ndarray]]] = None,  # PMF for h=5 (third week)
    actuals: List[int] = None,  # Actual demand for weeks 1..12
    initial_state: BacktestState = None,
    costs: Costs = None,
    pmf_grain: int = 500
) -> BacktestResult:
    """
    Run 12-week progressive backtest with L=2 lead time and 3-week protection.
    
    Timeline:
    - Week 1-10: Place orders (arrive at week 3-12)
    - Week 11-12: No new orders; just compute costs from pending orders
    - Week 1: Cost is uncontrollable (everyone has same state)
    - Week 2-12: Costs affected by our decisions
    
    Protection period: 3 weeks (h=3,4,5)
    - h=3: Week order arrives (covers demand in arrival week)
    - h=4: Next week after arrival
    - h=5: Third week (full 3-week protection)
    
    Args:
        store: Store ID
        product: Product ID
        model_name: Model name
        forecasts_h1: List of 12 PMFs for h=3 (arrival week demand)
        forecasts_h2: List of 12 PMFs for h=4 (next week demand)
        forecasts_h3: List of 12 PMFs for h=5 (third week demand) - optional
        actuals: List of 12 actual demands
        initial_state: Initial inventory state at start of week 1
        costs: Cost parameters
        pmf_grain: PMF support size (for validation)
    
    Returns:
        BacktestResult with per-week orders, costs, and totals
    """
    assert len(forecasts_h1) == 12, "Need 12 h1 forecasts"
    assert len(forecasts_h2) == 12, "Need 12 h2 forecasts"
    assert len(actuals) == 12, "Need 12 weeks of actual demand"
    
    state = initial_state.copy()
    weeks = []
    n_missing = 0
    
    for t in range(1, 13):  # Weeks 1-12
        week_idx = t - 1
        h1 = forecasts_h1[week_idx]
        h2 = forecasts_h2[week_idx]
        demand_actual = actuals[week_idx]
        
        state_before = state.copy()
        
        # Get h5 forecast if available
        h5 = forecasts_h3[week_idx] if forecasts_h3 is not None else None
        
        # Check for missing forecasts
        if h1 is None or h2 is None:
            # Missing forecast: place order of 0
            q_t = 0
            expected_cost = 0.0
            pmf_residual = 0.0
            n_missing += 1
        elif t >= 11:
            # Weeks 11-12: No new orders (would arrive after horizon)
            q_t = 0
            expected_cost = 0.0
            pmf_residual = 1.0 - (h1.sum() + h2.sum()) / 2.0
        else:
            # Weeks 1-10: Optimize order
            # Normalize and check PMFs
            h1 = _safe_pmf(h1)
            h2 = _safe_pmf(h2)
            
            # Use 3-week aggregation if h5 is available
            if h5 is not None and len(h5) > 0:
                h5 = _safe_pmf(h5)
                pmf_residual = max(1.0 - h1.sum(), 1.0 - h2.sum(), 1.0 - h5.sum())
                
                # Calculate current position (what's available + what's coming)
                current_position = state.on_hand + state.intransit_1 + state.intransit_2
                
                # Use simple VN2-style policy (not complex 3-week aggregation)
                # VN2 uses 4-week coverage and achieves €5,248
                # Let's try 4 weeks to match VN2
                q_t, expected_cost = choose_order_simple_vn2(
                    h1, h2, h5,
                    current_position,
                    costs,
                    coverage_weeks=4.0  # Match VN2's 4-week coverage
                )
            else:
                # Fall back to 2-period optimization
                pmf_residual = max(1.0 - h1.sum(), 1.0 - h2.sum())
                
                # Choose order using current state
                # I0 = on_hand, Q1 = intransit_1, Q2 = intransit_2
                q_t, expected_cost = choose_order_L2(
                    h1, h2,
                    state.on_hand,
                    state.intransit_1,
                    state.intransit_2,
                    costs,
                    micro_refine=True
                )
        
        # Compute realized cost for current week
        # Week t cost depends on:
        # - Starting inventory: on_hand + intransit_1
        # - Demand: demand_actual
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
            state_after=state_after,
            pmf_residual=pmf_residual
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
        weeks=weeks,
        total_expected_cost=total_expected,
        total_realized_cost=total_realized,
        total_expected_cost_excl_w1=total_expected_excl_w1,
        total_realized_cost_excl_w1=total_realized_excl_w1,
        n_weeks=12,
        n_missing_forecasts=n_missing,
        diagnostics={
            'initial_state': initial_state,
            'final_state': state,
            'max_pmf_residual': max(w.pmf_residual for w in weeks)
        }
    )


def quantiles_to_pmf(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    grain: int = 500
) -> np.ndarray:
    """
    Convert quantile forecast to discrete PMF via interpolation.
    
    Args:
        quantiles: Array of quantile values (e.g., 13 quantiles from model)
        quantile_levels: Array of quantile levels (e.g., [0.01, 0.05, ..., 0.99])
        grain: Maximum support for PMF (0 to grain-1 inclusive)
    
    Returns:
        pmf: Array of length grain with probabilities
    """
    # Ensure quantiles are non-negative and sorted
    quantiles = np.maximum(quantiles, 0)
    quantiles = np.sort(quantiles)
    
    # Create CDF by interpolating quantile function
    # Add boundary points for extrapolation
    q_extended = np.concatenate([[0], quantile_levels, [1]])
    v_extended = np.concatenate([[0], quantiles, [quantiles[-1]]])
    
    # Interpolate CDF at integer points
    support = np.arange(grain)
    cdf = np.interp(support, v_extended, q_extended)
    
    # Convert CDF to PMF via differencing
    pmf = np.diff(cdf, prepend=0)
    
    # Ensure valid PMF (non-negative, sums to ~1)
    pmf = np.maximum(pmf, 0)
    pmf_sum = pmf.sum()
    if pmf_sum > 0:
        pmf = pmf / pmf_sum
    else:
        # Degenerate case: put all mass at zero
        pmf = np.zeros(grain)
        pmf[0] = 1.0
    
    return pmf

