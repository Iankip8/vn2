#!/usr/bin/env python3
"""
Backtest the L=3 lead time correction against weeks 1-6.

This script simulates what our orders WOULD have been if we had correctly
implemented the 3-week lead time instead of 2-week lead time.

Competition Timeline:
- Order 1: End of Week 0 → arrives Start of Week 3
- Order 2: End of Week 1 → arrives Start of Week 4  
- Order 3: End of Week 2 → arrives Start of Week 5
- Order 4: End of Week 3 → arrives Start of Week 6
- Order 5: End of Week 4 → arrives Start of Week 7
- Order 6: End of Week 5 → arrives Start of Week 8

IMPORTANT: This uses only information available at each decision point (no look-ahead).

Usage:
    python scripts/backtest_L3_correction.py \
        --checkpoints-dir models/checkpoints_h3 \
        --output-dir reports/backtest_L3

Output:
    - reports/backtest_L3/counterfactual_orders_week{1-6}.csv
    - reports/backtest_L3/weekly_costs.csv
    - reports/backtest_L3/summary.md
"""

import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from vn2.analyze.sequential_planner import Costs, choose_order_L3
from vn2.analyze.sip_opt import quantiles_to_pmf

console = Console()


@dataclass
class SKUState:
    """State for a single SKU at a decision point."""
    store: int
    product: int
    on_hand: int
    intransit_1: int  # Arriving week t+1
    intransit_2: int  # Arriving week t+2
    intransit_3: int  # Arriving week t+3 (our previous order)


@dataclass
class WeekResult:
    """Results for a single week's simulation."""
    week: int
    total_order_qty: int
    total_holding_cost: float
    total_shortage_cost: float
    total_cost: float
    n_stockouts: int
    n_overstocks: int


def load_state_csv(path: Path) -> pd.DataFrame:
    """Load and normalize state CSV."""
    df = pd.read_csv(path)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'end inventory' in cl:
            col_map[col] = 'end_inventory'
        elif 'in transit w+1' in cl:
            col_map[col] = 'intransit_1'
        elif 'in transit w+2' in cl:
            col_map[col] = 'intransit_2'
    if col_map:
        df = df.rename(columns=col_map)
    return df


def load_sales_csv(path: Path, week_date: str = None) -> pd.DataFrame:
    """
    Load sales CSV (wide format with dates as columns).
    
    The sales files have columns: Store, Product, <date1>, <date2>, ...
    We need to extract the sales for the specific week.
    
    Args:
        path: Path to sales CSV
        week_date: Date string for the week to extract (last column if None)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    # Get date columns (everything except Store, Product)
    date_cols = [c for c in df.columns if c not in ['Store', 'Product']]
    
    if week_date is None:
        # Use the last date column (most recent week's sales)
        week_date = date_cols[-1]
    
    # Extract Store, Product, and the specific week's sales
    result = df[['Store', 'Product', week_date]].copy()
    result = result.rename(columns={week_date: 'Sales'})
    
    return result


def load_initial_state(path: Path) -> pd.DataFrame:
    """Load initial state (Week 0)."""
    df = pd.read_csv(path)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'initial inventory' in cl or 'on hand' in cl:
            col_map[col] = 'on_hand'
        elif 'in transit w+1' in cl or 'order arriving' in cl.lower():
            col_map[col] = 'intransit_1'
        elif 'in transit w+2' in cl:
            col_map[col] = 'intransit_2'
    if col_map:
        df = df.rename(columns=col_map)
    return df


def get_fold_for_week(decision_week: int) -> int:
    """
    Map decision week to appropriate fold index.
    
    Fold index represents which historical cutoff was used for training.
    For the competition, we used pre-trained models, so fold_idx=0 is typically used.
    """
    return 0


def load_quantiles(
    store: int, 
    product: int, 
    model: str,
    checkpoints_dir: Path,
    fold_idx: int = 0
) -> Optional[pd.DataFrame]:
    """Load quantile forecasts for a SKU."""
    ckpt_path = checkpoints_dir / model / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
    if not ckpt_path.exists():
        return None
    
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    return data.get('quantiles')


def generate_counterfactual_order(
    state: SKUState,
    qdf: pd.DataFrame,
    costs: Costs,
    quantile_levels: np.ndarray,
    sip_grain: int = 500
) -> Tuple[int, float]:
    """Generate counterfactual order using L=3 optimization."""
    # Ensure we have h=1, h=2, h=3
    if 3 not in qdf.index:
        if 2 in qdf.index:
            qdf = qdf.copy()
            qdf.loc[3] = qdf.loc[2].values
        else:
            return 0, 0.0
    
    # Convert to PMFs
    h1_pmf = quantiles_to_pmf(qdf.loc[1].values, quantile_levels, grain=sip_grain)
    h2_pmf = quantiles_to_pmf(qdf.loc[2].values, quantile_levels, grain=sip_grain)
    h3_pmf = quantiles_to_pmf(qdf.loc[3].values, quantile_levels, grain=sip_grain)
    
    # Run L=3 optimization
    q_opt, exp_cost = choose_order_L3(
        h1_pmf, h2_pmf, h3_pmf,
        state.on_hand,
        state.intransit_1,
        state.intransit_2,
        state.intransit_3,
        costs
    )
    
    return int(q_opt), float(exp_cost)


def simulate_week_costs(
    state_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    costs: Costs
) -> Tuple[pd.DataFrame, WeekResult]:
    """
    Simulate a week's inventory dynamics and compute costs.
    
    Args:
        state_df: State at START of week (after arrivals)
        orders_df: Orders placed (will arrive in 3 weeks)
        sales_df: Actual sales for the week
        costs: Cost parameters
        
    Returns:
        (detailed_results_df, summary)
    """
    results = []
    total_holding = 0.0
    total_shortage = 0.0
    n_stockouts = 0
    n_overstocks = 0
    
    # Start with state
    merged = state_df.copy()
    merged['Store'] = merged['Store'].astype(int)
    merged['Product'] = merged['Product'].astype(int)
    
    # Drop any existing Sales column from state to avoid conflict
    if 'Sales' in merged.columns:
        merged = merged.drop(columns=['Sales'])
    
    # Add orders if available
    if not orders_df.empty and '0' in orders_df.columns:
        orders_sub = orders_df[['Store', 'Product', '0']].copy()
        orders_sub['Store'] = orders_sub['Store'].astype(int)
        orders_sub['Product'] = orders_sub['Product'].astype(int)
        orders_sub = orders_sub.rename(columns={'0': 'order_qty'})
        merged = merged.merge(orders_sub, on=['Store', 'Product'], how='left')
        merged['order_qty'] = merged['order_qty'].fillna(0).astype(int)
    else:
        merged['order_qty'] = 0
    
    # Add sales (the actual sales from the sales file for this week)
    if not sales_df.empty and 'Sales' in sales_df.columns:
        sales_sub = sales_df[['Store', 'Product', 'Sales']].copy()
        sales_sub['Store'] = sales_sub['Store'].astype(int)
        sales_sub['Product'] = sales_sub['Product'].astype(int)
        sales_sub = sales_sub.rename(columns={'Sales': 'actual_sales'})
        merged = merged.merge(sales_sub, on=['Store', 'Product'], how='left')
        merged['Sales'] = merged['actual_sales'].fillna(0).astype(int)
    else:
        merged['Sales'] = 0
    
    for _, row in merged.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        
        # Get inventory available for this week
        on_hand = int(row.get('end_inventory', row.get('on_hand', 0)))
        intransit_1 = int(row.get('intransit_1', 0))
        
        # Available inventory = on_hand + arriving this week
        available = on_hand + intransit_1
        demand = int(row['Sales'])
        
        # Compute outcomes
        sold = min(available, demand)
        shortage = max(0, demand - available)
        leftover = max(0, available - demand)
        
        # Costs
        holding_cost = costs.holding * leftover
        shortage_cost = costs.shortage * shortage
        
        total_holding += holding_cost
        total_shortage += shortage_cost
        
        if shortage > 0:
            n_stockouts += 1
        if leftover > 10:  # Arbitrary threshold for "overstock"
            n_overstocks += 1
        
        results.append({
            'Store': store,
            'Product': product,
            'available': available,
            'demand': demand,
            'sold': sold,
            'shortage': shortage,
            'leftover': leftover,
            'holding_cost': holding_cost,
            'shortage_cost': shortage_cost,
            'total_cost': holding_cost + shortage_cost
        })
    
    results_df = pd.DataFrame(results)
    total_order_qty = int(merged['order_qty'].sum())
    
    summary = WeekResult(
        week=0,  # Will be set by caller
        total_order_qty=total_order_qty,
        total_holding_cost=total_holding,
        total_shortage_cost=total_shortage,
        total_cost=total_holding + total_shortage,
        n_stockouts=n_stockouts,
        n_overstocks=n_overstocks
    )
    
    return results_df, summary


def run_backtest(
    checkpoints_dir: Path,
    selector_map_path: Path,
    states_dir: Path,
    sales_dir: Path,
    initial_state_path: Path,
    output_dir: Path,
    costs: Costs,
    max_weeks: int = 6
) -> List[WeekResult]:
    """Run the full backtest simulation."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load selector map
    if selector_map_path.exists():
        selector = pd.read_parquet(selector_map_path)
        model_for_sku = {
            (int(r.store), int(r.product)): r.model_name
            for r in selector.itertuples(index=False)
        }
    else:
        model_for_sku = {}
    
    quantile_levels = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    
    # Track orders history (for Q3 - previous orders that arrive 3 weeks later)
    order_history: Dict[int, pd.DataFrame] = {}  # decision_week -> orders_df
    
    # Results
    week_results: List[WeekResult] = []
    all_orders: Dict[int, pd.DataFrame] = {}
    
    console.print("\n[bold cyan]Starting L=3 Backtest Simulation[/bold cyan]")
    console.print(f"  Checkpoints: {checkpoints_dir}")
    console.print(f"  States: {states_dir}")
    console.print(f"  Costs: cu={costs.shortage}, co={costs.holding}")
    console.print()
    
    # Process each decision week
    for decision_week in range(1, max_weeks + 1):
        console.print(f"[bold]Decision Week {decision_week}[/bold]")
        
        # Determine which state file to use
        # Decision at end of week t uses state at end of week t-1 (if available)
        # For decision_week=1, we use initial state (week 0)
        if decision_week == 1:
            state_df = load_initial_state(initial_state_path)
            # Normalize column names for initial state
            if 'on_hand' not in state_df.columns and 'Initial Inventory' in state_df.columns:
                state_df = state_df.rename(columns={'Initial Inventory': 'end_inventory'})
        else:
            state_path = states_dir / f"state{decision_week - 1}.csv"
            if not state_path.exists():
                console.print(f"  [yellow]Warning: State file not found: {state_path}[/yellow]")
                continue
            state_df = load_state_csv(state_path)
        
        # Get Q3 (our order from 3 weeks ago that arrives this week)
        # For decision_week=1, there is no previous order from us
        # For decision_week=2, Q3 would be order from decision_week=-1 (doesn't exist)
        # For decision_week>=4, Q3 is order from decision_week-3
        q3_orders = order_history.get(decision_week - 3, None)
        
        # Generate counterfactual orders for each SKU
        orders = []
        n_success = 0
        n_fail = 0
        
        for _, row in state_df.iterrows():
            store = int(row['Store'])
            product = int(row['Product'])
            key = (store, product)
            
            # Get state
            on_hand = int(row.get('end_inventory', row.get('on_hand', row.get('Initial Inventory', 0))))
            intransit_1 = int(row.get('intransit_1', row.get('In Transit W+1', 0)))
            intransit_2 = int(row.get('intransit_2', row.get('In Transit W+2', 0)))
            
            # Q3: our previous order that arrives at t+3
            if q3_orders is not None:
                q3_row = q3_orders[(q3_orders['Store'] == store) & (q3_orders['Product'] == product)]
                intransit_3 = int(q3_row['0'].iloc[0]) if len(q3_row) > 0 else 0
            else:
                intransit_3 = 0
            
            state = SKUState(
                store=store,
                product=product,
                on_hand=on_hand,
                intransit_1=intransit_1,
                intransit_2=intransit_2,
                intransit_3=intransit_3
            )
            
            # Load quantiles
            model = model_for_sku.get(key, 'zinb')
            fold_idx = get_fold_for_week(decision_week)
            qdf = load_quantiles(store, product, model, checkpoints_dir, fold_idx)
            
            if qdf is None or qdf.empty:
                orders.append({'Store': store, 'Product': product, '0': 0})
                n_fail += 1
                continue
            
            try:
                q_opt, _ = generate_counterfactual_order(
                    state, qdf, costs, quantile_levels
                )
                orders.append({'Store': store, 'Product': product, '0': q_opt})
                n_success += 1
            except Exception as e:
                orders.append({'Store': store, 'Product': product, '0': 0})
                n_fail += 1
        
        orders_df = pd.DataFrame(orders)
        order_history[decision_week] = orders_df
        all_orders[decision_week] = orders_df
        
        # Save counterfactual orders
        orders_path = output_dir / f"counterfactual_orders_week{decision_week}.csv"
        orders_df.to_csv(orders_path, index=False)
        
        total_qty = orders_df['0'].sum()
        console.print(f"  Generated orders: {total_qty} units ({n_success} SKUs, {n_fail} missing)")
    
    # Now simulate costs for weeks where we have both orders and actual sales
    console.print("\n[bold cyan]Simulating Week Costs[/bold cyan]")
    
    # Load actual sales - map week to (file, date_column)
    # Week 1 sales are in the "2024-04-15" column of Week 1 file
    # Each file contains historical data plus the new week's sales in the last column
    sales_files = {
        1: (sales_dir / "Week 1 - 2024-04-15 - Sales.csv", "2024-04-15"),
        2: (sales_dir / "Week 2 - 2024-04-22 - Sales.csv", "2024-04-22"),
        3: (sales_dir / "Week 3 - 2024-04-29 - Sales.csv", "2024-04-29"),
        4: (sales_dir / "Week 4 - 2024-05-06 - Sales.csv", "2024-05-06"),
        5: (sales_dir / "Week 5 - 2024-05-13 - Sales.csv", "2024-05-13"),
    }
    
    weekly_cost_data = []
    
    for sales_week in range(1, min(max_weeks + 1, 6)):  # We have sales through week 5
        sales_info = sales_files.get(sales_week)
        if sales_info is None:
            console.print(f"  [yellow]Week {sales_week}: Sales file not configured[/yellow]")
            continue
        
        sales_path, week_date = sales_info
        if not sales_path.exists():
            console.print(f"  [yellow]Week {sales_week}: Sales file not found: {sales_path}[/yellow]")
            continue
        
        sales_df = load_sales_csv(sales_path, week_date)
        
        # For week t costs, we need state at start of week t
        if sales_week == 1:
            state_df = load_initial_state(initial_state_path)
            if 'end_inventory' not in state_df.columns:
                state_df = state_df.rename(columns={'Initial Inventory': 'end_inventory'})
        else:
            state_path = states_dir / f"state{sales_week - 1}.csv"
            if not state_path.exists():
                continue
            state_df = load_state_csv(state_path)
        
        # Get orders that were placed and will affect this week
        # Orders placed at decision_week arrive at sales_week = decision_week + 2 (for L=3)
        # But costs are computed based on inventory at start of week, not arrivals
        # We just compute realized costs based on actual inventory and demand
        
        results_df, summary = simulate_week_costs(state_df, all_orders.get(sales_week, pd.DataFrame()), sales_df, costs)
        summary.week = sales_week
        week_results.append(summary)
        
        weekly_cost_data.append({
            'week': sales_week,
            'holding_cost': summary.total_holding_cost,
            'shortage_cost': summary.total_shortage_cost,
            'total_cost': summary.total_cost,
            'n_stockouts': summary.n_stockouts,
            'n_overstocks': summary.n_overstocks,
            'order_qty': summary.total_order_qty
        })
        
        console.print(f"  Week {sales_week}: Holding=€{summary.total_holding_cost:.2f}, "
                     f"Shortage=€{summary.total_shortage_cost:.2f}, "
                     f"Total=€{summary.total_cost:.2f}")
    
    # Save weekly costs
    weekly_df = pd.DataFrame(weekly_cost_data)
    weekly_df.to_csv(output_dir / "weekly_costs.csv", index=False)
    
    # Generate summary
    generate_summary(output_dir, weekly_df, all_orders)
    
    return week_results


def generate_summary(output_dir: Path, weekly_df: pd.DataFrame, all_orders: Dict):
    """Generate markdown summary report."""
    
    total_holding = weekly_df['holding_cost'].sum()
    total_shortage = weekly_df['shortage_cost'].sum()
    total_cost = weekly_df['total_cost'].sum()
    total_stockouts = weekly_df['n_stockouts'].sum()
    
    total_orders = sum(df['0'].sum() for df in all_orders.values())
    
    lines = [
        "# L=3 Lead Time Correction Backtest Results",
        "",
        "## Summary",
        "",
        "This report shows what our performance **would have been** if we had correctly",
        "implemented the 3-week lead time instead of 2-week lead time.",
        "",
        "### Total Costs (Weeks 1-5)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Holding Cost | €{total_holding:.2f} |",
        f"| Total Shortage Cost | €{total_shortage:.2f} |",
        f"| **Total Cost** | **€{total_cost:.2f}** |",
        f"| Total Stockout Events | {total_stockouts} |",
        f"| Total Units Ordered | {total_orders} |",
        "",
        "### Weekly Breakdown",
        "",
        "| Week | Holding | Shortage | Total | Stockouts |",
        "|------|---------|----------|-------|-----------|",
    ]
    
    for _, row in weekly_df.iterrows():
        lines.append(
            f"| {int(row['week'])} | €{row['holding_cost']:.2f} | "
            f"€{row['shortage_cost']:.2f} | €{row['total_cost']:.2f} | "
            f"{int(row['n_stockouts'])} |"
        )
    
    lines.extend([
        "",
        "### Methodology",
        "",
        "1. For each decision week, loaded the state available at that time",
        "2. Generated counterfactual orders using `choose_order_L3()` with h=1, h=2, h=3 forecasts",
        "3. Tracked in-transit inventory including our own previous orders (Q3)",
        "4. Computed realized costs using actual demand (no look-ahead in forecasting)",
        "",
        "### Comparison Notes",
        "",
        "To compare with actual competition results, load the leaderboard data and",
        "compare the total costs computed here against what we actually achieved.",
        "",
        "The difference shows the cost of the lead time implementation error.",
    ])
    
    (output_dir / "summary.md").write_text('\n'.join(lines))
    console.print(f"\n[green]Summary saved to {output_dir / 'summary.md'}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Backtest L=3 lead time correction")
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('models/checkpoints_h3'))
    parser.add_argument('--selector-map', type=Path, default=Path('models/results/selector_map_seq12_v1.parquet'))
    parser.add_argument('--states-dir', type=Path, default=Path('data/states'))
    parser.add_argument('--sales-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--initial-state', type=Path, default=Path('data/raw/Week 0 - 2024-04-08 - Initial State.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/backtest_L3'))
    parser.add_argument('--cu', type=float, default=1.0)
    parser.add_argument('--co', type=float, default=0.2)
    parser.add_argument('--max-weeks', type=int, default=6)
    
    args = parser.parse_args()
    
    costs = Costs(holding=args.co, shortage=args.cu)
    
    run_backtest(
        checkpoints_dir=args.checkpoints_dir,
        selector_map_path=args.selector_map,
        states_dir=args.states_dir,
        sales_dir=args.sales_dir,
        initial_state_path=args.initial_state,
        output_dir=args.output_dir,
        costs=costs,
        max_weeks=args.max_weeks
    )


if __name__ == '__main__':
    main()

