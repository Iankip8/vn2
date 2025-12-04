#!/usr/bin/env python3
"""
Full L=3 Simulation with Rolling State Propagation.

This script properly simulates what would have happened if we had used L=3 lead time:
1. Starts with initial state (Week 0)
2. For each decision week, generates order using L=3 optimization and CURRENT simulated state
3. Simulates inventory dynamics for the week
4. Propagates state forward to next week
5. Computes realized costs

This differs from the previous backtest which used actual historical state files.
"""

import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rich.console import Console

from vn2.analyze.sequential_planner import Costs, choose_order_L3
from vn2.analyze.sip_opt import quantiles_to_pmf

console = Console()


@dataclass
class SKUSimState:
    """Simulated state for a single SKU."""
    store: int
    product: int
    on_hand: int
    # In-transit queue: index 0 = arriving next week, 1 = in 2 weeks, 2 = in 3 weeks
    in_transit: List[int] = field(default_factory=lambda: [0, 0, 0])


def load_initial_state(path: Path) -> Dict[Tuple[int, int], SKUSimState]:
    """Load Week 0 initial state."""
    df = pd.read_csv(path)
    states = {}
    for _, row in df.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        on_hand = int(row.get('End Inventory', row.get('Start Inventory', 0)))
        it1 = int(row.get('In Transit W+1', 0))  # Arrives week 1
        it2 = int(row.get('In Transit W+2', 0))  # Arrives week 2
        states[(store, product)] = SKUSimState(
            store=store,
            product=product,
            on_hand=on_hand,
            in_transit=[it1, it2, 0]  # W+1, W+2, W+3
        )
    return states


def load_sales(sales_dir: Path, week: int) -> Dict[Tuple[int, int], int]:
    """Load actual sales for a specific week."""
    week_info = {
        1: ("Week 1 - 2024-04-15 - Sales.csv", "2024-04-15"),
        2: ("Week 2 - 2024-04-22 - Sales.csv", "2024-04-22"),
        3: ("Week 3 - 2024-04-29 - Sales.csv", "2024-04-29"),
        4: ("Week 4 - 2024-05-06 - Sales.csv", "2024-05-06"),
        5: ("Week 5 - 2024-05-13 - Sales.csv", "2024-05-13"),
    }
    
    if week not in week_info:
        return {}
    
    filename, date_col = week_info[week]
    path = sales_dir / filename
    if not path.exists():
        return {}
    
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    result = {}
    for _, row in df.iterrows():
        key = (int(row['Store']), int(row['Product']))
        result[key] = int(row[date_col])
    
    return result


def load_selector_map(path: Path) -> Dict[Tuple[int, int], str]:
    """Load model selector map."""
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {(int(r.store), int(r.product)): r.model_name for r in df.itertuples(index=False)}


def load_quantiles(store: int, product: int, model: str, checkpoints_dir: Path, fold_idx: int = 0) -> Optional[pd.DataFrame]:
    """Load quantile forecasts."""
    path = checkpoints_dir / model / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('quantiles')


def generate_order_L3(
    state: SKUSimState,
    qdf: pd.DataFrame,
    costs: Costs,
    quantile_levels: np.ndarray,
    sip_grain: int = 500
) -> int:
    """Generate optimal order using L=3 optimization."""
    if qdf is None or qdf.empty:
        return 0
    
    # Ensure h=1, h=2, h=3 exist
    if 3 not in qdf.index:
        if 2 in qdf.index:
            qdf = qdf.copy()
            qdf.loc[3] = qdf.loc[2].values
        else:
            return 0
    
    # Convert to PMFs
    h1_pmf = quantiles_to_pmf(qdf.loc[1].values, quantile_levels, grain=sip_grain)
    h2_pmf = quantiles_to_pmf(qdf.loc[2].values, quantile_levels, grain=sip_grain)
    h3_pmf = quantiles_to_pmf(qdf.loc[3].values, quantile_levels, grain=sip_grain)
    
    # State: on_hand, Q1 (arriving next week), Q2 (arriving in 2 weeks), Q3 (arriving in 3 weeks)
    I0 = state.on_hand
    Q1 = state.in_transit[0]
    Q2 = state.in_transit[1]
    Q3 = state.in_transit[2]
    
    try:
        q_opt, _ = choose_order_L3(h1_pmf, h2_pmf, h3_pmf, I0, Q1, Q2, Q3, costs)
        return int(q_opt)
    except:
        return 0


def simulate_week(
    states: Dict[Tuple[int, int], SKUSimState],
    sales: Dict[Tuple[int, int], int],
    pending_orders: Dict[Tuple[int, int], int],  # Orders that will be placed at END of this week
    costs: Costs
) -> Tuple[float, float, int]:
    """
    Simulate one week of inventory dynamics.
    
    Timeline for week W:
    - START of W: in_transit[0] arrives (this was placed end of W-3)
    - DURING W: demand served from on_hand + arrivals
    - END of W: new order placed (arrives start of W+3)
    
    in_transit semantics:
    - in_transit[0]: arriving START of NEXT week (placed end of week W-2)
    - in_transit[1]: arriving in 2 weeks (placed end of week W-1)  
    - in_transit[2]: arriving in 3 weeks (placed end of week W = now)
    
    Returns: (holding_cost, shortage_cost, stockouts)
    """
    total_holding = 0.0
    total_shortage = 0.0
    stockouts = 0
    
    for key, state in states.items():
        # 1. Arrivals at START of this week
        arriving = state.in_transit[0]
        available = state.on_hand + arriving
        
        # 2. Sales during week
        demand = sales.get(key, 0)
        sold = min(available, demand)
        shortage = max(0, demand - available)
        leftover = max(0, available - demand)
        
        # 3. Costs (holding on leftover at end of week)
        h_cost = costs.holding * leftover
        s_cost = costs.shortage * shortage
        total_holding += h_cost
        total_shortage += s_cost
        if shortage > 0:
            stockouts += 1
        
        # 4. Update on_hand to end-of-week position
        state.on_hand = leftover
        
        # 5. Shift in-transit (new orders are added separately after this function)
        # in_transit[0] was just received, shift everything left
        state.in_transit = [state.in_transit[1], state.in_transit[2], 0]
    
    return total_holding, total_shortage, stockouts


def run_full_simulation(
    initial_state_path: Path,
    sales_dir: Path,
    checkpoints_dir: Path,
    selector_map_path: Path,
    max_weeks: int = 5,
    costs: Costs = Costs()
) -> pd.DataFrame:
    """Run full simulation with rolling state propagation."""
    
    # Load initial state
    console.print("Loading initial state...")
    states = load_initial_state(initial_state_path)
    console.print(f"  {len(states)} SKUs")
    
    # Load selector map
    model_for_sku = load_selector_map(selector_map_path)
    console.print(f"  {len(model_for_sku)} model mappings")
    
    quantile_levels = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    
    results = []
    all_orders = {}  # Track all orders placed
    
    # Place Order 1 at End of Week 0 (before Week 1 happens)
    # Order 1 arrives at Start of Week 3
    console.print("\n[bold]End of Week 0: Place Order 1[/bold]")
    order1 = {}
    for key, state in states.items():
        model = model_for_sku.get(key, 'zinb')
        qdf = load_quantiles(state.store, state.product, model, checkpoints_dir)
        order = generate_order_L3(state, qdf, costs, quantile_levels)
        order1[key] = order
        # Add to in_transit[2] = arrives in 3 weeks = Week 3
        state.in_transit[2] = order
    
    all_orders[1] = sum(order1.values())
    console.print(f"  Order 1: {all_orders[1]} units → arrives Week 3")
    
    for week in range(1, max_weeks + 1):
        console.print(f"\n[bold]Week {week}[/bold]")
        
        # Load actual demand
        sales = load_sales(sales_dir, week)
        total_demand = sum(sales.values())
        console.print(f"  Demand: {total_demand} units")
        
        # What's arriving this week?
        total_arriving = sum(s.in_transit[0] for s in states.values())
        console.print(f"  Arriving: {total_arriving} units")
        
        # Simulate week - pass empty dict for new orders since we place them AFTER
        holding, shortage, stockouts = simulate_week(states, sales, {}, costs)
        total = holding + shortage
        
        console.print(f"  Costs: Holding=€{holding:.2f}, Shortage=€{shortage:.2f}, Total=€{total:.2f}")
        
        # Place next order at END of this week (Order N+1 where we're processing week N)
        # This order arrives at week + 3
        next_order_num = week + 1
        if next_order_num <= 6:  # Only 6 orders in competition
            console.print(f"  Placing Order {next_order_num} (arrives Week {week + 3})...")
            next_order = {}
            for key, state in states.items():
                model = model_for_sku.get(key, 'zinb')
                qdf = load_quantiles(state.store, state.product, model, checkpoints_dir)
                order = generate_order_L3(state, qdf, costs, quantile_levels)
                next_order[key] = order
                # Add to pipeline - will arrive in 3 weeks
                state.in_transit[2] = order
            
            all_orders[next_order_num] = sum(next_order.values())
            console.print(f"    Order {next_order_num}: {all_orders[next_order_num]} units")
        
        results.append({
            'week': week,
            'demand': total_demand,
            'arriving': total_arriving,
            'holding_cost': holding,
            'shortage_cost': shortage,
            'total_cost': total,
            'stockouts': stockouts
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Full L=3 simulation with state propagation')
    parser.add_argument('--initial-state', type=Path,
                        default=Path('data/raw/Week 0 - 2024-04-08 - Initial State.csv'))
    parser.add_argument('--sales-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('models/checkpoints_h3'))
    parser.add_argument('--selector-map', type=Path,
                        default=Path('models/results/selector_map_seq12_v1.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/backtest_L3'))
    parser.add_argument('--max-weeks', type=int, default=5)
    parser.add_argument('--cu', type=float, default=1.0)
    parser.add_argument('--co', type=float, default=0.2)
    
    args = parser.parse_args()
    
    costs = Costs(holding=args.co, shortage=args.cu)
    
    console.print("[bold cyan]Full L=3 Simulation with Rolling State Propagation[/bold cyan]\n")
    
    results = run_full_simulation(
        args.initial_state,
        args.sales_dir,
        args.checkpoints_dir,
        args.selector_map,
        args.max_weeks,
        costs
    )
    
    # Summary
    console.print("\n[bold green]Simulation Summary[/bold green]")
    total_holding = results['holding_cost'].sum()
    total_shortage = results['shortage_cost'].sum()
    total_cost = results['total_cost'].sum()
    
    console.print(f"  Total Holding: €{total_holding:.2f}")
    console.print(f"  Total Shortage: €{total_shortage:.2f}")
    console.print(f"  [bold]Total Cost: €{total_cost:.2f}[/bold]")
    
    # Compare with actual
    # Our actual week 1-5: €913.80 + €931.60 + €1780.40 + €1004.20 = €4629.20 (approx from W5 cumulative - W1-2)
    actual_costs = {
        'weeks_1_2': 913.80,
        'week_3': 931.60,
        'week_4': 1780.40,
        'week_5': 1004.20,
    }
    actual_total = sum(actual_costs.values())
    
    console.print(f"\n[bold]Comparison with Actual (L=2)[/bold]")
    console.print(f"  Our Actual Cost (W1-5): €{actual_total:.2f}")
    console.print(f"  L=3 Simulated Cost: €{total_cost:.2f}")
    console.print(f"  Difference: €{actual_total - total_cost:.2f} ({'better' if total_cost < actual_total else 'worse'})")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / 'full_simulation_results.csv', index=False)
    
    console.print(f"\n[green]Results saved to {args.output_dir / 'full_simulation_results.csv'}[/green]")


if __name__ == '__main__':
    main()

