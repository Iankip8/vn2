#!/usr/bin/env python3
"""
Generate next order using correct L=3 lead time optimization.

Competition Rule: "orders are made at the end of week X and received at the start of week X+3"

This means:
- Order placed at END of week t arrives at START of week t+3
- We need forecasts for weeks t+1, t+2, and t+3 (h=1, h=2, h=3)
- We need to track 3 in-transit orders: Q1 (arriving t+1), Q2 (arriving t+2), Q3 (arriving t+3)

Usage:
    python scripts/generate_order_L3.py \
        --state-file data/states/state3.csv \
        --checkpoints-dir models/checkpoints_h3 \
        --output data/submissions/order4_L3.csv

    # With overrides
    python scripts/generate_order_L3.py \
        --state-file data/states/state5.csv \
        --checkpoints-dir models/checkpoints_h3 \
        --selector-map models/results/selector_map_seq12_v1.parquet \
        --guardrail-overrides reports/guardrail_overrides.csv \
        --output data/submissions/order6_L3.csv
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from vn2.analyze.sequential_planner import Costs, choose_order_L3
from vn2.analyze.sip_opt import quantiles_to_pmf

console = Console()


def load_state_csv(state_path: Path) -> pd.DataFrame:
    """Load state CSV with proper column handling for L=3 lead time."""
    df = pd.read_csv(state_path)
    
    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'end inventory' in cl:
            col_map[col] = 'end_inventory'
        elif 'in transit w+1' in cl:
            col_map[col] = 'intransit_1'  # Arriving week t+1
        elif 'in transit w+2' in cl:
            col_map[col] = 'intransit_2'  # Arriving week t+2
        # Note: For L=3, we need to track Q3 (arriving week t+3) but this
        # may not be in state files from competition. We'll handle this.
    
    if col_map:
        df = df.rename(columns=col_map)
    
    return df


def generate_order_L3(
    state_path: Path,
    output_path: Path,
    selector_map_path: Path = None,
    checkpoints_dir: Path = None,
    guardrail_overrides_path: Path = None,
    cu: float = 1.0,
    co: float = 0.2,
    sip_grain: int = 500,
    fold_idx: int = 0,
    previous_order_path: Path = None,  # Path to our previous order (arrives at t+3)
    test_mode: bool = False
) -> tuple:
    """
    Generate next order using L=3 lead time optimization.
    
    Args:
        state_path: Path to current state CSV
        output_path: Path for output order CSV
        selector_map_path: Path to model selector parquet
        checkpoints_dir: Path to h=3 checkpoints directory
        guardrail_overrides_path: Optional path to guardrail overrides
        cu: Shortage cost (default 1.0)
        co: Holding cost (default 0.2)
        sip_grain: PMF grain size
        fold_idx: Fold index for forecasts
        previous_order_path: Path to our previous order (needed for Q3)
        test_mode: Enable verbose output
        
    Returns:
        (orders_df, expected_cost)
    """
    # Default paths
    if selector_map_path is None:
        selector_map_path = Path('models/results/selector_map_seq12_v1.parquet')
    if checkpoints_dir is None:
        checkpoints_dir = Path('models/checkpoints_h3')
    
    console.print(f"[bold blue]📦 Generating Order with L=3 Lead Time[/bold blue]")
    console.print(f"  State: {state_path}")
    console.print(f"  Checkpoints: {checkpoints_dir}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Costs: cu={cu}, co={co}")
    
    # Load state
    state = load_state_csv(state_path)
    console.print(f"  Loaded {len(state)} SKUs from state")
    
    # Load selector
    if selector_map_path.exists():
        selector = pd.read_parquet(selector_map_path)
        model_for_sku = {
            (int(r.store), int(r.product)): r.model_name 
            for r in selector.itertuples(index=False)
        }
    else:
        model_for_sku = {}
    console.print(f"  Loaded {len(model_for_sku)} SKU-model mappings")
    
    # Load guardrail overrides
    guardrails = {}
    if guardrail_overrides_path and guardrail_overrides_path.exists():
        guard_df = pd.read_csv(guardrail_overrides_path)
        for _, row in guard_df.iterrows():
            key = (int(row.get('Store', row.get('store'))), 
                   int(row.get('Product', row.get('product'))))
            guardrails[key] = float(row.get('service_level_override', 0.8333))
        console.print(f"  Loaded {len(guardrails)} guardrail overrides")
    
    # Load previous order if provided (for Q3)
    previous_orders = {}
    if previous_order_path and previous_order_path.exists():
        prev_df = pd.read_csv(previous_order_path)
        for _, row in prev_df.iterrows():
            key = (int(row['Store']), int(row['Product']))
            previous_orders[key] = int(row.get('0', row.get('Order', 0)))
        console.print(f"  Loaded {len(previous_orders)} previous orders for Q3")
    
    # Configuration
    base_costs = Costs(holding=co, shortage=cu)
    quantile_levels = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 
                                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    
    # Generate orders
    orders = []
    expected_costs = []
    successful = 0
    missing_h3 = 0
    
    for _, row in state.iterrows():
        store = int(row['Store'])
        product = int(row['Product'])
        key = (store, product)
        
        # Get state
        I0 = int(row.get('end_inventory', row.get('End Inventory', 0)))
        Q1 = int(row.get('intransit_1', row.get('In Transit W+1', 0)))
        Q2 = int(row.get('intransit_2', row.get('In Transit W+2', 0)))
        Q3 = previous_orders.get(key, 0)  # Our previous order, arriving at t+3
        
        # Get model
        model = model_for_sku.get(key, 'zinb')
        ckpt = checkpoints_dir / model / f'{store}_{product}' / f'fold_{fold_idx}.pkl'
        
        if not ckpt.exists():
            orders.append({'Store': store, 'Product': product, '0': 0})
            continue
        
        try:
            with open(ckpt, 'rb') as f:
                data = pickle.load(f)
            qdf = data.get('quantiles')
            
            # Require h=1, h=2, and h=3 forecasts
            if qdf is None or qdf.empty:
                orders.append({'Store': store, 'Product': product, '0': 0})
                continue
            
            if 1 not in qdf.index or 2 not in qdf.index or 3 not in qdf.index:
                missing_h3 += 1
                # Fall back to h=2 if h=3 missing
                if 3 not in qdf.index and 2 in qdf.index:
                    qdf.loc[3] = qdf.loc[2].values  # Use h=2 as proxy for h=3
            
            # Convert quantiles to PMFs
            h1_pmf = quantiles_to_pmf(qdf.loc[1].values, quantile_levels, grain=sip_grain)
            h2_pmf = quantiles_to_pmf(qdf.loc[2].values, quantile_levels, grain=sip_grain)
            h3_pmf = quantiles_to_pmf(qdf.loc[3].values, quantile_levels, grain=sip_grain)
            
            # Apply guardrail override (adjust costs to achieve target service level)
            if key in guardrails:
                s_target = guardrails[key]
                # Service level s = cu / (cu + co) => cu = co * s / (1 - s)
                adjusted_cu = co * s_target / (1.0 - s_target)
                costs = Costs(holding=co, shortage=adjusted_cu)
            else:
                costs = base_costs
            
            # Optimize order with L=3
            q_opt, exp_cost = choose_order_L3(
                h1_pmf, h2_pmf, h3_pmf,
                I0, Q1, Q2, Q3,
                costs
            )
            
            orders.append({'Store': store, 'Product': product, '0': int(q_opt)})
            expected_costs.append(exp_cost)
            successful += 1
            
        except Exception as e:
            if test_mode:
                console.print(f"[yellow]Warning: {store}_{product} failed: {e}[/yellow]")
            orders.append({'Store': store, 'Product': product, '0': 0})
    
    # Save output
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(output_path, index=False)
    
    # Summary
    portfolio_cost = sum(expected_costs)
    total_units = orders_df['0'].sum()
    nonzero_count = (orders_df['0'] > 0).sum()
    
    console.print(f"\n[bold green]✅ Order Generated (L=3 Lead Time)[/bold green]")
    console.print(f"  Total units: {total_units}")
    console.print(f"  SKUs with orders: {nonzero_count}")
    console.print(f"  Mean order per SKU: {orders_df['0'].mean():.2f}")
    console.print(f"  Expected portfolio cost: €{portfolio_cost:.2f}")
    console.print(f"  SKUs with forecasts: {successful}")
    if missing_h3 > 0:
        console.print(f"  [yellow]SKUs missing h=3 (used h=2 fallback): {missing_h3}[/yellow]")
    
    if test_mode and nonzero_count > 0:
        nonzero = orders_df[orders_df['0'] > 0].head(10)
        console.print(f"\n[yellow]Sample non-zero orders:[/yellow]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Store")
        table.add_column("Product")
        table.add_column("Quantity")
        for _, r in nonzero.iterrows():
            table.add_row(str(r['Store']), str(r['Product']), str(r['0']))
        console.print(table)
    
    return orders_df, portfolio_cost


def main():
    parser = argparse.ArgumentParser(
        description="Generate order with L=3 lead time optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--state-file', type=Path, required=True,
                        help='Path to current state CSV')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output path for order CSV')
    parser.add_argument('--checkpoints-dir', type=Path, 
                        default=Path('models/checkpoints_h3'),
                        help='Path to h=3 checkpoints directory')
    parser.add_argument('--selector-map', type=Path, default=None,
                        help='Path to model selector parquet')
    parser.add_argument('--guardrail-overrides', type=Path, default=None,
                        help='Path to guardrail overrides CSV')
    parser.add_argument('--previous-order', type=Path, default=None,
                        help='Path to previous order CSV (for Q3)')
    parser.add_argument('--cu', type=float, default=1.0,
                        help='Shortage cost')
    parser.add_argument('--co', type=float, default=0.2,
                        help='Holding cost')
    parser.add_argument('--sip-grain', type=int, default=500,
                        help='PMF grain size')
    parser.add_argument('--fold-idx', type=int, default=0,
                        help='Fold index for forecasts')
    parser.add_argument('--test', action='store_true',
                        help='Enable verbose test mode')
    
    args = parser.parse_args()
    
    if not args.state_file.exists():
        console.print(f"[bold red]Error: State file not found: {args.state_file}[/bold red]")
        return 1
    
    try:
        generate_order_L3(
            state_path=args.state_file,
            output_path=args.output,
            selector_map_path=args.selector_map,
            checkpoints_dir=args.checkpoints_dir,
            guardrail_overrides_path=args.guardrail_overrides,
            cu=args.cu,
            co=args.co,
            sip_grain=args.sip_grain,
            fold_idx=args.fold_idx,
            previous_order_path=args.previous_order,
            test_mode=args.test
        )
        return 0
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

