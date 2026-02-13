"""
Evaluate VN2 Benchmark Approach

This script evaluates the exact VN2 benchmark approach:
- Seasonal moving average (13-week MA with 52-week seasonality)
- Simple order-up-to policy (4-week coverage)
- Target: €5,248

This validates that our framework can reproduce the benchmark result.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from rich import print as rprint

from vn2.config import load_config
from vn2.forecast.models.vn2_benchmark import VN2BenchmarkForecaster
from vn2.analyze.sequential_backtest import (
    run_12week_backtest,
    reconstruct_initial_state,
    load_actual_demand,
    quantiles_to_pmf,
    BacktestState
)
from vn2.analyze.sequential_planner import Costs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_sku(
    sku_id: tuple,
    demand_df: pd.DataFrame,
    initial_state: pd.Series,
    cfg: dict,
    forecaster: VN2BenchmarkForecaster
) -> pd.DataFrame:
    """
    Run VN2 benchmark evaluation for a single SKU.
    
    Args:
        sku_id: (store, product) tuple
        demand_df: Full demand history
        initial_state: Initial inventory state for this SKU
        cfg: Configuration dict
        forecaster: VN2 benchmark forecaster
        
    Returns:
        DataFrame with weekly results for this SKU
    """
    store, product = sku_id
    
    # Get demand history for this SKU
    sku_demand = demand_df.loc[sku_id]
    
    # Reconstruct initial state
    try:
        state = BacktestState(
            on_hand=int(initial_state['End Inventory']),
            in_transit_w1=int(initial_state['In Transit W+1']),
            in_transit_w2=int(initial_state['In Transit W+2'])
        )
    except Exception as e:
        logger.warning(f"SKU {sku_id}: Failed to get initial state, using defaults: {e}")
        state = BacktestState(on_hand=0, in_transit_w1=0, in_transit_w2=0)
    
    # Get actual demand for backtest period
    actual_demand_dict = load_actual_demand(sku_demand, holdout_weeks=12)
    
    # Create costs object
    costs = Costs(
        holding=0.2,
        shortage=1.0
    )
    
    # Run backtest for 12 weeks
    all_forecasts_h1 = []
    all_forecasts_h2 = []
    
    for epoch in range(12):
        # Get forecast date (start of each epoch)
        all_dates = sku_demand.index.sort_values()
        forecast_date = all_dates[-(12 - epoch)]
        
        # Get historical demand up to this point
        history = sku_demand[sku_demand.index <= forecast_date]
        
        # Generate forecast
        forecast_df = forecaster.predict(
            sku_id=sku_id,
            demand_history=history,
            forecast_date=forecast_date
        )
        
        # Convert to PMFs for h=3,4,5
        h3_pmf = None
        h4_pmf = None  
        h5_pmf = None
        
        for h in [3, 4, 5]:
            h_forecast = forecast_df[forecast_df['horizon'] == h]
            if len(h_forecast) > 0:
                pmf = quantiles_to_pmf(
                    h_forecast['quantile'].values,
                    h_forecast['value'].values,
                    sip_grain=500
                )
                if h == 3:
                    h3_pmf = pmf
                elif h == 4:
                    h4_pmf = pmf
                elif h == 5:
                    h5_pmf = pmf
        
        # Store PMFs for this epoch
        all_forecasts_h1.append(h3_pmf)
        all_forecasts_h2.append(h4_pmf)
    
    # Run backtest
    result = run_12week_backtest(
        store=store,
        product=product,
        forecasts_h1=all_forecasts_h1,
        forecasts_h2=all_forecasts_h2,
        initial_state=state,
        actual_demand=actual_demand_dict,
        costs=costs,
        order_method="simple_vn2_4week"
    )
    
    # Convert to DataFrame
    rows = []
    for week_result in result.weeks:
        rows.append({
            'store': store,
            'product': product,
            'week': week_result.week_num,
            'demand': week_result.actual_demand,
            'order': week_result.order_placed,
            'on_hand': week_result.ending_on_hand,
            'holding_cost': week_result.holding_cost,
            'shortage_cost': week_result.shortage_cost
        })
    
    return pd.DataFrame(rows)


def main():
    """Run VN2 benchmark evaluation."""
    
    logger.info("=" * 80)
    logger.info("VN2 BENCHMARK EVALUATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Approach: Seasonal 13-week MA + 4-week coverage order-up-to")
    logger.info("Target cost: €5,248")
    logger.info("")
    
    # Load configuration
    cfg = load_config()
    forecast_config = cfg['forecast']
    
    # Load demand data
    logger.info("Loading demand data...")
    demand_path = Path("data/processed/demand_imputed.parquet")
    demand_df = pd.read_parquet(demand_path)
    demand_df = demand_df.set_index(['store', 'product'])
    demand_df.columns = pd.to_datetime(demand_df.columns)
    demand_df = demand_df.sort_index()
    
    logger.info(f"Loaded demand for {len(demand_df)} SKUs")
    logger.info(f"Weeks: {demand_df.columns[0]} to {demand_df.columns[-1]}")
    
    # Load initial state
    logger.info("Loading initial state...")
    state_path = Path("Play VN2/Data/Week 0 - 2024-04-08 - Initial State.csv")
    initial_state_df = pd.read_csv(state_path)
    initial_state_df = initial_state_df.set_index(['Store', 'Product'])
    logger.info(f"Loaded initial state for {len(initial_state_df)} SKUs")
    
    # Initialize VN2 benchmark forecaster
    logger.info("")
    logger.info("Initializing VN2 Benchmark forecaster...")
    forecaster = VN2BenchmarkForecaster(forecast_config)
    logger.info("✓ VN2 Benchmark forecaster ready (no training required)")
    
    # Get SKU list
    skus = [(store, prod) for store, prod in demand_df.index]
    logger.info(f"")
    logger.info(f"Evaluating {len(skus)} SKUs...")
    logger.info("")
    
    # Run evaluation in parallel
    results = Parallel(n_jobs=11, verbose=10)(
        delayed(evaluate_sku)(
            sku_id=sku,
            demand_df=demand_df,
            initial_state=initial_state_df.loc[sku] if sku in initial_state_df.index else None,
            cfg=cfg,
            forecaster=forecaster
        )
        for sku in skus
    )
    
    # Combine results
    results_df = pd.concat(results, ignore_index=True)
    
    # Analyze results
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    
    total_cost = results_df[['holding_cost', 'shortage_cost']].sum().sum()
    total_holding = results_df['holding_cost'].sum()
    total_shortage = results_df['shortage_cost'].sum()
    
    logger.info(f"")
    logger.info(f"Total Cost:      €{total_cost:,.0f}")
    logger.info(f"  Holding Cost:  €{total_holding:,.0f} ({100*total_holding/total_cost:.1f}%)")
    logger.info(f"  Shortage Cost: €{total_shortage:,.0f} ({100*total_shortage/total_cost:.1f}%)")
    logger.info(f"")
    logger.info(f"VN2 Benchmark:   €5,248")
    logger.info(f"Gap:             €{total_cost - 5248:,.0f} ({100*(total_cost/5248 - 1):.1f}%)")
    logger.info("")
    
    if abs(total_cost - 5248) / 5248 < 0.05:
        logger.info("✓ SUCCESS! Reproduced VN2 benchmark within 5%")
        logger.info("✓ Framework validated - can proceed with confidence")
    elif abs(total_cost - 5248) / 5248 < 0.15:
        logger.info("⚠ CLOSE - Within 15% of benchmark")
        logger.info("  May have minor implementation differences")
    else:
        logger.info("❌ GAP TOO LARGE - Implementation issue detected")
        logger.info("  Need to debug framework")
    
    # Save results
    output_dir = Path("models/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "vn2_benchmark_results.parquet"
    results_df.to_parquet(output_file)
    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    
    # Summary statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("DETAILED BREAKDOWN")
    logger.info("=" * 80)
    
    # Per-SKU statistics
    sku_costs = results_df.groupby(['store', 'product'])[['holding_cost', 'shortage_cost']].sum()
    sku_costs['total_cost'] = sku_costs['holding_cost'] + sku_costs['shortage_cost']
    
    logger.info(f"")
    logger.info(f"Per-SKU Cost Distribution:")
    logger.info(f"  Mean:   €{sku_costs['total_cost'].mean():.2f}")
    logger.info(f"  Median: €{sku_costs['total_cost'].median():.2f}")
    logger.info(f"  Std:    €{sku_costs['total_cost'].std():.2f}")
    logger.info(f"  Min:    €{sku_costs['total_cost'].min():.2f}")
    logger.info(f"  Max:    €{sku_costs['total_cost'].max():.2f}")
    
    # Weekly breakdown
    weekly_costs = results_df.groupby('week')[['holding_cost', 'shortage_cost']].sum()
    weekly_costs['total_cost'] = weekly_costs['holding_cost'] + weekly_costs['shortage_cost']
    
    logger.info(f"")
    logger.info(f"Weekly Cost Breakdown:")
    for week in sorted(results_df['week'].unique()):
        wk_data = weekly_costs.loc[week]
        logger.info(f"  Week {week:2d}: €{wk_data['total_cost']:6,.0f} "
                   f"(H: €{wk_data['holding_cost']:5,.0f}, "
                   f"S: €{wk_data['shortage_cost']:5,.0f})")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    

if __name__ == "__main__":
    main()
