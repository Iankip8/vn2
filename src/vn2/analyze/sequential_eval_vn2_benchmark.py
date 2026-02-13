"""
VN2 Benchmark Sequential Evaluation

Evaluates the VN2 benchmark approach using the existing sequential backtest framework.
Target: Reproduce €5,248 cost from competition benchmark.
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from rich import print as rprint

from vn2.forecast.models.vn2_benchmark import VN2BenchmarkForecaster
from vn2.analyze.sequential_backtest import (
    run_12week_backtest,
    quantiles_to_pmf,
    BacktestState
)
from vn2.analyze.sequential_planner import Costs


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_sku(
    sku_id: tuple,
    demand_df: pd.DataFrame,
    initial_state: pd.Series,
    holdout_weeks: int
) -> pd.DataFrame:
    """Evaluate VN2 benchmark for one SKU."""
    store, product = sku_id
    
    # Get SKU demand
    sku_demand = demand_df.loc[sku_id]
    
    # Initial state
    try:
        state = BacktestState(
            week=0,
            on_hand=int(initial_state['End Inventory']),
            intransit_1=int(initial_state['In Transit W+1']),
            intransit_2=int(initial_state['In Transit W+2'])
        )
    except:
        state = BacktestState(week=0, on_hand=0, intransit_1=0, intransit_2=0)
    
    # Get actual demand
    all_dates = sku_demand.index.sort_values()
    backtest_dates = all_dates[-holdout_weeks:]
    actual_demand_dict = {i+1: sku_demand[date] for i, date in enumerate(backtest_dates)}
    
    # Create forecaster
    forecast_config = type('obj', (object,), {
        'horizon': 12,
        'quantiles': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    })()
    forecaster = VN2BenchmarkForecaster(forecast_config)
    
    # Generate forecasts for each epoch
    all_forecasts_h1 = []
    all_forecasts_h2 = []
    
    for epoch in range(holdout_weeks):
        forecast_date = all_dates[-(holdout_weeks - epoch)]
        history = sku_demand[sku_demand.index <= forecast_date]
        
        forecast_df = forecaster.predict(
            sku_id=sku_id,
            demand_history=history,
            forecast_date=forecast_date
        )
        
        # Convert to PMFs for h=3,4
        h3_pmf = None
        h4_pmf = None
        
        for h in [3, 4]:
            h_forecast = forecast_df[forecast_df['horizon'] == h]
            if len(h_forecast) > 0:
                pmf = quantiles_to_pmf(
                    quantiles=h_forecast['value'].values,
                    quantile_levels=h_forecast['quantile'].values,
                    grain=500
                )
                if h == 3:
                    h3_pmf = pmf
                else:
                    h4_pmf = pmf
        
        all_forecasts_h1.append(h3_pmf)
        all_forecasts_h2.append(h4_pmf)
    
    # Run backtest
    costs = Costs(holding=0.2, shortage=1.0)
    result = run_12week_backtest(
        store=store,
        product=product,
        model_name="vn2_benchmark",
        forecasts_h1=all_forecasts_h1,
        forecasts_h2=all_forecasts_h2,
        forecasts_h3=None,
        actuals=list(actual_demand_dict.values()),
        initial_state=state,
        costs=costs,
        pmf_grain=500
    )
    
    # Convert to DataFrame
    rows = []
    for week_result in result.weeks:
        # Calculate holding and shortage costs from realized cost
        # (realized_cost includes both holding and shortage)
        if week_result.realized_cost is not None:
            rows.append({
                'store': store,
                'product': product,
                'week': week_result.week,
                'demand': week_result.demand_actual if week_result.demand_actual is not None else 0,
                'order': week_result.order_placed,
                'on_hand': week_result.state_after.on_hand if week_result.state_after else 0,
                'holding_cost': 0,  # Will calculate from realized_cost
                'shortage_cost': 0,  # Will calculate from realized_cost
                'total_cost': week_result.realized_cost
            })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate VN2 Benchmark')
    parser.add_argument('--demand', required=True, help='Path to demand parquet')
    parser.add_argument('--state', required=True, help='Path to initial state CSV')
    parser.add_argument('--output', required=True, help='Path to output parquet')
    parser.add_argument('--n-jobs', type=int, default=11, help='Number of parallel jobs')
    parser.add_argument('--holdout', type=int, default=12, help='Holdout weeks')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("VN2 BENCHMARK EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Demand: {args.demand}")
    logger.info(f"State: {args.state}")
    logger.info(f"Holdout weeks: {args.holdout}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    logger.info("")
    
    # Load data
    logger.info("Loading data...")
    demand_long = pd.read_parquet(args.demand)
    
    # Convert to wide format (pivot)
    logger.info("Converting to wide format...")
    demand_df = demand_long.pivot(
        index=['Store', 'Product'],
        columns='week_date',
        values='sales'
    )
    demand_df.index.names = ['store', 'product']
    demand_df.columns = pd.to_datetime(demand_df.columns)
    demand_df = demand_df.sort_index()
    
    state_df = pd.read_csv(args.state)
    state_df = state_df.set_index(['Store', 'Product'])
    state_df.index.names = ['store', 'product']
    
    logger.info(f"Loaded {len(demand_df)} SKUs")
    
    # Evaluate
    logger.info("Running evaluation...")
    skus = [(s, p) for s, p in demand_df.index]
    
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_sku)(
            sku_id=sku,
            demand_df=demand_df,
            initial_state=state_df.loc[sku] if sku in state_df.index else None,
            holdout_weeks=args.holdout
        )
        for sku in skus
    )
    
    # Combine and analyze
    results_df = pd.concat(results, ignore_index=True)
    
    total_cost = results_df['total_cost'].sum()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Cost:      €{total_cost:,.0f}")
    logger.info(f"")
    logger.info(f"VN2 Benchmark:   €5,248")
    logger.info(f"Gap:             €{total_cost - 5248:,.0f} ({100*(total_cost/5248 - 1):.1f}%)")
    
    if abs(total_cost - 5248) / 5248 < 0.05:
        logger.info("")
        logger.info("✓ SUCCESS! Reproduced VN2 benchmark within 5%")
    elif abs(total_cost - 5248) / 5248 < 0.15:
        logger.info("")
        logger.info("⚠ CLOSE - Within 15% of benchmark")
    else:
        logger.info("")
        logger.info("❌ GAP TOO LARGE - Implementation differs")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)
    logger.info("")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
