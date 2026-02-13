"""
Test Ensemble Forecasting Approach

Blend model forecasts with recent average to reduce over-estimation:
    final_forecast = α × model_forecast + (1-α) × recent_avg

Tests different α values (0.3, 0.5, 0.7) to find optimal blend
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VN2Costs:
    """Cost structure for VN2 competition"""
    holding_cost: float = 0.2
    shortage_cost: float = 1.0


def load_vn2_data() -> pd.DataFrame:
    """Load VN2 competition data"""
    data_path = Path("data/processed/demand_imputed.parquet")
    return pd.read_parquet(data_path)


def load_model_forecasts(store: int, product: int, model: str = 'slurp_stockout_aware') -> Dict:
    """Load model forecasts for a SKU"""
    forecast_dir = Path(f"models/results/{model}")
    
    forecasts = {}
    for horizon in [3, 4, 5]:
        forecast_path = forecast_dir / f"quantiles_h{horizon}.parquet"
        if not forecast_path.exists():
            return None
            
        df = pd.read_parquet(forecast_path)
        sku_forecasts = df[(df['Store'] == store) & (df['Product'] == product)]
        
        if len(sku_forecasts) == 0:
            return None
            
        # Get median (q=0.50)
        forecasts[f'h{horizon}'] = sku_forecasts[sku_forecasts['quantile'] == 0.50]['value'].values[0]
    
    return forecasts


def get_recent_average(historical_demand: np.ndarray, weeks: int = 13) -> float:
    """Compute recent average demand"""
    if len(historical_demand) < weeks:
        return np.mean(historical_demand) if len(historical_demand) > 0 else 0.0
    return np.mean(historical_demand[-weeks:])


def ensemble_forecast(model_forecast: float, recent_avg: float, alpha: float) -> float:
    """
    Blend model forecast with recent average
    
    Args:
        model_forecast: Model's median forecast
        recent_avg: Recent 13-week average
        alpha: Weight for model (1-alpha for recent average)
        
    Returns:
        Blended forecast
    """
    return alpha * model_forecast + (1 - alpha) * recent_avg


def simple_order_policy(forecast: float, position: int, coverage_weeks: int = 3) -> int:
    """Simple order-up-to policy"""
    order_up_to = coverage_weeks * forecast
    return int(max(0, np.round(order_up_to - position)))


def run_ensemble_backtest(
    df: pd.DataFrame,
    costs: VN2Costs,
    alpha: float,
    model: str = 'slurp_stockout_aware',
    coverage_weeks: int = 3
) -> pd.DataFrame:
    """Run sequential backtest with ensemble forecasts"""
    
    results = []
    skus = df.groupby(['Store', 'Product']).size().index.tolist()
    
    logger.info(f"Running ensemble backtest (α={alpha}) for {len(skus)} SKUs")
    
    for idx, (store, product) in enumerate(skus, 1):
        if idx % 100 == 0:
            logger.info(f"  Processing SKU {idx}/{len(skus)}")
        
        # Load model forecasts
        model_forecasts = load_model_forecasts(store, product, model)
        if model_forecasts is None:
            continue
        
        # Get SKU data
        sku_data = df[(df['Store'] == store) & (df['Product'] == product)].copy()
        sku_data = sku_data.sort_values('week').reset_index(drop=True)
        
        # Initialize state
        initial_idx = sku_data[sku_data['week'] == 78].index
        if len(initial_idx) == 0:
            continue
            
        initial_row = sku_data.iloc[initial_idx[0]]
        inventory = int(initial_row['on_hand'])
        transit = [int(initial_row['in_transit_W+1']), int(initial_row['in_transit_W+2'])]
        
        total_cost = 0.0
        orders = []
        demands = []
        
        # Run 12-week backtest
        for week_idx in range(12):
            current_week = 78 + week_idx
            
            # Get historical demand
            historical_data = sku_data[sku_data['week'] < current_week]
            if len(historical_data) == 0:
                break
            historical_demand = historical_data['demand'].values
            
            # Compute recent average
            recent_avg = get_recent_average(historical_demand, weeks=13)
            
            # Ensemble forecast: blend model with recent average
            # Use average of h3, h4, h5 model forecasts
            model_forecast = np.mean([model_forecasts['h3'], model_forecasts['h4'], model_forecasts['h5']])
            final_forecast = ensemble_forecast(model_forecast, recent_avg, alpha)
            
            # Order decision
            position = inventory + sum(transit)
            if week_idx < 10:
                order = simple_order_policy(final_forecast, position, coverage_weeks)
            else:
                order = 0
            orders.append(order)
            
            # Receive arrivals
            if len(transit) >= 2:
                inventory += transit.pop(0)
            
            # Observe demand
            demand_row = sku_data[sku_data['week'] == current_week]
            if len(demand_row) == 0:
                break
            actual_demand = int(demand_row.iloc[0]['demand'])
            demands.append(actual_demand)
            
            # Satisfy demand
            shortage = max(0, actual_demand - inventory)
            inventory = max(0, inventory - actual_demand)
            
            # Calculate costs
            total_cost += inventory * costs.holding_cost + shortage * costs.shortage_cost
            
            # Add order to transit
            if week_idx < 10:
                transit.append(order)
        
        results.append({
            'Store': store,
            'Product': product,
            'total_cost': total_cost,
            'total_demand': sum(demands),
            'total_ordered': sum(orders),
            'alpha': alpha
        })
    
    return pd.DataFrame(results)


def main():
    """Test different ensemble weights"""
    
    logger.info("Starting Ensemble Forecast Test")
    logger.info("="*60)
    
    # Load data
    df = load_vn2_data()
    costs = VN2Costs()
    
    # Test different alpha values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    all_results = []
    
    for alpha in alphas:
        logger.info(f"\nTesting α={alpha:.1f} (model weight)")
        logger.info(f"  Forecast = {alpha:.1f}×model + {1-alpha:.1f}×recent_avg")
        
        results_df = run_ensemble_backtest(
            df=df,
            costs=costs,
            alpha=alpha,
            coverage_weeks=3
        )
        
        total_cost = results_df['total_cost'].sum()
        logger.info(f"  Total Cost: €{total_cost:,.2f}")
        logger.info(f"  vs VN2 benchmark: {(total_cost/5248 - 1)*100:+.1f}%")
        
        all_results.append({
            'alpha': alpha,
            'total_cost': total_cost,
            'num_skus': len(results_df),
            'avg_cost': total_cost / len(results_df)
        })
    
    # Summary
    summary_df = pd.DataFrame(all_results)
    
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE FORECAST RESULTS SUMMARY")
    logger.info("="*60)
    logger.info("\nCost by Ensemble Weight (α):")
    logger.info(f"{'α':<8} {'Model%':<10} {'Recent%':<10} {'Cost':<12} {'vs VN2':<10}")
    logger.info("-"*60)
    
    for _, row in summary_df.iterrows():
        alpha = row['alpha']
        cost = row['total_cost']
        vs_vn2 = (cost / 5248 - 1) * 100
        logger.info(f"{alpha:<8.1f} {alpha*100:<10.0f} {(1-alpha)*100:<10.0f} €{cost:<11,.0f} {vs_vn2:+6.1f}%")
    
    # Find best
    best_idx = summary_df['total_cost'].idxmin()
    best_alpha = summary_df.loc[best_idx, 'alpha']
    best_cost = summary_df.loc[best_idx, 'total_cost']
    
    logger.info("\n" + "="*60)
    logger.info(f"✅ Best Result: α={best_alpha:.1f}")
    logger.info(f"   Cost: €{best_cost:,.2f}")
    logger.info(f"   vs VN2: {(best_cost/5248 - 1)*100:+.1f}%")
    logger.info(f"   vs Baseline (€9,950): {(best_cost/9950 - 1)*100:+.1f}%")
    logger.info("="*60)
    
    # Save results
    output_path = Path("models/results/ensemble_forecast_summary.parquet")
    summary_df.to_parquet(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
