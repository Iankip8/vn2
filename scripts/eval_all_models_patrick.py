"""
Evaluate all trained models using Patrick's corrected policy.

Implements Section 6.1 evaluation matrix:
1. Load forecasts for each model (h=3,4,5 quantiles)
2. For EACH model, evaluate under TWO policies:
   a) Density-aware SIP (Patrick's MC aggregation)
   b) Point + service-level (traditional approach)
3. Compute Jensen gap: Δcost = (point policy) - (SIP)
4. Generate leaderboards and cohort analyses

Usage:
    uv run python scripts/eval_all_models_patrick.py
    uv run python scripts/eval_all_models_patrick.py --models slurp_surd lightgbm_quantile
    uv run python scripts/eval_all_models_patrick.py --output results_v5.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
from math import ceil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/eval_all_models_patrick.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Costs:
    """VN2 competition cost structure."""
    holding: float = 0.20  # €0.20 per unit held
    shortage: float = 1.00  # €1.00 per unit short
    
    @property
    def critical_fractile(self) -> float:
        """Newsvendor critical fractile τ = c_s/(c_h + c_s)."""
        return self.shortage / (self.holding + self.shortage)


def aggregate_weekly_distributions_mc(
    quantiles_df: pd.DataFrame,
    quantile_levels: List[float],
    protection_weeks: int = 3,
    lead_weeks: int = 2,
    n_samples: int = 10000
) -> Tuple[float, float]:
    """
    Patrick's Fix #3: MC aggregation over 3-week protection period.
    
    Args:
        quantiles_df: DataFrame with horizons [3,4,5] as index, quantiles as columns
        quantile_levels: List of quantile probabilities
        protection_weeks: 3 (L+R weeks)
        lead_weeks: 2 (lead time)
        n_samples: 10,000 MC samples
    
    Returns:
        (mu, sigma): Mean and std of aggregated demand
    """
    horizons = list(range(lead_weeks + 1, lead_weeks + protection_weeks + 1))
    samples = []
    
    for _ in range(n_samples):
        total = 0
        for h in horizons:
            # Sample from this week's distribution
            q_vals = quantiles_df.loc[h].values
            u = np.random.rand()
            demand = np.interp(u, quantile_levels, q_vals)
            total += demand
        samples.append(total)
    
    return np.mean(samples), np.std(samples)


def evaluate_sku_density_sip(
    quantiles_h3: np.ndarray,
    quantiles_h4: np.ndarray,
    quantiles_h5: np.ndarray,
    quantile_levels: List[float],
    costs: Costs,
    actual_demand: float,
    current_position: float = 0
) -> Dict:
    """
    Evaluate SKU using Patrick's corrected density-aware SIP policy.
    
    This is the CORRECT implementation with all 3 fixes:
    1. Protection period = 3 weeks (h=3,4,5)
    2. Critical fractile τ = 0.833 explicit
    3. MC aggregation over 3 weeks
    """
    # Build quantiles DataFrame in CORRECT format (horizons as index)
    quantiles_df = pd.DataFrame({
        q: [quantiles_h3[i], quantiles_h4[i], quantiles_h5[i]]
        for i, q in enumerate(quantile_levels)
    }, index=[3, 4, 5])
    
    # Patrick's MC aggregation
    mu, sigma = aggregate_weekly_distributions_mc(
        quantiles_df, quantile_levels,
        protection_weeks=3, lead_weeks=2, n_samples=10000
    )
    
    # Newsvendor with explicit critical fractile
    tau = costs.critical_fractile  # 0.833
    z = stats.norm.ppf(tau)
    S = mu + z * sigma  # Base stock level
    
    # Order quantity (integer)
    order = max(0, ceil(S - current_position))
    
    # Realized cost
    ending_inventory = current_position + order - actual_demand
    if ending_inventory >= 0:
        cost = costs.holding * ending_inventory
    else:
        cost = costs.shortage * abs(ending_inventory)
    
    return {
        'order': order,
        'base_stock': S,
        'mu': mu,
        'sigma': sigma,
        'tau': tau,
        'cost': cost,
        'policy': 'density_sip'
    }


def evaluate_sku_point_service(
    quantiles_h3: np.ndarray,
    quantiles_h4: np.ndarray,
    quantiles_h5: np.ndarray,
    quantile_levels: List[float],
    costs: Costs,
    actual_demand: float,
    current_position: float = 0
) -> Dict:
    """
    Evaluate SKU using traditional point + service-level policy.
    
    This represents common industry practice:
    - Use median as point forecast
    - Target critical fractile with Normal approximation
    - No MC aggregation (just sum means and vars)
    """
    # Point forecasts (median = q0.5)
    median_idx = quantile_levels.index(0.5)
    med_h3 = quantiles_h3[median_idx]
    med_h4 = quantiles_h4[median_idx]
    med_h5 = quantiles_h5[median_idx]
    
    # Simple aggregation (assume independence - WRONG but common)
    mu_point = med_h3 + med_h4 + med_h5
    
    # Estimate variance from quantile spread (rough approx)
    q25_idx = quantile_levels.index(0.2) if 0.2 in quantile_levels else quantile_levels.index(0.3)
    q75_idx = quantile_levels.index(0.8) if 0.8 in quantile_levels else quantile_levels.index(0.7)
    
    iqr_h3 = quantiles_h3[q75_idx] - quantiles_h3[q25_idx]
    iqr_h4 = quantiles_h4[q75_idx] - quantiles_h4[q25_idx]
    iqr_h5 = quantiles_h5[q75_idx] - quantiles_h5[q25_idx]
    
    # IQR ≈ 1.349σ for normal
    sigma_h3 = iqr_h3 / 1.349
    sigma_h4 = iqr_h4 / 1.349
    sigma_h5 = iqr_h5 / 1.349
    
    # Sum variances (assume independence)
    sigma_point = np.sqrt(sigma_h3**2 + sigma_h4**2 + sigma_h5**2)
    
    # Newsvendor
    tau = costs.critical_fractile
    z = stats.norm.ppf(tau)
    S = mu_point + z * sigma_point
    
    # Order
    order = max(0, ceil(S - current_position))
    
    # Realized cost
    ending_inventory = current_position + order - actual_demand
    if ending_inventory >= 0:
        cost = costs.holding * ending_inventory
    else:
        cost = costs.shortage * abs(ending_inventory)
    
    return {
        'order': order,
        'base_stock': S,
        'mu': mu_point,
        'sigma': sigma_point,
        'tau': tau,
        'cost': cost,
        'policy': 'point_service'
    }


def evaluate_model(model_name: str, quantile_levels: List[float], costs: Costs):
    """
    Evaluate a single model under BOTH policies.
    
    Returns DataFrame with columns:
        store, product, policy, cost, order, base_stock, mu, sigma
    """
    logger.info(f"\nEvaluating {model_name}...")
    
    # Load model forecasts
    forecast_path = Path(f'models/results/{model_name}_quantiles.parquet')
    if not forecast_path.exists():
        logger.warning(f"  Forecasts not found: {forecast_path}")
        logger.warning(f"  Skipping {model_name}")
        return None
    
    forecasts = pd.read_parquet(forecast_path)
    logger.info(f"  Loaded {len(forecasts)} SKU forecasts")
    
    # Load actual demand
    actuals = pd.read_parquet('data/processed/demand_long.parquet')
    eval_week = actuals['week'].max()  # Use latest week for evaluation
    actuals_eval = actuals[actuals['week'] == eval_week][['store', 'product', 'demand']]
    
    results = []
    
    for _, row in forecasts.iterrows():
        store, product = row['store'], row['product']
        
        # Get quantiles for h=3,4,5
        q3 = [row[f'q{int(q*100)}_h3'] for q in quantile_levels]
        q4 = [row[f'q{int(q*100)}_h4'] for q in quantile_levels]
        q5 = [row[f'q{int(q*100)}_h5'] for q in quantile_levels]
        
        # Actual demand (3-week total)
        actual_total = actuals_eval[
            (actuals_eval['store'] == store) & (actuals_eval['product'] == product)
        ]['demand'].sum()
        
        # Evaluate under density-aware SIP
        result_sip = evaluate_sku_density_sip(
            np.array(q3), np.array(q4), np.array(q5),
            quantile_levels, costs, actual_total
        )
        result_sip.update({'store': store, 'product': product, 'model': model_name})
        results.append(result_sip)
        
        # Evaluate under point + service-level
        result_point = evaluate_sku_point_service(
            np.array(q3), np.array(q4), np.array(q5),
            quantile_levels, costs, actual_total
        )
        result_point.update({'store': store, 'product': product, 'model': model_name})
        results.append(result_point)
    
    return pd.DataFrame(results)


def compute_jensen_gap(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Jensen gap per model: Δ = cost(point) - cost(SIP).
    
    Positive values mean density-aware SIP is better.
    """
    jensen_gaps = []
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        for store, product in model_data[['store', 'product']].drop_duplicates().values:
            sku_data = model_data[
                (model_data['store'] == store) & (model_data['product'] == product)
            ]
            
            cost_sip = sku_data[sku_data['policy'] == 'density_sip']['cost'].values[0]
            cost_point = sku_data[sku_data['policy'] == 'point_service']['cost'].values[0]
            
            jensen_gap = cost_point - cost_sip  # Positive = SIP wins
            
            jensen_gaps.append({
                'model': model,
                'store': store,
                'product': product,
                'cost_sip': cost_sip,
                'cost_point': cost_point,
                'jensen_gap': jensen_gap,
                'jensen_gap_pct': 100 * jensen_gap / max(cost_point, 0.01)
            })
    
    return pd.DataFrame(jensen_gaps)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models with Patrick's policy")
    parser.add_argument('--models', nargs='+', default=None,
                       help='Models to evaluate (default: all in results/)')
    parser.add_argument('--output', default='models/results/eval_patrick_all_models.parquet',
                       help='Output file for detailed results')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PATRICK'S CORRECTED POLICY - ALL MODELS EVALUATION")
    logger.info("=" * 80)
    logger.info("Protection period: 3 weeks (h=3,4,5)")
    logger.info("Critical fractile: τ = 0.833")
    logger.info("Aggregation: Monte Carlo (10,000 samples)")
    logger.info("Policies: Density-aware SIP vs Point+Service-Level")
    logger.info("=" * 80)
    
    costs = Costs()
    quantile_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Get models to evaluate
    if args.models:
        models = args.models
    else:
        # Find all forecast files
        results_dir = Path('models/results')
        forecast_files = list(results_dir.glob('*_quantiles.parquet'))
        models = [f.stem.replace('_quantiles', '') for f in forecast_files]
    
    logger.info(f"\nModels to evaluate ({len(models)}):")
    for m in models:
        logger.info(f"  - {m}")
    
    # Evaluate each model
    all_results = []
    for model in models:
        df = evaluate_model(model, quantile_levels, costs)
        if df is not None:
            all_results.append(df)
    
    if not all_results:
        logger.error("No models evaluated successfully!")
        return 1
    
    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"\nTotal evaluations: {len(results_df)}")
    
    # Compute Jensen gaps
    jensen_df = compute_jensen_gap(results_df)
    
    # Leaderboard by policy
    logger.info("\n" + "=" * 80)
    logger.info("LEADERBOARD - DENSITY-AWARE SIP (Patrick's corrected policy)")
    logger.info("=" * 80)
    sip_leaderboard = results_df[results_df['policy'] == 'density_sip'].groupby('model').agg({
        'cost': ['sum', 'mean', 'std', 'count']
    }).round(2)
    sip_leaderboard.columns = ['total_cost', 'mean_cost', 'std_cost', 'n_skus']
    sip_leaderboard = sip_leaderboard.sort_values('total_cost')
    print(sip_leaderboard)
    
    logger.info("\n" + "=" * 80)
    logger.info("LEADERBOARD - POINT + SERVICE-LEVEL (traditional)")
    logger.info("=" * 80)
    point_leaderboard = results_df[results_df['policy'] == 'point_service'].groupby('model').agg({
        'cost': ['sum', 'mean', 'std', 'count']
    }).round(2)
    point_leaderboard.columns = ['total_cost', 'mean_cost', 'std_cost', 'n_skus']
    point_leaderboard = point_leaderboard.sort_values('total_cost')
    print(point_leaderboard)
    
    logger.info("\n" + "=" * 80)
    logger.info("JENSEN GAP SUMMARY (positive = SIP wins)")
    logger.info("=" * 80)
    jensen_summary = jensen_df.groupby('model').agg({
        'jensen_gap': ['sum', 'mean', 'median', 'std'],
        'jensen_gap_pct': ['mean', 'median']
    }).round(2)
    jensen_summary.columns = ['total_gap', 'mean_gap', 'median_gap', 'std_gap', 'mean_pct', 'median_pct']
    jensen_summary = jensen_summary.sort_values('total_gap', ascending=False)
    print(jensen_summary)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved detailed results: {output_path}")
    
    jensen_path = output_path.parent / output_path.name.replace('.parquet', '_jensen.parquet')
    jensen_df.to_parquet(jensen_path, index=False)
    logger.info(f"Saved Jensen gaps: {jensen_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info("  1. Analyze cohort performance (by sparsity, stockout rate, etc.)")
    logger.info("  2. Plot Jensen gap distributions")
    logger.info("  3. Test hypotheses H1-H4 from paper")
    logger.info("  4. Update paper Section 7 with results")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
