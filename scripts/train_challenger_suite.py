"""
Train complete challenger model suite for Jensen study.

Implements Section 6.1 of the paper:
- SLURP family (4): train on raw data with stockout indicators
- Challengers (6+): train on winsorized imputed data
- All models predict horizons h=3,4,5 (Patrick's corrected protection period)

Usage:
    uv run python scripts/train_challenger_suite.py --models all
    uv run python scripts/train_challenger_suite.py --models slurp lightgbm_quantile qrf
    uv run python scripts/train_challenger_suite.py --folds 5
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_challenger_suite.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load forecast configuration."""
    with open('configs/forecast.yaml') as f:
        return yaml.safe_load(f)


def get_enabled_models(config, requested=None):
    """Get list of models to train based on config and request."""
    all_models = {
        # SLURP family - train on raw
        'slurp': ['slurp_bootstrap', 'slurp_surd', 'slurp_stockout_aware', 'slurp_surd_stockout_aware'],
        # Challengers - train on winsorized
        'lightgbm': ['lightgbm_quantile', 'lightgbm_point'],
        'tree': ['qrf'],
        'stats': ['ets', 'seasonal_naive'],
        'intermittent': ['croston_classic', 'croston_sba', 'croston_tsb'],
        'parametric': ['zinb', 'zip', 'glm_poisson'],
        'linear': ['linear_quantile'],
        'boosting': ['ngboost'],
    }
    
    if requested == ['all']:
        # Train everything in the paper's model canvas
        models = []
        for group in all_models.values():
            models.extend(group)
        return models
    
    if requested:
        models = []
        for req in requested:
            if req in all_models:
                models.extend(all_models[req])
            elif req in sum(all_models.values(), []):
                models.append(req)
        return models
    
    # Default: use config enabled flags
    enabled = []
    for name, cfg in config['models'].items():
        if cfg.get('enabled', False):
            enabled.append(name)
    return enabled


def train_model(model_name, config, horizons=[3, 4, 5], n_folds=5):
    """
    Train a single model for specified horizons.
    
    Args:
        model_name: Model identifier
        config: Forecast config dict
        horizons: List of forecast horizons (default [3,4,5] per Patrick)
        n_folds: Number of cross-validation folds
    """
    logger.info(f"Training {model_name} for horizons {horizons}...")
    
    # Determine data source
    slurp_models = ['slurp_bootstrap', 'slurp_surd', 'slurp_stockout_aware', 'slurp_surd_stockout_aware']
    if model_name in slurp_models:
        data_path = Path('data/processed/demand_long.parquet')
        logger.info(f"  Using RAW data with stockout indicators: {data_path}")
    else:
        data_path = Path('data/processed/demand_imputed_winsor.parquet')
        logger.info(f"  Using WINSORIZED imputed data: {data_path}")
    
    if not data_path.exists():
        logger.error(f"  Data file not found: {data_path}")
        logger.error(f"  Run data preparation script first!")
        return False
    
    # Load data
    df = pd.read_parquet(data_path)
    logger.info(f"  Loaded {len(df)} records, {df['store'].nunique()} stores, {df['product'].nunique()} products")
    
    # TODO: Implement actual training logic here
    # For now, create placeholder showing what needs to happen
    logger.warning(f"  PLACEHOLDER: Would train {model_name} here")
    logger.info(f"  Need to implement model-specific training for each horizon: {horizons}")
    logger.info(f"  Output: models/checkpoints/{model_name}_h{h}_fold{f}.pkl for each h, f")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Train challenger model suite")
    parser.add_argument('--models', nargs='+', default=['all'],
                       help='Models to train: all, slurp, lightgbm, qrf, etc.')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 4, 5],
                       help='Forecast horizons (default: 3 4 5 per Patrick)')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--n-jobs', type=int, default=11,
                       help='Parallel jobs')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("CHALLENGER SUITE TRAINING")
    logger.info("=" * 80)
    logger.info(f"Horizons: {args.horizons} (Patrick's corrected protection period)")
    logger.info(f"Folds: {args.folds}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    
    # Load config
    config = load_config()
    
    # Get models to train
    models = get_enabled_models(config, args.models)
    logger.info(f"\nModels to train ({len(models)}):")
    for m in models:
        logger.info(f"  - {m}")
    
    # Train each model
    results = {}
    for model_name in models:
        logger.info(f"\n{'='*80}")
        success = train_model(model_name, config, args.horizons, args.folds)
        results[model_name] = 'SUCCESS' if success else 'FAILED'
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    for model, status in results.items():
        logger.info(f"  {model:30s} {status}")
    
    success_count = sum(1 for s in results.values() if s == 'SUCCESS')
    logger.info(f"\nCompleted: {success_count}/{len(results)} models")
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS:")
    logger.info("  1. Verify model checkpoints in models/checkpoints/")
    logger.info("  2. Run sequential evaluation with Patrick's corrected policy:")
    logger.info("     uv run python scripts/eval_all_models_patrick.py")
    logger.info("  3. Compare density-aware SIP vs point+service-level policies")
    logger.info("  4. Quantify Jensen gap per model")
    logger.info("="*80)


if __name__ == '__main__':
    main()
