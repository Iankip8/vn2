"""
Forecast loader for sequential backtest.

Loads forecast checkpoints and converts quantiles to PMFs.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from .sequential_backtest import quantiles_to_pmf


def load_forecasts_for_sku(
    store: int,
    product: int,
    model_name: str,
    checkpoints_dir: Path,
    n_folds: int = 12,
    quantile_levels: Optional[np.ndarray] = None,
    pmf_grain: int = 500,
    bias_correction: float = 1.0  # NO bias correction by default (was 0.747)
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    Load h3, h4, h5 PMFs for a SKU from forecast checkpoints.
    
    Args:
        store: Store ID
        product: Product ID
        model_name: Model name (e.g., 'zinb', 'qrf', 'naive4')
        checkpoints_dir: Path to checkpoints directory
        n_folds: Number of folds (weeks) to load
        quantile_levels: Quantile levels (default: [0.01, 0.05, ..., 0.99])
        pmf_grain: PMF support size
        bias_correction: Multiplicative bias correction for systematic demand decline
    
    Returns:
        (h3_pmfs, h4_pmfs, h5_pmfs): Lists of PMFs for horizons 3, 4, 5 (or None if missing)
    """
    if quantile_levels is None:
        quantile_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    model_dir = checkpoints_dir / model_name
    sku_dir = model_dir / f"{store}_{product}"
    
    if not sku_dir.exists():
        # SKU not found: return all None
        return [None] * n_folds, [None] * n_folds, [None] * n_folds
    
    h3_pmfs = []
    h4_pmfs = []
    h5_pmfs = []
    
    for fold_idx in range(n_folds):
        fold_file = sku_dir / f"fold_{fold_idx}.pkl"
        
        if not fold_file.exists():
            h3_pmfs.append(None)
            h4_pmfs.append(None)
            h5_pmfs.append(None)
            continue
        
        try:
            with open(fold_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Extract quantiles DataFrame
            quantiles_df = checkpoint.get('quantiles')
            
            if quantiles_df is None or len(quantiles_df) < 3:
                h3_pmfs.append(None)
                h4_pmfs.append(None)
                h5_pmfs.append(None)
                continue
            
            # Apply bias correction for systematic demand decline
            # 80.5% of SKUs have declining demand (recent vs historical)
            # Median ratio: 0.747
            quantiles_df = quantiles_df * bias_correction
            
            # Convert h=3, h=4, h=5 to PMFs (3-week protection period)
            q3 = quantiles_df.loc[3].values if 3 in quantiles_df.index else None
            q4 = quantiles_df.loc[4].values if 4 in quantiles_df.index else None
            q5 = quantiles_df.loc[5].values if 5 in quantiles_df.index else None
            
            if q3 is not None and len(q3) == len(quantile_levels):
                pmf3 = quantiles_to_pmf(q3, quantile_levels, pmf_grain)
                h3_pmfs.append(pmf3)
            else:
                h3_pmfs.append(None)
            
            if q4 is not None and len(q4) == len(quantile_levels):
                pmf4 = quantiles_to_pmf(q4, quantile_levels, pmf_grain)
                h4_pmfs.append(pmf4)
            else:
                h4_pmfs.append(None)
            
            if q5 is not None and len(q5) == len(quantile_levels):
                pmf5 = quantiles_to_pmf(q5, quantile_levels, pmf_grain)
                h5_pmfs.append(pmf5)
            else:
                h5_pmfs.append(None)
            
        except Exception as e:
            print(f"Warning: Failed to load {fold_file}: {e}")
            h3_pmfs.append(None)
            h4_pmfs.append(None)
            h5_pmfs.append(None)
    
    return h3_pmfs, h4_pmfs, h5_pmfs


def get_available_models(checkpoints_dir: Path) -> List[str]:
    """
    Get list of available model names in checkpoints directory.
    
    Args:
        checkpoints_dir: Path to checkpoints directory
    
    Returns:
        List of model names
    """
    if not checkpoints_dir.exists():
        return []
    
    models = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
            # Check if it has any SKU subdirectories
            sku_dirs = [d for d in item.iterdir() if d.is_dir() and '_' in d.name]
            if sku_dirs:
                models.append(item.name)
    
    return sorted(models)


def get_available_skus(checkpoints_dir: Path, model_name: str) -> List[Tuple[int, int]]:
    """
    Get list of available SKUs for a model.
    
    Args:
        checkpoints_dir: Path to checkpoints directory
        model_name: Model name
    
    Returns:
        List of (store, product) tuples
    """
    model_dir = checkpoints_dir / model_name
    if not model_dir.exists():
        return []
    
    skus = []
    for sku_dir in model_dir.iterdir():
        if sku_dir.is_dir() and '_' in sku_dir.name:
            try:
                store, product = sku_dir.name.split('_')
                skus.append((int(store), int(product)))
            except ValueError:
                continue
    
    return sorted(skus)

