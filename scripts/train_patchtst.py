#!/usr/bin/env python3
"""
Train PatchTST (Patch Time Series Transformer) model.

PatchTST is a transformer-based architecture that:
1. Patches time series into subseries-level segments
2. Uses channel-independence for efficiency
3. Provides strong performance on various forecasting benchmarks

References:
- Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

Requirements:
- pytorch
- neuralforecast (nixtla's implementation)

Usage:
    python scripts/train_patchtst.py \
        --demand-path data/processed/demand_long.parquet \
        --output-root models/checkpoints_h3 \
        --max-epochs 50 \
        --patch-len 8

Note: GPU is strongly recommended for training.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])

# Check for NeuralForecast availability
NEURALFORECAST_AVAILABLE = False
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.losses.pytorch import MQLoss
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    pass


def load_demand(path: Path) -> pd.DataFrame:
    """Load and prepare demand data for NeuralForecast."""
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'demand' not in df.columns and 'sales' in df.columns:
        df['demand'] = df['sales']
    
    # NeuralForecast expects specific column names
    df['unique_id'] = df['store'].astype(str) + '_' + df['product'].astype(str)
    
    # Rename for NeuralForecast format
    df = df.rename(columns={'week': 'ds', 'demand': 'y'})
    
    # Ensure ds is datetime
    if df['ds'].dtype != 'datetime64[ns]':
        df['ds'] = pd.to_datetime(df['ds'])
    
    df = df.sort_values(['unique_id', 'ds'])
    
    return df[['unique_id', 'ds', 'y']]


def train_patchtst(
    df: pd.DataFrame,
    output_dir: Path,
    horizon: int = 3,
    input_size: int = 52,
    patch_len: int = 8,
    stride: int = 4,
    hidden_size: int = 64,
    n_heads: int = 4,
    encoder_layers: int = 2,
    max_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    gpus: int = 0
) -> Dict:
    """Train PatchTST model."""
    
    # Create PatchTST model with quantile loss for probabilistic forecasts
    model = PatchTST(
        h=horizon,
        input_size=input_size,
        patch_len=patch_len,
        stride=stride,
        hidden_size=hidden_size,
        n_heads=n_heads,
        encoder_layers=encoder_layers,
        max_steps=max_epochs * (len(df) // batch_size + 1),
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss=MQLoss(level=list((QUANTILE_LEVELS * 100).astype(int))),
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        random_seed=42,
    )
    
    # Create NeuralForecast wrapper
    nf = NeuralForecast(
        models=[model],
        freq='W'
    )
    
    print("Training PatchTST...")
    nf.fit(df=df)
    
    # Save model
    model_path = output_dir / 'patchtst_model'
    nf.save(path=str(model_path), overwrite=True)
    
    return {
        'model': nf,
        'model_path': str(model_path)
    }


def generate_forecasts_and_save(
    nf,
    df: pd.DataFrame,
    output_dir: Path,
    horizon: int = 3
) -> int:
    """Generate forecasts and save as checkpoints."""
    
    print("Generating forecasts...")
    forecasts = nf.predict(df=df)
    
    # NeuralForecast returns forecasts with quantile columns like 'PatchTST-lo-5', 'PatchTST-hi-95'
    # We need to convert to our standard format
    
    n_saved = 0
    unique_ids = df['unique_id'].unique()
    
    for uid in unique_ids:
        store, product = map(int, uid.split('_'))
        
        sku_dir = output_dir / 'patchtst' / f'{store}_{product}'
        sku_dir.mkdir(parents=True, exist_ok=True)
        
        # Get forecasts for this SKU
        sku_fcst = forecasts[forecasts['unique_id'] == uid]
        
        if len(sku_fcst) == 0:
            continue
        
        # Extract quantiles and reshape to (horizon, n_quantiles)
        # Column names depend on the loss function configuration
        quantile_cols = [c for c in sku_fcst.columns if 'PatchTST' in c]
        
        # Build quantiles DataFrame
        # For each horizon step, extract the quantile values
        quantiles_data = {}
        for h in range(1, horizon + 1):
            if h <= len(sku_fcst):
                row_vals = []
                for q in QUANTILE_LEVELS:
                    # Try to find matching column
                    q_pct = int(q * 100)
                    col_lo = f'PatchTST-lo-{q_pct}'
                    col_hi = f'PatchTST-hi-{q_pct}'
                    col_median = 'PatchTST-median'
                    
                    if col_lo in sku_fcst.columns:
                        row_vals.append(sku_fcst[col_lo].iloc[h-1])
                    elif col_hi in sku_fcst.columns:
                        row_vals.append(sku_fcst[col_hi].iloc[h-1])
                    elif col_median in sku_fcst.columns and q == 0.5:
                        row_vals.append(sku_fcst[col_median].iloc[h-1])
                    else:
                        # Fallback: use mean if available
                        mean_col = [c for c in sku_fcst.columns if 'mean' in c.lower()]
                        if mean_col:
                            row_vals.append(sku_fcst[mean_col[0]].iloc[h-1])
                        else:
                            row_vals.append(0.0)
                
                quantiles_data[h] = row_vals
        
        if not quantiles_data:
            continue
        
        quantiles_df = pd.DataFrame(quantiles_data).T
        quantiles_df.columns = QUANTILE_LEVELS
        quantiles_df.index.name = 'step'
        
        # Ensure monotonicity and non-negativity
        for idx in quantiles_df.index:
            row = quantiles_df.loc[idx].values
            row = np.maximum.accumulate(row)
            row = np.maximum(row, 0)
            quantiles_df.loc[idx] = row
        
        # Save for each fold
        for fold_idx in range(12):
            path = sku_dir / f'fold_{fold_idx}.pkl'
            with open(path, 'wb') as f:
                pickle.dump({'quantiles': quantiles_df}, f)
        
        n_saved += 1
    
    return n_saved


def main():
    parser = argparse.ArgumentParser(description="Train PatchTST forecasting model")
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=52, help="Lookback window size")
    parser.add_argument("--patch-len", type=int, default=8, help="Patch length")
    parser.add_argument("--stride", type=int, default=4, help="Patch stride")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs (0 for CPU)")
    parser.add_argument("--max-skus", type=int, default=None, help="Limit SKUs for testing")
    args = parser.parse_args()
    
    if not NEURALFORECAST_AVAILABLE:
        print("ERROR: neuralforecast is not available.")
        print("Install with: pip install neuralforecast")
        return
    
    print(f"Loading demand data from {args.demand_path}")
    df = load_demand(args.demand_path)
    
    if args.max_skus:
        unique_ids = df['unique_id'].unique()[:args.max_skus]
        df = df[df['unique_id'].isin(unique_ids)]
    
    print(f"  {len(df['unique_id'].unique())} SKUs, {len(df)} observations")
    print(f"  Using {'GPU' if args.gpus > 0 else 'CPU'}")
    
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    # Train model
    result = train_patchtst(
        df=df,
        output_dir=args.output_root,
        horizon=args.horizon,
        input_size=args.input_size,
        patch_len=args.patch_len,
        stride=args.stride,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gpus=args.gpus
    )
    
    # Generate forecasts and save checkpoints
    n_saved = generate_forecasts_and_save(
        nf=result['model'],
        df=df,
        output_dir=args.output_root,
        horizon=args.horizon
    )
    
    print(f"\nPatchTST training complete!")
    print(f"  Model saved to: {result['model_path']}")
    print(f"  Checkpoints saved: {n_saved} SKUs")
    print(f"  Output: {args.output_root / 'patchtst'}")


if __name__ == "__main__":
    main()

