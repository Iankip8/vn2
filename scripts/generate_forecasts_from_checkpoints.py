#!/usr/bin/env python3
"""
Generate forecasts from trained checkpoint files for h=3,4,5 (Patrick's corrected horizons).
This extracts quantile predictions from the 4 fully-trained models.
"""

import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
CHECKPOINT_DIR = Path("/home/ian/vn2/models/checkpoints")
OUTPUT_DIR = Path("/home/ian/vn2/models/results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Models to process (4 fully trained + zinb partially trained)
MODELS = ["ets", "lightgbm_quantile", "slurp_bootstrap", "slurp_stockout_aware", "zinb"]

# Horizons for Patrick's corrected SIP (h=3,4,5 for 3-week protection period)
HORIZONS = [3, 4, 5]

# Quantiles we need for SIP
QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

def load_checkpoint(checkpoint_path):
    """Load a trained model checkpoint."""
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def extract_forecasts_from_checkpoint(checkpoint, horizons=HORIZONS):
    """Extract quantile forecasts for specified horizons."""
    try:
        # Checkpoint structure varies by model type
        if hasattr(checkpoint, 'predict_quantiles'):
            # Direct quantile prediction
            forecasts = {}
            for h in horizons:
                forecasts[h] = checkpoint.predict_quantiles(quantiles=QUANTILES, horizon=h)
            return forecasts
        elif hasattr(checkpoint, 'quantiles_'):
            # Pre-computed quantiles (e.g., SLURP models)
            return {h: checkpoint.quantiles_[h-1] for h in horizons if h-1 < len(checkpoint.quantiles_)}
        elif 'quantiles' in checkpoint:
            # Dictionary format
            return {h: checkpoint['quantiles'][h] for h in horizons if h in checkpoint['quantiles']}
        else:
            return None
    except Exception as e:
        print(f"Error extracting forecasts: {e}")
        return None

def main():
    """Generate forecast files for each model."""
    
    all_results = []
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        model_dir = CHECKPOINT_DIR / model_name
        if not model_dir.exists():
            print(f"⚠️  No checkpoints found for {model_name}")
            continue
        
        # Find all SKU directories
        sku_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
        print(f"Found {len(sku_dirs)} SKUs with checkpoints")
        
        model_forecasts = []
        
        for sku_idx, sku_dir in enumerate(sku_dirs):
            if sku_idx % 50 == 0:
                print(f"  Processing SKU {sku_idx+1}/{len(sku_dirs)}...")
            # Parse SKU ID from directory name (format: "store_product")
            sku_parts = sku_dir.name.split('_')
            if len(sku_parts) != 2:
                continue
            store_id, product_id = int(sku_parts[0]), int(sku_parts[1])
            
            # Load checkpoints for all folds (we'll average later)
            fold_files = sorted(sku_dir.glob("fold_*.pkl"))
            
            if len(fold_files) == 0:
                continue
            
            # For simplicity, use fold_0 (out-of-sample validation fold)
            checkpoint_path = sku_dir / "fold_0.pkl"
            if not checkpoint_path.exists():
                checkpoint_path = fold_files[0]  # Use first available
            
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint is None:
                continue
            
            # Extract forecasts for h=3,4,5
            forecasts = extract_forecasts_from_checkpoint(checkpoint, HORIZONS)
            if forecasts is None:
                continue
            
            # Store forecast data
            for horizon, quantile_forecasts in forecasts.items():
                if quantile_forecasts is None:
                    continue
                    
                for i, q in enumerate(QUANTILES):
                    if isinstance(quantile_forecasts, dict):
                        value = quantile_forecasts.get(q, None)
                    elif isinstance(quantile_forecasts, (list, tuple)) and i < len(quantile_forecasts):
                        value = quantile_forecasts[i]
                    else:
                        value = None
                    
                    if value is not None:
                        model_forecasts.append({
                            'model': model_name,
                            'store_id': store_id,
                            'product_id': product_id,
                            'horizon': horizon,
                            'quantile': q,
                            'forecast': float(value)
                        })
        
        if model_forecasts:
            # Save model-specific forecasts
            df_model = pd.DataFrame(model_forecasts)
            output_file = OUTPUT_DIR / f"{model_name}_quantiles_h345.parquet"
            df_model.to_parquet(output_file, index=False)
            print(f"✅ Saved {len(df_model)} forecasts to {output_file.name}")
            all_results.append(df_model)
        else:
            print(f"⚠️  No forecasts extracted for {model_name}")
    
    # Combine all models
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        output_file = OUTPUT_DIR / "all_models_quantiles_h345.parquet"
        df_all.to_parquet(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"✅ COMPLETE: Saved {len(df_all)} total forecasts")
        print(f"   File: {output_file}")
        print(f"   Models: {df_all['model'].unique().tolist()}")
        print(f"   SKUs: {df_all[['store_id', 'product_id']].drop_duplicates().shape[0]}")
        print(f"   Horizons: {sorted(df_all['horizon'].unique())}")
        print(f"{'='*60}")
    else:
        print("\n⚠️  No forecasts were generated!")

if __name__ == "__main__":
    main()
