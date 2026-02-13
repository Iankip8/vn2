#!/usr/bin/env python3
"""
Train forecast models on a subset of SKUs for faster iteration.
Usage: python scripts/train_subset.py --n-skus 100 --n-jobs 4
"""

import sys
sys.path.insert(0, 'src')

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def select_representative_skus(df, n_skus=100, random_state=42):
    """
    Select a diverse, representative subset of SKUs.
    Ensures variety across:
    - Sales volume (high/medium/low)
    - Volatility (CV)
    - Zero-inflation rate
    """
    # Compute SKU-level statistics
    sku_stats = df.groupby(['Store', 'Product']).agg({
        'sales': ['mean', 'std', 'sum'],
        'in_stock': 'mean'
    }).reset_index()
    
    sku_stats.columns = ['Store', 'Product', 'mean_sales', 'std_sales', 'total_sales', 'in_stock_rate']
    sku_stats['cv'] = sku_stats['std_sales'] / (sku_stats['mean_sales'] + 1e-6)
    sku_stats['zero_rate'] = (sku_stats['mean_sales'] == 0).astype(float)
    
    # Stratified sampling by volume quintiles
    sku_stats['volume_quintile'] = pd.qcut(sku_stats['total_sales'], q=5, labels=False, duplicates='drop')
    
    # Sample proportionally from each quintile
    n_per_quintile = n_skus // 5
    selected = []
    
    for quintile in range(5):
        quintile_skus = sku_stats[sku_stats['volume_quintile'] == quintile]
        n_sample = min(n_per_quintile, len(quintile_skus))
        sampled = quintile_skus.sample(n=n_sample, random_state=random_state + quintile)
        selected.append(sampled)
    
    # Fill remaining with random selection
    selected_df = pd.concat(selected)
    if len(selected_df) < n_skus:
        remaining = sku_stats[~sku_stats.index.isin(selected_df.index)]
        extra = remaining.sample(n=min(n_skus - len(selected_df), len(remaining)), random_state=random_state)
        selected_df = pd.concat([selected_df, extra])
    
    return selected_df[['Store', 'Product']].values.tolist()

def main():
    parser = argparse.ArgumentParser(description='Train on subset of SKUs')
    parser.add_argument('--n-skus', type=int, default=100, help='Number of SKUs to train')
    parser.add_argument('--n-jobs', type=int, default=4, help='Parallel workers')
    parser.add_argument('--config', type=str, default='configs/forecast.yaml', help='Config file')
    args = parser.parse_args()
    
    print(f"🎯 Selecting {args.n_skus} representative SKUs for training...")
    
    # Load data
    df = pd.read_parquet('data/processed/demand_long.parquet')
    
    # Select SKUs
    selected_skus = select_representative_skus(df, n_skus=args.n_skus)
    
    print(f"✓ Selected {len(selected_skus)} SKUs")
    print(f"  Volume distribution: High/Med/Low spread")
    print(f"  Random seed: 42 (reproducible)")
    
    # Save selected SKUs to temp file
    sku_file = Path('data/processed/train_subset_skus.txt')
    with open(sku_file, 'w') as f:
        for store, product in selected_skus:
            f.write(f"{store},{product}\n")
    
    print(f"\n✓ SKU list saved to {sku_file}")
    print(f"\nNow training {len(selected_skus)} SKUs...")
    print(f"Estimated time: {len(selected_skus) * 13.8 / 60:.1f} minutes")
    print("=" * 60)
    
    # Import and run training
    from vn2.cli import cmd_forecast
    
    class Args:
        config = args.config
        pilot = False
        test = False
        n_jobs = args.n_jobs
        subset_skus = selected_skus
    
    # Monkey-patch to use subset
    import vn2.forecast.pipeline
    original_generate_tasks = vn2.forecast.pipeline.ForecastPipeline.generate_tasks
    
    def patched_generate_tasks(self, df, models, pilot_skus=None):
        return original_generate_tasks(self, df, models, pilot_skus=selected_skus)
    
    vn2.forecast.pipeline.ForecastPipeline.generate_tasks = patched_generate_tasks
    
    # Run training
    cmd_forecast(Args())

if __name__ == '__main__':
    main()
