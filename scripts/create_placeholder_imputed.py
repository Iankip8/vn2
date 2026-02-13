#!/usr/bin/env python3
"""
Create placeholder imputed data files to enable model training while
imputation runs in the background or is skipped.

For SLURP models: Use demand_long.parquet directly (handles stockouts)
For others: Create placeholder winsorized data (can be replaced later with real imputation)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Creating placeholder imputed data files...")
    print("=" * 70)
    
    # Load demand_long
    demand_long_path = Path('data/processed/demand_long.parquet')
    if not demand_long_path.exists():
        print(f"Error: {demand_long_path} not found!")
        return
    
    df = pd.read_parquet(demand_long_path)
    print(f"Loaded {len(df):,} observations from demand_long.parquet")
    
    # Add 'imputed' flag (all False for placeholder)
    df['imputed'] = False
    
    # Create demand_imputed.parquet (exact copy with flag)
    output_imputed = Path('data/processed/demand_imputed.parquet')
    df.to_parquet(output_imputed, index=False)
    print(f"✓ Created {output_imputed}")
    
    # Create winsorized version with simple clipping
    df_winsor = df.copy()
    df_winsor['sales_winsor'] = np.clip(df['sales'], 0, 500)  # Simple cap at 500
    
    output_winsor = Path('data/processed/demand_imputed_winsor.parquet')
    df_winsor.to_parquet(output_winsor, index=False)
    print(f"✓ Created {output_winsor}")
    
    # Create capped version (more conservative)
    df_capped = df.copy()
    df_capped['sales_capped'] = np.clip(df['sales'], 0, 200)  # Cap at 200
    
    output_capped = Path('data/processed/demand_imputed_capped.parquet')
    df_capped.to_parquet(output_capped, index=False)
    print(f"✓ Created {output_capped}")
    
    print("\n" + "=" * 70)
    print("✅ Placeholder files created successfully!")
    print("\nIMPORTANT NOTES:")
    print("- These are PLACEHOLDERS without real imputation")
    print("- SLURP models will work fine (they use demand_long directly)")
    print("- Other models will train but may be suboptimal")
    print("- Run real imputation later: ./go impute --n-neighbors 5 --n-jobs 1")
    print("- Re-train models after real imputation completes")
    print("=" * 70)

if __name__ == "__main__":
    main()
