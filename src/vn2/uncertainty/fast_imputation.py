"""
Fast stockout imputation using vectorized seasonal-index approach.

This replaces the computationally expensive profile-matching method with
a simple, fast seasonal median imputation that can handle large datasets.

Recommended by Patrick McDonald as replacement for hanging extract_profile_features().
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings


def compute_seasonal_indices(
    df: pd.DataFrame,
    method: str = 'median'
) -> pd.DataFrame:
    """
    Compute seasonal indices (week-of-year patterns) for all SKUs.
    
    This is a vectorized, groupwise precompute that gets reused for all imputations.
    Much faster than per-observation seasonal stats.
    
    Args:
        df: DataFrame with columns [Store, Product, week, retail_week, sales, in_stock]
        method: 'median' or 'trimmed_mean' for aggregation
        
    Returns:
        DataFrame with multi-index (Store, Product, retail_week) and column 'seasonal_index'
    """
    # Filter to non-stockout weeks only
    non_stockout = df[df['in_stock'] == True].copy()
    
    if len(non_stockout) == 0:
        warnings.warn("No non-stockout observations found for seasonal index computation")
        return pd.DataFrame()
    
    # Group by SKU and retail week
    if method == 'median':
        seasonal = non_stockout.groupby(['Store', 'Product', 'retail_week'])['sales'].median()
    elif method == 'trimmed_mean':
        # 10% trimmed mean (remove top/bottom 10%)
        seasonal = non_stockout.groupby(['Store', 'Product', 'retail_week'])['sales'].apply(
            lambda x: x.quantile([0.1, 0.9]).mean() if len(x) >= 5 else x.median()
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    seasonal_df = seasonal.reset_index()
    seasonal_df.columns = ['Store', 'Product', 'retail_week', 'seasonal_index']
    
    return seasonal_df


def impute_stockout_seasonal(
    row: pd.Series,
    seasonal_indices: pd.DataFrame,
    fallback_quantile: float = 0.5
) -> float:
    """
    Impute a single stockout observation using seasonal index.
    
    Args:
        row: Row with stockout (must have Store, Product, retail_week, sales)
        seasonal_indices: Precomputed seasonal indices from compute_seasonal_indices()
        fallback_quantile: Quantile to use if no seasonal match found
        
    Returns:
        Imputed demand value
    """
    store = row['Store']
    product = row['Product']
    retail_week = row['retail_week']
    
    # Look up seasonal index
    match = seasonal_indices[
        (seasonal_indices['Store'] == store) &
        (seasonal_indices['Product'] == product) &
        (seasonal_indices['retail_week'] == retail_week)
    ]
    
    if len(match) > 0:
        return match['seasonal_index'].iloc[0]
    
    # Fallback 1: Use overall SKU median from seasonal indices
    sku_indices = seasonal_indices[
        (seasonal_indices['Store'] == store) &
        (seasonal_indices['Product'] == product)
    ]
    
    if len(sku_indices) > 0:
        return sku_indices['seasonal_index'].median()
    
    # Fallback 2: Use observed stock level (conservative)
    return row['sales']


def impute_all_stockouts_seasonal(
    df: pd.DataFrame,
    method: str = 'median',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Replace all stockout observations with seasonal-index imputed values.
    
    This is the fast, vectorized replacement for the slow profile-matching approach.
    
    Args:
        df: DataFrame with demand data
        method: 'median' or 'trimmed_mean'
        verbose: Print progress
        
    Returns:
        DataFrame with imputed values and 'imputed' flag
    """
    if verbose:
        n_stockouts = (~df['in_stock']).sum()
        print(f"📊 Imputing {n_stockouts:,} stockouts using seasonal {method}...")
    
    # Step 1: Vectorized precompute of seasonal indices (FAST)
    if verbose:
        print("   Computing seasonal indices (vectorized, groupwise)...")
    seasonal_indices = compute_seasonal_indices(df, method=method)
    
    if verbose:
        print(f"   ✓ Computed {len(seasonal_indices):,} seasonal patterns")
    
    # Step 2: Impute stockouts
    df_imputed = df.copy()
    df_imputed['imputed'] = False
    
    stockout_mask = ~df['in_stock']
    
    if verbose:
        print(f"   Imputing {stockout_mask.sum():,} observations...")
    
    # Apply imputation to stockout rows
    df_imputed.loc[stockout_mask, 'sales'] = df[stockout_mask].apply(
        lambda row: impute_stockout_seasonal(row, seasonal_indices),
        axis=1
    )
    df_imputed.loc[stockout_mask, 'imputed'] = True
    
    if verbose:
        imputed_count = df_imputed['imputed'].sum()
        pct = 100 * imputed_count / len(df)
        print(f"   ✓ Imputed {imputed_count:,} observations ({pct:.1f}%)")
        
        # Summary stats
        original_mean = df[stockout_mask]['sales'].mean()
        imputed_mean = df_imputed[stockout_mask]['sales'].mean()
        print(f"   Mean demand: {original_mean:.2f} (censored) → {imputed_mean:.2f} (imputed)")
    
    return df_imputed


def apply_winsorization(
    df: pd.DataFrame,
    lower: float = 0.0,
    upper: float = 500.0,
    column: str = 'sales'
) -> pd.DataFrame:
    """
    Apply simple winsorization (clipping) to demand values.
    
    Args:
        df: DataFrame with demand
        lower: Lower bound
        upper: Upper bound
        column: Column to clip
        
    Returns:
        DataFrame with clipped values
    """
    df_winsor = df.copy()
    df_winsor[column] = df[column].clip(lower=lower, upper=upper)
    return df_winsor


def create_imputed_variants(
    df: pd.DataFrame,
    output_dir: str = 'data/processed',
    method: str = 'median',
    verbose: bool = True
) -> dict:
    """
    Create all imputed data variants needed for training.
    
    This replaces the slow placeholder approach with real imputation.
    
    Args:
        df: Raw demand data with stockouts
        output_dir: Where to save outputs
        method: Seasonal aggregation method
        verbose: Print progress
        
    Returns:
        Dict with paths to created files
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*60)
        print("Creating imputed data variants (FAST seasonal-index method)")
        print("="*60)
    
    # Base imputation
    df_imputed = impute_all_stockouts_seasonal(df, method=method, verbose=verbose)
    
    # Variant 1: Base imputed
    path_imputed = output_dir / 'demand_imputed.parquet'
    df_imputed.to_parquet(path_imputed)
    if verbose:
        print(f"\n✓ Created {path_imputed}")
    
    # Variant 2: Winsorized (0-500 cap)
    df_winsor = apply_winsorization(df_imputed, lower=0, upper=500)
    path_winsor = output_dir / 'demand_imputed_winsor.parquet'
    df_winsor.to_parquet(path_winsor)
    if verbose:
        print(f"✓ Created {path_winsor}")
    
    # Variant 3: Capped (conservative 0-200)
    df_capped = apply_winsorization(df_imputed, lower=0, upper=200)
    path_capped = output_dir / 'demand_imputed_capped.parquet'
    df_capped.to_parquet(path_capped)
    if verbose:
        print(f"✓ Created {path_capped}")
    
    if verbose:
        print("\n" + "="*60)
        print("✅ Fast imputation complete!")
        print("="*60)
        print("\nIMPORTANT NOTES:")
        print("- Imputation rate: {:.1f}%".format(100 * df_imputed['imputed'].sum() / len(df)))
        print("- Method: Seasonal {} (week-of-year)".format(method))
        print("- No profile-matching overhead (vectorized groupwise)")
        print("- Ready for model training")
        print("="*60)
    
    return {
        'imputed': str(path_imputed),
        'winsor': str(path_winsor),
        'capped': str(path_capped)
    }


if __name__ == '__main__':
    """Quick test of fast imputation"""
    import sys
    from pathlib import Path
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'demand_long.parquet'
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    
    print(f"Loaded {len(df):,} observations")
    print(f"Stockouts: {(~df['in_stock']).sum():,} ({100*(~df['in_stock']).mean():.1f}%)")
    
    # Run fast imputation
    paths = create_imputed_variants(df, method='median', verbose=True)
    
    print("\nGenerated files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
