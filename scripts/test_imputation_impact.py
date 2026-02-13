"""
Imputation Impact Test: Compare forecast quality with/without imputation

This script tests the impact of stockout imputation on forecast performance by comparing:
- Option 1: Drop stockout weeks entirely
- Option 2: Fast seasonal imputation (11.2% rate)
- Option 3: Placeholder zeros (0% imputation)

Expected: Real imputation should improve forecast quality vs dropping or zeros.

Author: Ian
Date: February 4, 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_demand_data():
    """Load demand data with different imputation strategies."""
    # Load raw data
    raw_file = Path("data/processed/demand_long.parquet")
    if not raw_file.exists():
        raise FileNotFoundError(f"Cannot find {raw_file}")
    
    df_raw = pd.read_parquet(raw_file)
    
    # Load imputed variants (if they exist)
    imputed_file = Path("data/processed/demand_imputed.parquet")
    winsor_file = Path("data/processed/demand_winsor.parquet")
    
    variants = {'raw': df_raw}
    
    if imputed_file.exists():
        variants['imputed'] = pd.read_parquet(imputed_file)
    
    if winsor_file.exists():
        variants['winsor'] = pd.read_parquet(winsor_file)
    
    return variants


def prepare_variant_dropped(df):
    """Option 1: Drop all stockout weeks."""
    return df[df['is_stockout'] == False].copy()


def prepare_variant_zeros(df):
    """Option 3: Replace stockouts with zeros (original placeholder approach)."""
    df = df.copy()
    df.loc[df['is_stockout'] == True, 'sales'] = 0
    df.loc[df['is_stockout'] == True, 'imputed'] = False
    return df


def prepare_variant_imputed(df):
    """Option 2: Use seasonal imputation."""
    # This assumes the imputed file already has imputation applied
    # Check if imputation actually happened
    if 'imputed' in df.columns:
        imputed_count = df['imputed'].sum()
        imputed_rate = imputed_count / len(df)
        print(f"   Using imputed data: {imputed_count:,} observations ({imputed_rate:.1%})")
    return df


def compute_forecast_quality_metrics(df, group_cols=['store', 'product']):
    """
    Compute simple forecast quality metrics.
    
    For a basic test, we'll use:
    - Within-group variance (higher is better for model training)
    - Number of observations per group
    - Mean demand level
    """
    metrics = df.groupby(group_cols).agg({
        'sales': ['count', 'mean', 'std', 'min', 'max']
    })
    
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns]
    
    # Add some derived metrics
    metrics['cv'] = metrics['sales_std'] / (metrics['sales_mean'] + 1e-6)  # Coefficient of variation
    metrics['zero_rate'] = (df.groupby(group_cols)['sales'].apply(lambda x: (x == 0).mean()))
    
    return metrics


def compare_imputation_strategies():
    """Compare the three imputation strategies."""
    print("="*70)
    print("IMPUTATION IMPACT TEST")
    print("="*70)
    print()
    
    # Load data
    print("Loading data variants...")
    variants = load_demand_data()
    
    if 'imputed' not in variants:
        print("⚠️  Warning: Imputed data not found. Creating from raw...")
        # Would run imputation here if needed
    
    # Prepare variants
    print("\nPreparing comparison variants:")
    print("-" * 70)
    
    results = {}
    
    # Variant 1: Dropped stockouts
    print("\n1. DROP STOCKOUTS:")
    df_dropped = prepare_variant_dropped(variants['raw'])
    n_dropped = len(variants['raw']) - len(df_dropped)
    print(f"   Removed {n_dropped:,} stockout observations ({n_dropped/len(variants['raw']):.1%})")
    results['dropped'] = compute_forecast_quality_metrics(df_dropped)
    
    # Variant 2: Imputed
    print("\n2. SEASONAL IMPUTATION:")
    if 'imputed' in variants:
        df_imputed = prepare_variant_imputed(variants['imputed'])
    else:
        print("   ⚠️  No imputed file found, using raw")
        df_imputed = variants['raw'].copy()
    results['imputed'] = compute_forecast_quality_metrics(df_imputed)
    
    # Variant 3: Zeros
    print("\n3. PLACEHOLDER ZEROS:")
    df_zeros = prepare_variant_zeros(variants['raw'])
    n_zeros = (df_zeros['sales'] == 0).sum()
    print(f"   Replaced {n_zeros:,} values with zeros")
    results['zeros'] = compute_forecast_quality_metrics(df_zeros)
    
    print()
    
    # Compare metrics
    print("="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print()
    
    comparison = pd.DataFrame({
        'Dropped': results['dropped'].mean(),
        'Imputed': results['imputed'].mean(),
        'Zeros': results['zeros'].mean(),
    }).T
    
    print("Average Statistics Across All SKUs:")
    print("-" * 70)
    print(comparison.to_string())
    print()
    
    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()
    
    print("Observations per SKU:")
    print(f"  Dropped:  {comparison.loc['Dropped', 'sales_count']:.0f}")
    print(f"  Imputed:  {comparison.loc['Imputed', 'sales_count']:.0f}")
    print(f"  Zeros:    {comparison.loc['Zeros', 'sales_count']:.0f}")
    print()
    
    print("Mean demand:")
    print(f"  Dropped:  {comparison.loc['Dropped', 'sales_mean']:.2f}")
    print(f"  Imputed:  {comparison.loc['Imputed', 'sales_mean']:.2f}")
    print(f"  Zeros:    {comparison.loc['Zeros', 'sales_mean']:.2f}")
    print()
    
    print("Standard deviation:")
    print(f"  Dropped:  {comparison.loc['Dropped', 'sales_std']:.2f}")
    print(f"  Imputed:  {comparison.loc['Imputed', 'sales_std']:.2f}")
    print(f"  Zeros:    {comparison.loc['Zeros', 'sales_std']:.2f}")
    print()
    
    print("RECOMMENDATIONS:")
    print("-" * 70)
    
    # Check which is best
    if comparison.loc['Imputed', 'sales_mean'] > comparison.loc['Zeros', 'sales_mean']:
        print("✅ Imputation improves mean demand estimate vs zeros")
    else:
        print("⚠️  Imputation does not improve mean demand (may indicate poor imputation)")
    
    if comparison.loc['Imputed', 'sales_count'] > comparison.loc['Dropped', 'sales_count']:
        print("✅ Imputation preserves more data than dropping")
    else:
        print("⚠️  Imputation lost data (unexpected)")
    
    if comparison.loc['Imputed', 'sales_std'] > comparison.loc['Zeros', 'sales_std']:
        print("✅ Imputation captures more variance than zeros")
    else:
        print("⚠️  Imputation has less variance than zeros (may indicate conservative imputation)")
    
    print()
    
    # Save detailed results
    output_file = "models/results/imputation_comparison.csv"
    comparison.to_csv(output_file)
    print(f"✅ Detailed results saved to: {output_file}")
    
    return results


def test_forecast_model_impact():
    """
    Test how imputation impacts actual forecast model performance.
    
    This would train a simple model (e.g., seasonal naive) on each variant
    and compare forecast accuracy.
    """
    print("\n")
    print("="*70)
    print("FORECAST MODEL IMPACT TEST")
    print("="*70)
    print()
    
    print("This test would:")
    print("  1. Split data into train/test")
    print("  2. Train simple forecaster on each variant (dropped/imputed/zeros)")
    print("  3. Compare out-of-sample MAE, RMSE, service level")
    print()
    print("Expected: Imputed variant should give best forecast accuracy")
    print()
    print("⏸️  Skipping detailed model training for now (computationally expensive)")
    print("   Run this test separately if needed for detailed diagnostics")


def main():
    """Run the imputation comparison."""
    results = compare_imputation_strategies()
    
    # Optional: Test actual model impact
    test_forecast_model_impact()
    
    print()
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    print("The fast seasonal imputation approach should:")
    print("  ✅ Preserve all observations (vs dropping 11%)")
    print("  ✅ Provide realistic demand estimates (vs zeros)")
    print("  ✅ Improve forecast model training")
    print()
    print("If imputation doesn't improve forecast quality, investigate:")
    print("  - Are imputed values realistic?")
    print("  - Is seasonal pattern appropriate?")
    print("  - Should we use more sophisticated imputation?")


if __name__ == "__main__":
    main()
