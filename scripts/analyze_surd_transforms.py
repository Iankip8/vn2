#!/usr/bin/env python3
"""
Analyze SURD transforms to understand their effectiveness.

The issue: Many transformed_cv values are negative, which is invalid for CV.
This happens when log-transformed data has a negative mean (log of values < 1).

We need to:
1. Recalculate CV properly (using absolute values or better metrics)
2. Determine if transforms actually reduce variance
3. Understand when transforms help vs hurt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_demand_data() -> pd.DataFrame:
    """Load demand data to recalculate CV properly."""
    demand_path = Path('data/processed/demand_long.parquet')
    if not demand_path.exists():
        # Try alternative paths
        alt_paths = [
            Path('data/processed/demand_imputed.parquet'),
            Path('data/processed/demand_imputed_capped.parquet'),
        ]
        for p in alt_paths:
            if p.exists():
                demand_path = p
                break
    
    if not demand_path.exists():
        print(f"Warning: Could not find demand data. Using transforms file only.")
        return None
    
    df = pd.read_parquet(demand_path)
    
    # Standardize column names
    if 'Sales' in df.columns:
        df = df.rename(columns={'Sales': 'sales'})
    if 'demand' in df.columns:
        df = df.rename(columns={'demand': 'sales'})
    
    return df

def calculate_cv_properly(series: np.ndarray, transform_name: str = 'identity') -> tuple[float, float]:
    """
    Calculate CV properly for transformed data.
    
    Returns: (cv, transformed_series)
    """
    # Apply transform
    if transform_name == 'log':
        transformed = np.log(np.maximum(series, 1e-10))
    elif transform_name == 'log1p':
        transformed = np.log1p(series)
    elif transform_name == 'sqrt':
        transformed = np.sqrt(np.maximum(series, 0))
    elif transform_name == 'cbrt':
        transformed = np.cbrt(series)
    else:  # identity
        transformed = series.copy()
    
    # Calculate CV: std / mean
    # For CV to be meaningful, we need positive mean or use absolute value
    mean_val = transformed.mean()
    std_val = transformed.std()
    
    if abs(mean_val) < 1e-9:
        cv = np.inf if std_val > 0 else 0.0
    else:
        # Use absolute value of mean for CV calculation
        # This makes sense because we care about relative variability
        cv = std_val / abs(mean_val)
    
    return cv, transformed

def analyze_surd_effectiveness():
    """Main analysis function."""
    print("=" * 80)
    print("SURD Transforms Effectiveness Analysis")
    print("=" * 80)
    
    # Load transforms
    transforms_path = Path('data/processed/surd_transforms.parquet')
    surd_df = pd.read_parquet(transforms_path)
    
    print(f"\nLoaded {len(surd_df)} SKU transforms")
    print(f"Columns: {list(surd_df.columns)}")
    
    # Load demand data if available
    demand_df = load_demand_data()
    
    # Analyze current metrics
    print("\n" + "=" * 80)
    print("1. CURRENT METRICS ANALYSIS")
    print("=" * 80)
    
    print(f"\nOriginal CV statistics:")
    print(f"  Mean: {surd_df['original_cv'].mean():.4f}")
    print(f"  Median: {surd_df['original_cv'].median():.4f}")
    print(f"  Min: {surd_df['original_cv'].min():.4f}")
    print(f"  Max: {surd_df['original_cv'].max():.4f}")
    
    print(f"\nTransformed CV statistics (as stored):")
    print(f"  Mean: {surd_df['transformed_cv'].mean():.4f}")
    print(f"  Median: {surd_df['transformed_cv'].median():.4f}")
    print(f"  Min: {surd_df['transformed_cv'].min():.4f}")
    print(f"  Max: {surd_df['transformed_cv'].max():.4f}")
    
    negative_cv = (surd_df['transformed_cv'] < 0).sum()
    print(f"\n⚠️  PROBLEM: {negative_cv} SKUs ({negative_cv/len(surd_df)*100:.1f}%) have negative transformed_cv")
    print("   This is mathematically invalid for CV (std/mean cannot be negative)")
    print("   This happens when log-transformed data has negative mean (log of values < 1)")
    
    # Recalculate CV properly if we have demand data
    if demand_df is not None:
        print("\n" + "=" * 80)
        print("2. RECALCULATING CV PROPERLY")
        print("=" * 80)
        
        # Group by Store, Product
        store_col = 'Store' if 'Store' in demand_df.columns else 'store'
        product_col = 'Product' if 'Product' in demand_df.columns else 'product'
        
        if store_col not in demand_df.columns or product_col not in demand_df.columns:
            print(f"Warning: Could not find Store/Product columns. Available: {list(demand_df.columns)}")
            return
        
        results = []
        for idx, row in surd_df.iterrows():
            store = row['Store']
            product = row['Product']
            transform = row['best_transform']
            
            # Get SKU data
            sku_data = demand_df[
                (demand_df[store_col] == store) & 
                (demand_df[product_col] == product)
            ].copy()
            
            if len(sku_data) < 2:
                continue
            
            sales = sku_data['sales'].values
            
            # Calculate original CV
            orig_cv = sales.std() / (sales.mean() + 1e-9)
            
            # Calculate transformed CV properly
            trans_cv, _ = calculate_cv_properly(sales, transform)
            
            # Calculate proper CV reduction
            if orig_cv > 0:
                cv_reduction_pct = (orig_cv - trans_cv) / orig_cv
            else:
                cv_reduction_pct = 0.0
            
            results.append({
                'Store': store,
                'Product': product,
                'best_transform': transform,
                'original_cv': orig_cv,
                'transformed_cv_corrected': trans_cv,
                'cv_reduction_pct': cv_reduction_pct,
                'n_observations': len(sku_data),
            })
        
        corrected_df = pd.DataFrame(results)
        
        print(f"\nRecalculated CV for {len(corrected_df)} SKUs")
        print(f"\nCorrected Transformed CV statistics:")
        print(f"  Mean: {corrected_df['transformed_cv_corrected'].mean():.4f}")
        print(f"  Median: {corrected_df['transformed_cv_corrected'].median():.4f}")
        print(f"  Min: {corrected_df['transformed_cv_corrected'].min():.4f}")
        print(f"  Max: {corrected_df['transformed_cv_corrected'].max():.4f}")
        
        print(f"\nCV Reduction (corrected):")
        print(f"  Mean: {corrected_df['cv_reduction_pct'].mean():.4f} ({corrected_df['cv_reduction_pct'].mean()*100:.2f}%)")
        print(f"  Median: {corrected_df['cv_reduction_pct'].median():.4f} ({corrected_df['cv_reduction_pct'].median()*100:.2f}%)")
        print(f"  Min: {corrected_df['cv_reduction_pct'].min():.4f} ({corrected_df['cv_reduction_pct'].min()*100:.2f}%)")
        print(f"  Max: {corrected_df['cv_reduction_pct'].max():.4f} ({corrected_df['cv_reduction_pct'].max()*100:.2f}%)")
        
        # Count improvements
        improved = (corrected_df['cv_reduction_pct'] > 0).sum()
        worsened = (corrected_df['cv_reduction_pct'] < 0).sum()
        no_change = (corrected_df['cv_reduction_pct'] == 0).sum()
        
        print(f"\n📊 Effectiveness Summary:")
        print(f"  ✅ Improved (CV reduced): {improved} ({improved/len(corrected_df)*100:.1f}%)")
        print(f"  ❌ Worsened (CV increased): {worsened} ({worsened/len(corrected_df)*100:.1f}%)")
        print(f"  ➖ No change: {no_change} ({no_change/len(corrected_df)*100:.1f}%)")
        
        # By transform type
        print(f"\n📊 Effectiveness by Transform Type:")
        for transform in corrected_df['best_transform'].unique():
            subset = corrected_df[corrected_df['best_transform'] == transform]
            improved_pct = (subset['cv_reduction_pct'] > 0).mean() * 100
            mean_reduction = subset['cv_reduction_pct'].mean() * 100
            print(f"  {transform:8s}: {len(subset):3d} SKUs, {improved_pct:5.1f}% improved, mean reduction: {mean_reduction:6.2f}%")
        
        # Merge with original
        merged = surd_df.merge(
            corrected_df[['Store', 'Product', 'transformed_cv_corrected', 'cv_reduction_pct']],
            on=['Store', 'Product'],
            how='left'
        )
        
        # Save corrected analysis
        output_path = Path('reports/surd_analysis_corrected.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path)
        print(f"\n💾 Saved corrected analysis to {output_path}")
        
        # Create visualization
        create_visualizations(corrected_df, merged)
        
    else:
        print("\n⚠️  Could not load demand data to recalculate CV")
        print("   Analysis limited to stored metrics")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

def create_visualizations(corrected_df: pd.DataFrame, merged_df: pd.DataFrame):
    """Create visualization plots."""
    output_dir = Path('reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. CV reduction distribution
    ax = axes[0, 0]
    ax.hist(corrected_df['cv_reduction_pct'] * 100, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(corrected_df['cv_reduction_pct'].mean() * 100, color='green', 
               linestyle='--', linewidth=2, label=f'Mean: {corrected_df["cv_reduction_pct"].mean()*100:.1f}%')
    ax.set_xlabel('CV Reduction (%)')
    ax.set_ylabel('Number of SKUs')
    ax.set_title('Distribution of CV Reduction (Corrected)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Original vs Transformed CV
    ax = axes[0, 1]
    sample = corrected_df.sample(min(200, len(corrected_df)))
    ax.scatter(sample['original_cv'], sample['transformed_cv_corrected'], 
               alpha=0.5, s=20)
    max_cv = max(corrected_df['original_cv'].max(), corrected_df['transformed_cv_corrected'].max())
    ax.plot([0, max_cv], [0, max_cv], 'r--', linewidth=2, label='No improvement')
    ax.set_xlabel('Original CV')
    ax.set_ylabel('Transformed CV (Corrected)')
    ax.set_title('Original vs Transformed CV')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Effectiveness by transform
    ax = axes[1, 0]
    transform_effectiveness = corrected_df.groupby('best_transform')['cv_reduction_pct'].agg(['mean', 'median', 'std'])
    x_pos = np.arange(len(transform_effectiveness))
    ax.bar(x_pos, transform_effectiveness['mean'] * 100, 
           yerr=transform_effectiveness['std'] * 100, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(transform_effectiveness.index)
    ax.set_ylabel('Mean CV Reduction (%)')
    ax.set_title('Effectiveness by Transform Type')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Comparison: stored vs corrected
    ax = axes[1, 1]
    if 'transformed_cv' in merged_df.columns:
        # Only plot where both exist
        valid = merged_df.dropna(subset=['transformed_cv', 'transformed_cv_corrected'])
        if len(valid) > 0:
            ax.scatter(valid['transformed_cv'], valid['transformed_cv_corrected'], 
                      alpha=0.5, s=20)
            ax.set_xlabel('Stored Transformed CV (may be negative)')
            ax.set_ylabel('Corrected Transformed CV')
            ax.set_title('Stored vs Corrected CV Values')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'surd_effectiveness_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved visualization to {output_path}")
    plt.close()

if __name__ == '__main__':
    analyze_surd_effectiveness()

