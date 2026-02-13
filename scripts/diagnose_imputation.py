#!/usr/bin/env python3
"""
Diagnostic script to test and debug stockout imputation logic.
Tests the imputation pipeline step-by-step to identify where zeros are coming from.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from vn2.uncertainty.stockout_imputation import (
    find_neighbor_profiles,
    extract_profile_features,
    impute_stockout_sip,
    _impute_single_stockout,
    impute_all_stockouts
)

print("🔍 STOCKOUT IMPUTATION DIAGNOSTIC")
print("=" * 60)

# Load data
print("\n1. Loading data...")
df = pd.read_parquet('data/processed/demand_long.parquet')
print(f"   ✓ Loaded {len(df):,} observations")
print(f"   Stores: {df['Store'].nunique()}, Products: {df['Product'].nunique()}")

# Check stockouts
stockouts = df[df['in_stock'] == False]
print(f"\n2. Stockout analysis:")
print(f"   Total stockouts: {len(stockouts):,} ({len(stockouts)/len(df)*100:.1f}%)")
print(f"   SKUs with stockouts: {stockouts.groupby(['Store', 'Product']).size().shape[0]}")

if len(stockouts) == 0:
    print("\n⚠️  NO STOCKOUTS FOUND!")
    print("   The 'in_stock' column might be missing or all True")
    print(f"   Unique values in 'in_stock': {df['in_stock'].unique()}")
    sys.exit(1)

# Test a single stockout
print(f"\n3. Testing single stockout imputation:")
test_stockout = stockouts.iloc[0]
print(f"   Test case: Store {test_stockout['Store']}, Product {test_stockout['Product']}, Week {test_stockout['week']}")
print(f"   Observed sales (stock level): {test_stockout['sales']}")
print(f"   In stock flag: {test_stockout['in_stock']}")

# Test neighbor finding
sku_id = (test_stockout['Store'], test_stockout['Product'])
week = test_stockout['week']
print(f"\n4. Finding neighbors...")
neighbors = find_neighbor_profiles(sku_id, week, df, n_neighbors=20)
print(f"   Neighbors found: {len(neighbors)}")

if len(neighbors) == 0:
    print("   ⚠️  NO NEIGHBORS FOUND - This will cause fallback behavior")
    print("   Possible issues:")
    print("   - retail_week column missing or misaligned")
    print("   - No historical non-stockout data available")
    print("   - Window too restrictive")
else:
    print(f"   ✓ Found {len(neighbors)} neighbors")
    print(f"   Neighbor sales range: {neighbors['sales'].min():.2f} - {neighbors['sales'].max():.2f}")
    print(f"   Neighbor sales mean: {neighbors['sales'].mean():.2f}")

# Test quantile levels
q_levels = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
print(f"\n5. Testing SIP imputation...")

try:
    sip = impute_stockout_sip(
        sku_id=sku_id,
        week=week,
        stock_level=test_stockout['sales'],
        q_levels=q_levels,
        df=df,
        transform_name='log',
        n_neighbors=20
    )
    
    print(f"   ✓ SIP generated successfully")
    print(f"   SIP quantile values:")
    print(f"     Q0.01:  {sip[0.01]:.4f}")
    print(f"     Q0.50:  {sip[0.50]:.4f} (median)")
    print(f"     Q0.99:  {sip[0.99]:.4f}")
    print(f"   SIP range: {sip.min():.4f} - {sip.max():.4f}")
    print(f"   SIP mean: {sip.mean():.4f}")
    
    if sip.mean() == 0.0:
        print("\n   ❌ CRITICAL: SIP is all zeros!")
        print("   This is the root cause of the bug.")
        print(f"   All SIP values: {sip.values}")
    elif sip.mean() < test_stockout['sales']:
        print(f"\n   ⚠️  WARNING: Imputed demand ({sip.mean():.2f}) < observed stock ({test_stockout['sales']:.2f})")
        print("   Stockout demand should be HIGHER than stock level")
    else:
        print(f"\n   ✓ Imputed demand ({sip.mean():.2f}) > stock ({test_stockout['sales']:.2f}) - Looks correct")
        
except Exception as e:
    print(f"   ❌ ERROR during imputation: {e}")
    import traceback
    traceback.print_exc()

# Test historical data availability for this SKU
print(f"\n6. Checking historical data for test SKU:")
sku_history = df[(df['Store'] == sku_id[0]) & (df['Product'] == sku_id[1]) & (df['week'] < week)]
print(f"   Historical observations: {len(sku_history)}")
if len(sku_history) > 0:
    print(f"   Historical sales: mean={sku_history['sales'].mean():.2f}, std={sku_history['sales'].std():.2f}")
    print(f"   Historical stockouts: {(~sku_history['in_stock']).sum()}")
else:
    print("   ⚠️  COLD START - No historical data available!")

# Test the parallel wrapper
print(f"\n7. Testing parallel wrapper function:")
surd_transforms = pd.DataFrame()  # Empty = use default 'log' transform
try:
    key, sip_result = _impute_single_stockout(
        test_stockout, df, surd_transforms, q_levels, n_neighbors=20
    )
    if sip_result is None:
        print("   ❌ Wrapper returned None")
    else:
        print(f"   ✓ Wrapper succeeded")
        print(f"   Result key: {key}")
        print(f"   Result median: {sip_result[0.5]:.4f}")
except Exception as e:
    print(f"   ❌ Wrapper failed: {e}")

# Summary
print(f"\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY:")
print("=" * 60)
print(f"Stockouts detected: {len(stockouts):,}")
print(f"Neighbors available: {'Yes' if len(neighbors) > 0 else 'NO (CRITICAL)'}")
print(f"SIP generation: {'Success' if 'sip' in locals() and sip.mean() > 0 else 'FAILED (returns zeros)'}")

if 'sip' in locals() and sip.mean() == 0:
    print("\n🔥 ROOT CAUSE IDENTIFIED: Imputation returns zeros")
    print("   Check:")
    print("   1. Are all neighbors being filtered out?")
    print("   2. Is the transform/inverse transform working?")
    print("   3. Is splice_tail_from_neighbors broken?")
    print("   4. Are fallback paths returning zeros?")
    
print("\n" + "=" * 60)
