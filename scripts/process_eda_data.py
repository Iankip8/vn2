#!/usr/bin/env python3
"""Extract essential data processing from EDA notebook to create required processed files"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("📊 Processing raw data for pipeline...")

# Create processed directory if it doesn't exist
Path('data/processed').mkdir(parents=True, exist_ok=True)

# Load raw files
print("Loading raw data...")
sales = pd.read_csv('data/raw/Week 0 - 2024-04-08 - Sales.csv')
stock = pd.read_csv('data/raw/Week 0 - In Stock.csv')
master = pd.read_csv('data/raw/Week 0 - Master.csv')

# Melt to long format
print("Reshaping to long format...")
sales_long = sales.melt(
    id_vars=['Store', 'Product'], 
    var_name='week', 
    value_name='sales'
)
sales_long['week_date'] = pd.to_datetime(sales_long['week'])
sales_long['year'] = sales_long['week_date'].dt.year
sales_long['retail_week'] = sales_long['week_date'].dt.isocalendar().week
sales_long['month'] = sales_long['week_date'].dt.month

stock_long = stock.melt(
    id_vars=['Store', 'Product'], 
    var_name='week', 
    value_name='in_stock'
)

# Merge
df = sales_long.merge(
    stock_long, 
    on=['Store', 'Product', 'week']
).merge(
    master,
    on=['Store', 'Product']
).sort_values(['Store', 'Product', 'week']).reset_index(drop=True)

# Memory optimization
print("Optimizing memory usage...")
df['Store'] = df['Store'].astype('int16')
df['Product'] = df['Product'].astype('int16')
df['sales'] = df['sales'].astype('float32')
df['year'] = df['year'].astype('int16')
df['retail_week'] = df['retail_week'].astype('int8')
df['month'] = df['month'].astype('int8')
df['in_stock'] = df['in_stock'].astype('bool')

for col in ['ProductGroup', 'Division', 'Department', 'DepartmentGroup', 'StoreFormat', 'Format']:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f"Data shape: {df.shape}")
print(f"Date range: {df['week_date'].min()} to {df['week_date'].max()}")
print(f"Stores: {df['Store'].nunique()}, Products: {df['Product'].nunique()}")

# Save demand_long.parquet (required for imputation)
print("Saving demand_long.parquet...")
df.to_parquet('data/processed/demand_long.parquet', index=False)

# Create basic SLURP structure (simplified version)
print("Creating basic SLURP structure...")
slurp_df = df.copy()
slurp_df.to_parquet('data/processed/slurp_master.parquet', index=False)

# Create summary stats
print("Computing summary statistics...")
summary = df.groupby(['Store', 'Product']).agg({
    'sales': ['mean', 'std', 'min', 'max', 'sum'],
    'in_stock': 'mean'
}).reset_index()
summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
summary.to_parquet('data/processed/summary_stats.parquet', index=False)

print("\n✅ Processed data saved:")
print(f"  - data/processed/demand_long.parquet ({len(df):,} rows)")
print(f"  - data/processed/slurp_master.parquet")
print(f"  - data/processed/summary_stats.parquet")
print("\n✅ Ready for next steps: imputation, forecasting")
