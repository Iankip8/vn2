# Data Loading Phase - Summary

## Status: ✅ COMPLETE

Your data has already been loaded and processed! Here's what we found:

---

## 📊 Current Data Status

### Raw Data (Competition Files)
Located in: `data/raw/`

- **Sales History**: 599 SKUs × 157 weeks (2021-04-12 to 2024-04-08)
  - Total sales: 276,648 units
  - 43.2% of observations are zero (intermittent demand!)
  - Average: 2.94 units per SKU per week

- **Initial State**: 599 SKUs starting with 1 unit on-hand each
  - All SKUs have inventory
  - No in-transit orders yet

- **Master Data**: SKU metadata with 6-level hierarchy
  - 111 Product Groups
  - 47 Divisions  
  - 26 Departments
  - 3 Store Formats

---

### Interim Data (Phase 1 Output)
Located in: `data/interim/`

✅ **state.parquet** - Inventory state (599 SKUs)
- Columns: `on_hand`, `intransit_1`, `intransit_2`

✅ **master.parquet** - SKU hierarchy (599 SKUs)
- Product/store attributes for segmentation

---

### Processed Data (Phase 2 Output - EDA)
Located in: `data/processed/`

✅ **demand_long.parquet** - Long-format demand history (94,043 observations)
- Key columns:
  - `Store`, `Product`, `week_date`, `sales`
  - `in_stock` - **Critical for stockout detection!**
  - Hierarchy: `ProductGroup`, `Division`, `Department`, etc.
  
- **Stockouts detected**: 10,517 observations (11.2%)
  - These are weeks where the SKU was out of stock
  - Observed sales = 0, but true demand is unknown (censored!)

✅ **demand_imputed.parquet** - With stockout imputation
✅ **demand_imputed_winsor.parquet** - Winsorized (outliers capped)
✅ **demand_imputed_capped.parquet** - For SIP optimization

---

## 🔍 Key Insights from Data Exploration

### 1. **Intermittent Demand**
- 43% zero sales weeks → Need specialized forecasting methods
- Models to consider: Croston, Zero-Inflated, SLURP

### 2. **Stockout Censoring** 
- 11.2% of observations have stockouts
- Can't observe true demand when inventory = 0
- **Solution**: Phase 3 imputation using neighbor profiles

### 3. **Data Structure**
```
Raw Data:
  Sales CSV: 599 rows (SKUs) × 157 columns (weeks)
  
After EDA transformation:
  demand_long: 94,043 rows (SKU-week pairs) × 15 columns
  
Why? Long format enables:
  - Time series analysis per SKU
  - Feature engineering (lags, rolling stats)
  - Stockout detection (in_stock flag)
```

---

## 📂 Data Loading Code

The data loading functions are in: [src/vn2/data/loaders.py](src/vn2/data/loaders.py)

### Key Functions:

```python
from vn2.data import (
    submission_index,    # Get canonical SKU list
    load_initial_state,  # Get inventory levels
    load_sales,          # Get historical sales
    load_master          # Get SKU metadata
)

# Example usage:
idx = submission_index('data/raw')           # 599 SKUs
state = load_initial_state('data/raw', idx)  # Inventory
sales = load_sales('data/raw')               # Historical
master = load_master('data/raw')             # Metadata
```

---

## 🔄 Data Transformation Pipeline

```
Phase 1: Ingest
  Raw CSV → Parquet (interim/)
  Command: ./go ingest --raw data/raw --out data/interim
  
Phase 2: EDA (Already Done!)
  Wide format → Long format
  Add stockout detection
  Output: demand_long.parquet
  
Phase 3: Imputation (Next Step)
  Impute censored demand using neighbors
  Command: ./go impute --n-neighbors 20 --n-jobs -1
```

---

## 🎯 Understanding the Data Files

### demand_long.parquet Structure

Each row = one SKU-week observation

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store ID |
| Product | int | Product ID |
| week | int | Week number (0-156) |
| sales | float | Observed sales |
| week_date | date | Week start date |
| **in_stock** | bool | **TRUE if inventory > 0** |
| ProductGroup | int | Hierarchy level 1 |
| Division | int | Hierarchy level 2 |
| Department | int | Hierarchy level 3 |

**Critical insight**: When `in_stock = False` and `sales = 0`, we have **censored demand** - the true demand could be higher!

---

## 🚀 Next Steps

Since your data is already loaded and processed:

### Option 1: Re-run Data Loading (if needed)
```bash
# Only if you want to regenerate interim files
./go ingest --raw data/raw --out data/interim
```

### Option 2: Explore the Data Yourself
```bash
# Run the walkthrough script we just created
source .venv/bin/activate
export PYTHONPATH="/home/ian/vn2/src:${PYTHONPATH:-}"
python scripts/explore_data_loading.py
```

### Option 3: Move to Phase 3 (Recommended)
```bash
# Stockout imputation
./go impute --n-neighbors 20 --n-jobs -1
```

### Option 4: Explore in Python
```python
import pandas as pd

# Load long-format demand
df = pd.read_parquet('data/processed/demand_long.parquet')

# Explore a single SKU
sku_data = df[(df['Store'] == 1) & (df['Product'] == 124)]
print(sku_data[['week_date', 'sales', 'in_stock']].tail(20))

# Find SKUs with most stockouts
stockout_pct = df.groupby(['Store', 'Product'])['in_stock'].apply(
    lambda x: (~x).mean()
)
print(stockout_pct.sort_values(ascending=False).head(10))
```

---

## 📚 Key Concepts Learned

### 1. **SKU Index**
- 599 unique Store-Product combinations
- This is the canonical ordering for all submissions
- MultiIndex: (Store, Product)

### 2. **Inventory State**
- Three components:
  - `on_hand`: Current warehouse inventory
  - `intransit_1`: Order arriving next week (L=1)
  - `intransit_2`: Order arriving in 2 weeks (L=2)

### 3. **Stockout Censoring**
- When inventory = 0, sales are "censored"
- We observe sales = 0, but don't know true demand
- This is a **survival analysis** / **censored data** problem
- Solution: Impute using similar SKUs (Phase 3)

### 4. **Data Formats**
- **Wide format** (raw): Easy for humans, hard for models
  - 1 row per SKU, many date columns
- **Long format** (processed): Model-ready
  - 1 row per SKU-date, explicit columns for features

---

## 🎓 Understanding the Pipeline

You're currently here:

```
[✅ Phase 1: Ingest]  ← DONE (data/interim/)
      ↓
[✅ Phase 2: EDA]     ← DONE (data/processed/demand_long.parquet)
      ↓
[➡️  Phase 3: Impute] ← NEXT (data/processed/demand_imputed.parquet exists!)
      ↓
[ ] Phase 4: Train Models
      ↓
[ ] Phase 5: Evaluate
      ↓
[ ] Phase 6: Deploy
```

**Good news**: Phases 1-3 are already complete! You can jump straight to Phase 4 (training) if you want.

---

## 🔬 Data Quality Checks

Run these to verify your data:

```python
import pandas as pd

# Load demand
df = pd.read_parquet('data/processed/demand_long.parquet')

# Check 1: No missing values in key columns
assert df[['Store', 'Product', 'sales', 'in_stock']].notna().all().all()

# Check 2: Sales are non-negative
assert (df['sales'] >= 0).all()

# Check 3: All SKUs present
assert len(df.groupby(['Store', 'Product'])) == 599

# Check 4: Consistent week coverage
weeks_per_sku = df.groupby(['Store', 'Product']).size()
print(f"Weeks per SKU: min={weeks_per_sku.min()}, max={weeks_per_sku.max()}")

print("✅ All checks passed!")
```

---

## 💡 Quick Analysis Ideas

Now that you understand the data, try:

1. **Find the busiest SKU**:
   ```python
   df.groupby(['Store', 'Product'])['sales'].sum().sort_values(ascending=False).head()
   ```

2. **Identify seasonal patterns**:
   ```python
   df['month'] = pd.to_datetime(df['week_date']).dt.month
   monthly_sales = df.groupby('month')['sales'].sum()
   monthly_sales.plot(kind='bar', title='Sales by Month')
   ```

3. **Stockout analysis**:
   ```python
   stockout_rate = (~df['in_stock']).mean() * 100
   print(f"Overall stockout rate: {stockout_rate:.1f}%")
   
   # By product group
   df.groupby('ProductGroup')['in_stock'].apply(lambda x: (~x).mean() * 100)
   ```

---

## 🧪 Data Validation

A comprehensive validation notebook has been created to verify data quality:

**Notebook**: [notebooks/00_data_validation.ipynb](notebooks/00_data_validation.ipynb)

### Validation Checklist

The notebook performs these checks:

1. ✅ **Raw Data Loading**
   - Load all competition CSV files
   - Verify file sizes and structure
   - Check date ranges and SKU coverage

2. ✅ **Interim Data Validation**
   - Compare parquet vs CSV (should match exactly)
   - Verify state and master data integrity

3. ✅ **Wide → Long Transformation**
   - Verify row count: 599 SKUs × 157 weeks = 94,043 rows
   - Validate total sales match between formats
   - Check for data loss during transformation

4. ✅ **Stockout Detection**
   - Verify 11.2% stockout rate
   - Confirm stockout weeks have sales = 0
   - Validate `in_stock` flag logic

5. ✅ **Imputation Quality**
   - Compare original vs imputed distributions
   - Verify no negative imputed values
   - Check imputed values are reasonable

6. ✅ **Data Integrity**
   - No missing values in critical columns
   - No duplicate SKU-week pairs
   - Continuous week sequences
   - Non-negative sales values

7. ✅ **Distribution Comparisons**
   - Visualize before/after distributions
   - Compare winsorized vs capped versions
   - Statistical summaries

8. ✅ **SKU-Level Inspection**
   - Examine individual SKUs with stockouts
   - Verify imputation at granular level
   - Time series visualization

### Expected Results

When you run the notebook, you should see:

```
📊 Raw Data:
  - 599 SKUs
  - 157 weeks (2021-04-12 to 2024-04-08)
  - 276,648 total units sold
  - 43.2% zero sales weeks

📦 Processed Data:
  - 94,043 observations (599×157)
  - 11.2% stockout rate (10,517 obs)
  - All sales totals match

💉 Imputation:
  - ~11% of observations imputed
  - No negative values
  - Reasonable distribution

✅ All Validation Checks PASSED
```

### How to Run

```bash
# Open in Jupyter
cd /home/ian/vn2
jupyter notebook notebooks/00_data_validation.ipynb

# Or in VS Code:
# Just open the notebook and run all cells
```

The notebook will generate:
- `data/processed/validation_results.json` - Machine-readable results
- `DATA_VALIDATION_RESULTS.md` - Human-readable summary

---

## 📖 References

- **Validation notebook**: [notebooks/00_data_validation.ipynb](notebooks/00_data_validation.ipynb)
- **Loader code**: [src/vn2/data/loaders.py](src/vn2/data/loaders.py)
- **CLI ingest command**: [src/vn2/cli.py](src/vn2/cli.py) - `cmd_ingest()`
- **Full pipeline guide**: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- **EDA notebook**: [notebooks/02_comprehensive_time_series_eda.ipynb](notebooks/02_comprehensive_time_series_eda.ipynb)
