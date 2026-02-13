# 🚨 Data Validation Issue Found

**Date**: February 2, 2026  
**Status**: ❌ IMPUTATION NOT COMPLETE

---

## Problem Summary

The validation notebook revealed a **critical issue**: **Stockout imputation has not been run**.

### Evidence:

- **Expected**: ~11% imputation rate (10,517 stockout observations)
- **Actual**: 0% imputation rate (0 imputed observations)
- **Impact**: Models will train on censored data (sales = 0 during stockouts)

### Why This Matters:

When inventory = 0, observed sales = 0, but **true demand is unknown**. Without imputation:
- Forecasts will be biased LOW
- Stockout periods look like "zero demand" periods
- Model will underestimate demand variability
- Inventory decisions will be suboptimal

---

## Root Cause Analysis

The files exist but were likely created as placeholders:
```
data/processed/demand_imputed.parquet         ✓ exists
data/processed/demand_imputed_winsor.parquet  ✓ exists  
data/processed/demand_imputed_capped.parquet  ✓ exists
```

However, the `imputed` flag in all files = `False` for all observations.

**Conclusion**: The `./go impute` command was not run, or ran but failed silently.

---

## Solution

### Step 1: Run Imputation

```bash
cd /home/ian/vn2
./go impute --n-neighbors 20 --n-jobs -1
```

**What this does**:
1. Loads `demand_long.parquet` (94,043 observations)
2. Identifies 10,517 stockout observations (11.2%)
3. For each stockout:
   - Finds K=20 similar SKUs based on sales patterns
   - Uses their demand profiles to create probability distributions
   - Generates quantile forecasts (13 levels)
4. Saves imputed data with `imputed=True` flag
5. Creates full SIP library in `imputed_sips.parquet`

**Expected runtime**: ~5-15 minutes (depends on CPU cores)

### Step 2: Re-validate

After imputation completes, re-run the validation notebook:

```bash
# Open in Jupyter or VS Code
jupyter notebook notebooks/00_data_validation.ipynb

# Run all cells (Ctrl+Shift+Enter)
```

**Expected results after fix**:
```
✅ Imputation rate: 11.2%
✅ Imputed observations: 10,517
✅ No negative imputed values
✅ Reasonable imputed distributions
```

### Step 3: Verify & Proceed

Once validation passes, proceed to model training:

```bash
./go forecast --config configs/forecast.yaml --n-jobs 12
```

---

## Technical Details

### What the Imputation Process Does

**Input**: `demand_long.parquet`
- 94,043 observations
- 10,517 stockouts (in_stock = False, sales = 0)

**Process**: For each stockout observation
1. **Find neighbors**: Identify K=20 SKUs with similar:
   - Sales volume
   - Seasonality pattern
   - Trend characteristics
   - Coefficient of variation

2. **Build SIP**: Create Stochastic Information Packet
   - Sample from neighbor distributions
   - Preserve correlations
   - Generate quantiles: [0.01, 0.05, 0.1, ..., 0.95, 0.99]

3. **Impute**: Replace sales=0 with quantile forecasts
   - Median (50th percentile) as point estimate
   - Full distribution for uncertainty quantification

**Output**: `demand_imputed.parquet`
- Same 94,043 observations
- 10,517 with `imputed=True` flag
- Stockout obs now have sales > 0 (imputed values)

### Why K-Nearest Neighbors?

- **Profile-based**: Uses entire sales history, not just point values
- **Non-parametric**: No distributional assumptions
- **Preserves patterns**: Captures SKU-specific characteristics
- **Robust**: Works well for intermittent demand

---

## Validation Checklist

After running imputation, verify:

- [ ] Imputation rate ≈ 11.2% (matches stockout rate)
- [ ] 10,517 observations have `imputed=True`
- [ ] No negative imputed values
- [ ] Imputed mean close to original mean (2-3 units)
- [ ] Stockout obs now have sales > 0
- [ ] File `imputed_sips.parquet` exists
- [ ] All validation checks PASS

---

## Current Status

```
Phase 1: Data Ingestion     ✅ COMPLETE
Phase 2: EDA Processing      ✅ COMPLETE
Phase 3: Imputation          ❌ NOT RUN
Phase 4: Model Training      ⏸️  BLOCKED
```

**Next Action**: Run `./go impute --n-neighbors 20 --n-jobs -1`

---

## Files Modified

When imputation runs successfully, these files will be updated:

- `data/processed/demand_imputed.parquet` - Main imputed dataset
- `data/processed/demand_imputed_winsor.parquet` - Winsorized version
- `data/processed/demand_imputed_capped.parquet` - Capped for optimization
- `data/processed/imputed_sips.parquet` - Full SIP library
- `data/processed/imputation_summary.parquet` - Quality metrics

---

## References

- **Validation notebook**: [notebooks/00_data_validation.ipynb](notebooks/00_data_validation.ipynb)
- **Imputation code**: [src/vn2/forecast/imputation_pipeline.py](src/vn2/forecast/imputation_pipeline.py)
- **CLI command**: [src/vn2/cli.py](src/vn2/cli.py) - `cmd_impute_stockouts()`
- **Pipeline guide**: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Phase 3
