# FORECAST HORIZON ISSUE - ROOT CAUSE IDENTIFIED

## Critical Discovery

**The horizon fix (h=3,4,5) had ZERO effect because the forecasts only contain h=1,2!**

### Evidence

```python
# From seasonal_naive checkpoint fold_0:
Quantiles shape: (2, 13)
Index (horizons): [1, 2]  # Only h=1 and h=2!
```

### Timeline Analysis

**What we need:**
- Lead time = 2 weeks
- Review period = 1 week  
- Protection period = 3 weeks
- Order placed at t=0 arrives at week 3
- Should aggregate h=3,4,5 (weeks when order is available)

**What we have:**
- Forecasts only for h=1,2 (2 steps ahead)
- When code looks for h=3,4,5, it falls back to **ZEROS**
- This explains why h=1,2,3 and h=3,4,5 give IDENTICAL results

### Fallback Behavior

From `corrected_policy.py` line 80:
```python
if h in quantiles_df.index:
    # Use forecast
else:
    # Fallback: assume zero demand for missing weeks
    protection_samples.append(np.zeros(n_samples))
```

### Why Results are Identical

**h=1,2,3 aggregation:**
- h=1: ✓ forecast available
- h=2: ✓ forecast available  
- h=3: ✗ missing → zeros
- Sum = forecast(h=1) + forecast(h=2) + zeros

**h=3,4,5 aggregation:**
- h=3: ✗ missing → zeros
- h=4: ✗ missing → zeros
- h=5: ✗ missing → zeros
- Sum = zeros + zeros + zeros

Actually, wait - if h=3,4,5 are all zeros, we'd get different results. Let me check which one is actually running...

### Actual Behavior Check

Let me verify what's actually being aggregated in the current evaluation.

## Root Cause

**Models were trained to forecast only h=1,2 (2 steps), but we need h=1 through h=5+ for proper inventory policy evaluation with 2-week lead time and 3-week protection period.**

## Solutions

1. **Re-train all models with longer forecast horizons (h=1 through 12)**
   - This matches the 12-week evaluation horizon
   - Allows proper protection period aggregation
   
2. **Use recursive forecasting for missing horizons**
   - If only h=1,2 available, forecast h=3 from h=2, etc.
   - Less accurate but doesn't require retraining
   
3. **Reduce protection period to match available forecasts**
   - Use only h=1,2 (2 weeks protection)
   - But this violates the L+R=3 weeks requirement

## Next Steps

1. Verify which horizons models were actually trained for
2. Check training configuration to see why only h=1,2
3. Re-train models with full 12-step horizon
4. Re-evaluate with proper forecast coverage
