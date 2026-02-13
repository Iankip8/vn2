# Sequential Evaluation Debugging Summary

## Problem
Our sequential evaluation costs are **1.90x higher** than VN2 benchmark:
- **Our best model**: €9,950 (slurp_stockout_aware with 2-period newsvendor)
- **VN2 benchmark**: €5,248 (simple 4-week moving average order-up-to)

## Root Causes Identified

### 1. Systematic Demand Decline (Minor Impact)
- **Finding**: 80.5% of SKUs have declining demand (recent vs historical)
- **Median ratio**: 0.747 (recent is 75% of historical)
- **Test**: Applied 0.747× bias correction to forecasts
- **Result**: Made things WORSE (€9,950 → €11,086, +11.4%)
- **Conclusion**: Models may have already adapted to recent trends, or 0.747 is too aggressive

### 2. Algorithm Complexity (Major Impact)
**Hypothesis**: Simpler is better for this problem

**VN2 Benchmark Approach**:
```python
# Very simple
order_up_to = 4 weeks × recent_average_demand
order = max(0, order_up_to - inventory_position)
```

**Our Approach**:
- Complex probabilistic newsvendor optimization
- PMF-based expected cost minimization
- Critical fractile calculations
- Result: More complex but WORSE performance

## Tests Conducted

### Test 1: Bias Correction (0.747×)
- **Config**: Multiply all quantiles by 0.747
- **Result**: €10,045 → €11,086 (+10.4% WORSE)
- **Reason**: Too aggressive, caused shortages

### Test 2: 3-Week PMF Aggregation  
- **Config**: Convolve h=3,4,5 PMFs, use critical fractile 0.833
- **Result**: €9,950 → €11,326 (+13.8% WORSE)
- **Reason**: Convolved PMF has higher variance → orders at 83.3% quantile are too high

### Test 3: Simple VN2-Style (RUNNING)
- **Config**: order_up_to = 3 weeks × median(h3, h4, h5)
- **Status**: Evaluation in progress (876/2396 tasks complete)
- **Expected**: Should perform better than complex newsvendor

## Sample SKU Analysis (Store 0, Product 126)

**Initial State**:
- On-hand: 3 units
- In-transit W+1: 0 units  
- In-transit W+2: 3 units
- **Total position**: 6 units

**Actual Demand** (12 weeks): `[0,0,0,2,2,0,0,0,0,0,2,2]` = **8 units**

**Model Behavior**:
- **Baseline (2-period)**: Ordered 13 units total, cost €8.00
- **VN2 simple (4-week)**: Would order ~4 units total, cost €8.20
- **Problem**: Our model ordered 62% more than needed (13 vs 8)

**Forecast Analysis**:
- Historical mean (52 weeks): 2.77 units/week
- Recent mean (13 weeks): 0.77 units/week
- Model forecast median (h=3): 2.00 units/week
- **Model is using historical trend, not recent!**

## Key Insights

1. **Simpler Forecasting Wins**: VN2's simple 4-week moving average beats our probabilistic models
   
2. **Critical Fractile Too High**: Using 0.833 (83% service level) with inflated forecasts → over-ordering
   
3. **Forecast Accuracy Issue**: Models forecast 2.0 units/week, actual recent is 0.77 → 2.6x over-forecast
   
4. **PMF Aggregation Amplifies**: Convolving PMFs increases variance, making problem worse

5. **Structural Break**: Demand has systematically declined but models trained on historical data

## Next Steps

### Option A: Match VN2's Simplicity
- Use simple coverage-based policy (median × weeks)
- Test with different coverage levels (3, 4, 5 weeks)
- **Currently testing**: 3-week coverage with forecast medians

### Option B: Fix Probabilistic Approach
- Use lower critical fractile (try 0.50 or 0.70 instead of 0.833)
- Apply conservative forecast adjustment
- Use simple sum instead of convolution for aggregation

### Option C: Hybrid Approach
- Simple order-up-to for base stock
- Add safety stock based on forecast uncertainty
- Target: €5,248 VN2 benchmark

## Files Modified

1. `src/vn2/analyze/forecast_loader.py`
   - Added bias_correction parameter
   - Load h=3,4,5 instead of h=1,2
   - Apply multiplicative bias adjustment

2. `src/vn2/analyze/sequential_backtest.py`
   - Added `choose_order_simple_vn2()` function
   - Added `choose_order_3week()` with PMF aggregation
   - Modified to accept forecasts_h3 parameter

3. `scripts/test_bias_correction.py`
   - Test harness for different approaches

## Results Summary

| Approach | slurp_stockout_aware Cost | vs Baseline | vs VN2 Benchmark |
|----------|---------------------------|-------------|------------------|
| Baseline (2-period, h=3,4) | €9,950 | - | +89.6% |
| With 0.747 bias correction | €11,086 | +11.4% worse | +111.3% |
| 3-week PMF aggregation | €11,326 | +13.8% worse | +115.9% |
| Simple VN2-style (3-week) | **TBD** | TBD | TBD |
| **VN2 Benchmark** | **€5,248** | **-47.2%** | **0%** |

## Conclusion

The complex probabilistic newsvendor approach is consistently underperforming VN2's simple moving average. This suggests:

1. **Forecast quality is the bottleneck**, not optimization sophistication
2. **Simple policies are more robust** to forecast errors
3. **Over-engineering backfired** - added complexity without benefit

**Recommendation**: Implement VN2's exact simple approach as baseline, then incrementally add sophistication only if it improves performance.
