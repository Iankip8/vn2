# L=3 Lead Time Error Analysis

## Executive Summary

We identified and corrected a lead time implementation error in our VN2 competition code. After thorough backtesting, we found that **the error had minimal impact on our results** (~1.6% or €125 over 8 weeks).

**Key Finding**: The lead time error was NOT the primary cause of our 110th place finish. Our core issue was forecast quality, not the timing of order arrivals.

---

## The Error

### Competition Rule
> "Orders are made at the end of week X and received at the start of week X+3"

### Our Implementation
- We implemented **L=2** lead time (order arrives at week t+2)
- We should have implemented **L=3** lead time (order arrives at week t+3)
- Our forecasts only covered h=1 and h=2 horizons, missing h=3

### Technical Details
- `choose_order_L2()` optimized for demand at week t+2
- Should have been `choose_order_L3()` optimizing for demand at week t+3
- Model selector was trained on h=1/h=2 accuracy, not h=3

---

## Correction Implemented

### Code Changes
1. **`src/vn2/forecast/models/base.py`** - Updated `ForecastConfig.horizon` from 2 to 3
2. **`src/vn2/analyze/sequential_planner.py`** - Added `choose_order_L3()` function
3. **`scripts/regenerate_h3_forecasts.py`** - Regenerated 121,998 checkpoints with h=3 forecasts
4. **`scripts/generate_order_L3.py`** - New order generation using L=3 optimization
5. **`scripts/full_L3_simulation.py`** - Full simulation with rolling state propagation
6. **`scripts/optimized_L3_simulation.py`** - Simulation with h=3-optimized model selection

### New Checkpoints
- Directory: `models/checkpoints_h3/`
- Contains h=1, h=2, h=3 forecasts for all models

---

## Backtest Results

### Methodology
1. Corrected lead time from L=2 to L=3
2. Re-selected models based on h=3 historical accuracy (using only pre-competition data)
3. Ran full simulation with rolling state propagation
4. Estimated weeks 6-8 based on weeks 3-5 improvement ratio

### Weekly Comparison

| Week | Actual (L=2) | Corrected (L=3) | Difference |
|------|-------------|-----------------|------------|
| 1-2 | €913.80 | €913.80 | €0.00 (baseline) |
| 3 | €931.60 | €1,001.60 | +€70.00 |
| 4 | €1,780.40 | €1,328.60 | **-€451.80** |
| 5 | €1,004.20 | €1,318.60 | +€314.40 |
| **1-5 Total** | **€4,630.00** | **€4,562.60** | -€67.40 |

### 8-Week Estimate

| Metric | Value |
|--------|-------|
| Actual 8-week cost (L=2) | €7,787.40 |
| Estimated 8-week cost (L=3) | €7,662.73 |
| **Improvement** | **€124.67 (1.6%)** |
| Actual rank | 110th |
| Estimated rank with fix | ~105-108th |

---

## Why the Fix Didn't Help More

### 1. Forecast Quality Was the Real Issue
- Our h=3 forecasts were not significantly better than h=2 forecasts
- Some models showed massive over/under-prediction regardless of horizon
- Example: SKU 63_124 had h3_Q80=2 but actual demand was 85-144 units

### 2. Model Selection Limitations
- Original selector was optimized for h=1/h=2 accuracy
- h=3-optimized selector only improved results by ~1.8%
- No single model was consistently best across all SKUs

### 3. Week-to-Week Variance
- Week 4 improved significantly (-€452)
- But Week 5 got worse (+€314)
- Net effect nearly canceled out

---

## Comparison to Winners

| Team | 8-Week Cost | Gap to Us |
|------|-------------|-----------|
| Winner (Bartosz Szabłowski) | €4,677.00 | -€3,110 |
| Our Actual | €7,787.40 | - |
| Our Corrected Estimate | €7,662.73 | - |
| Gap remaining after fix | | ~€2,985 |

**The lead time fix would have closed only €125 of the €3,110 gap to the winner.**

---

## Lessons Learned

### 1. Lead Time Semantics Are Critical
Always verify:
- When orders are placed (start vs. end of period)
- When orders arrive (start vs. end of period)  
- How this translates to forecast horizons needed

### 2. Forecast Quality > Optimization
- We had sophisticated PMF-based newsvendor optimization
- But optimization can't overcome poor forecast inputs
- "Garbage in, garbage out" applies to inventory optimization

### 3. Model Selection Matters, But Has Limits
- Per-SKU model selection can improve results
- But if all models are poor, selection can't fix it
- Need fundamentally better forecasting approaches

### 4. Validate End-to-End
- We validated individual components (forecasts, optimization)
- But didn't validate the full pipeline against competition rules
- End-to-end testing would have caught the L=3 error earlier

---

## Files Created

| File | Purpose |
|------|---------|
| `models/checkpoints_h3/` | H=3 forecast checkpoints |
| `scripts/regenerate_h3_forecasts.py` | Regenerate checkpoints with h=3 |
| `scripts/generate_order_L3.py` | Generate orders with L=3 optimization |
| `scripts/full_L3_simulation.py` | Full simulation with state propagation |
| `scripts/optimized_L3_simulation.py` | Simulation with h=3 model selection |
| `scripts/backtest_L3_correction.py` | Initial backtest harness |
| `reports/backtest_L3/` | Simulation results and analysis |
| `reports/backtest_L3_optimized/` | H=3-optimized simulation results |

---

## Conclusion

The L=3 lead time error was a genuine bug that we should have caught earlier. However, fixing it would have improved our results by only ~1.6%, moving us from rank 110 to approximately rank 105-108.

**The primary cause of our poor performance was forecast quality, not the lead time implementation.**

Future improvements should focus on:
1. Better forecasting models (especially for intermittent/volatile demand)
2. Improved feature engineering
3. Ensemble methods that can adapt to different demand patterns
4. More rigorous end-to-end validation against competition rules

