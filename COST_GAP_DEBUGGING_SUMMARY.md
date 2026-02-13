# Cost Gap Debugging Summary
**Date**: February 5, 2026  
**Objective**: Reduce sequential evaluation costs from €10,045 to match VN2 benchmark of €5,248 (1.91x gap)

---

## Problem Statement

After implementing Patrick's 4 recommendations and training models with h=12 horizons, our sequential evaluation showed costs significantly higher than the VN2 competition benchmark:

- **Our best model (slurp_stockout_aware)**: €10,045
- **VN2 benchmark (simple policy)**: €5,248
- **Gap**: 1.91x (91% higher costs)

The VN2 competition uses:
- 599 SKUs
- 6-week sequential ordering with weekly demand revelation
- L=2 week lead time, R=1 week review period, T=3 week protection period
- Costs: €0.20/unit/week holding, €1.00/unit shortage
- Critical fractile: τ = 0.833 (83.3% service level)

---

## Root Cause Analysis

### 1. SKU-Level Investigation (Store 0, Product 126)

**Actual demand pattern** (12 weeks): `[0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2]`
- Total: 8 units
- Mean: 0.67 units/week
- Recent 13-week average: 0.77 units/week

**Our model behavior**:
- Initial position: 6 units (sufficient for 8 units demand)
- Orders placed: `[5, 0, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0]`
- Total ordered: 13 units
- **Over-ordered by 5 units (62% excess)**
- Total cost: €25.80 (vs VN2 simple: €8.20)

**VN2 simple policy**:
- Order-up-to level: 4 weeks × 0.77 = 3.08 units
- Week 1 order: 0 (position 6 > 3.08)
- Total ordered: 4 units
- Total cost: €8.20 (€7.20 holding + €1.00 shortage)

**Key finding**: Our model ordered **3.25x more** than VN2 for this SKU.

---

### 2. Forecast Over-Estimation

**Model forecasts vs actual demand**:

| Metric | Forecast (h=3) | Actual Recent | Ratio |
|--------|---------------|---------------|-------|
| Median (q=0.50) | 2.00 | 0.77 | 2.6x higher |
| 90th percentile | 5.00 | - | 6.5x higher |

**Historical context**:
- Last 52 weeks mean: 2.77 units/week
- Last 13 weeks mean: 0.77 units/week
- **Demand dropped to 28% of historical level**

The model trained on historical data (mean ~2.77) is forecasting median of 2.0, which was reasonable historically but not for the recent structural break.

---

### 3. Dataset-Wide Demand Decline

**Analysis of all 599 SKUs**:
- **80.5% of SKUs** have declining demand (recent < historical)
- **Median ratio**: 0.747 (recent is 75% of historical)
- **21% of SKUs**: demand dropped by >50%

**Demand ratio quantiles**:
- 10th percentile: 0.36
- 25th percentile: 0.53
- 50th percentile: 0.75
- 75th percentile: 0.93
- 90th percentile: 1.14

**Implication**: Models trained on historical data systematically over-forecast, leading to over-ordering and excess holding costs.

---

## Experiments and Results

### Baseline: 2-Period Newsvendor with h=3,4

**Implementation**:
- Uses `choose_order_L2()` from sequential_planner.py
- Optimizes for 2-period newsvendor (week t+1 only)
- Uses h=3 and h=4 forecasts (not aggregating h=5)
- PMF-based fractile optimization at τ=0.833

**Results**:
- slurp_stockout_aware: **€9,950** ✓ (slightly better than original €10,045)
- slurp_bootstrap: €10,016
- lightgbm_quantile: €10,236
- zinb: €11,240

**Note**: Switching from h=1,2 to h=3,4,5 gave a small -0.9% improvement.

---

### Experiment 1: Bias Correction (0.747×)

**Hypothesis**: Apply multiplicative bias correction to account for 75% demand decline.

**Implementation**:
```python
# In forecast_loader.py
quantiles_df = quantiles_df * 0.747  # Reduce forecasts by 25%
```

**Results**: ❌ **Made things WORSE**
- slurp_stockout_aware: €11,086 (+11.4% worse than baseline)
- Total portfolio: €44,229

**Why it failed**: 
- Models at h=3,4,5 may have already adapted to recent trends during training
- 0.747 correction was too aggressive
- Caused under-ordering → increased shortage costs
- Shortages are 5x more expensive than holding (€1.00 vs €0.20)

---

### Experiment 2: 3-Week PMF Aggregation

**Hypothesis**: Properly aggregate h=3+4+5 using convolution for full 3-week protection period.

**Implementation**:
```python
def aggregate_3week_pmf(h3_pmf, h4_pmf, h5_pmf):
    # Convolve h3 and h4
    agg_pmf = np.convolve(h3_pmf, h4_pmf)
    # Convolve with h5
    agg_pmf = np.convolve(agg_pmf, h5_pmf)
    return agg_pmf

# Find order-up-to at critical fractile 0.833
order_up_to = np.searchsorted(cdf_agg, 0.833)
```

**Results**: ❌ **Made things MUCH WORSE**
- slurp_stockout_aware: €11,326 (+13.8% worse than baseline)
- Total portfolio: €53,803

**Why it failed**:
- Convolved PMF has much higher variance than individual PMFs
- Var(X+Y+Z) = Var(X) + Var(Y) + Var(Z) for independent variables
- Higher variance → higher 83.3rd percentile → over-ordering
- Example: If h3, h4, h5 each have q90=5, aggregated q90 could be 15+
- Over-ordering → excessive holding costs

---

### Experiment 3: Simple VN2-Style (3-Week Coverage)

**Hypothesis**: Use VN2's simple approach - order up to N weeks × forecast median.

**Implementation**:
```python
def choose_order_simple_vn2(h3_pmf, h4_pmf, h5_pmf, current_position, coverage_weeks=3):
    # Get median from each PMF
    median_h3 = pmf_quantile(h3_pmf, 0.50)
    median_h4 = pmf_quantile(h4_pmf, 0.50)
    median_h5 = pmf_quantile(h5_pmf, 0.50)
    
    # Average median across horizons
    avg_median = (median_h3 + median_h4 + median_h5) / 3
    
    # Order up to coverage_weeks × median
    order_up_to = coverage_weeks * avg_median
    order_qty = max(0, order_up_to - current_position)
    
    return int(order_qty)
```

**Results**: ✓ **IMPROVEMENT!**
- slurp_stockout_aware: **€9,543** (-4.1% vs baseline €9,950)
- Improvement over complex newsvendor approach
- Still +81.8% vs VN2 benchmark (€5,248)

**Why it worked**:
- Simpler is better - uses median (50th percentile) not 83.3rd
- Avoids variance explosion from convolution
- More conservative ordering
- Actual VN2 uses recent 13-week average, not model forecasts

---

### Experiment 4: VN2-Style with 4-Week Coverage

**Hypothesis**: Match VN2's 4-week coverage (vs our 3-week protection period).

**Implementation**:
```python
# order_up_to = 4 weeks × median(h3, h4, h5)
coverage_weeks = 4
```

**Results**: ❌ **Slightly worse than 3-week**
- slurp_stockout_aware: **€10,194** (+2.5% vs baseline)
- Worse than 3-week coverage (€9,543)
- Still +94.2% vs VN2 benchmark

**Why it failed**:
- 4 weeks is too much coverage given our forecasts
- Our forecasts still over-estimate (median 2.0 vs actual 0.77)
- 4 × 2.0 = 8.0 order-up-to level (vs VN2's 4 × 0.77 = 3.08)

---

## Key Insights

### 1. Simpler is Better
- **Simple order-up-to** (median × weeks) beats complex newsvendor optimization
- VN2 achieves €5,248 with simple 4-week × recent average
- Our best: €9,543 with 3-week × forecast median
- Complex PMF convolution made things worse due to variance explosion

### 2. Forecast Quality > Algorithm Sophistication
- The 1.82x remaining gap (€9,543 vs €5,248) is primarily forecast quality
- Our models forecast median 2.0, actual recent average 0.77 (2.6x over-estimation)
- VN2 uses simple recent average which adapts faster to demand changes
- No amount of algorithm tuning can fix over-forecasting

### 3. Structural Break in Demand
- Demand systematically declined across 80.5% of SKUs
- Models trained on historical data don't adapt quickly enough
- Need either:
  - More recent-weighted training data
  - Online learning / adaptive forecasting
  - Simpler recent-average forecasts like VN2

### 4. Conservative Ordering Can Backfire
- High critical fractile (0.833) + over-forecasting = excessive ordering
- 3-week PMF aggregation amplifies variance
- Lower coverage (3 weeks) beats higher coverage (4 weeks) when forecasts are biased high

---

## Algorithm Comparison

| Approach | Method | Cost | vs Baseline | vs VN2 |
|----------|--------|------|-------------|---------|
| **Baseline** | 2-period newsvendor (h=3,4) | €9,950 | - | +89.6% |
| **Bias 0.747** | Baseline × 0.747 correction | €11,086 | +11.4% ❌ | +111.2% |
| **3-week PMF agg** | Convolve h3+h4+h5, τ=0.833 | €11,326 | +13.8% ❌ | +115.8% |
| **Simple 3-week** | median × 3 weeks | **€9,543** | **-4.1%** ✓ | +81.8% |
| **Simple 4-week** | median × 4 weeks | €10,194 | +2.5% ❌ | +94.2% |
| **VN2 Benchmark** | recent_avg × 4 weeks | €5,248 | -47.3% | - |

---

## Recommendations

### Immediate Actions

1. **Use Simple 3-Week Approach**
   - Best result achieved: €9,543
   - 4.1% better than baseline newsvendor
   - Much simpler and more interpretable
   - Code: `order_up_to = 3 * median(h3, h4, h5)`

2. **Consider Lower Coverage**
   - 3 weeks beats 4 weeks when forecasts are biased high
   - Could test 2-2.5 week coverage
   - Or use adaptive coverage based on forecast uncertainty

### Strategic Improvements Needed

3. **Improve Forecast Quality** (highest priority)
   - Current forecasts over-estimate by 2.6x for declining-demand SKUs
   - Options:
     - Retrain with more recent data (last 13-26 weeks only)
     - Add demand trend features to models
     - Use exponential smoothing / adaptive methods
     - Simple recent average (like VN2)
   
4. **Implement Adaptive Bias Correction**
   - Not global 0.747 (too aggressive)
   - Per-SKU recent/historical ratio
   - Or ensemble with simple recent-average forecast
   
5. **Test Hybrid Approach**
   - Use model forecasts for SKUs with stable demand
   - Use simple recent average for declining-demand SKUs
   - Selector based on demand trend (increasing vs declining)

### Alternative Approaches to Consider

6. **VN2's Exact Algorithm**
   - Implement exact VN2 benchmark: `order_up_to = 4 × recent_13week_avg`
   - Should achieve ~€5,248 if our simulation is correct
   - Baseline to beat before using complex models

7. **Quantile Regression at Lower Levels**
   - Instead of τ=0.833, try τ=0.50 or 0.70
   - Lower service level → less over-ordering
   - May reduce costs if shortage rate is currently very low

8. **Ensemble with Simple Forecasts**
   - `final_forecast = 0.5 × model_forecast + 0.5 × recent_average`
   - Blend sophistication with recency bias
   - Should reduce over-estimation

---

## Code Changes Implemented

### Modified Files

1. **`src/vn2/analyze/forecast_loader.py`**
   - Changed to load h=3,4,5 instead of h=1,2
   - Added `bias_correction` parameter (default 1.0)
   - Returns 3 PMF lists instead of 2

2. **`src/vn2/analyze/sequential_backtest.py`**
   - Added `convolve_pmfs()` for PMF convolution
   - Added `aggregate_3week_pmf()` for h3+h4+h5 aggregation
   - Added `choose_order_3week()` for aggregated PMF optimization
   - Added `choose_order_simple_vn2()` for simple median × weeks approach
   - Updated `run_12week_backtest()` to accept `forecasts_h3` parameter
   - Logic to use different ordering policies based on flags

3. **`src/vn2/analyze/sequential_eval.py`**
   - Updated to use new 3-PMF loader
   - Added `bias_correction` to SequentialConfig
   - Passes h3, h4, h5 to backtest function

### Test Scripts Created

1. **`scripts/test_bias_correction.py`** - Tests different approaches
2. **`scripts/test_simple_vn2_style.py`** - Tests VN2-style simple approach

### Results Files Generated

1. `models/results/sequential_results_seq12_h12.parquet` - Baseline (h=1,2)
2. `models/results/sequential_results_baseline_h345.parquet` - Baseline (h=3,4,5)  
3. `models/results/sequential_results_test_bias_correction.parquet` - With 0.747 bias
4. `models/results/sequential_results_3week_aggregation.parquet` - PMF convolution
5. `models/results/sequential_results_vn2_style.parquet` - Simple 3-week
6. `models/results/sequential_results_vn2_4week.parquet` - Simple 4-week

---

## Lessons Learned

1. **Always establish simple baselines first**
   - VN2's simple approach should have been implemented first
   - Complex algorithms don't help if forecasts are wrong

2. **Data quality > Model complexity**
   - Structural break in demand (25% decline) is the real issue
   - Sophisticated newsvendor can't fix bad forecasts

3. **Variance matters for high-quantile optimization**
   - PMF convolution increases variance
   - Higher variance → higher fractile values → over-ordering
   - Simple averaging (median × weeks) is more robust

4. **Cost asymmetry drives behavior**
   - Shortage cost (€1.00) is 5x holding cost (€0.20)
   - Critical fractile 0.833 is quite conservative
   - But over-forecasting makes it too conservative

5. **Recent data > Historical data for fast-changing environments**
   - 13-week recent average would likely beat our 52-week trained models
   - Online learning or sliding window training needed

---

## Next Steps

1. ✅ **Completed**: Identified root cause (over-forecasting due to demand decline)
2. ✅ **Completed**: Tested multiple ordering algorithms
3. ✅ **Completed**: Found best current approach (simple 3-week: €9,543)
4. 🔄 **In Progress**: Document findings (this file)
5. ⏭️ **Next**: Implement VN2 exact benchmark to validate
6. ⏭️ **Next**: Retrain models with recent data weighting
7. ⏭️ **Next**: Test hybrid model/simple ensemble
8. ⏭️ **Next**: Generate competition submission with best approach

---

## Technical Details

### VN2 Competition Timeline

```
Week 0: Initial state known
  ↓
Week 1: Place order Q1 → arrives Week 3
  ↓ Observe demand D1
Week 2: Place order Q2 → arrives Week 4  
  ↓ Observe demand D2
Week 3: Q1 arrives, Place order Q3 → arrives Week 5
  ↓ Observe demand D3
...
Week 10: Place order Q10 → arrives Week 12
  ↓
Week 11-12: No new orders, just fulfill demand
```

### Cost Calculation

```
Holding cost = €0.20 × end_inventory × num_weeks
Shortage cost = €1.00 × shortage_units × num_weeks
Total cost = Σ(holding + shortage) over 12 weeks
```

### Critical Fractile Derivation

```
τ = cu / (cu + co) = 1.0 / (1.0 + 0.2) = 0.833
```
Where:
- cu = underage cost (shortage) = €1.00
- co = overage cost (holding) = €0.20

---

**Document created**: February 5, 2026  
**Total experiments**: 5  
**Best result**: €9,543 (Simple 3-week approach)  
**Remaining gap to VN2**: 81.8% (€9,543 vs €5,248)  
**Primary bottleneck**: Forecast over-estimation due to demand structural break
