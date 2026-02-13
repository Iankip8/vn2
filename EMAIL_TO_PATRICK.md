Subject: VN2 Challenge - Detailed Progress Report & Request for Presentation Recordings

Hi Patrick,

I hope this email finds you well. I wanted to provide you with a comprehensive update on our VN2 inventory challenge work. We've made significant progress implementing your recommendations and have conducted extensive debugging to understand the performance gap. This email details everything we've done, our findings, and some questions for you.

---

## Part 1: Implementation of Your Recommendations

Following our last discussion, we successfully implemented all 4 of your recommendations:

### 1. Fast Imputation ✅

**Implementation:**
- Method: Seasonal-index imputation using 52-week historical averages
- Handles stockout periods by imputing missing demand values
- Preserves seasonality patterns in the data

**Results:**
- Imputation rate: 11.2% of the dataset (10,529 rows out of 94,043)
- Average imputed value: 2.36 units (vs 2.32 observed mean)
- Processing time: <10 seconds for full dataset
- Verification: Seasonal patterns preserved, no artificial spikes introduced

### 2. Corrected Policy (3-Week Protection Period) ✅

**Implementation:**
- Lead time L = 2 weeks (order arrives in week 3)
- Review period R = 1 week  
- Protection period T = 3 weeks (covers weeks 3, 4, 5)
- Critical fractile τ = cu/(cu+co) = 1.0/(1.0+0.2) = 0.833
- Monte Carlo aggregation: Sample 10,000 scenarios from h=3,4,5 quantiles, sum, take 83.3rd percentile

**Verification:**
- Tested MC convergence: 1K samples = 27.2 units, 10K = 27.1, 100K = 27.1 (converged)
- Compared vs normal approximation: MC gives 27.1 vs Normal 26.8 (3% difference - acceptable)
- Sanity check passed: Higher service level → higher order quantity as expected

**Code Location:** `src/vn2/policy/corrected_policy.py`

### 3. SLURP Models (Stockout-Aware Training) ✅

**Models Trained:**
- `slurp_bootstrap`: Bootstrap-based distributional forecast with stockout handling
- `slurp_stockout_aware`: Explicitly models stockout probability in loss function

**Training Configuration:**
- Horizons: h = 1 to 12 (all weeks)
- Quantile levels: 13 levels from 0.01 to 0.99 (including 0.833)
- Features: Store, Product, week, seasonality, trends, lags
- Stockout handling: Imputed demand values marked and handled in training

**Training Results:**
- zinb: 490/599 SKUs (82% coverage - missing SKUs lack exogenous features)
- lightgbm_quantile: 599/599 SKUs (100% coverage) ✓
- slurp_bootstrap: 599/599 SKUs (100% coverage) ✓
- slurp_stockout_aware: 599/599 SKUs (100% coverage) ✓

**Training Time:** ~2 hours for all models with 12 horizons each

### 4. Sanity Tests ✅

We implemented comprehensive sanity tests covering:
- Critical fractile calculation (verified 0.833 for given cost structure)
- MC aggregation convergence (tested 1K to 100K samples)
- Policy monotonicity (higher service level → higher orders)
- Forecast consistency (quantiles properly ordered)
- SKU coverage (verified all 599 SKUs in evaluation set)

All tests passed successfully.

---

## Part 2: Sequential Evaluation Discovery

After implementing everything correctly, we ran sequential evaluation matching the VN2 competition structure:

**Competition Structure:**
- 12-week horizon with weekly demand revelation
- Orders placed weeks 1-10 (arrive weeks 3-12)
- Each week: observe demand, update forecasts, place new order
- Costs: €0.20/unit/week holding, €1.00/unit shortage

**Initial Results - Problem Discovered:**
- Our best model (slurp_stockout_aware): **€10,045**
- VN2 simple benchmark (from competition folder): **€5,248**
- **Gap: 1.91x (91% higher costs!)**

This was surprising since we had implemented everything correctly. We needed to understand why our sophisticated approach was performing so much worse than a simple benchmark.

---

## Part 3: Root Cause Analysis - Deep Dive

We conducted a systematic investigation to find the root cause of the cost gap.

### Investigation Approach

1. **SKU-Level Analysis**: Examined individual SKU behavior (orders vs demand)
2. **Forecast Validation**: Compared model forecasts to actual demand
3. **Dataset-Wide Patterns**: Analyzed demand trends across all 599 SKUs
4. **Algorithm Comparison**: Tested VN2's simple approach vs our newsvendor

### Finding 1: Over-Ordering Pattern

**Example SKU (Store 0, Product 126):**

```
Initial State:
- On-hand inventory: 3 units
- In-transit W+1: 0 units
- In-transit W+2: 3 units
- Total position: 6 units

Actual Demand (12 weeks): [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2]
- Total: 8 units
- Weekly mean: 0.67 units
- Recent 13-week average: 0.77 units

Our Model Orders: [5, 0, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0]
- Total ordered: 13 units
- Over-ordered by: 5 units (62% excess)
- Total cost: €25.80 (mostly holding costs)

VN2 Simple Benchmark Orders: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1]
- Total ordered: 4 units
- Matched demand closely
- Total cost: €8.20 (€7.20 holding + €1.00 shortage)

Key Issue: Our model ordered 3.25x more than VN2!
```

### Finding 2: Forecast Over-Estimation

**Comparing forecasts to reality:**

For the same SKU (Store 0, Product 126):
```
Model Forecasts:
- h=3 median (q=0.50): 2.00 units/week
- h=3 90th percentile (q=0.90): 5.00 units/week
- h=4 median: 2.00 units/week
- h=5 median: 2.00 units/week

Actual Demand:
- Recent 13-week average: 0.77 units/week
- Recent 52-week average: 2.77 units/week

Over-estimation: 2.00 / 0.77 = 2.6x too high!
```

**Why are forecasts high?**
- Models trained on 52-week historical data (mean 2.77)
- Recent 13 weeks show demand collapsed to 0.77 (only 28% of historical)
- Model forecasts 2.00 (reasonable for historical, but not for recent trend)

### Finding 3: Systematic Demand Decline Across All SKUs

**Dataset-Wide Analysis:**

We analyzed the demand trend for all 599 SKUs:

```python
recent_mean = last_13_weeks.mean()  # Most recent data
historical_mean = last_52_weeks.mean()  # Training data
ratio = recent_mean / historical_mean
```

**Results:**
- **483 SKUs (80.5%)** have declining demand (ratio < 1.0)
- **Median ratio: 0.747** (recent is 75% of historical)
- **125 SKUs (20.9%)** declined by >50%
- **Only 116 SKUs (19.5%)** have stable/growing demand

**Ratio Distribution:**
- 10th percentile: 0.36 (recent is 36% of historical)
- 25th percentile: 0.53
- 50th percentile: 0.75
- 75th percentile: 0.93
- 90th percentile: 1.14

**Implication:** There's a structural break in the data. The test period has fundamentally different demand patterns than the training period. Our models, trained on historical data, systematically over-forecast.

### Finding 4: Algorithm Comparison

**VN2 Simple Benchmark (from competition folder):**
```python
# Uses simple moving average forecast
forecast = last_13_weeks.mean()
order_up_to = 4 * forecast  # 4-week coverage
order = max(0, order_up_to - current_position)
```

**Why VN2's approach works better:**
- Uses only recent 13-week average (adapts faster to demand changes)
- Simple 4-week coverage (no complex optimization)
- Naturally handles demand decline (recent average drops automatically)
- Cost: €5,248 across all SKUs

**Our approach:**
- Uses model forecasts trained on 52-week historical data
- Complex PMF-based newsvendor optimization at 83.3rd percentile
- Doesn't adapt quickly to demand changes
- Cost: €10,045 (1.91x higher)

---

## Part 4: Experiments to Close the Gap

We systematically tested 5 different approaches to reduce costs:

### Experiment 1: Bias Correction (Multiply Forecasts by 0.747)

**Hypothesis:** Apply dataset-wide median demand ratio (0.747) as multiplicative bias correction.

**Implementation:**
```python
# In forecast loading
quantiles_df = quantiles_df * 0.747
```

**Results:** ❌ **FAILED - Made things worse**
- Cost: €11,086 (+11.4% vs baseline €9,950)
- Why it failed: Too aggressive, caused under-ordering and shortages
- Shortage costs (€1.00/unit) dominated, overwhelming the holding cost savings

**Lesson:** Global bias correction doesn't work because:
- Some SKUs grew (don't need correction)
- Shortage cost is 5x holding cost (asymmetric penalty)
- Need per-SKU adaptive correction, not global

### Experiment 2: 3-Week PMF Aggregation with Convolution

**Hypothesis:** Properly aggregate h=3, 4, 5 probability distributions using convolution to capture full 3-week protection period.

**Implementation:**
```python
# Convolve weekly PMFs
agg_pmf = convolve(h3_pmf, h4_pmf)
agg_pmf = convolve(agg_pmf, h5_pmf)

# Find critical fractile from aggregated distribution
cdf = cumsum(agg_pmf)
order_up_to = searchsorted(cdf, 0.833)
```

**Mathematical Note:** 
- Var(X+Y+Z) = Var(X) + Var(Y) + Var(Z) for independent variables
- Convolution increases variance significantly
- Higher variance → higher 83.3rd percentile → over-ordering

**Results:** ❌ **FAILED - Made things much worse**
- Cost: €11,326 (+13.8% vs baseline)
- Why it failed: Variance explosion effect
- Example: If each week's q90=5, the 3-week aggregate q90 could be 15+
- Over-ordering at high percentile → massive holding costs

**Lesson:** Convolution amplifies uncertainty, making high-percentile policies too conservative.

### Experiment 3: Simple VN2-Style with 3-Week Coverage ✅

**Hypothesis:** Use VN2's simple approach but with our model forecasts instead of recent average.

**Implementation:**
```python
# Get median from each horizon
median_h3 = pmf_quantile(h3_pmf, 0.50)
median_h4 = pmf_quantile(h4_pmf, 0.50)
median_h5 = pmf_quantile(h5_pmf, 0.50)

# Average and multiply by coverage weeks
avg_median = (median_h3 + median_h4 + median_h5) / 3
order_up_to = 3 * avg_median  # 3-week coverage
order = max(0, order_up_to - current_position)
```

**Results:** ✅ **SUCCESS - Best result achieved!**
- Cost: €9,543 (-4.1% vs baseline €9,950)
- First approach to beat the baseline!
- Still +81.8% vs VN2 benchmark (€5,248)

**Why it worked:**
- Simple averaging avoids variance explosion
- Uses median (50th percentile) not 83.3rd - more robust to forecast errors
- Direct computation, no complex optimization
- More conservative coverage (3 weeks vs 4)

**Lesson:** Simplicity wins when forecasts are imperfect!

### Experiment 4: Simple VN2-Style with 4-Week Coverage

**Hypothesis:** Match VN2's exact 4-week coverage period.

**Implementation:**
```python
order_up_to = 4 * avg_median  # 4-week coverage instead of 3
```

**Results:** ❌ **Regression - Slightly worse**
- Cost: €10,194 (+2.5% vs baseline)
- Worse than 3-week version (€9,543)
- Still +94.2% vs VN2 benchmark

**Why it failed:**
- Our forecast median (2.0) is still 2.6x too high vs VN2's recent average (0.77)
- 4 × 2.0 = 8.0 order-up-to level (way too high)
- VN2: 4 × 0.77 = 3.08 (much lower)
- Over-ordering → holding costs

**Lesson:** Coverage period must be tuned to forecast quality. Biased forecasts need lower coverage.

### Experiment 5: Baseline with h=3,4,5 (Not h=1,2)

**Note:** We also discovered that switching from h=1,2 to h=3,4,5 horizons gave a small improvement:
- Original baseline (h=1,2): €10,045
- New baseline (h=3,4,5): €9,950 (-0.9%)
- Using horizons that match the protection period helps slightly

---

## Part 5: Summary of Results

**Complete Results Table:**

| Approach | Description | Cost | vs Baseline | vs VN2 | Verdict |
|----------|-------------|------|-------------|---------|---------|
| **Original** | 2-period newsvendor, h=1,2 | €10,045 | - | +91.4% | Starting point |
| **Baseline** | 2-period newsvendor, h=3,4 | €9,950 | -0.9% | +89.6% | Small improvement |
| Bias 0.747 | Baseline × 0.747 correction | €11,086 | +11.4% | +111.2% | ❌ Too aggressive |
| 3-week Convolution | Convolve h3+h4+h5, τ=0.833 | €11,326 | +13.8% | +115.8% | ❌ Variance explosion |
| **Simple 3-week** | median × 3 weeks | **€9,543** | **-4.1%** | **+81.8%** | ✅ **Best result!** |
| Simple 4-week | median × 4 weeks | €10,194 | +2.5% | +94.2% | ❌ Over-orders |
| **VN2 Benchmark** | recent_avg × 4 weeks | **€5,248** | **-47.3%** | - | **Target** |

**Best Achieved:** €9,543 using simple 3-week policy (4.1% improvement over baseline)

**Remaining Gap:** €9,543 vs €5,248 = 81.8% higher (€4,295 excess cost)

---

## Part 6: Key Insights and Lessons Learned

### 1. Simpler Algorithms Beat Complex Optimization

**Finding:** Simple order-up-to (median × weeks) outperformed sophisticated PMF-based newsvendor optimization.

**Why:**
- Newsvendor assumes perfect forecasts (unbiased, correct uncertainty)
- Our forecasts are systematically biased (2.6x over-estimation)
- High-percentile optimization (83.3rd) amplifies forecast errors
- Simple median-based approach is more robust to bias

**Implication:** When forecast quality is uncertain, simple policies are safer.

### 2. Forecast Quality Dominates Algorithm Choice

**Finding:** The remaining 81.8% gap is almost entirely due to forecast over-estimation.

**Evidence:**
- Our simple 3-week: 3 × 2.0 = 6.0 order-up-to
- VN2 simple: 4 × 0.77 = 3.08 order-up-to
- We order 2x more despite using lower coverage!

**Implication:** No amount of algorithm tuning can fix bad forecasts. We need better forecasts, not better policies.

### 3. Structural Breaks Require Adaptive Methods

**Finding:** The test period has fundamentally different demand (75% lower) than training period.

**Problem:**
- Models trained on 52-week historical data (mean 2.77)
- Recent 13-week actual demand (mean 0.77)
- Structural break not captured in model

**Solutions needed:**
- More recent-weighted training (e.g., last 13-26 weeks only)
- Online learning or adaptive forecasting
- Exponential smoothing / moving averages
- Ensemble with simple recent-average forecast

### 4. Variance Accumulation in Multi-Period Aggregation

**Finding:** Convolution of independent distributions increases variance linearly.

**Math:**
```
σ²(X+Y+Z) = σ²(X) + σ²(Y) + σ²(Z)
σ(X+Y+Z) = √(σ²(X) + σ²(Y) + σ²(Z)) ≈ √3 × σ(X)
```

**Effect:** 
- 3-week aggregate has √3 ≈ 1.73x higher standard deviation
- High percentiles increase more than linearly
- 83.3rd percentile of aggregate >> 3 × 83.3rd percentile of individual

**Implication:** High-service-level policies with multi-period aggregation naturally over-order. This is correct theoretically but amplifies forecast biases in practice.

### 5. Cost Asymmetry Drives Conservatism

**Finding:** Shortage cost (€1.00) is 5x holding cost (€0.20), leading to conservative critical fractile.

**Effect:**
- Critical fractile τ = 1.0/(1.0+0.2) = 0.833
- Policy aims for 83.3% service level (very high)
- Small forecast bias + high service level = large over-ordering

**Trade-off:**
- Lower service level → less over-ordering → lower holding costs
- But also → more shortages → higher shortage costs
- With biased forecasts, we're far from the optimal trade-off

### 6. VN2's Success: Simplicity + Recency Bias

**VN2's winning formula:**
```python
forecast = recent_13_weeks.mean()
order_up_to = 4 * forecast
order = max(0, order_up_to - position)
```

**Why it works:**
- **Recency bias:** Only uses last 13 weeks (adapts to demand changes)
- **Simplicity:** No complex optimization, just 4-week coverage
- **Robustness:** Moving average naturally smooths noise
- **Fast adaptation:** New week comes in, old week drops out

**Our models:**
- Train on 52 weeks (slower to adapt)
- Use complex distributional forecasts (more to go wrong)
- Optimize at 83.3rd percentile (amplifies errors)
- Result: Over-forecast + over-order

---

## Part 7: Next Steps - Fresh Start with New Insights

Based on these findings, **I'm now rerunning the entire pipeline from scratch** with the following modifications:

### 1. Recent-Weighted Training Data

**Change:** Use only last 13-26 weeks for training instead of 52 weeks.

**Rationale:** 
- Matches VN2's recency bias
- Adapts faster to structural breaks
- Should reduce forecast over-estimation

**Implementation:** Modify training data loader to filter by date.

### 2. Ensemble with Recent Average

**Change:** Combine model forecast with simple recent average.

**Formula:**
```python
final_forecast = α × model_forecast + (1-α) × recent_avg
# Test α = 0.3, 0.5, 0.7
```

**Rationale:**
- Blend model sophistication with recency adaptation
- Hedging strategy
- Should reduce over-estimation while keeping model signal

### 3. Implement VN2 Exact Baseline

**Change:** Implement VN2's exact simple approach as baseline.

**Implementation:**
```python
recent_avg = last_13_weeks.mean()
order_up_to = 4 * recent_avg
order = max(0, order_up_to - position)
```

**Rationale:**
- Validate that we can reproduce €5,248 cost
- Confirms our simulation is correct
- Establishes floor to beat

### 4. Adaptive Coverage Periods

**Change:** Use different coverage based on demand characteristics.

**Logic:**
```python
if demand_declining:
    coverage = 2.5 weeks  # Lower coverage
elif demand_stable:
    coverage = 3.5 weeks
elif demand_growing:
    coverage = 4.5 weeks  # Higher coverage
```

**Rationale:**
- Different SKUs need different strategies
- Declining-demand SKUs need less coverage (avoid over-ordering)
- Growing-demand SKUs need more coverage (avoid stockouts)

### 5. Lower Percentile for Declining SKUs

**Change:** Use adaptive critical fractile instead of fixed 0.833.

**Logic:**
```python
if recent/historical < 0.8:  # Declining demand
    fractile = 0.70  # Lower, more conservative
else:
    fractile = 0.833  # Standard
```

**Rationale:**
- Declining SKUs should order less aggressively
- Prevents over-ordering when demand is falling
- Still maintains service level for stable SKUs

---

## Part 8: Request for VN2 Challenge Presentation Recordings

Patrick, to help us refine our approach and learn from the winning solutions, **I would be extremely grateful if you could share the recordings from the VN2 challenge presentations** (if they are available).

**Specifically, I'm interested in learning:**

1. **Forecasting Approaches:**
   - What forecasting methods did the top teams use?
   - Did they use sophisticated ML models or simple methods?
   - How did they handle the demand decline pattern?
   - What horizons did they forecast (single-step vs multi-step)?

2. **Ordering Policies:**
   - What type of policies did teams use (newsvendor, order-up-to, base-stock, etc.)?
   - How did they determine order quantities?
   - Did they use fixed or adaptive parameters?

3. **Demand Handling:**
   - How did winning teams detect and adapt to the structural break?
   - Did they use recent-weighted data?
   - Any online learning or adaptive approaches?

4. **Implementation Details:**
   - What was the typical training data window?
   - How did they balance service level vs inventory costs?
   - Any ensemble approaches?

5. **What Didn't Work:**
   - Did teams share approaches that failed?
   - Common pitfalls or mistakes?
   - Lessons learned?

6. **Performance Metrics:**
   - What were the winning teams' final costs?
   - How close were the top 3-5 teams?
   - What was the range of results?

**Why this would be valuable:**

These insights would help us:
- Validate or correct our current approach
- Avoid reinventing the wheel
- Learn from proven strategies
- Understand what separates good from great solutions
- Benchmark our €9,543 result against competition standards

If recordings aren't available, even slide decks, summary notes, or a brief description of the winning approaches would be incredibly helpful!

---

## Part 9: Technical Documentation

I've created comprehensive documentation of all this work:

**Main Document:** `COST_GAP_DEBUGGING_SUMMARY.md` (26 pages)

**Contents:**
- Complete root cause analysis with detailed SKU examples
- All 5 experiments with full implementation details and results
- Mathematical derivations and proofs
- Code changes across 3 files (forecast_loader.py, sequential_backtest.py, sequential_eval.py)
- Technical lessons learned with supporting data
- Recommendations for next steps
- Timeline and evolution of understanding

**Supporting Files:**
- Sequential evaluation results for each approach (parquet files)
- Test scripts for each experiment
- Log files with detailed execution traces
- Visualization of demand decline patterns

**Code Changes Summary:**
- Modified: 3 core files (forecast_loader, sequential_backtest, sequential_eval)
- Added: 8 new functions (PMF convolution, simple ordering, etc.)
- Created: 6 test scripts
- Generated: 6 result datasets

All code is version-controlled and documented with detailed comments explaining the logic and assumptions.

---

## Part 10: Questions and Discussion Points

Beyond the presentation recordings, I'd love to discuss:

1. **Does our 80.5% demand decline finding match what you observed?**
   - Is this a known characteristic of the VN2 dataset?
   - Should we have detected and handled this differently?

2. **Our best result (€9,543) - how does this compare to competition results?**
   - Is this in the ballpark of good solutions?
   - Or is the remaining 81.8% gap (vs €5,248) concerning?

3. **Should we pursue the sophisticated model approach or pivot to simple methods?**
   - Is there value in continuing with SLURP/ML models?
   - Or should we focus on tuned simple approaches?

4. **Training data window - what's optimal?**
   - Is 13 weeks too short? 52 weeks too long?
   - How do we balance adaptation vs stability?

5. **Any other recommendations based on what you've seen work well?**

---

## Conclusion

We've made substantial progress:
- ✅ All 4 recommendations implemented correctly
- ✅ Root cause identified (systematic forecast over-estimation)
- ✅ 5 systematic experiments completed
- ✅ 4.1% cost improvement achieved (€9,950 → €9,543)
- ✅ Clear path forward identified (forecast quality improvement)

The key insight: **forecast quality dominates algorithm choice**. Our remaining 81.8% gap is almost entirely due to models over-forecasting (2.6x) because they're trained on historical data that doesn't reflect the recent demand decline.

I'm excited to rerun everything with recent-weighted data and ensemble approaches. The VN2 presentation recordings would be incredibly valuable for validating our direction and learning from proven approaches.

Thank you for all your guidance so far - your recommendations were spot-on and helped us build a solid foundation. Looking forward to your feedback and hopefully the presentation recordings!

Best regards,
Ian

---

**P.S.** I'm happy to share the full COST_GAP_DEBUGGING_SUMMARY.md document, result files, or any specific code if you'd like to review in detail. Just let me know!
