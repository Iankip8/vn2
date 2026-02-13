# VN2 Next Steps - Post Email to Patrick

## Current Status ✅

**Completed:**
- ✅ All 4 Patrick priorities implemented
- ✅ Models trained (599/599 SKUs for SLURP models)
- ✅ Sequential evaluation working correctly
- ✅ Root cause identified (demand decline + forecast over-estimation)
- ✅ 5 experiments completed and documented
- ✅ Best current result: €9,543 (simple 3-week approach)
- ✅ Comprehensive documentation created (COST_GAP_DEBUGGING_SUMMARY.md)
- ✅ Email to Patrick sent requesting presentation recordings

**Current Gap:**
- Our best: €9,543
- VN2 benchmark: €5,248
- **Remaining gap: 81.8% (€4,295 excess cost)**

---

## Root Cause Summary

**The Problem:**
- 80.5% of SKUs have declining demand (median ratio 0.747)
- Our models trained on 52-week historical data (mean demand 2.77)
- Recent 13-week actual demand much lower (mean 0.77)
- **Model over-forecasts by 2.6x**

**Why VN2's Simple Approach Works:**
```python
# VN2's winning formula
forecast = recent_13_weeks.mean()  # Adapts automatically to demand changes
order_up_to = 4 * forecast         # Simple 4-week coverage
order = max(0, order_up_to - position)
```

**Why Our Sophisticated Approach Fails:**
- Uses model forecasts (trained on 52 weeks, biased high)
- Complex PMF-based newsvendor at 83.3rd percentile
- Amplifies forecast errors
- Result: Over-ordering → holding costs

---

## Recommended Next Actions

### Option A: Implement VN2 Exact Baseline (Recommended First Step)

**Goal:** Validate we can reproduce €5,248 benchmark

**Approach:** Add `forecast_mode` parameter to existing framework

**Implementation:**

1. **Modify `src/vn2/analyze/sequential_eval.py`:**
```python
@dataclass
class SequentialConfig:
    # ... existing fields ...
    forecast_mode: str = 'model'  # 'model' or 'recent_avg'
    recent_avg_weeks: int = 13     # For recent_avg mode
```

2. **Modify `src/vn2/analyze/forecast_loader.py`:**
```python
def load_forecasts_for_sku(
    ...,
    mode: str = 'model',
    recent_avg_weeks: int = 13
):
    if mode == 'recent_avg':
        # Compute recent average from historical data
        recent_demand = historical_data[-recent_avg_weeks:].mean()
        # Return as uniform PMF centered on recent average
        ...
    else:
        # Load model forecasts (existing logic)
        ...
```

3. **Run test:**
```bash
uv run python scripts/run_sequential_eval.py --forecast_mode recent_avg
```

**Expected Result:** Should get ~€5,248 (within 5%)

**Why This Matters:**
- Validates our simulation framework is correct
- Confirms the gap is forecast quality, not implementation bugs
- Establishes a solid baseline to beat

---

### Option B: Ensemble Forecasts (Quick Win)

**Goal:** Blend model with recent average to reduce over-estimation

**Formula:**
```python
final_forecast = α × model_forecast + (1-α) × recent_avg
```

**Test different α values:** 0.0, 0.3, 0.5, 0.7, 1.0

**Implementation:**
```bash
# Create test script
uv run python scripts/test_ensemble_forecast.py
```

**Expected Result:** α=0.3-0.5 should beat our current €9,543

**Already Created:** `scripts/test_ensemble_forecast.py` (needs minor fixes for data structure)

---

### Option C: Retrain with Recent Data

**Goal:** Reduce forecast over-estimation by using recent training window

**Changes:**
- Current: Train on 52 weeks
- New: Train on last 13-26 weeks only

**Implementation:**
1. Modify training data loader to filter by date
2. Retrain all models with recent data only
3. Re-run sequential evaluation

**Trade-off:**
- ✅ Adapts faster to demand changes
- ❌ Less training data → potentially higher variance

---

### Option D: Adaptive Coverage

**Goal:** Use different coverage periods based on SKU characteristics

**Logic:**
```python
# Detect demand trend
recent_mean = last_13_weeks.mean()
historical_mean = last_52_weeks.mean()
ratio = recent_mean / historical_mean

# Adaptive coverage
if ratio < 0.8:  # Declining demand
    coverage_weeks = 2.5
elif ratio < 1.2:  # Stable
    coverage_weeks = 3.5
else:  # Growing
    coverage_weeks = 4.5
```

**Implementation:** Modify `choose_order_simple_vn2()` function

---

## Recommended Sequence

### Phase 1: Validation (This Week)
1. **Implement Option A** (VN2 exact baseline with forecast_mode parameter)
2. Run and verify we can reproduce €5,248
3. If successful → Our framework is correct ✅
4. If not → Debug simulation logic

### Phase 2: Quick Wins (Next)
5. **Test Option B** (Ensemble forecasts)
6. Find best α that minimizes cost
7. **Expected:** €7,000-8,000 (significant improvement)

### Phase 3: Model Improvements (After Patrick's Response)
8. Wait for presentation recordings from Patrick
9. Learn from winning teams' approaches
10. **Implement Option C** (retrain with recent data) or other insights
11. **Test Option D** (adaptive coverage)
12. Combine best approaches

### Phase 4: Final Optimization
13. Fine-tune hyperparameters
14. Test on competition leaderboard
15. Submit final solution

---

## Files Created (Do Not Override Core Files)

**New Test Scripts:**
- `scripts/test_vn2_exact_baseline.py` - Informational script explaining the approach
- `scripts/test_ensemble_forecast.py` - Test ensemble forecasting (needs data structure fix)

**Documentation:**
- `COST_GAP_DEBUGGING_SUMMARY.md` - Complete 26-page analysis of all work
- `EMAIL_TO_PATRICK.md` - Comprehensive email to Patrick (sent)
- `NEXT_STEPS.md` - This file

**No Core Files Modified** ✅

---

## Quick Reference: Key Metrics

| Approach | Cost | vs Baseline | vs VN2 |
|----------|------|-------------|--------|
| Original (h=1,2) | €10,045 | - | +91.4% |
| Baseline (h=3,4,5) | €9,950 | -0.9% | +89.6% |
| Simple 3-week | €9,543 | -4.1% | +81.8% |
| **VN2 Benchmark** | **€5,248** | **-47.3%** | **-** |
| **Target** | **~€5,500** | **~-45%** | **~+5%** |

---

## Question for You

**Which option would you like to implement first?**

**Option A (Recommended):** Add `forecast_mode` parameter to validate VN2 baseline
- Pros: Validates framework, minimal code changes, reversible
- Cons: Requires modifying core files (but cleanly with parameter)

**Option B:** Test ensemble forecasts with existing structure
- Pros: No core file changes needed
- Cons: Needs data structure compatibility fixes first

**Option C:** Create completely separate fork of evaluation pipeline
- Pros: Zero risk to existing code
- Cons: Code duplication, harder to maintain

Let me know and I'll implement it!
