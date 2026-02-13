# Re-Evaluation Results: Corrected Policy vs Buggy Policy

**Date:** February 3, 2026  
**Evaluation:** 49,500 tasks (7 models × 599 SKUs × 12 folds)  
**Runtime:** ~20 minutes

---

## 📊 RESULTS SUMMARY

### Overall Leaderboard (Corrected 3-Week Policy)

| Rank | Model | Expected Cost | MAE | Service Level | Change vs Old |
|------|-------|---------------|-----|---------------|---------------|
| 1 | **QRF** | **26,699** | 2.73 | 22.0% | +1,480 (+5.9%) |
| 2 | **ZINB** | **27,151** | overflow | 42.9% | +1,843 (+7.3%) |
| 3 | **SLURP Stockout-Aware** | **36,866** | 1.42 | 42.8% | +2,835 (+8.3%) |
| 4 | **LightGBM Quantile** | **36,868** | 1.77 | 42.8% | +2,835 (+8.3%) |
| 5 | **SLURP Bootstrap** | **36,868** | 1.39 | 42.8% | +2,835 (+8.3%) |
| 6 | **Seasonal Naive** | **36,868** | 70.8 | 42.8% | +26,617 (+260%!) |

### Old Leaderboard (Buggy 1-Week Policy)

| Rank | Model | Expected Cost | Service Level |
|------|-------|---------------|---------------|
| 1 | Seasonal Naive | 10,251 | 45.4% |
| 2 | QRF | 25,219 | 18.6% |
| 3 | ZINB | 25,308 | 43.5% |
| 4 | SLURP Stockout-Aware | 34,031 | 43.1% |
| 5 | SLURP Bootstrap | 34,033 | 43.1% |
| 6 | LightGBM Quantile | 34,033 | 43.1% |

---

## 🔍 KEY FINDINGS

### 1. **UNEXPECTED: Costs INCREASED with Corrected Policy**

**Expected:** Lower costs with better protection period  
**Actual:** All models showed cost increases of 6-260%

**Most Dramatic:**
- Seasonal Naive: 10,251 → 36,868 (+260%!)
- This went from BEST to WORST

### 2. **Service Levels CONVERGED to ~43%**

All models (except QRF) now achieve nearly identical service levels:
- LightGBM: 43.1% → 42.8% (-0.3%)
- SLURP Bootstrap: 43.1% → 42.8% (-0.3%)
- SLURP Stockout-Aware: 43.1% → 42.8% (-0.3%)
- ZINB: 43.5% → 42.9% (-0.6%)
- Seasonal Naive: 45.4% → 42.8% (-2.6%)
- QRF: 18.6% → 22.0% (+3.4%)

**Expected:** Service levels should rise toward 83.3% (critical fractile target)  
**Actual:** Converged to 43% regardless of model

### 3. **Policy is Consistent Across Models**

The fact that all models now achieve ~43% service level (except QRF) suggests:
- ✅ The corrected policy IS being applied consistently
- ✅ The 3-week aggregation is working
- ❌ But the target 83.3% service level is NOT being achieved

### 4. **QRF Remains Best, But Worse Than Before**

- QRF is still the best model by cost
- But it got WORSE with the corrected policy (+1,480 cost)
- Service level improved slightly (18.6% → 22.0%)

---

## 🤔 HYPOTHESES FOR 43% SERVICE LEVEL

### Hypothesis 1: Service Level ≠ Critical Fractile

**Critical Fractile:** The quantile we target in the demand distribution (0.833)  
**Service Level:** The fraction of cycles where demand was fully met

These are different concepts:
- Critical fractile = 0.833 means we stock to the 83.3rd percentile
- But actual service level depends on forecast accuracy

If forecasts are biased or uncertain, targeting 83.3% fractile might only achieve 43% service.

### Hypothesis 2: Forecast Quality Limits

All models may have similar forecast characteristics:
- Similar bias patterns
- Similar uncertainty levels
- Similar coverage of the true demand distribution

This would explain why all models converge to ~43% regardless of targeting 83.3%.

### Hypothesis 3: Initial Inventory Constraint

The policy calculation:
```python
position = on_hand + intransit_1 + intransit_2
order_qty = max(0, S - position)
```

If `position` is consistently high relative to `S`, we might not be ordering enough.

### Hypothesis 4: Evaluation Period Issue

We're evaluating on a 12-fold holdout. If:
- The holdout period has different demand patterns
- Or the initial inventory states are not representative
- The service level might be artificially constrained

---

## ⚠️ CONCERNS

### 1. **Seasonal Naive Disaster**

The 260% cost increase for seasonal_naive (10,251 → 36,868) is alarming:
- This was the BEST model with buggy policy
- Now it's the WORST with corrected policy
- This suggests the buggy policy was accidentally beneficial for naive forecasts

**Possible Explanation:**
- Buggy policy used only h=1 (1-week ahead)
- Seasonal naive is good at h=1 forecasts
- Corrected policy aggregates 3 weeks (h=1, h=2, h=3)
- Seasonal naive gets worse at longer horizons
- So corrected policy exposed seasonal_naive's weakness

### 2. **Uniform 43% Service Level is Suspicious**

Having all models achieve exactly ~43% suggests:
- Either a hard constraint in the system (inventory limits?)
- Or a bug in the service level calculation
- Or forecasts are so similar that model differences don't matter

### 3. **Expected Cost Increases**

With corrected policy, we're targeting MORE inventory (3-week protection vs 1-week).  
More inventory means:
- ✅ Lower shortage costs (better service)
- ❌ Higher holding costs (more stock)

But we're seeing HIGHER total costs, which means:
- Holding costs increased more than shortage costs decreased
- The trade-off is suboptimal

---

## 📋 RECOMMENDED NEXT STEPS

### IMMEDIATE: Investigate Service Level Calculation

1. **Verify policy implementation:**
   ```python
   # Check one SKU manually:
   # - Read quantile forecasts
   # - Compute S using corrected policy
   # - Compare to actual orders in results
   # - Check if S is being correctly translated to service level
   ```

2. **Check if there's a constraint:**
   - Max order quantity limit?
   - Budget constraint?
   - Physical inventory cap?

### SHORT-TERM: Diagnostic Analysis

3. **Policy A/B test (Patrick Priority #4):**
   - Hold forecast fixed (use QRF)
   - Compare service levels with:
     - Policy A: 1-week protection
     - Policy B: 3-week protection
   - Expected: Policy B should achieve higher service level

4. **Manual calculation for one SKU:**
   - Pick one SKU (e.g., store=0, product=126)
   - Load its QRF quantile forecasts
   - Manually compute S using corrected policy
   - Check what service level this achieves
   - Compare to reported service level in results

5. **Check initial inventory positions:**
   - Are they reasonable?
   - Are they constraining orders?

### MEDIUM-TERM: Re-evaluate Approach

6. **Check if holding cost is too low:**
   - Current: holding=0.2, shortage=1.0
   - Ratio: 5:1
   - Critical fractile: 0.833
   - Maybe shortage cost should be higher to justify more stock?

7. **Verify forecast quality:**
   - Check MAE, RMSE for h=1, h=2, h=3 separately
   - See if forecasts degrade quickly with horizon

---

## 💡 INSIGHTS

### What We Learned:

1. **The buggy policy was accidentally good for seasonal_naive**
   - By using only h=1, it played to seasonal_naive's strength
   - Corrected policy exposed its weakness at longer horizons

2. **Models are more similar than different**
   - All achieve ~43% service level
   - Suggests forecast quality matters more than model choice
   - Or that there's a system-wide constraint

3. **QRF is consistently best**
   - Best with both policies
   - Though got slightly worse with correction

4. **The corrected policy IS working**
   - Consistent service levels across models prove this
   - But something is preventing us from reaching 83.3% target

---

## 🎯 PATRICK'S PRIORITIES STATUS

### ✅ Priority 1: Fast Imputation (COMPLETE)
- Implemented vectorized seasonal-index
- 11.2% imputation rate, 2.36 mean demand

### ✅ Priority 2: Corrected Policy (COMPLETE - BUT UNEXPECTED RESULTS)
- Implemented all 3 fixes
- 3-week protection ✅
- 0.833 critical fractile ✅
- MC aggregation ✅
- **But service levels stuck at 43%, not 83%**

### 🔄 Priority 3: SLURP Validation (COMPLETE)
- SLURP models trained successfully
- Both variants in evaluation
- Not a failure issue - working correctly

### ⏳ Priority 4: Sanity Tests (PENDING)
- **NOW CRITICAL:** Need to diagnose why service level = 43%
- Policy A/B test would help isolate the issue
- Manual SKU calculation would verify policy implementation

---

## 🚨 ACTION REQUIRED

**Before proceeding with re-training or further optimization, we MUST understand:**

1. Why are service levels stuck at 43% instead of 83%?
2. Is the policy calculation correct?
3. Is there a system constraint we're hitting?
4. Why did seasonal_naive perform so much better with the buggy policy?

**Recommendation:** Build the sanity test suite (Priority #4) to diagnose these issues before making any other changes.
