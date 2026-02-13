# VN2 Project Status - Patrick's Recommendations Implementation

**Date:** February 3, 2026  
**Status:** 2/4 Priorities Complete, Ready for Re-Training

---

## ✅ COMPLETED

### Priority 1: Fast Stockout Imputation
- **Implementation:** `src/vn2/uncertainty/fast_imputation.py`
- **Method:** Vectorized seasonal-index (week-of-year median)
- **Performance:** 
  - Imputation rate: **11.2%** (10,517 observations)
  - Mean imputed demand: **2.36** (vs 0.00 with placeholders)
  - Execution time: <10 seconds (vs infinite hang)
- **Files Generated:**
  - `data/processed/demand_imputed.parquet`
  - `data/processed/demand_imputed_winsor.parquet`
  - `data/processed/demand_imputed_capped.parquet`

### Priority 2: Corrected Ordering Policy
- **Implementation:** `src/vn2/policy/corrected_policy.py`
- **Integration:** `src/vn2/analyze/model_eval.py` (UPDATED)
- **Fixes Applied:**
  1. ✅ Protection period = 3 weeks (lead + review)
  2. ✅ Critical fractile = 0.833 explicit
  3. ✅ Monte Carlo aggregation over full horizon
- **Impact:** Should dramatically improve service level (was 21.8% with buggy 1-week protection)

---

## 🔍 INVESTIGATED

### Priority 3: SLURP Models Status

**FINDING: SLURP models DID train successfully!**

Evidence:
```bash
$ ls models/checkpoints/ | grep slurp
slurp_bootstrap/          (601 SKUs trained)
slurp_stockout_aware/     (601 SKUs trained)
```

Evaluation results confirm:
```python
Models evaluated: ['lightgbm_quantile', 'qrf', 'seasonal_naive', 
                   'slurp_bootstrap', 'slurp_stockout_aware', 'zinb']
```

**Why they weren't in training_results.parquet:**
- SLURP models train on `df_raw` (different data stream)
- Other models train on `df_winsor`
- Training results only captured winsor-trained models
- Both data streams work correctly, just separate logging

**Action Required:**
- ✅ No urgent fix needed - models trained successfully
- 📝 Optional: Update training logger to capture both streams

### Missing SLURP Variants
Two SLURP variants are defined but not enabled:
1. `slurp_surd` - SLURP with SURD transforms, no stockout handling
2. `slurp_surd_stockout_aware` - SLURP with both SURD and stockout handling

**Configuration check:**
```yaml
# configs/forecast.yaml
slurp_bootstrap:
  enabled: true    # ✅ TRAINED
  
slurp_stockout_aware:
  enabled: true    # ✅ TRAINED
  
slurp_surd:
  enabled: false   # ❌ NOT ENABLED
  
slurp_surd_stockout_aware:
  enabled: false   # ❌ NOT ENABLED
```

**Recommendation:** Enable these for next training run to get full SLURP comparison.

---

## ⏳ PENDING

### Priority 4: Sanity Test Suite

**Required Tests:**

#### A) SKU×Store Golden Test
Create: `test/test_policy_sanity.py`
```python
def test_protection_period_aggregation():
    # Verify 3-week aggregation vs 1-week
    assert protection_weeks == 3
    
def test_critical_fractile():
    # Verify 0.833 quantile targeting
    assert abs(tau - 0.833) < 0.001
    
def test_base_stock_accounting():
    # Verify inventory position calculation
    assert position == on_hand + intransit_1 + intransit_2
```

#### B) Policy A/B Test
Create: `scripts/compare_policy_ab.py`
```python
# Hold forecast fixed (seasonal_naive)
# Compare:
#   Policy A: Old (1-week protection)
#   Policy B: New (3-week protection)
# Expected: Policy B should win significantly
```

#### C) Imputation On/Off Check
Create: `scripts/test_imputation_impact.py`
```python
# Compare QRF/ZINB with:
#   Option 1: Drop stockout weeks
#   Option 2: Fast seasonal imputation
#   Option 3: Old placeholder (0% imputation)
# Measure impact on forecast quality
```

---

## 📊 CURRENT MODEL STATUS

### Trained Models (6 total)
| Model | Checkpoints | In Eval | Notes |
|-------|------------|---------|-------|
| lightgbm_quantile | ✅ | ✅ | Trained on winsor data |
| linear_quantile | ✅ | ❌ | Trained but not in eval |
| qrf | ✅ | ✅ | Best performer (25,219 cost) |
| zinb | ✅ | ✅ | Numerical overflow issues |
| slurp_bootstrap | ✅ | ✅ | Trained on raw data |
| slurp_stockout_aware | ✅ | ✅ | Trained on raw data |

### NOT Trained (2 variants)
| Model | Config | Reason |
|-------|--------|--------|
| slurp_surd | enabled: false | Not enabled in config |
| slurp_surd_stockout_aware | enabled: false | Not enabled in config |

### Baseline (from previous run)
| Model | Source | Notes |
|-------|--------|-------|
| seasonal_naive | Old evaluation | Best cost (10,251) - using buggy policy? |

---

## 🎯 NEXT STEPS

### Immediate Actions

1. **Re-evaluate with Corrected Policy**
   ```bash
   uv run ./go eval-models --n-jobs 2 --holdout 12 --n-sims 1000
   ```
   - Uses corrected policy in model_eval.py
   - Should see service levels increase toward 83%
   - Expected costs should decrease

2. **Re-train with Real Imputation**
   ```bash
   # Generate fresh imputed data
   uv run python src/vn2/uncertainty/fast_imputation.py
   
   # Re-train non-SLURP models
   uv run ./go forecast --config configs/forecast.yaml --n-jobs 2 --resume
   ```
   - Uses real imputation (not placeholders)
   - Should improve ZINB, QRF, lightgbm_quantile

3. **Enable Missing SLURP Variants**
   - Edit `configs/forecast.yaml`:
     ```yaml
     slurp_surd:
       enabled: true
     slurp_surd_stockout_aware:
       enabled: true
     ```
   - Train these models for complete comparison

### Validation Sequence

1. **Policy A/B Test** (validate fix works)
   ```bash
   uv run python scripts/compare_policy_ab.py
   ```
   Expected: Policy B (corrected) >> Policy A (buggy)

2. **Imputation Impact Test** (validate fix works)
   ```bash
   uv run python scripts/test_imputation_impact.py
   ```
   Expected: Real imputation > drop stockouts > placeholders

3. **Golden Test Suite** (regression prevention)
   ```bash
   uv run pytest test/test_policy_sanity.py -v
   ```
   Expected: All tests pass

### Expected Improvements After Re-Training

| Metric | Before (Buggy) | After (Fixed) | Change |
|--------|---------------|---------------|--------|
| QRF Service Level | 21.8% | ~83% | **+61.2pp** |
| QRF Expected Cost | 25,219 | <15,000? | **-40%?** |
| ZINB MAE | 4e11 (overflow) | <10 | **Fixed** |
| Seasonal_naive | 10,251 | Unknown | Baseline |

---

## 📁 FILES MODIFIED

### New Files Created
- `src/vn2/uncertainty/fast_imputation.py` - Vectorized imputation
- `src/vn2/policy/corrected_policy.py` - Fixed policy math
- `PATRICK_RECOMMENDATIONS_IMPLEMENTATION.md` - Implementation doc
- `PROJECT_STATUS.md` - This file

### Modified Files
- `src/vn2/analyze/model_eval.py` - Integrated corrected policy

### Data Files Updated
- `data/processed/demand_imputed.parquet` - Real imputation (11.2%)
- `data/processed/demand_imputed_winsor.parquet` - Real imputation + winsor
- `data/processed/demand_imputed_capped.parquet` - Real imputation + cap

---

## ❓ QUESTIONS FOR PATRICK

1. **Re-evaluation Priority:**
   - Should I re-evaluate existing models with corrected policy first?
   - Or re-train everything from scratch with both fixes?

2. **Seasonal_naive Baseline:**
   - The 10,251 cost - was this evaluated with buggy or corrected policy?
   - Should I re-evaluate baseline to ensure fair comparison?

3. **SLURP SURD Variants:**
   - Should I enable and train `slurp_surd` and `slurp_surd_stockout_aware`?
   - Or focus on the 2 SLURP models already trained?

4. **Sanity Tests:**
   - Should I build the full test suite before re-training?
   - Or proceed with re-training and validate afterward?

---

## 🎉 SUMMARY

**Completed:**
- ✅ Fast imputation (11.2% rate, 2.36 mean)
- ✅ Corrected policy (3-week protection, 0.833 quantile, MC aggregation)
- ✅ Policy integrated into evaluation code
- ✅ SLURP investigation (they DID train!)

**Ready to Execute:**
- 🔄 Re-evaluate with corrected policy
- 🔄 Re-train with real imputation
- 🔄 Enable missing SLURP variants
- ⏳ Build sanity test suite

**Expected Impact:**
- Service levels: 21.8% → ~83% (**+61pp**)
- Expected costs: Significant reduction across all models
- Model rankings: Likely to change completely

**Confidence Level:** High - both fixes address root causes identified by Patrick.
