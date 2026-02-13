# Email to Patrick McDonald - VN2 Project Update

**Subject:** VN2 Project Update: Training Complete, Imputation Issue Persists

---

Dear Patrick,

I wanted to update you on my progress with the VN2 inventory optimization project and discuss some challenges I've encountered.

## Progress Summary

I've successfully completed the following phases:

1. **Data Pipeline Validation** (✅ Complete)
   - Verified all 599 SKUs across 157 weeks (94,043 observations)
   - Confirmed stockout detection: 10,517 observations (11.2%) correctly flagged
   - Sales totals validated between raw and processed data

2. **Model Training** (✅ Complete - 4 models)
   - Successfully trained: `lightgbm_quantile`, `linear_quantile`, `qrf`, `zinb`
   - Training success rate: 96.2% (27,658/28,752 tasks)
   - All models trained across 12 cross-validation folds

3. **Model Evaluation** (✅ Complete)
   - Completed full backtesting evaluation with 1,000 Monte Carlo simulations
   - Generated performance metrics and cost analysis

## Challenges Encountered

### Critical Issue: Stockout Imputation Failure

As I mentioned in my previous communication about "Mean Demand (Imputed Stockout) = 0.000", the imputation process continues to fail. I've now identified the root cause:

**Problem:** The `extract_profile_features()` function in `stockout_imputation.py` hangs indefinitely when computing seasonal statistics for 10,517 stockout observations. This is computationally infeasible on my 2-core laptop.

**Attempts Made:**
- Tried 5 different parameter combinations (n_neighbors: 2-20, n_jobs: 1-4)
- All attempts hung at line 116 in `stockout_imputation.py`
- Process consistently blocks during profile feature extraction

**Workaround Implemented:**
- Created placeholder imputed files to unblock training
- All observations flagged as `imputed=False`
- Applied simple winsorization (0-500 cap) for non-SLURP models

**Impact:**
- ✅ SLURP models can use `demand_long.parquet` directly (no imputation needed)
- ⚠️ Non-SLURP models trained with placeholders (suboptimal performance expected)

## Evaluation Results

**Current Leaderboard (by Expected Cost):**

| Rank | Model | Expected Cost | MAE | Service Level |
|------|-------|---------------|-----|---------------|
| 1 | seasonal_naive (baseline) | 10,251 | 27.93 | 45.4% |
| 2 | **qrf** (NEW) | **25,219** | 2.85 | 21.8% |
| 3 | **zinb** (NEW) | **25,308** | overflow | 43.3% |
| 4 | slurp_stockout_aware (old) | 34,031 | 1.41 | 43.1% |
| 5 | lightgbm_quantile (NEW) | 34,033 | 1.77 | 43.1% |

**Key Observations:**
- QRF and ZINB outperform old SLURP evaluations
- ZINB shows numerical overflow issues (4e11 MAE) but competitive cost
- All models significantly underperform the seasonal_naive baseline
- SLURP models (high priority) did not train in this run despite being enabled

## Questions & Next Steps

I would appreciate your guidance on the following:

1. **Imputation Issue:**
   - Do you have recommendations for optimizing `extract_profile_features()` for larger datasets?
   - Should I attempt to run imputation overnight on a more powerful machine?
   - Or should I proceed with the current placeholder approach?

2. **SLURP Models:**
   - The 4 SLURP variants (slurp_bootstrap, slurp_surd, slurp_stockout_aware, slurp_surd_stockout_aware) are enabled in config but didn't train. Should I investigate why or run them separately?

3. **Model Performance:**
   - The seasonal_naive baseline (10,251 cost) significantly outperforms all trained models. Does this suggest:
     - Need for hyperparameter tuning?
     - Data quality issues?
     - Wrong optimization objective?

4. **Next Priorities:**
   - Should I focus on:
     - Resolving imputation and re-training non-SLURP models?
     - Training SLURP models separately?
     - Hyperparameter optimization for existing models?
     - Generating submission with current best model (qrf)?

## Request for Meeting

Would you be available for a brief call this week to discuss these challenges and align on priorities? I want to ensure I'm focusing on the right areas given the project timeline.

Thank you for your continued guidance.

Best regards,
Ian

---

**Attachments (if needed):**
- Training results: `/home/ian/vn2/models/results/training_results.parquet`
- Evaluation results: `/home/ian/vn2/models/results/leaderboards.parquet`
- Imputation diagnostics: `/home/ian/vn2/IMPUTATION_ISSUE.md`
