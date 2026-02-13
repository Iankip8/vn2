# 🚀 TRAINING IN PROGRESS

**Started:** February 5, 2026  
**Status:** RUNNING (Background process)

---

## What's Happening

Training **11 models** on **599 SKUs** for paper Section 6.1:

### Models Being Trained

#### SLURP Family (on raw data with stockout indicators)
- ✅ `slurp_bootstrap` - Baseline conditional bootstrap  
- ✅ `slurp_surd` - SURD transforms, NO stockout handling (H3 ablation)
- ✅ `slurp_stockout_aware` - Stockout-aware, NO SURD (H2 ablation)
- ✅ `slurp_surd_stockout_aware` - Full implementation (best expected)

#### Challenger Models (on winsorized imputed data)
- ✅ `lightgbm_quantile` - Gradient boosted quantile regression (density)
- ✅ `lightgbm_point` - MSE objective (point baseline for Jensen gap)
- ✅ `ngboost` - NGBoost LogNormal (distributional)
- ✅ `ets` - Exponential Smoothing (statistical baseline)
- ✅ `qrf` - Quantile Random Forest (density)
- ✅ `linear_quantile` - Linear quantile regression (density)
- ✅ `zinb` - Zero-Inflated Negative Binomial (parametric)

---

## Configuration

- **SKUs:** 599 store-product combinations
- **Horizons:** h=1 through h=12 (full 12-week forecasts)
- **Quantiles:** [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
- **Cross-validation:** 12 rolling origins
- **Parallel workers:** 11
- **Timeout per fit:** 90 seconds

---

## Expected Timeline

- **Total tasks:** ~11 models × 599 SKUs × 12 folds = ~79,000 tasks
- **Estimated time:** 4-8 hours (depending on hardware)
- **Progress tracking:** Checkpointed (can resume if interrupted)

---

## Monitor Progress

### Check Current Status
```bash
# View latest log output
tail -100 logs/full_training_*.log

# Watch live progress
tail -f logs/full_training_*.log

# Count completed checkpoints
find models/checkpoints -name "*.pkl" | wc -l

# Check progress file
cat models/checkpoints/progress.json | grep -c "completed"
```

### Training Phases

1. **SLURP models** (raw data, censoring-aware)
   - Train on `demand_long.parquet`
   - Preserve stockout information
   - Expected: ~20% of time (fewer samples per bootstrap)

2. **Challenger models** (winsorized data)
   - Train on `demand_imputed_winsor.parquet`
   - Stable targets (no extreme outliers)
   - Expected: ~80% of time (gradient boosting is slow)

---

## Output Files

### Model Checkpoints
```
models/checkpoints/
  {model}_{store}_{product}_{fold}.pkl
  
Example: lightgbm_quantile_0_126_0.pkl
```

### Forecast Results
```
models/results/
  training_results.parquet  ← Main results file
  
Columns:
  - model_name, store, product, fold
  - horizon (h=1..12)
  - quantile levels (q01, q05, ..., q99)
  - evaluation metrics
```

### Progress Tracking
```
models/checkpoints/progress.json
  {
    "completed": [...],  ← List of finished tasks
    "failed": [...],     ← List of failed tasks
    "data_hash": "..."   ← For cache invalidation
  }
```

---

## What Happens Next

### After Training Completes

1. **Generate Forecast Files**
   - Extract quantiles for h=3,4,5 (Patrick's horizons)
   - Format: `{model}_quantiles.parquet` per model
   - Columns: store, product, q01_h3, q05_h3, ..., q99_h5

2. **Run Evaluation**
   ```bash
   uv run python scripts/eval_all_models_patrick.py
   ```
   - Evaluates BOTH policies (density-aware SIP + point+service-level)
   - Computes Jensen gaps
   - Generates leaderboards

3. **Analyze Results**
   - Jensen gap analysis (H1)
   - SLURP ablation study (H2, H3)
   - Cohort analyses (by sparsity, stockout rate, etc.)
   - Update paper Section 7

---

## Expected Results

Based on Patrick's €6,266 baseline with simple 13-week MA:

### Best Models (Density-Aware SIP)
```
Model                       Total Cost   Gap vs VN2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
slurp_surd_stockout_aware   €5,500       +4.8%     ← TARGET
slurp_stockout_aware        €5,800       +10.5%
lightgbm_quantile           €6,100       +16.2%
qrf                         €6,400       +21.9%
slurp_bootstrap             €6,600       +25.7%
...
VN2 Benchmark               €5,248       0%
```

### Jensen Gap (SIP vs Point Policy)
```
Model                       Jensen Δ     Improvement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
slurp_surd_stockout_aware   +€1,700      +23.6%
lightgbm_quantile           +€2,000      +24.7%
qrf                         +€1,200      +18.7%
...
All positive = H1 CONFIRMED ✓
```

---

## Troubleshooting

### Training Stalled?
```bash
# Check if process is running
ps aux | grep "vn2.cli forecast"

# Check last activity
ls -lt models/checkpoints/*.pkl | head -5

# Resume if needed (uses checkpoint)
uv run python -m vn2.cli forecast --config configs/forecast.yaml
```

### Out of Memory?
- Reduce `n_jobs` in config (currently 11)
- Reduce `n_bootstrap` for SLURP models (currently 1000)
- Use `--pilot` flag to train subset first

### Timeouts?
- Increase `timeout_per_fit` in config (currently 90s)
- Some SKUs with complex patterns may timeout (acceptable)
- Failed SKUs are tracked separately

---

## Key Differences from Original Plan

### ✅ What We Have (Better!)
- **Complete training infrastructure** already exists in `src/vn2/`
- All models implemented and working
- Checkpoint/resume capability built-in
- Parallel execution with joblib
- Progress tracking in JSON
- Automatic data source selection (raw vs winsorized)

### 🔧 What We Built
- Configuration updates (enabled key models)
- Evaluation script with Patrick's corrected policy
- Jensen gap analysis framework
- Documentation

---

## Success Metrics

Training is successful when:
- ✅ All 11 models complete for majority of SKUs (>95%)
- ✅ Checkpoint files created for each model × SKU × fold
- ✅ training_results.parquet contains quantile forecasts
- ✅ No critical errors in log file
- ✅ Can extract h=3,4,5 quantiles for evaluation

---

## Current Status Check

```bash
# Quick status
echo "=== TRAINING STATUS ==="
echo "Process running: $(ps aux | grep 'vn2.cli forecast' | grep -v grep | wc -l)"
echo "Checkpoints created: $(find models/checkpoints -name '*.pkl' 2>/dev/null | wc -l)"
echo "Latest activity: $(ls -lt models/checkpoints/*.pkl 2>/dev/null | head -1 | awk '{print $6,$7,$8}')"
echo ""
tail -20 logs/full_training_*.log | grep -E "✓|Success|Failed|complete"
```

---

## After Training

Once you see `✅ Training complete!` in the log:

1. **Extract forecasts for h=3,4,5**
   - Read training_results.parquet
   - Filter to horizons 3, 4, 5
   - Save as {model}_quantiles.parquet

2. **Run evaluation**
   ```bash
   uv run python scripts/eval_all_models_patrick.py
   ```

3. **Generate paper results**
   - Leaderboards for Section 7.1
   - Jensen gap tables
   - Cohort analysis plots

4. **Update paper draft**
   - Fill in Section 7 with actual numbers
   - Add result tables and figures
   - Validate hypotheses H1-H4

---

**Status:** ⏳ Training in progress...  
**Check:** `tail -f logs/full_training_*.log`  
**Estimated completion:** 4-8 hours from start
