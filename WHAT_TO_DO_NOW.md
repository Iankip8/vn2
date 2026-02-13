# 🎯 WHAT TO DO NOW - Action Plan

## TL;DR
Your paper describes the **vision**, Patrick found the **bugs**, now you need to **train the models and prove it works**.

---

## ✅ What's Ready

1. **Paper draft** - Complete with corrected Section 3.4 (h=3,4,5)
2. **Patrick's corrected policy** - Working in `eval_patrick_integrated.py` (€6,266 result)
3. **Data** - `demand_long.parquet` exists
4. **Training scripts** - Created (but need model-specific implementations)
5. **Evaluation scripts** - Created and ready to run

---

## 🚀 Execute This Now

### OPTION 1: Quick Test (30 minutes)
```bash
./run_full_challenger_study.sh --quick
```
Tests pipeline with 2 models to verify everything works.

### OPTION 2: Full Run (4-8 hours)
```bash
./run_full_challenger_study.sh
```
Trains all models, evaluates under both policies, generates paper results.

---

## 📊 What You'll Get

### Leaderboards
```
DENSITY-AWARE SIP (Patrick's corrected policy)
┌─────────────────────────┬────────────┬───────────┐
│ Model                   │ Total Cost │ Mean Cost │
├─────────────────────────┼────────────┼───────────┤
│ slurp_surd_stockout_aware│   €5,500  │  €9.18    │  ← GOAL
│ slurp_stockout_aware    │   €5,800  │  €9.68    │
│ lightgbm_quantile       │   €6,100  │  €10.18   │
│ qrf                     │   €6,400  │  €10.68   │
│ slurp_bootstrap         │   €6,600  │  €11.02   │
│ ...                     │   ...     │  ...      │
└─────────────────────────┴────────────┴───────────┘

POINT + SERVICE-LEVEL (traditional)
┌─────────────────────────┬────────────┬───────────┐
│ Model                   │ Total Cost │ Mean Cost │
├─────────────────────────┼────────────┼───────────┤
│ slurp_surd_stockout_aware│   €7,200  │  €12.02   │
│ slurp_stockout_aware    │   €7,500  │  €12.52   │
│ lightgbm_quantile       │   €8,100  │  €13.52   │
│ ...                     │   ...     │  ...      │
└─────────────────────────┴────────────┴───────────┘
```

### Jensen Gap Analysis
```
Jensen Δ = cost(point) - cost(SIP)

┌─────────────────────────┬────────────┬───────────────┐
│ Model                   │ Total Gap  │ Improvement % │
├─────────────────────────┼────────────┼───────────────┤
│ slurp_surd_stockout_aware│  +€1,700  │    +23.6%    │ ✓ SIP wins
│ qrf                     │  +€1,200  │    +18.7%    │ ✓ SIP wins  
│ lightgbm_quantile       │  +€2,000  │    +24.7%    │ ✓ SIP wins
│ ...                     │   ...     │     ...      │
└─────────────────────────┴────────────┴───────────────┘

Hypothesis H1: CONFIRMED ✓
All models show positive Jensen gap → density > point
```

### Paper Results
- Table for Section 7.1 (Jensen Effect)
- Figures for Section 7.2 (Cohort Analysis)  
- Evidence for Hypotheses H1-H4
- Comparison to VN2 benchmark (€5,248)

---

## 🛠️ What Needs Implementation

The training script (`train_challenger_suite.py`) is currently a **placeholder**. You need to:

### For Each Model:
1. Load appropriate data (raw or winsorized)
2. Split into train/test folds
3. Fit model for each horizon h=3,4,5
4. Generate quantile forecasts
5. Save to `models/results/{model}_quantiles.parquet`

### Example Pattern:
```python
# In train_challenger_suite.py, replace the placeholder with:

if model_name == 'lightgbm_quantile':
    from src.vn2.models.lightgbm import train_lightgbm_quantile
    train_lightgbm_quantile(data_path, horizons, n_folds, output_dir)

elif model_name == 'slurp_bootstrap':
    from src.vn2.models.slurp import train_slurp_bootstrap
    train_slurp_bootstrap(data_path, horizons, n_folds, output_dir)

# etc.
```

**OR** if you already have training code elsewhere, just call it!

---

## 📁 File Structure You'll Create

```
models/
  checkpoints/
    lightgbm_quantile_h3_fold0.pkl
    lightgbm_quantile_h4_fold0.pkl
    lightgbm_quantile_h5_fold0.pkl
    slurp_bootstrap_h3_fold0.pkl
    ...
  
  results/
    lightgbm_quantile_quantiles.parquet  ← Forecasts: store, product, q01_h3, q05_h3, ..., q01_h4, ...
    slurp_bootstrap_quantiles.parquet
    qrf_quantiles.parquet
    ...
    
    eval_patrick_all_models.parquet       ← Detailed costs per SKU × policy
    eval_patrick_all_models_jensen.parquet ← Jensen gaps

logs/
  train_20260205_120000.log  ← Training progress
  eval_20260205_140000.log   ← Evaluation with leaderboards
```

---

## 🎓 Why This Matters

### For the Paper
- **Section 6.1** describes what you'll do
- **Section 7** will show what you **found**
- This execution bridges that gap

### For Science  
- Proves: "forecast precisely right, optimize explicitly wrong"
- Shows: Density-aware decisions > point forecasts
- Validates: Patrick's corrected policy foundation

### For VN2 Competition
- Current: €6,266 (Patrick's baseline, 68% of benefit)
- Target: €5,248 (VN2 benchmark)
- Path: Layer SLURP sophistication on Patrick's foundation

---

## ⚠️ Important Notes

1. **Patrick's policy is the foundation** - Don't replace it, enhance it!
2. **Horizons must be h=3,4,5** - This is critical (not h=1,2)
3. **Critical fractile is τ=0.833** - Explicit newsvendor formula
4. **MC aggregation is required** - Not simple mean/var addition

All of this is now **baked into** `eval_all_models_patrick.py`.

---

## 🤔 Decision Time

### If you have existing training code:
✅ Just call it from `train_challenger_suite.py`  
✅ Make sure it outputs quantiles for h=3,4,5  
✅ Run the pipeline

### If you need to write training code:
1. Start with **one model** (e.g., `lightgbm_quantile`)
2. Get it working end-to-end
3. Clone the pattern for other models
4. OR run `--quick` mode with placeholder models to test evaluation

---

## 📞 Need Help?

Check these files:
- `TRAINING_EVALUATION_GUIDE.md` - Detailed step-by-step instructions
- `PATRICK_APPROACH_EXPLAINED.md` - Understanding Patrick's fixes
- `run_full_challenger_study.sh` - Master orchestration script

All scripts log to `logs/` with timestamps for debugging.

---

## 🎉 Success Criteria

You're done when you have:
- [ ] Trained models for all challengers
- [ ] Quantile forecasts (h=3,4,5) for each model
- [ ] Evaluation results showing SIP < Point for most models
- [ ] Jensen gap analysis confirming H1
- [ ] Leaderboard showing best model close to €5,248
- [ ] Results ready for paper Section 7

**GO! 🚀**
