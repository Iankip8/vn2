# Complete Training & Evaluation Guide

**Date:** February 5, 2026  
**Purpose:** Train all models from paper Section 6.1 and evaluate using Patrick's corrected policy

---

## Quick Start

### Option 1: Full Pipeline (Recommended)
```bash
# Run everything: data prep, training, evaluation, analysis
./run_full_challenger_study.sh
```

**Time:** 4-8 hours (depending on hardware)  
**Output:** Complete results for paper Section 7

### Option 2: Quick Test (Subset)
```bash
# Test with just 2 models to verify pipeline
./run_full_challenger_study.sh --quick
```

**Time:** ~30 minutes  
**Output:** Proof of concept

---

## Step-by-Step Manual Execution

### Step 1: Data Preparation
```bash
# Create winsorized data for non-SLURP models
uv run python scripts/create_winsorized_data.py
```

**Input:** `data/processed/demand_long.parquet` (raw with stockouts)  
**Output:** `data/processed/demand_imputed_winsor.parquet` (SURD transform-space winsorized)

### Step 2: Train Models

#### Train All Models
```bash
uv run python scripts/train_challenger_suite.py \
    --models all \
    --horizons 3 4 5 \
    --folds 5
```

#### Train Specific Model Groups
```bash
# SLURP family only
uv run python scripts/train_challenger_suite.py --models slurp

# Gradient boosting models
uv run python scripts/train_challenger_suite.py --models lightgbm ngboost

# Tree-based models
uv run python scripts/train_challenger_suite.py --models qrf

# All challengers (no SLURP)
uv run python scripts/train_challenger_suite.py --models lightgbm tree stats parametric
```

**Output:** Model checkpoints in `models/checkpoints/`

### Step 3: Evaluate All Models
```bash
uv run python scripts/eval_all_models_patrick.py
```

**What it does:**
1. Loads forecasts for each model (h=3,4,5 quantiles)
2. Evaluates under **TWO policies**:
   - **Density-aware SIP**: Patrick's corrected MC aggregation
   - **Point + service-level**: Traditional industry practice
3. Computes **Jensen gap**: Δcost = (point policy) - (SIP)
4. Generates leaderboards

**Output:** 
- `models/results/eval_patrick_all_models.parquet` (detailed results)
- `models/results/eval_patrick_all_models_jensen.parquet` (Jensen gaps)

### Step 4: Analyze Results
```bash
# View leaderboards
cat logs/eval_*log | grep -A 20 "LEADERBOARD"

# Jensen gap summary
python -c "
import pandas as pd
df = pd.read_parquet('models/results/eval_patrick_all_models_jensen.parquet')
print(df.groupby('model')['jensen_gap'].describe())
"
```

---

## Model Groups (from Paper Section 6.1)

### SLURP Family (Train on Raw Data)
- `slurp_bootstrap` - Baseline conditional bootstrap
- `slurp_surd` - SURD transforms, no stockout handling
- `slurp_stockout_aware` - Censoring-aware, no SURD
- `slurp_surd_stockout_aware` - Full implementation

**Data:** `demand_long.parquet` (raw with `in_stock` indicator)

### Challengers (Train on Winsorized Data)

#### Gradient Boosting
- `lightgbm_quantile` - Quantile regression (density)
- `lightgbm_point` - MSE objective (point baseline)
- `ngboost` - NGBoost LogNormal (density)

#### Tree-Based
- `qrf` - Quantile Random Forest (density)

#### Statistical
- `ets` - Exponential Smoothing (density via PIs)
- `seasonal_naive` - Seasonal naive baseline

#### Intermittent Demand
- `croston_classic` - Classic Croston
- `croston_sba` - Syntetos-Boylan Approximation
- `croston_tsb` - Teunter-Syntetos-Babai

#### Parametric
- `zinb` - Zero-Inflated Negative Binomial
- `zip` - Zero-Inflated Poisson
- `glm_poisson` - GLM Poisson (deferred)

#### Linear
- `linear_quantile` - Linear quantile regression

**Data:** `demand_imputed_winsor.parquet` (SURD transform-space μ+3σ winsorization)

---

## Patrick's Corrected Policy (What Makes This Different)

### The 3 Critical Fixes

**Fix #1: Protection Period = 3 Weeks**
```python
# WRONG (original): horizons = [1, 2]
# CORRECT (Patrick): horizons = [3, 4, 5]
protection_weeks = lead_time + review_period  # 2 + 1 = 3
```

**Fix #2: Explicit Critical Fractile**
```python
tau = costs.shortage / (costs.holding + costs.shortage)  # 0.833
# €1.00 shortage / (€0.20 holding + €1.00 shortage) = 0.833
```

**Fix #3: Monte Carlo Aggregation**
```python
# Sample 10,000 scenarios from weekly distributions
# Sum demand across weeks 3, 4, 5
# Compute mean μ and std σ of aggregated demand
mu, sigma = aggregate_weekly_distributions_mc(
    quantiles_df, quantile_levels,
    protection_weeks=3, n_samples=10000
)
```

**Result:** €6,266 vs €9,193 (31.8% improvement)

---

## Expected Outcomes (for Paper Section 7)

### H1: Jensen Gap Hypothesis
- **Prediction:** Density-aware SIP beats point+service-level
- **Measure:** Positive Jensen gaps across all models
- **Paper content:** Jensen delta distributions, cohort analysis

### H2: Stockout Awareness Hypothesis
- **Prediction:** SLURP stockout-aware < SLURP baseline on high-stockout cohorts
- **Measure:** Cost reduction on censored SKUs
- **Paper content:** Ablation study results

### H3: SURD Effect Hypothesis
- **Prediction:** SLURP SURD < SLURP identity on interval sharpness
- **Measure:** Coverage @ critical fractile, interval width
- **Paper content:** Transform selection impact

### H4: Sequential Consistency Hypothesis  
- **Prediction:** SIP with correct lead times discriminates better than proxies
- **Measure:** Model rankings stability, correlation with realized cost
- **Paper content:** Validation of evaluation methodology

---

## Troubleshooting

### "demand_long.parquet not found"
```bash
# Run data preprocessing
uv run python scripts/process_eda_data.py
```

### "No forecasts found for model X"
The training script is currently a **placeholder**. You need to:
1. Check if model training code exists in `src/vn2/models/`
2. Implement model-specific training if missing
3. Generate forecast outputs to `models/results/{model}_quantiles.parquet`

### "Model training fails"
Check logs in `logs/train_*.log` for specific errors. Common issues:
- Missing dependencies (install with `uv pip install`)
- Insufficient memory (reduce `n_jobs` in config)
- Data format mismatches (verify column names)

---

## File Locations

### Input Data
- `data/processed/demand_long.parquet` - Raw demand with stockout indicators
- `data/processed/demand_imputed_winsor.parquet` - Winsorized for challengers

### Training Outputs
- `models/checkpoints/{model}_h{horizon}_fold{fold}.pkl` - Trained model checkpoints

### Forecast Outputs
- `models/results/{model}_quantiles.parquet` - Quantile forecasts (h=3,4,5)

### Evaluation Outputs
- `models/results/eval_patrick_all_models.parquet` - Detailed costs per SKU × policy
- `models/results/eval_patrick_all_models_jensen.parquet` - Jensen gaps per SKU

### Logs
- `logs/train_*.log` - Training logs
- `logs/eval_*.log` - Evaluation logs with leaderboards
- `logs/winsorize_*.log` - Data preparation logs

---

## Next Steps After Evaluation

1. **Generate Visualizations**
   - Jensen gap distributions (histograms, boxplots)
   - Cohort analysis plots (by sparsity, stockout rate, CV)
   - Leaderboard comparisons

2. **Update Paper Section 7**
   - Table 1: Model leaderboard (SIP policy)
   - Table 2: Jensen gap summary
   - Figure 1-4: Cohort analyses (as in current draft)

3. **Test Remaining Hypotheses**
   - H2: Ablation study (SLURP variants)
   - H3: SURD transform analysis
   - H4: Sequential consistency validation

4. **Layer Sophistication**
   - Now that Patrick's foundation works (€6,266)
   - Add full SIP/SLURP features to close remaining gap to €5,248
   - Implement advanced stockout imputation
   - Add SURD variance stabilization refinements

---

## Summary

**What you're doing:**
- Training 9+ models from paper Section 6.1
- Evaluating under **both** density-aware SIP and point+service-level policies
- Using **Patrick's corrected policy** (h=3,4,5, τ=0.833, MC aggregation)
- Quantifying the **Jensen gap** to prove density > point
- Generating results for paper Section 7 (hypotheses testing)

**Why it matters:**
- Validates the paper's core claim: "forecast precisely right, optimize explicitly wrong"
- Proves that density-aware decisions materially reduce cost vs point forecasts
- Shows that Patrick's simple corrections get 68% of the benefit before adding sophistication

**The payoff:**
- Publishable results for INFORMS Analytics+ submission
- Reproducible evidence that modeling uncertainty correctly > chasing point accuracy
- Foundation for layering advanced SIP/SLURP features on top
