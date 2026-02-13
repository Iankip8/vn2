# VN2 Pipeline: Raw Data → Final Models

Complete end-to-end workflow for the VN2 inventory optimization system.

---

## Overview

This pipeline transforms raw sales data into optimized inventory orders using:
- Probabilistic demand forecasting
- SIP-based uncertainty quantification
- Multi-period dynamic programming
- Cost-based model evaluation

---

## Phase 1: Data Ingestion

**Command:**
```bash
./go ingest --raw data/raw --out data/interim
```

**What it does:**
- Loads raw sales/inventory data
- Creates submission index (SKU list)
- Generates initial state (on_hand, in_transit inventory)
- Loads master data (SKU metadata)

**Outputs:**
- `data/interim/state.parquet` - Current inventory levels
- `data/interim/master.parquet` - SKU metadata

**Entry point:** `cmd_ingest()` in [src/vn2/cli.py](src/vn2/cli.py#L43)

---

## Phase 2: Exploratory Data Analysis (EDA)

**Via notebooks** (manual step):
```bash
# Open and run the notebook
jupyter notebook notebooks/02_comprehensive_time_series_eda.ipynb
```

**What it does:**
- Analyzes sales patterns across SKUs
- Identifies stockout periods (censored demand)
- Computes SURD transforms (optimal transformations for each SKU)
- Creates long-format demand history

**Outputs:**
- `data/processed/demand_long.parquet` - Sales history with `in_stock` flag
- `data/processed/surd_transforms.parquet` - Optimal transformations per SKU
- `data/processed/master.parquet` - Processed metadata

**Why this matters:** Stockouts censor true demand - we see sales=0 but actual demand may be higher.

---

## Phase 3: Stockout Imputation

**Command:**
```bash
./go impute --config configs/uncertainty.yaml --n-neighbors 20 --n-jobs -1
```

**What it does:**
1. Loads `demand_long.parquet` with stockout indicators
2. For each stockout event:
   - Finds K similar SKUs (nearest neighbors based on sales patterns)
   - Uses those neighbors' demand profiles to impute uncertainty distributions
3. Generates full quantile forecasts for censored periods using SIP
4. Creates training data with imputed values

**Key concept - SIP (Stochastic Information Packets):**
- Instead of single imputed values, creates full probability distributions
- Preserves correlations between SKUs
- Enables downstream uncertainty quantification

**Outputs:**
- `data/processed/demand_imputed.parquet` - Training data with imputed quantiles
- `data/processed/imputed_sips.parquet` - Full uncertainty distributions
- `data/processed/imputation_summary.parquet` - Quality metrics

**Entry point:** `cmd_impute_stockouts()` in [src/vn2/cli.py](src/vn2/cli.py#L257)

---

## Phase 4: Data Preprocessing Variants

**Scripts** (run separately as needed):
```bash
# Create winsorized version (caps extreme outliers)
python scripts/create_winsorized_data.py

# Create capped version (for SIP optimization)
# (prevents infeasible large demands in optimization)
```

**Outputs:**
- `data/processed/demand_imputed_winsor.parquet` - Stable for model training
- `data/processed/demand_imputed_capped.parquet` - For SIP optimization

**Why multiple versions?**
- **Winsorized:** Improves model stability by capping extreme outliers
- **Capped:** Prevents optimization from considering unrealistic demand scenarios
- **Raw:** Preserves censoring information for stockout-aware models

---

## Phase 5: Forecast Model Training

**Command:**
```bash
./go forecast --config configs/forecast.yaml --n-jobs 12
```

**Test mode** (single SKU for verification):
```bash
./go forecast --config configs/forecast.yaml --test --n-jobs 1
```

**Pilot mode** (subset for quick testing):
```bash
./go forecast --config configs/forecast.yaml --pilot --n-jobs 4
```

---

### Models Trained

#### 1. Baseline Models
- **Croston (Classic, SBA, TSB)** - Specialized for intermittent demand
- **Seasonal Naive** - Simple seasonal benchmark
- **Naive 4-week** - Rolling average

#### 2. Statistical Models
- **ZIP/ZINB** - Zero-Inflated Poisson/Negative Binomial
- **ETS** - Exponential Smoothing State Space
- **GLM** - Poisson/Negative Binomial regression

#### 3. Machine Learning Models
- **LightGBM Quantile** - Gradient boosting for quantiles
- **LightGBM Point** - Standard MSE-based (baseline)
- **QRF** - Quantile Random Forest
- **Linear Quantile** - Quantile regression
- **NGBoost** - Probabilistic gradient boosting
- **KNN Profile** - Nearest neighbor forecasts

#### 4. SLURP Models (Advanced)
- **SLURP Bootstrap** - Bootstrap-based density forecasts
- **SLURP SURD** - With optimal transformations (no stockout handling)
- **SLURP Stockout-Aware** - Censoring-aware bootstrap
- **SLURP SURD + Stockout** - Combines both features

---

### Training Process

**For each SKU:**
1. Extract historical demand
2. Create lag features, calendar features
3. Fit model to historical data
4. Generate quantile forecasts for H=1 to H=12 weeks
5. Save checkpoint with forecasts

**Data routing:**
- **SLURP models:** Train on `demand_long.parquet` (raw, censoring-aware)
- **Other models:** Train on `demand_imputed_winsor.parquet` (stable, imputed)

**Parallelization:**
- Processes SKUs in parallel (configurable workers)
- Timeout protection per SKU fit
- Progress tracking with checkpoints

**Outputs:**
- `models/checkpoints/{model_name}/{store}_{product}/fold_{i}.pkl`
  - Contains: quantile forecasts, metadata, fit diagnostics
- `models/results/training_results.parquet` - Training summary

**Entry point:** `cmd_forecast()` in [src/vn2/cli.py](src/vn2/cli.py#L399)

---

## Phase 6: Model Evaluation

**Command:**
```bash
./go eval-models --holdout 8 --n-sims 500 --use-sip-optimization --sip-grain 1000 --n-jobs 12
```

**Options:**
- `--holdout 8` - Number of rolling-origin validation folds
- `--n-sims 500` - Monte Carlo samples for policy evaluation
- `--use-sip-optimization` - Use SIP-based newsvendor optimization
- `--sip-grain 1000` - PMF discretization (max support)
- `--resume` - Resume from checkpoint
- `--aggregate` - Only run aggregation step

---

### Evaluation Process

**For each fold (rolling origin):**
1. **Load forecasts** from checkpoint for this fold's date
2. **Convert quantiles → PMF** at specified grain
3. **Optimize order quantity:**
   - Without `--use-sip-optimization`: Base-stock policy (service level)
   - With `--use-sip-optimization`: SIP newsvendor optimization per SKU
4. **Simulate inventory:**
   - Apply order decision
   - Observe realized demand
   - Compute costs: `0.2 * excess + 1.0 * shortfall`
5. **Aggregate** across SKUs and folds

**Rolling-origin validation:**
```
Fold 0: Train on [--------] → Forecast week 1
Fold 1: Train on [---------] → Forecast week 2
Fold 2: Train on [----------] → Forecast week 3
...
Fold 7: Train on [---------------] → Forecast week 8
```

**Outputs:**
- `models/results/eval_folds_v4_sip.parquet` - Per-fold, per-SKU costs
- `models/results/leaderboards_v4_sip.parquet` - Aggregated model rankings
- `models/results/eval_progress.json` - Checkpoint for resuming
- `models/results/eval_summary_stats.parquet` - Distribution statistics

**Entry point:** `cmd_eval_models()` in [src/vn2/cli.py](src/vn2/cli.py#L317)

---

## Phase 7: Sequential Evaluation (Multi-Period)

**Command:**
```bash
./go sequential-eval --checkpoints models/checkpoints --holdout 12 --n-jobs 12 --sip-grain 500
```

**Options:**
- `--holdout 12` - Number of sequential epochs to simulate
- `--cu 1.0` - Shortage (underage) cost per unit
- `--co 0.2` - Holding (overage) cost per unit
- `--sip-grain 500` - PMF grain for optimization

---

### Sequential Decision Process

**Key difference from Phase 6:** Multi-period lookahead with lead time L=2

**For each epoch (week):**

1. **Load state:**
   - I₀ = current on-hand inventory
   - Q₁ = order arriving next week
   - Q₂ = order arriving in 2 weeks

2. **Load forecasts:**
   - h=1 demand distribution (next week)
   - h=2 demand distribution (2 weeks ahead)

3. **Convert to PMFs** at grain resolution

4. **Dynamic programming optimization:**
   - Considers 2-period lookahead
   - Accounts for lead time L=2, review period R=1
   - Chooses order Q₀ to minimize expected total cost

5. **Simulate:**
   - Observe realized demand d₁
   - Update inventory: I₁ = max(0, I₀ + Q₁ - d₁)
   - Update in-transit: Q₁ ← Q₂, Q₂ ← Q₀
   - Compute period costs

6. **Advance to next epoch**

**This simulates realistic operations:**
- Orders placed today arrive in 2 weeks
- Must forecast 2 weeks ahead
- Sequential decisions build on previous outcomes

**Outputs:**
- `models/results/seq12_folds.parquet` - Per-epoch, per-SKU results
- `models/results/seq12_leaderboard.parquet` - Model rankings by total cost
- `models/results/seq12_summary.parquet` - Aggregate statistics

**Entry point:** `cmd_sequential_eval()` in [src/vn2/cli.py](src/vn2/cli.py#L926)

---

## Phase 8: Ensemble Models

**Command:**
```bash
# Build per-SKU selector ensemble
./go ensemble-eval --stage selector \
  --eval-folds models/results/eval_folds_v4_sip.parquet \
  --selector-map models/results/per_sku_selector_map.parquet \
  --out-suffix _v4_ens_selector

# Aggregate results
./go ensemble-eval --stage selector --out-suffix _v4_ens_selector --aggregate
```

---

### Ensemble Strategies

#### 1. Selector Ensemble
**Concept:** Each SKU uses its own best model

**Process:**
1. For each SKU, identify best-performing model from eval_folds
2. Create selector map: SKU → Model
3. Rebuild fold results using per-SKU selections

**When to use:** When different SKU types benefit from different models

#### 2. Cohort Ensemble
**Concept:** Group SKUs into cohorts, assign best model per cohort

**Process:**
1. Segment SKUs by characteristics:
   - Demand rate (high/medium/low)
   - Coefficient of variation (stable/variable)
   - Zero ratio (intermittent vs continuous)
   - Stockout pattern
2. For each cohort, select best model
3. Apply cohort rules to assign models

**When to use:** When you want interpretable rules and generalization

---

### Example Cohort Rules
```
IF demand_rate = 'high' AND cv = 'low' 
   → Use LightGBM Quantile

IF demand_rate = 'low' AND zero_ratio = 'high'
   → Use Croston SBA

IF stockout_pattern = 'frequent'
   → Use SLURP Stockout-Aware
```

**Outputs:**
- `models/results/eval_folds_v4_ens_selector.parquet` - Ensemble fold results
- `models/results/leaderboards_v4_ens_selector.parquet` - Ensemble rankings
- `models/results/cohort_rules.parquet` - Cohort assignment rules (if cohort stage)

**Entry point:** `cmd_ensemble_eval()` in [src/vn2/cli.py](src/vn2/cli.py#L916)

---

## Phase 9: Final Order Generation

**Command:**
```bash
./go today-order \
  --model slurp_surd_stockout_aware \
  --state data/interim/state.parquet \
  --out data/submissions/order_today.csv \
  --cu 1.0 --co 0.2 --sip-grain 500
```

---

### Order Generation Process

**For each SKU:**

1. **Load current state:**
   - I₀ = current on-hand inventory
   - Q₁ = order arriving in 1 week
   - Q₂ = order arriving in 2 weeks

2. **Load latest forecasts:**
   - From checkpoint: `fold_0` (most recent training)
   - Extract h=1 and h=2 quantile forecasts

3. **Convert to PMFs** at grain resolution

4. **Optimize order:**
   - Use L=2 dynamic programming
   - Minimize expected cost given state and forecasts
   - Returns Q₀ = order quantity to place now

5. **Export to submission format**

**Output format:**
```csv
Store,Product,q_now
1,101,45
1,102,0
1,103,23
...
```

**Entry point:** `cmd_today_order()` in [src/vn2/cli.py](src/vn2/cli.py#L975)

---

## Key Data Artifacts

| **Artifact** | **Location** | **Purpose** |
|-------------|-------------|------------|
| Raw sales | `data/raw/` | Competition input data |
| State | `data/interim/state.parquet` | Current inventory levels |
| Demand long | `data/processed/demand_long.parquet` | Historical sales with stockout flags |
| SURD transforms | `data/processed/surd_transforms.parquet` | Optimal transformations per SKU |
| Imputed demand | `data/processed/demand_imputed.parquet` | Training data with imputed stockouts |
| Winsorized | `data/processed/demand_imputed_winsor.parquet` | Stable training data (outliers capped) |
| Capped | `data/processed/demand_imputed_capped.parquet` | For SIP optimization |
| Checkpoints | `models/checkpoints/{model}/{sku}/fold_{i}.pkl` | Trained forecasts per SKU |
| Eval folds | `models/results/eval_folds_*.parquet` | Validation results |
| Leaderboards | `models/results/leaderboards_*.parquet` | Model rankings by cost |
| Submissions | `data/submissions/` | Final order files |

---

## Complete Workflow Example

### First-Time Setup

```bash
# 1. Ingest raw data
./go ingest --raw data/raw --out data/interim

# 2. Run EDA notebook (manual)
jupyter notebook notebooks/02_comprehensive_time_series_eda.ipynb
# → Generates demand_long.parquet, surd_transforms.parquet

# 3. Impute stockouts
./go impute --n-neighbors 20 --n-jobs -1

# 4. Create winsorized version (if not already done)
python scripts/create_winsorized_data.py
```

### Model Development Cycle

```bash
# 5. Train models (pilot mode first)
./go forecast --config configs/forecast.yaml --pilot --n-jobs 4

# 6. Quick evaluation
./go eval-models --holdout 4 --use-sip-optimization --n-jobs 4

# 7. Full training
./go forecast --config configs/forecast.yaml --n-jobs 12

# 8. Full evaluation with SIP
./go eval-models --holdout 8 --use-sip-optimization --n-sims 500 --n-jobs 12

# 9. Aggregate results
./go eval-models --aggregate
```

### Advanced Evaluation

```bash
# 10. Sequential multi-period evaluation
./go sequential-eval --holdout 12 --sip-grain 500 --n-jobs 12

# 11. Build ensemble
./go ensemble-eval --stage selector --aggregate

# 12. Compare all approaches
python scripts/compare_evaluation_runs.py
```

### Production Deployment

```bash
# 13. Generate today's orders
./go today-order \
  --model slurp_surd_stockout_aware \
  --state data/interim/state.parquet \
  --out data/submissions/order_$(date +%Y%m%d).csv

# 14. Analyze order
python scripts/analyze_submitted_order.py data/submissions/order_20260202.csv
```

---

## Understanding Key Concepts

### SIP (Stochastic Information Packets)
- Represents uncertainty as discrete probability distributions
- Preserves correlations between variables
- Enables efficient sampling and optimization

### SLURP (Stochastic Locally Uniform Resource Packages)
- Extension of SIP for correlated scenarios
- Maintains dependency structure across SKUs
- Used for stockout imputation and forecast uncertainty

### SURD (Scaled Unit Root Distribution)
- Optimal transformation for each SKU's demand
- Examples: log, sqrt, identity
- Improves forecast accuracy for non-normal data

### Stockout Censoring
- When inventory = 0, observed sales understates true demand
- Must impute "latent demand" using similar SKUs
- Critical for accurate forecasting

### Newsvendor Optimization
- Classic inventory problem: how much to order?
- Balances holding costs vs shortage costs
- Uses demand distribution (not just mean)

### Critical Fractile
- Optimal service level = Cu / (Cu + Co)
- Example: Cu=1.0, Co=0.2 → target 83.3% service level
- Where Cu = shortage cost, Co = holding cost

---

## Configuration Files

### configs/base.yaml
- Basic simulation parameters
- Cost structure
- Lead time and review period

### configs/uncertainty.yaml
- SIP configuration (quantile levels, horizon)
- Imputation parameters
- Monte Carlo settings

### configs/forecast.yaml
- Model selection (enable/disable)
- Hyperparameters per model
- Training settings (n_jobs, timeout)
- Pilot mode configuration

---

## Monitoring and Debugging

### Check Training Progress
```bash
# View training results
python -c "import pandas as pd; print(pd.read_parquet('models/results/training_results.parquet'))"
```

### Check Evaluation Progress
```bash
# View evaluation checkpoint
cat models/results/eval_progress.json

# Resume interrupted evaluation
./go eval-models --holdout 8 --use-sip-optimization --resume --n-jobs 12
```

### Validate Forecasts
```bash
# Check specific SKU checkpoint
python -c "
import pickle
with open('models/checkpoints/slurp_surd/1_101/fold_0.pkl', 'rb') as f:
    cp = pickle.load(f)
    print(cp['quantiles'])
"
```

### Analyze Results
```bash
# Compare models
python scripts/compare_evaluation_runs.py

# Analyze leaderboard
python scripts/analyze_leaderboards.py
```

---

## Troubleshooting

### Issue: Out of memory during training
**Solution:** Reduce `n_jobs` or use `--pilot` mode
```bash
./go forecast --config configs/forecast.yaml --n-jobs 4
```

### Issue: Evaluation too slow
**Solution:** Reduce `n_sims` or `sip_grain`
```bash
./go eval-models --n-sims 100 --sip-grain 500 --n-jobs 8
```

### Issue: Some SKUs fail during training
**Solution:** Check timeout setting in configs/forecast.yaml
```yaml
compute:
  timeout_per_fit: 300  # Increase if needed
```

### Issue: Missing checkpoints
**Solution:** Retrain specific models
```bash
# Edit configs/forecast.yaml to enable only needed models
./go forecast --config configs/forecast.yaml --n-jobs 12
```

---

## Next Steps

1. **Review results:** Check leaderboards to see which models perform best
2. **Iterate:** Enable/disable models based on performance
3. **Tune:** Adjust hyperparameters in configs/forecast.yaml
4. **Ensemble:** Combine top models for better robustness
5. **Deploy:** Use best model for production orders

---

## Additional Resources

- **Developer Guide:** [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **SIP Implementation:** [SIP_IMPLEMENTATION.md](SIP_IMPLEMENTATION.md)
- **SURD Documentation:** [docs/SURD/](docs/SURD/)
- **Evaluation Guide:** [SEQUENTIAL_EVAL_GUIDE.md](SEQUENTIAL_EVAL_GUIDE.md)
