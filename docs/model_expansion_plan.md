# Forecast Model Expansion Plan

## Objective
Integrate five additional forecasting strategies to improve the Week 1–6 backtest:
iETS, TSB, PatchTST, DeepAR, and a meta-router ensemble. Each model must provide
quantile forecasts compatible with the existing harness (13 quantiles → PMF).

---

## 1. Classical Intermittent Models

### iETS (Intermittent ETS State Space)
- **Scope**: Model demand sizes + intervals as separate state components per
  Syntetos et al. [1][2].
- **Implementation Steps**
  1. Port open-source iETS equations (transition + measurement update) into a
     training script under `scripts/train_iets.py`.
  2. Fit per-SKU models using the historical demand trajectory up to fold_k.
  3. Export quantile forecasts (h+1, h+2) into
     `models/checkpoints/iets/{store}_{product}/fold_{k}.pkl`.
  4. Update selector generation to include iETS (feature: intermittent flag).
- **Data Requirements**: Weekly demand history, zero-run lengths, CV.

### TSB (Teunter–Syntetos–Babai)
- Already have croston_* baselines; TSB adds a second smoothing parameter for
  demand occurrence [17].
- **Steps**
  1. Implement StatsForecast TSB logic (demand probability smoothing, demand size).
  2. Produce quantile outputs via bootstrap or analytical approximations.
  3. Save under `models/checkpoints/tsb/…`.

---

## 2. Deep Probabilistic Models

### PatchTST
- **Why**: Captures non-linearities, covariates (promo, store attributes), and
  handles mixed intermittency.
- **Steps**
  1. Prepare training windows (patches) per SKU cohort (e.g., high CV vs. low CV).
  2. Fine-tune a pretrained PatchTST using nixtla’s implementation.
  3. Export quantile predictions for horizons h+1/h+2.
  4. Register checkpoints at `models/checkpoints/patchtst/…`.

### DeepAR
- **Steps**
  1. Use GluonTS or PyTorch Forecasting’s DeepAR; include covariates (season,
     store features, guardrail flags).
  2. Train per cohort; validate via CRPS/pinball.
  3. Export quantile checkpoints `deepar/<sku>/fold_k.pkl`.
  4. Add to selector map.

---

## 3. Meta-Router / Ensemble

- **Goal**: Dynamically choose the best model per SKU-week using observed
  diagnostics (CV, zero-share, Wasserstein, weighted pinball, service flags).
- **Steps**
  1. Build feature table from historical weeks (actual cost vs. each model’s cost).
  2. Train a classifier/regressor to pick the minimal-cost model.
  3. Emit `reports/selector_overrides_router_w1_w6.csv` for the harness.
  4. Optionally create blended forecasts (weighted average) when classifier
     confidence < threshold.

---

## 4. Evaluation Pipeline

1. For each new model/router combination, run the 6-week harness via a dedicated
   spec (e.g., `configs/backtests/iets_only.yaml`, `patchtst_router.yaml`).
2. Capture `run_metrics.csv`, `weekly_summary.csv`, `classification_metrics`.
3. Compare against baseline using:
   - Total cost (holding + shortage)
   - Value-weighted pinball
   - Stockout precision/recall (classification tool)
   - Wasserstein distance segments
4. Document results in `reports/model_expansion_summary.md`.

---

## 5. Resource Plan

- **Training**: PatchTST/DeepAR each require GPU or multi-core CPU; schedule
  12-core/64 GB box with checkpoint/resume (approx 2–4 hours per cohort).
- **iETS/TSB**: CPU-friendly; can batch through SKUs within minutes.
- **Router**: Light training; run nightly after diagnostics pipeline.
- **Evaluation**: Reuse harness (already deterministic, spec-driven).

---

## References
1. https://openforecast.org/2023/09/08/iets-state-space-model-for-intermittent-demand-forecasting/
2. https://www.sciencedirect.com/science/article/pii/S0925527323002451
17. https://nixtlaverse.nixtla.io/statsforecast/docs/models/tsb.html


