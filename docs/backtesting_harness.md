# Backtesting Harness

This document explains the flexible harness for replaying inventory strategies
across historical weeks while respecting the 2-week lead time semantics.

---

## Overview

`scripts/run_backtest_harness.py` consumes a YAML specification and produces a
fully-audited simulation run under `reports/backtests/<run_id>/`. The harness:

- uses the selector + guardrail overrides from the diagnostics workflow
- loads only the data that would have been available at each decision week
- generates counterfactual orders via `choose_order_L2` (PMF-based newsvendor)
- simulates realized holding + shortage costs using actual demand
- stores per-SKU metrics, weekly summaries, and a Markdown report for audit
- tags each run with metadata (spec copy, CLI args, timestamps, logging)

The system is extensible: additional forecasting strategies (bias corrections,
value-weighted prioritisation, challenger selectors) can be implemented by
augmenting the spec without modifying historical data.

---

## Spec Structure

Specs live under `configs/backtests/` (see
`example_value_weighted.yaml`). Key sections:

```yaml
run_id: value_weighted_w1_w6
weeks: [1,2,3,4,5,6]

paths:
  initial_state: data/raw/Week 0 - 2024-04-08 - Initial State.csv
  state_dir: data/states
  checkpoints_dir: models/checkpoints
  selector_map: models/results/selector_map_seq12_v1.parquet
  selector_overrides: reports/selector_overrides_w6.csv
  guardrail_overrides: reports/guardrail_overrides_w6.csv
  output_root: reports/backtests

week_data:        # optional per-week overrides
  - week: 1
    state: data/states/state1.csv
  - week: 2
    state: data/states/state2.csv

strategy:
  costs:
    holding: 0.2
    shortage: 1.0
  forecast:
    sip_grain: 500
    gaussian_weight_band: [0.73, 0.93]
    selector_override_enabled: true
    guardrails_enabled: true
  allocation:
    service_bias: 0.0
    default_service_level: 0.8333

logging:
  level: INFO
  tag_namespace: diagnostics_w6
```

- `run_id` + optional `tag` become `run_id_tag_timestamp` for output namespace.
- `paths.initial_state` should be the Week 0 state (end-of-week inventory) so
  the harness reconstructs the start of Week 1 correctly.
- `week_data` entries override `state_dir` pattern on a per-week basis (useful
  if filenames differ or new weeks arrive).
- Strategy block controls SIP granularities, Gaussian-weighted pinball band,
  selector overrides, guardrail overrides, and baseline costs.

---

## CLI Usage

```bash
python scripts/run_backtest_harness.py \
  --run-spec configs/backtests/example_value_weighted.yaml

# Optional overrides:
python scripts/run_backtest_harness.py \
  --run-spec configs/backtests/example_value_weighted.yaml \
  --tag quick_eval \
  --weeks 3 4 5 \
  --run-id vw_w1_w6_lab

# Dry-run just validates the spec (no outputs):
python scripts/run_backtest_harness.py \
  --run-spec configs/backtests/example_value_weighted.yaml \
  --dry-run
```

CLI parameters:

- `--run-spec`: YAML file path (required)
- `--run-id`: override for spec run_id
- `--tag`: appended tag for experimentation
- `--weeks`: custom subset of weeks (space separated)
- `--dry-run`: load config, build lookups, skip simulation

---

## Outputs

Each run writes to `reports/backtests/<run_id_timestamp>/`:

| File | Description |
| ---- | ----------- |
| `run_spec.yaml` | exact spec copy used |
| `run.log` | structured logs (INFO + DEBUG for troubleshooting) |
| `run_metadata.json` | CLI args, paths, timestamps, output pointers |
| `weekly_summary.csv` | per-week units, costs, baseline deltas |
| `run_metrics.csv` | per-SKU metrics (orders, demand, costs, weighted pinball) |
| `orders_week*.csv` | counterfactual orders for each week |
| `run_summary.md` | Markdown overview with cumulative stats |

The metadata JSON makes it easy to trace the inputs, enabling reproducible
analysis or additional ex post computations (e.g., leaderboard correlation).

---

## Extending the Harness

1. **New forecasting models**: train + drop checkpoints into
   `models/checkpoints/<model_name>/<store>_<product>/fold_*.pkl`. Update the
   selector map / overrides or add new strategy toggles to pick the challenger.

2. **Alternative allocation strategies**: add new knobs (e.g., service level
   overrides per cohort, stockout-aware selectors) to the spec and extend the
   order generation logic to interpret them. The harness already applies custom
   service levels by translating target service to `cu/(cu+co)` as part of
   the `choose_order_L2` call.

3. **Additional metrics**: `run_metrics.csv` stores demand, expected cost,
   realized cost, guardrail flags, and Gaussian-weighted pinball. Plug this into
   downstream notebooks (`reports/forecast_diagnostics_analysis.md`) to rerun
   skill-vs-luck or guardrail tuning analyses.

4. **Future weeks**: once Week 7/8 state + sales files arrive, drop them into
   `data/states`, add entries to `week_data`, and rerun with `--weeks 1 2 .. 8`.

5. **Audit logging**: `logging.level` can be upgraded to `DEBUG` in the spec for
   temporary introspection; the harness logs sample sales/missed series to
   confirm demand ingestion per week.

---

## Practical Tips

- Keep guardrail + selector override CSVs under `reports/` with run-specific
  names so future experiments can mix-and-match override sets.
- Use git tags/branches per backtest campaign to keep configs + outputs in sync.
- After each harness run, review `run_summary.md` and `weekly_summary.csv`
  before diving into per-SKU pivots; the delta vs. baseline highlights whether
  the new strategy improved or hurt cost/service.
- Pair harness runs with the `forecast_diagnostics_analysis.md` report to
  contextualize which cohorts drive gains/losses (CV bins, demand rate, zero
  share, etc.).

---

This harness is purpose-built for rapid experimentation in the contest setting,
but the spec-driven design makes it straightforward to integrate new models,
service policies, and evaluation metrics as new data (Weeks 7–8, leaderboard
insights, challenger models) becomes available.


