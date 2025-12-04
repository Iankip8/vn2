#!/usr/bin/env python3
"""
Forecast diagnostics (Weeks 1-6).

Computes value-weighted pinball, value-weighted precision/recall, and Wasserstein distance
per SKU/week using only information available at the time (fold_{week-1}).
Outputs per-SKU metrics CSV and a Markdown summary.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from vn2.analyze.sequential_backtest import quantiles_to_pmf
from vn2.analyze.sequential_planner import Costs

MU = 0.8333
SIGMA = 0.05
BAND_LOW = 0.73
BAND_HIGH = 0.93
SHORTAGE_COST = 1.0
HOLDING_COST = 0.2


def gaussian_weights(levels: np.ndarray) -> np.ndarray:
    weights = np.exp(-0.5 * ((levels - MU) / SIGMA) ** 2)
    mask = (levels >= BAND_LOW) & (levels <= BAND_HIGH)
    weights *= mask
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def load_selector_map(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[['store', 'product', 'model_name']].copy()


def load_state_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if 'start inventory' in cl:
            rename[col] = 'start_inventory'
        elif cl == 'sales':
            rename[col] = 'sales'
        elif 'missed sales' in cl:
            rename[col] = 'missed_sales'
    if rename:
        df = df.rename(columns=rename)
    return df


def load_quantiles(model: str, store: int, product: int, fold_idx: int, checkpoints_dir: Path):
    ckpt = checkpoints_dir / model / f'{store}_{product}' / f'fold_{fold_idx}.pkl'
    if not ckpt.exists():
        return None
    with open(ckpt, 'rb') as f:
        data = pickle.load(f)
    return data.get('quantiles')


def compute_metrics_for_week(week: int,
                             state_df: pd.DataFrame,
                             selector_map: pd.DataFrame,
                             checkpoints_dir: Path,
                             q_levels: np.ndarray,
                             weights: np.ndarray) -> pd.DataFrame:
    fold_idx = week - 1
    entries = []
    model_lookup = {(int(row.store), int(row.product)): row.model_name for row in selector_map.itertuples(index=False)}
    for rec in state_df.itertuples(index=False):
        store = int(rec[0])
        product = int(rec[1])
        start_inv = float(getattr(rec, 'start_inventory', getattr(rec, 'Start Inventory', 0.0)))
        sales = float(getattr(rec, 'sales', getattr(rec, 'Sales', 0.0)))
        missed = float(getattr(rec, 'missed_sales', getattr(rec, 'Missed Sales', 0.0)))
        actual = sales + missed
        model = model_lookup.get((store, product))
        if model is None:
            continue
        qdf = load_quantiles(model, store, product, fold_idx, checkpoints_dir)
        if qdf is None or 1 not in qdf.index:
            continue
        q_series = qdf.loc[1]
        pmf = quantiles_to_pmf(q_series.values, q_levels, grain=500)
        support = np.arange(len(pmf))
        diffs = actual - q_series.values
        pinball_losses = np.maximum(q_levels * diffs, (q_levels - 1.0) * diffs)
        weighted_pinball = float(np.sum(weights * pinball_losses)) if weights.sum() > 0 else float(np.mean(pinball_losses))
        wasserstein = float(np.sum(pmf * np.abs(support - actual)))
        prob_stockout = float(pmf[support > start_inv].sum()) if start_inv < support[-1] else 0.0
        predicted_stockout = prob_stockout >= 0.2
        actual_stockout = missed > 0
        weight_stock = SHORTAGE_COST
        tp_weight = weight_stock if (predicted_stockout and actual_stockout) else 0.0
        pred_weight = weight_stock if predicted_stockout else 0.0
        actual_weight = weight_stock if actual_stockout else 0.0
        entries.append({
            'week': week,
            'store': store,
            'product': product,
            'model': model,
            'actual_demand': actual,
            'start_inventory': start_inv,
            'missed_sales': missed,
            'weighted_pinball': weighted_pinball,
            'wasserstein': wasserstein,
            'prob_stockout': prob_stockout,
            'pred_stockout': predicted_stockout,
            'actual_stockout': actual_stockout,
            'tp_weight': tp_weight,
            'pred_weight': pred_weight,
            'actual_weight': actual_weight,
        })
    return pd.DataFrame(entries)


def summarize_metrics(df: pd.DataFrame) -> dict:
    summary = {}
    if df.empty:
        return summary
    summary['n_records'] = len(df)
    summary['weighted_pinball_mean'] = float(df['weighted_pinball'].mean())
    summary['weighted_pinball_median'] = float(df['weighted_pinball'].median())
    summary['wasserstein_mean'] = float(df['wasserstein'].mean())
    summary['wasserstein_median'] = float(df['wasserstein'].median())
    total_tp = df['tp_weight'].sum()
    total_pred = df['pred_weight'].sum()
    total_actual = df['actual_weight'].sum()
    summary['value_precision'] = float(total_tp / total_pred) if total_pred > 0 else None
    summary['value_recall'] = float(total_tp / total_actual) if total_actual > 0 else None
    return summary


def write_summary_md(path: Path, summary_rows: list, overall: dict, missing_weeks: list):
    lines = ['# Forecast Diagnostics Summary', '']
    if missing_weeks:
        lines.append(f"Pending weeks (missing forecasts/state): {', '.join(missing_weeks)}")
        lines.append('')
    lines.append('## Week-by-Week Metrics')
    lines.append('Week | Records | Mean Pinball | Median Pinball | Mean Wasserstein | Median Wasserstein | Value Precision | Value Recall')
    lines.append('---- | -------:| ------------:| --------------:| ----------------:| -----------------:| ---------------:| -------------:')
    for row in summary_rows:
        prec = row.get('value_precision')
        rec = row.get('value_recall')
        prec_str = f"{prec:.2f}" if prec is not None else 'NA'
        rec_str = f"{rec:.2f}" if rec is not None else 'NA'
        lines.append(f"{row['week']} | {row['n_records']} | {row['weighted_pinball_mean']:.2f} | {row['weighted_pinball_median']:.2f} | {row['wasserstein_mean']:.2f} | {row['wasserstein_median']:.2f} | {prec_str} | {rec_str}")
    lines.append('')
    lines.append('## Overall Statistics (Weeks Processed)')
    for key, value in overall.items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Forecast diagnostics (Weeks 1-6)')
    parser.add_argument('--selector-map', type=Path, default=Path('models/results/selector_map_seq12_v1.parquet'))
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('models/checkpoints'))
    parser.add_argument('--state-dir', type=Path, default=Path('data/states'))
    parser.add_argument('--weeks', type=int, default=6)
    parser.add_argument('--out-csv', type=Path, default=Path('reports/forecast_diagnostics_metrics.csv'))
    parser.add_argument('--out-md', type=Path, default=Path('reports/forecast_diagnostics_summary.md'))
    args = parser.parse_args()

    selector_map = load_selector_map(args.selector_map)
    q_levels = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    weights = gaussian_weights(q_levels)

    all_metrics = []
    week_summaries = []
    missing_weeks = []

    for week in range(1, args.weeks + 1):
        state_path = args.state_dir / f'state{week}.csv'
        if not state_path.exists():
            missing_weeks.append(str(week))
            continue
        state_df = load_state_csv(state_path)
        df = compute_metrics_for_week(week, state_df, selector_map, args.checkpoints_dir, q_levels, weights)
        if df.empty:
            missing_weeks.append(str(week))
            continue
        all_metrics.append(df)
        summary = summarize_metrics(df)
        summary['week'] = week
        week_summaries.append(summary)

    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.out_csv, index=False)
        overall = {
            'records_total': len(metrics_df),
            'avg_weighted_pinball': float(metrics_df['weighted_pinball'].mean()),
            'avg_wasserstein': float(metrics_df['wasserstein'].mean()),
        }
    else:
        metrics_df = pd.DataFrame()
        overall = {'records_total': 0}

    write_summary_md(args.out_md, week_summaries, overall, missing_weeks)
    print(f"✓ Metrics written: {args.out_csv} ({len(metrics_df)} rows)")
    print(f"✓ Summary written: {args.out_md}")


if __name__ == '__main__':
    main()
{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}