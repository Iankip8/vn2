#!/usr/bin/env python3
"""
Compute weighted/unweighted precision, recall, and F1 for stockout/overstock classification.

Data inputs:
  - Diagnostics CSV (e.g., reports/forecast_diagnostics_metrics.csv) with predicted probs.
  - Harness run CSV (e.g., run_metrics.csv from backtest harness) with realized costs.

Outputs:
  - Classification metrics CSV (per week + overall).
  - Markdown summary report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

CU = 1.0   # shortage cost weight
CO = 0.2   # holding cost weight


def load_diagnostics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    required = {"week", "store", "product", "prob_stockout"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Diagnostics file missing columns: {missing}")
    if "pred_stockout" not in df.columns:
        df["pred_stockout"] = df["prob_stockout"] >= 0.2
    return df


def load_harness_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    required = {"week", "store", "product", "holding_cost", "shortage_cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Harness metrics missing columns: {missing}")
    return df


def add_actual_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["actual_stockout"] = df["shortage_cost"] > 1e-9
    df["actual_overstock"] = df["holding_cost"] > 1e-9
    return df


def compute_weights(df: pd.DataFrame, weeks: int = 6) -> pd.DataFrame:
    week_weight = (weeks - df["week"] + 1) / weeks
    df["stockout_weight"] = week_weight * CU
    df["overstock_weight"] = week_weight * CO
    return df


def weighted_metrics(df: pd.DataFrame,
                     actual_col: str,
                     pred_col: str,
                     weight_col: str) -> Dict[str, float]:
    preds = df[pred_col]
    actuals = df[actual_col]
    weights = df[weight_col]
    tp = ((preds) & (actuals)) * weights
    fp = ((preds) & (~actuals)) * weights
    fn = ((~preds) & (actuals)) * weights
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()
    precision = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else np.nan
    recall = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else np.nan
    f1 = float(2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else np.nan
    return {
        "tp": float(tp_sum),
        "fp": float(fp_sum),
        "fn": float(fn_sum),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def unweighted_metrics(df: pd.DataFrame,
                       actual_col: str,
                       pred_col: str) -> Dict[str, float]:
    preds = df[pred_col]
    actuals = df[actual_col]
    tp = int(((preds) & (actuals)).sum())
    fp = int(((preds) & (~actuals)).sum())
    fn = int(((~preds) & (actuals)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) > 0 else np.nan
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    for target, pred_col, weight_col in [
            ("stockout", "pred_stockout", "stockout_weight"),
            ("overstock", "pred_overstock", "overstock_weight")]:
        actual_col = f"actual_{target}"
        weighted = weighted_metrics(df, actual_col, pred_col, weight_col)
        unweighted = unweighted_metrics(df, actual_col, pred_col)
        summaries.append({
            "target": target,
            "metric": "weighted",
            **weighted
        })
        summaries.append({
            "target": target,
            "metric": "unweighted",
            **unweighted
        })

    overall = pd.DataFrame(summaries)

    per_week_rows = []
    for week, week_df in df.groupby("week"):
        for target, pred_col, weight_col in [
                ("stockout", "pred_stockout", "stockout_weight"),
                ("overstock", "pred_overstock", "overstock_weight")]:
            actual_col = f"actual_{target}"
            row = {"week": week, "target": target}
            row.update(weighted_metrics(week_df, actual_col, pred_col, weight_col))
            per_week_rows.append(row)

    per_week = pd.DataFrame(per_week_rows)
    return overall, per_week


def write_markdown(path: Path, overall: pd.DataFrame, per_week: pd.DataFrame) -> None:
    lines = [
        "# Classification Metrics",
        "",
        "## Overall",
        "| Target | Metric | Precision | Recall | F1 |",
        "| ------ | ------ | --------: | -----: | --:|"
    ]
    for _, row in overall.iterrows():
        lines.append(f"| {row['target']} | {row['metric']} | "
                     f"{row['precision'] if not np.isnan(row['precision']) else 'NA':>8} | "
                     f"{row['recall'] if not np.isnan(row['recall']) else 'NA':>7} | "
                     f"{row['f1'] if not np.isnan(row['f1']) else 'NA':>6} |")
    lines.append("")
    lines.append("## By Week (weighted)")
    lines.append("| Week | Target | Precision | Recall | F1 |")
    lines.append("| ----:| ------ | --------: | -----: | --:|")
    for _, row in per_week.iterrows():
        lines.append(f"| {int(row['week'])} | {row['target']} | "
                     f"{row['precision'] if not np.isnan(row['precision']) else 'NA':>8} | "
                     f"{row['recall'] if not np.isnan(row['recall']) else 'NA':>7} | "
                     f"{row['f1'] if not np.isnan(row['f1']) else 'NA':>6} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compute classification metrics for stockout/overstock")
    parser.add_argument("--diagnostics", type=Path, required=True,
                        help="Path to forecast diagnostics metrics CSV")
    parser.add_argument("--harness-metrics", type=Path, required=True,
                        help="Path to harness run metrics CSV (run_metrics.csv)")
    parser.add_argument("--prob-overstock-threshold", type=float, default=0.1,
                        help="Probability threshold to tag overstock (default: 0.1)")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/classification_metrics.csv"))
    parser.add_argument("--output-weekly-csv", type=Path, default=Path("reports/classification_metrics_by_week.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/classification_metrics_summary.md"))
    parser.add_argument("--weeks", type=int, default=6,
                        help="Number of weeks in horizon (default: 6)")
    args = parser.parse_args()

    diag = load_diagnostics(args.diagnostics)
    harness = load_harness_metrics(args.harness_metrics)
    merged = pd.merge(diag, harness,
                      on=["week", "store", "product"],
                      suffixes=("_diag", "_run"))
    merged = add_actual_flags(merged)
    merged["pred_overstock"] = merged["prob_stockout"] <= args.prob_overstock_threshold
    merged = compute_weights(merged, weeks=args.weeks)

    overall, per_week = summarize(merged)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output_csv, index=False)
    per_week.to_csv(args.output_weekly_csv, index=False)
    write_markdown(args.output_md, overall, per_week)

    print(f"✓ Overall metrics -> {args.output_csv}")
    print(f"✓ Weekly metrics -> {args.output_weekly_csv}")
    print(f"✓ Markdown summary -> {args.output_md}")


if __name__ == "__main__":
    main()


