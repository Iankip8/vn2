#!/usr/bin/env python3
"""
Analyze forecast diagnostics metrics and produce cohort summaries + markdown report.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_cohort_map(demand_path: Path) -> pd.DataFrame:
    if not demand_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(demand_path)
    grp = df.groupby(['Store', 'Product'])['sales']
    stats = grp.agg(['mean', 'std', 'count'])
    stats = stats.rename(columns={'mean': 'rate', 'std': 'std', 'count': 'n'})
    stats['cv'] = stats['std'] / stats['rate'].replace(0, np.nan)
    stats = stats.reset_index().rename(columns={'Store': 'store', 'Product': 'product'})
    stats['rate_bin'] = pd.qcut(stats['rate'].fillna(0), q=4, duplicates='drop', labels=['R1', 'R2', 'R3', 'R4'])
    stats['cv_bin'] = pd.qcut(stats['cv'].fillna(0), q=4, duplicates='drop', labels=['CV1', 'CV2', 'CV3', 'CV4'])
    return stats[['store', 'product', 'rate_bin', 'cv_bin']]


def compute_week_stats(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('week').agg(
        records=('weighted_pinball', 'count'),
        mean_pinball=('weighted_pinball', 'mean'),
        median_pinball=('weighted_pinball', 'median'),
        mean_wass=('wasserstein', 'mean'),
        median_wass=('wasserstein', 'median'),
        stockout_rate=('actual_stockout', 'mean'),
        predicted_stockout_rate=('pred_stockout', 'mean'),
        value_precision=('tp_weight', lambda x: x.sum()),
        pred_weight=('pred_weight', 'sum'),
        actual_weight=('actual_weight', 'sum'),
    ).reset_index()
    agg['value_precision'] = agg.apply(lambda r: r['value_precision'] / r['pred_weight'] if r['pred_weight'] > 0 else np.nan, axis=1)
    agg['value_recall'] = agg.apply(lambda r: r['value_precision'] * r['pred_weight'] / r['actual_weight'] if r['actual_weight'] > 0 else np.nan, axis=1)
    return agg


def compute_cohort_stats(metrics: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    if cohorts.empty:
        return pd.DataFrame()
    merged = metrics.merge(cohorts, on=['store', 'product'], how='left')
    grp = merged.groupby(['week', 'cv_bin']).agg(
        mean_pinball=('weighted_pinball', 'mean'),
        mean_wass=('wasserstein', 'mean'),
        stockout_rate=('actual_stockout', 'mean'),
        records=('weighted_pinball', 'count')
    ).reset_index()
    return grp


def top_miscalibrated(metrics: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics['score'] = metrics['weighted_pinball'] * metrics['wasserstein']
    return metrics.sort_values('score', ascending=False).head(n)[
        ['week', 'store', 'product', 'model', 'weighted_pinball', 'wasserstein', 'actual_stockout', 'prob_stockout']
    ]


def write_report(path: Path,
                 week_stats: pd.DataFrame,
                 cohort_stats: pd.DataFrame,
                 top_skus: pd.DataFrame,
                 overall: dict):
    lines = ['# Forecast Diagnostics Deep-Dive', '']
    lines.append('## Week-level Findings')
    lines.append('Week | Records | Mean Pinball | Median Pinball | Mean Wasserstein | Median Wasserstein | Value Precision | Value Recall | Actual Stockout % | Predicted Stockout %')
    lines.append('---- | -------:| ------------:| --------------:| ----------------:| -----------------:| ---------------:| -------------:| -----------------:| -------------------:')
    for _, row in week_stats.iterrows():
        lines.append(
            f"{int(row['week'])} | {int(row['records'])} | {row['mean_pinball']:.1f} | {row['median_pinball']:.2f} | "
            f"{row['mean_wass']:.2f} | {row['median_wass']:.2f} | "
            f"{(row['value_precision'] if pd.notna(row['value_precision']) else float('nan')):.2f} | "
            f"{(row['value_recall'] if pd.notna(row['value_recall']) else float('nan')):.2f} | "
            f"{row['stockout_rate']*100:.1f}% | {row['predicted_stockout_rate']*100:.1f}%"
        )
    lines.append('')

    lines.append('## Cohort Diagnostics (CV bins)')
    if cohort_stats.empty:
        lines.append('_Cohort metadata unavailable_')
    else:
        lines.append('Week | CV Bin | Records | Mean Pinball | Mean Wasserstein | Stockout Rate')
        lines.append('---- | ------:| -------:| ------------:| ----------------:| -------------:')
        for _, row in cohort_stats.iterrows():
            lines.append(
                f"{int(row['week'])} | {row['cv_bin']} | {int(row['records'])} | "
                f"{row['mean_pinball']:.1f} | {row['mean_wass']:.2f} | {row['stockout_rate']*100:.1f}%"
            )
    lines.append('')

    lines.append('## Most Miscalibrated SKUs (top 20)')
    lines.append('Week | Store | Product | Model | Pinball | Wasserstein | Actual Stockout | Forecasted Prob Stockout')
    lines.append('---- | -----:| -------:|:------ | -------:| -----------:|:----------------:| -------------------------:')
    for _, row in top_skus.iterrows():
        lines.append(
            f"{int(row['week'])} | {int(row['store'])} | {int(row['product'])} | {row['model']} | "
            f"{row['weighted_pinball']:.1f} | {row['wasserstein']:.2f} | {row['actual_stockout']} | {row['prob_stockout']:.2f}"
        )
    lines.append('')

    lines.append('## Strategy Considerations')
    lines.append('- **Improve density fidelity**: Use Wasserstein-aligned models or decision-focused ensembles for cohorts with high CV and high pinball/wasserstein.')
    lines.append('- **Stockout-aware training**: Emphasize value-weighted pinball near τ=0.83 and use precision/recall feedback to adjust guardrails dynamically.')
    lines.append('- **Allocation adjustments**: For SKUs with repeated high stockout rates and high Wasserstein, bias service levels upward and monitor calibration weekly.')
    lines.append('')

    lines.append('## Overall Stats')
    for key, value in overall.items():
        lines.append(f"- {key}: {value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze forecast diagnostics metrics')
    parser.add_argument('--metrics-csv', type=Path, default=Path('reports/forecast_diagnostics_metrics.csv'))
    parser.add_argument('--demand-parquet', type=Path, default=Path('data/processed/demand_long.parquet'))
    parser.add_argument('--out-cohort-csv', type=Path, default=Path('reports/forecast_diagnostics_cohort_stats.csv'))
    parser.add_argument('--out-top-csv', type=Path, default=Path('reports/forecast_diagnostics_top_skus.csv'))
    parser.add_argument('--out-md', type=Path, default=Path('reports/forecast_diagnostics_analysis.md'))
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    cohorts = load_cohort_map(args.demand_parquet)
    week_stats = compute_week_stats(metrics)
    cohort_stats = compute_cohort_stats(metrics, cohorts)
    top_skus = top_miscalibrated(metrics)

    args.out_cohort_csv.parent.mkdir(parents=True, exist_ok=True)
    if not cohort_stats.empty:
        cohort_stats.to_csv(args.out_cohort_csv, index=False)
    top_skus.to_csv(args.out_top_csv, index=False)

    overall = {
        'records_total': len(metrics),
        'avg_weighted_pinball': float(metrics['weighted_pinball'].mean()),
        'avg_wasserstein': float(metrics['wasserstein'].mean()),
    }
    write_report(args.out_md, week_stats, cohort_stats, top_skus, overall)
    print(f"✓ Analysis report written to {args.out_md}")


if __name__ == '__main__':
    main()


