#!/usr/bin/env python3
"""
Train TSB and iETS-style intermittent demand models and export quantile checkpoints.

The script reads `data/processed/demand_long.parquet` (Store, Product, week, demand),
trains the requested model for each SKU, and writes quantile forecasts for horizons
h+1 and h+2 in the standard checkpoint format consumed by the harness.

Usage:
    python scripts/train_intermit_models.py \
        --model tsb \
        --demand-path data/processed/demand_long.parquet \
        --output-root models/checkpoints \
        --max-folds 6 \
        --max-skus 100
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from statsforecast import StatsForecast
from statsforecast.models import TSB

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])


def load_demand(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns=lambda c: c.strip().lower())
    if "demand" not in df.columns and "sales" in df.columns:
        df["demand"] = df["sales"]
    required = {"store", "product", "week", "demand"}
    if not required.issubset(df.columns):
        raise ValueError(f"Demand file missing columns: {required - set(df.columns)}")
    df["week"] = pd.to_datetime(df["week"])
    df["unique_id"] = df["store"].astype(str) + "_" + df["product"].astype(str)
    df = df.sort_values(["unique_id", "week"])
    return df


def poisson_quantiles(mean: float) -> np.ndarray:
    lam = max(mean, 1e-6)
    return poisson.ppf(QUANTILE_LEVELS, lam).astype(float)


class SimpleIETS:
    """Lightweight iETS-like forecaster (demand size + interval smoothing)."""

    def __init__(self, alpha_size: float = 0.1, alpha_interval: float = 0.1):
        self.alpha_size = alpha_size
        self.alpha_interval = alpha_interval

    def forecast(self, series: np.ndarray, horizon: int = 2) -> np.ndarray:
        size_hat = np.mean(series[series > 0]) if np.any(series > 0) else 0.1
        interval_hat = max(1.0, len(series) / max(1, (series > 0).sum()))
        zeros_since_last = 0.0

        for value in series:
            if value > 0:
                size_hat = self.alpha_size * value + (1 - self.alpha_size) * size_hat
                interval_hat = self.alpha_interval * (zeros_since_last + 1) + (1 - self.alpha_interval) * interval_hat
                zeros_since_last = 0
            else:
                zeros_since_last += 1

        avg_interval = max(interval_hat, 1e-2)
        mean_demand = size_hat / avg_interval
        return np.full(horizon, mean_demand, dtype=float)


def tsb_forecast(series: pd.Series, horizon: int = 2) -> np.ndarray:
    sf_df = pd.DataFrame({
        "unique_id": "sku",
        "ds": series.index,
        "y": series.values
    })
    sf = StatsForecast(models=[TSB(alpha_d=0.2, alpha_p=0.2)], freq="W", n_jobs=1)
    fcst = sf.forecast(df=sf_df, h=horizon)
    values = fcst.filter(like="TSB").T.values.flatten()
    return np.maximum(values, 0.0)


def generate_quantiles(forecasts: np.ndarray) -> pd.DataFrame:
    rows = {}
    for horizon_idx, mean in enumerate(forecasts, start=1):
        rows[horizon_idx] = poisson_quantiles(mean)
    quantiles = pd.DataFrame(rows).T
    quantiles.columns = QUANTILE_LEVELS
    return quantiles


def save_checkpoint(output_dir: Path, store: int, product: int, fold_idx: int, quantiles: pd.DataFrame):
    ckpt_dir = output_dir / f"{store}_{product}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"fold_{fold_idx}.pkl"
    with open(path, "wb") as fp:
        pickle.dump({"quantiles": quantiles}, fp)


def main():
    parser = argparse.ArgumentParser(description="Train intermittent demand models (TSB/iETS)")
    parser.add_argument("--model", choices=["tsb", "iets"], required=True)
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--max-skus", type=int, default=None)
    args = parser.parse_args()

    demand = load_demand(args.demand_path)
    unique_ids = demand["unique_id"].unique()
    if args.max_skus:
        unique_ids = unique_ids[:args.max_skus]

    weeks = sorted(demand["week"].unique())
    model_dir = args.output_root / args.model

    for uid in unique_ids:
        store, product = map(int, uid.split("_"))
        series = demand[demand["unique_id"] == uid]
        y = series["demand"].values.astype(float)
        for fold_idx in range(min(args.max_folds, len(weeks))):
            cutoff = weeks[fold_idx]
            train_mask = series["week"] <= cutoff
            train_series = series.loc[train_mask, ["week", "demand"]].set_index("week")["demand"]
            if len(train_series) < 2:
                continue

            if args.model == "tsb":
                forecasts = tsb_forecast(train_series, horizon=2)
            else:
                forecasts = SimpleIETS().forecast(train_series.values.astype(float), horizon=2)

            quantiles = generate_quantiles(forecasts)
            save_checkpoint(model_dir, store, product, fold_idx, quantiles)


if __name__ == "__main__":
    main()

