#!/usr/bin/env python3
"""
Reusable backtesting harness for sequential inventory strategies (lead time L=3).

Features
--------
* YAML-based run specification (configurable weeks, strategy controls, data paths)
* Sequential simulation that respects temporal constraints (orders at week t use fold_{t-1})
* L=3 lead time: orders placed end of week t arrive start of week t+3
* Guardrail + selector overrides aligned with diagnostics pipeline
* Detailed logging + audit trail under reports/backtests/<run_id>
* Outputs per-SKU metrics, weekly summaries, and markdown run report
* Hypothesis test metrics (Jensen delta, stockout classification, etc.)

Example
-------
python scripts/run_backtest_harness.py \\
    --run-spec configs/backtests/example_value_weighted.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from vn2.analyze.sequential_planner import Costs, choose_order_L3
from vn2.analyze.sequential_backtest import quantiles_to_pmf

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])


def gaussian_weights(levels: np.ndarray, center: float, sigma: float,
                     low: float, high: float) -> np.ndarray:
    """Gaussian weights clipped to [low, high]."""
    weights = np.exp(-0.5 * ((levels - center) / sigma) ** 2)
    mask = (levels >= low) & (levels <= high)
    weights *= mask
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def normalize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename columns using case-/space-insensitive matching."""
    def norm(name: str) -> str:
        return _norm(name)

    normalized_mapping = {norm(src): dest for src, dest in mapping.items()}
    rename_map = {}
    for col in df.columns:
        key = norm(col)
        if key in normalized_mapping:
            rename_map[col] = normalized_mapping[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def get_column(df: pd.DataFrame, label: str, to_numeric: bool = True) -> pd.Series:
    """Return column by label (case/space-insensitive)."""
    target = _norm(label)
    for col in df.columns:
        if _norm(col) == target:
            series = df[col]
            if to_numeric:
                return pd.to_numeric(series, errors="coerce")
            return series
    raise KeyError(f"Column '{label}' not found in DataFrame")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


@dataclass
class HarnessArgs:
    run_spec: Path
    run_id: Optional[str]
    tag: Optional[str]
    weeks: Optional[List[int]]
    dry_run: bool


class BacktestHarness:
    def __init__(self, spec: Dict, cli_args: HarnessArgs):
        self.spec = spec
        self.cli_args = cli_args
        base_run_id = cli_args.run_id or spec.get("run_id") or "backtest_run"
        tag = cli_args.tag or spec.get("logging", {}).get("tag_namespace")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = base_run_id
        if tag:
            run_id = f"{run_id}_{tag}"
        run_id = f"{run_id}_{timestamp}"
        self.run_id = run_id

        paths_cfg = spec.get("paths", {})
        output_root = Path(paths_cfg.get("output_root", "reports/backtests"))
        self.run_dir = (output_root / run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Copy spec for audit
        self.spec_path_copy = self.run_dir / "run_spec.yaml"
        with open(self.spec_path_copy, "w") as fp:
            yaml.safe_dump(spec, fp, sort_keys=False)

        # Logging
        self.logger = logging.getLogger("backtest_harness")
        self.logger.setLevel(spec.get("logging", {}).get("level", "INFO"))
        self._configure_logging()

        self.logger.info("Run ID: %s", self.run_id)
        self.logger.info("Spec copy: %s", self.spec_path_copy)

        # Prepare metadata file (populated at end)
        self.metadata_path = self.run_dir / "run_metadata.json"

        # Load strategy controls
        self.costs = Costs(
            holding=float(spec.get("strategy", {}).get("costs", {}).get("holding", 0.2)),
            shortage=float(spec.get("strategy", {}).get("costs", {}).get("shortage", 1.0)),
        )
        self.strategy_cfg = spec.get("strategy", {})
        forecast_cfg = self.strategy_cfg.get("forecast", {})
        band = forecast_cfg.get("gaussian_weight_band", [0.73, 0.93])
        sigma = forecast_cfg.get("gaussian_weight_sigma", 0.05)
        self.weight_center = forecast_cfg.get("gaussian_weight_center", 0.8333)
        self.gaussian_weights = gaussian_weights(
            QUANTILE_LEVELS, self.weight_center, sigma, band[0], band[1]
        )
        self.sip_grain = int(forecast_cfg.get("sip_grain", 500))
        self.fold_offset = int(forecast_cfg.get("fold_offset", 0))
        self.missing_policy = self.strategy_cfg.get("allocation", {}).get(
            "missing_forecast_policy", "zero"
        )

        # Weeks to run
        spec_weeks = spec.get("weeks", [])
        if not spec_weeks:
            raise ValueError("spec must include a non-empty 'weeks' list")
        self.weeks = sorted(set(cli_args.weeks or spec_weeks))

        # Data paths
        def _opt_path(key: str) -> Optional[Path]:
            val = paths_cfg.get(key)
            return Path(val).resolve() if val else None

        self.paths = {
            "initial_state": Path(paths_cfg.get("initial_state")).resolve(),
            "state_dir": Path(paths_cfg.get("state_dir", "data/states")).resolve(),
            "checkpoints_dir": Path(paths_cfg.get("checkpoints_dir", "models/checkpoints")).resolve(),
            "selector_map": Path(paths_cfg.get("selector_map")).resolve(),
            "selector_overrides": _opt_path("selector_overrides"),
            "guardrail_overrides": _opt_path("guardrail_overrides"),
        }

        self.week_state_map = self._build_week_state_map(spec.get("week_data", []))

        # Lazy-loaded data
        self.selector_lookup = None
        self.guardrail_lookup = None
        self.quantile_cache: Dict[Tuple[str, int, int, int], Optional[pd.Series]] = {}

        # Runtime storage
        self.metrics_rows: List[Dict] = []
        self.weekly_rows: List[Dict] = []
        self.baseline_totals: List[Dict] = []

    def _configure_logging(self) -> None:
        """Configure logging handlers (console + file)."""
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        self.logger.addHandler(console)

        file_handler = logging.FileHandler(self.run_dir / "run.log")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)

    def _build_week_state_map(self, week_data: List[Dict]) -> Dict[int, Path]:
        """Resolve explicit per-week state file overrides."""
        mapping = {}
        for entry in week_data:
            wk = int(entry["week"])
            mapping[wk] = Path(entry["state"]).resolve()
        return mapping

    # ------------------------------------------------------------------ #
    # Data loaders
    # ------------------------------------------------------------------ #
    def _load_selector_lookup(self) -> Dict[Tuple[int, int], str]:
        df = pd.read_parquet(self.paths["selector_map"])
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        lookup = {(int(r.store), int(r.product)): r.model_name for r in df.itertuples(index=False)}
        # Apply overrides if requested
        override_path = self.paths.get("selector_overrides")
        if override_path and override_path.exists():
            ov = pd.read_csv(override_path)
            ov = normalize_columns(ov, {
                "Store": "store",
                "Product": "product",
                "best_challenger": "best_challenger",
                "recommend_switch": "recommend_switch"
            })
            mask = ov.get("recommend_switch")
            if mask is not None:
                ov = ov[mask == True]
            for _, row in ov.iterrows():
                key = (int(row["store"]), int(row["product"]))
                lookup[key] = row.get("best_challenger", lookup.get(key))
        return lookup

    def _load_guardrail_lookup(self) -> Dict[Tuple[int, int], float]:
        path = self.paths.get("guardrail_overrides")
        if not path or not path.exists():
            return {}
        df = pd.read_csv(path)
        df = normalize_columns(df, {
            "Store": "store",
            "Product": "product",
            "service_level_override": "service_level_override"
        })
        lookup = {}
        for _, row in df.iterrows():
            val = row.get("service_level_override")
            if pd.notna(val):
                lookup[(int(row["store"]), int(row["product"]))] = float(val)
        return lookup

    def _load_initial_state(self) -> pd.DataFrame:
        """Load initial inventory (end of week 0).
        
        For L=3 lead time, we track 3 in-transit columns:
        - intransit_1: arrives start of week 1 (placed end of week -2)
        - intransit_2: arrives start of week 2 (placed end of week -1)
        - intransit_3: arrives start of week 3 (placed end of week 0 - our first order)
        """
        df = load_csv(self.paths["initial_state"])
        store_series = get_column(df, "Store").astype(int)
        product_series = get_column(df, "Product").astype(int)
        end_inventory = get_column(df, "End Inventory").fillna(0).astype(float)
        intransit1 = get_column(df, "In Transit W+1").fillna(0).astype(float)
        intransit2 = get_column(df, "In Transit W+2").fillna(0).astype(float)
        # Initial state doesn't have intransit_3 - that's our first order
        intransit3 = pd.Series(0.0, index=range(len(store_series)))

        df_sorted = pd.DataFrame({
            "store": store_series,
            "product": product_series,
            "end_inventory": end_inventory,
            "intransit_1": intransit1,
            "intransit_2": intransit2,
            "intransit_3": intransit3
        }).sort_values(["store", "product"])

        idx = pd.MultiIndex.from_arrays(
            [df_sorted["store"].astype(int), df_sorted["product"].astype(int)],
            names=["store", "product"]
        )
        state = pd.DataFrame({
            "on_hand": df_sorted["end_inventory"].values,
            "intransit_1": df_sorted["intransit_1"].values,
            "intransit_2": df_sorted["intransit_2"].values,
            "intransit_3": df_sorted["intransit_3"].values,
        }, index=idx)
        return state

    def _resolve_state_path(self, week: int) -> Path:
        if week in self.week_state_map:
            return self.week_state_map[week]
        return (self.paths["state_dir"] / f"state{week}.csv").resolve()

    def _load_week_actuals(self, week: int) -> Tuple[pd.DataFrame, float]:
        """Load actual demand + baseline costs from state{week}.csv."""
        state_path = self._resolve_state_path(week)
        df = load_csv(state_path)

        store_series = get_column(df, "Store").astype(int)
        product_series = get_column(df, "Product").astype(int)
        sales = get_column(df, "Sales").fillna(0)
        missed = get_column(df, "Missed Sales").fillna(0)
        holding = get_column(df, "Holding Cost").fillna(0)
        shortage = get_column(df, "Shortage Cost").fillna(0)
        self.logger.debug("Week %d sample sales: %s", week, sales.head().tolist())
        self.logger.debug("Week %d sample missed: %s", week, missed.head().tolist())

        demand = (sales + missed).astype(float)
        baseline_cost = (holding + shortage).astype(float)

        df_sorted = pd.DataFrame({
            "store": store_series,
            "product": product_series,
            "demand": demand,
            "baseline_cost": baseline_cost
        }).sort_values(["store", "product"])

        idx = pd.MultiIndex.from_arrays([df_sorted["store"].astype(int), df_sorted["product"].astype(int)],
                                        names=["store", "product"])
        demand_df = pd.DataFrame({
            "demand": df_sorted["demand"].values,
            "baseline_cost": df_sorted["baseline_cost"].values
        }, index=idx)
        baseline_total = float(df_sorted["baseline_cost"].sum())
        return demand_df, baseline_total

    def _ensure_state_index(self, state: pd.DataFrame, required_idx: pd.Index) -> pd.DataFrame:
        """Ensure state has rows for every SKU in required_idx."""
        missing = required_idx.difference(state.index)
        if missing.empty:
            return state
        add = pd.DataFrame(0.0, index=missing, columns=state.columns)
        combined = pd.concat([state, add])
        return combined.sort_index()

    # ------------------------------------------------------------------ #
    # Forecast helpers
    # ------------------------------------------------------------------ #
    def _get_selector_lookup(self) -> Dict[Tuple[int, int], str]:
        if self.selector_lookup is None:
            self.selector_lookup = self._load_selector_lookup()
        return self.selector_lookup

    def _get_guardrail_lookup(self) -> Dict[Tuple[int, int], float]:
        if self.guardrail_lookup is None:
            self.guardrail_lookup = self._load_guardrail_lookup()
        return self.guardrail_lookup

    def _load_quantiles(self, model: str, store: int, product: int, fold_idx: int
                        ) -> Optional[pd.Series]:
        key = (model, store, product, fold_idx)
        if key in self.quantile_cache:
            return self.quantile_cache[key]
        ckpt = self.paths["checkpoints_dir"] / model / f"{store}_{product}" / f"fold_{fold_idx}.pkl"
        if not ckpt.exists():
            self.quantile_cache[key] = None
            return None
        try:
            data = pd.read_pickle(ckpt)
            qdf = data.get("quantiles")
            if qdf is None or qdf.empty:
                self.quantile_cache[key] = None
            else:
                self.quantile_cache[key] = qdf
        except Exception as exc:
            self.logger.warning("Failed to load %s: %s", ckpt, exc)
            self.quantile_cache[key] = None
        return self.quantile_cache[key]

    def _service_level_for_sku(self, store: int, product: int) -> float:
        guardrails = self._get_guardrail_lookup()
        base = self.strategy_cfg.get("allocation", {}).get("default_service_level", 0.8333)
        bias = self.strategy_cfg.get("allocation", {}).get("service_bias", 0.0)
        override = guardrails.get((store, product))
        if override is not None:
            return max(0.5, min(0.99, float(override) + bias))
        return max(0.5, min(0.99, base + bias))

    # ------------------------------------------------------------------ #
    # Order generation + simulation (L=3 lead time)
    # ------------------------------------------------------------------ #
    def _generate_orders_for_week(
        self,
        week: int,
        state: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate orders + expected costs for each SKU using L=3 optimization.
        
        L=3 LEAD TIME SEMANTICS:
        - Order placed at END of week t arrives at START of week t+3
        - We need h=1, h=2, h=3 forecasts to optimize
        - Q1 = intransit_1 (arrives week t+1, placed end of t-2)
        - Q2 = intransit_2 (arrives week t+2, placed end of t-1)
        - Q3 = intransit_3 (arrives week t+3, placed end of t - previous week's order)
        """
        selector = self._get_selector_lookup()
        records = []
        fold_idx = max(0, week - 1 + self.fold_offset)
        missing_forecasts = 0

        for (store, product), inv in state.sort_index().iterrows():
            model = selector.get((store, product))
            if not model:
                records.append({
                    "store": store,
                    "product": product,
                    "order_qty": 0,
                    "model": None,
                    "expected_cost": 0.0,
                    "service_level": self._service_level_for_sku(store, product),
                    "forecast_available": False,
                    "h1_quantiles": None,
                })
                missing_forecasts += 1
                continue

            qdf = self._load_quantiles(model, store, product, fold_idx)
            # For L=3, we need h=1, h=2, and h=3 forecasts
            if qdf is None or 1 not in qdf.index or 2 not in qdf.index:
                records.append({
                    "store": store,
                    "product": product,
                    "order_qty": 0,
                    "model": model,
                    "expected_cost": 0.0,
                    "service_level": self._service_level_for_sku(store, product),
                    "forecast_available": False,
                    "h1_quantiles": None,
                })
                missing_forecasts += 1
                continue

            h1_quantiles = qdf.loc[1].values
            h2_quantiles = qdf.loc[2].values
            # If h=3 not available, use h=2 as approximation
            h3_quantiles = qdf.loc[3].values if 3 in qdf.index else h2_quantiles
            
            h1_pmf = quantiles_to_pmf(h1_quantiles, QUANTILE_LEVELS, self.sip_grain)
            h2_pmf = quantiles_to_pmf(h2_quantiles, QUANTILE_LEVELS, self.sip_grain)
            h3_pmf = quantiles_to_pmf(h3_quantiles, QUANTILE_LEVELS, self.sip_grain)

            service_level = self._service_level_for_sku(store, product)

            if service_level != self.strategy_cfg.get("allocation", {}).get("default_service_level", 0.8333):
                co = self.costs.holding
                cu = co * service_level / (1.0 - service_level)
                costs = Costs(holding=co, shortage=cu)
            else:
                costs = self.costs

            # L=3 optimization with 3-week pipeline
            order_qty, expected_cost = choose_order_L3(
                h1_pmf, h2_pmf, h3_pmf,
                int(inv["on_hand"]),
                int(inv["intransit_1"]),  # Q1: arrives week t+1
                int(inv["intransit_2"]),  # Q2: arrives week t+2
                int(inv["intransit_3"]),  # Q3: arrives week t+3 (previous order)
                costs,
                micro_refine=True
            )

            records.append({
                "store": store,
                "product": product,
                "order_qty": int(order_qty),
                "model": model,
                "expected_cost": float(expected_cost),
                "service_level": service_level,
                "forecast_available": True,
                "h1_quantiles": h1_quantiles,
                "h2_quantiles": h2_quantiles,
                "h3_quantiles": h3_quantiles,
                "cost_holding": costs.holding,
                "cost_shortage": costs.shortage,
            })

        orders_df = pd.DataFrame(records).set_index(["store", "product"])
        summary = {
            "week": week,
            "skus": len(records),
            "orders_positive": int((orders_df["order_qty"] > 0).sum()),
            "units_ordered": float(orders_df["order_qty"].sum()),
            "missing_forecasts": missing_forecasts,
        }
        return orders_df, summary

    def _simulate_week(
        self,
        state: pd.DataFrame,
        demand: pd.Series,
        orders: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate one week of demand + new orders with L=3 pipeline.
        
        L=3 LEAD TIME:
        - intransit_1: arrives this week (start of week t)
        - intransit_2: arrives next week (start of week t+1)  
        - intransit_3: arrives week after (start of week t+2)
        - new orders placed now arrive at start of week t+3
        """
        # Align indices
        idx = state.index.union(demand.index).union(orders.index)
        state = state.reindex(idx, fill_value=0)
        demand = demand.reindex(idx, fill_value=0)
        orders = orders.reindex(idx, fill_value=0)

        # Inventory arriving this week
        received = state["intransit_1"]
        available = state["on_hand"] + received
        sold = np.minimum(available, demand)
        lost = demand - sold
        ending_on_hand = available - sold

        # Shift pipeline forward, add new orders at end
        next_state = pd.DataFrame({
            "on_hand": ending_on_hand,
            "intransit_1": state["intransit_2"],  # was arriving week t+1, now arrives week t
            "intransit_2": state["intransit_3"],  # was arriving week t+2, now arrives week t+1
            "intransit_3": orders                 # new orders arrive week t+3
        }, index=idx)

        holding_cost = ending_on_hand * self.costs.holding
        shortage_cost = lost * self.costs.shortage

        detail = pd.DataFrame({
            "available_start": state["on_hand"],
            "received": received,
            "demand": demand,
            "sold": sold,
            "lost": lost,
            "ending_on_hand": ending_on_hand,
            "order_qty": orders,
            "holding_cost": holding_cost,
            "shortage_cost": shortage_cost,
            "total_cost": holding_cost + shortage_cost
        }, index=idx)

        return next_state, detail

    # ------------------------------------------------------------------ #
    # Run pipeline
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if self.cli_args.dry_run:
            self.logger.info("Dry run requested; configuration validated only.")
            return

        # Preload helpers
        _ = self._get_selector_lookup()
        _ = self._get_guardrail_lookup()

        state = self._load_initial_state().fillna(0)
        self.logger.info("Loaded initial state with %d SKUs", len(state))

        for week in self.weeks:
            self.logger.info("=== Week %d ===", week)

            demand_df, baseline_total = self._load_week_actuals(week)
            state = self._ensure_state_index(state, demand_df.index).fillna(0)
            orders_df, order_summary = self._generate_orders_for_week(week, state)
            next_state, cost_detail = self._simulate_week(
                state, demand_df["demand"], orders_df["order_qty"]
            )

            self._record_week(
                week=week,
                orders=orders_df,
                demand_df=demand_df,
                cost_detail=cost_detail,
                baseline_total=baseline_total
            )

            self._write_week_orders_csv(week, orders_df)

            state = next_state
            self.logger.info(
                "Week %d summary: orders=%d units, cost=%.2f, baseline=%.2f",
                week, int(order_summary["units_ordered"]),
                self.weekly_rows[-1]["total_cost"],
                baseline_total
            )

        self._write_metrics_outputs()
        self._write_metadata()
        self.logger.info("Run completed: %s", self.run_dir)

    def _record_week(
        self,
        week: int,
        orders: pd.DataFrame,
        demand_df: pd.DataFrame,
        cost_detail: pd.DataFrame,
        baseline_total: float
    ) -> None:
        """Merge decision + outcome metrics."""
        records = []
        total_cost = float(cost_detail["total_cost"].sum())

        for (store, product), order_row in orders.iterrows():
            try:
                demand = float(demand_df.at[(store, product), "demand"])
            except KeyError:
                demand = 0.0
            order_qty = float(order_row["order_qty"])
            expected_cost = float(order_row.get("expected_cost", 0.0))
            holding_cost = float(cost_detail.at[(store, product), "holding_cost"])
            shortage_cost = float(cost_detail.at[(store, product), "shortage_cost"])
            weighted_pinball = None

            h1_quantiles = order_row.get("h1_quantiles")
            if h1_quantiles is not None and np.size(h1_quantiles) == len(QUANTILE_LEVELS):
                diffs = demand - h1_quantiles
                pinball = np.maximum(QUANTILE_LEVELS * diffs,
                                     (QUANTILE_LEVELS - 1.0) * diffs)
                weighted_pinball = float(
                    np.sum(self.gaussian_weights * pinball)
                ) if self.gaussian_weights.sum() > 0 else float(np.mean(pinball))

            records.append({
                "week": week,
                "store": store,
                "product": product,
                "model": order_row.get("model"),
                "order_qty": order_qty,
                "actual_demand": demand,
                "expected_cost": expected_cost,
                "holding_cost": holding_cost,
                "shortage_cost": shortage_cost,
                "total_cost": holding_cost + shortage_cost,
                "service_level": order_row.get("service_level"),
                "forecast_available": bool(order_row.get("forecast_available")),
                "weighted_pinball": weighted_pinball,
            })

        self.metrics_rows.extend(records)
        self.weekly_rows.append({
            "week": week,
            "skus": len(records),
            "total_cost": total_cost,
            "baseline_cost": baseline_total,
            "cost_delta": total_cost - baseline_total,
            "orders_positive": int(sum(r["order_qty"] > 0 for r in records)),
            "units_ordered": float(sum(r["order_qty"] for r in records)),
        })

    def _write_week_orders_csv(self, week: int, orders: pd.DataFrame) -> None:
        path = self.run_dir / f"orders_week{week}.csv"
        out = orders.drop(columns=["h1_quantiles", "h2_quantiles"], errors="ignore").reset_index()
        out.to_csv(path, index=False)

    def _write_metrics_outputs(self) -> None:
        metrics_df = pd.DataFrame(self.metrics_rows)
        metrics_path = self.run_dir / "run_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        weekly_df = pd.DataFrame(self.weekly_rows)
        weekly_path = self.run_dir / "weekly_summary.csv"
        weekly_df.to_csv(weekly_path, index=False)

        summary_md = self.run_dir / "run_summary.md"
        lines = [
            f"# Backtest Run: {self.run_id}",
            "",
            f"- Spec: `{self.spec_path_copy}`",
            f"- Metrics CSV: `{metrics_path}`",
            f"- Weekly Summary: `{weekly_path}`",
            "",
            "## Weekly Totals",
            "Week | Order Units | SKUs | Total Cost | Baseline | Delta",
            "---- | -----------:| ----:| ----------:| --------:| -----:",
        ]
        for row in self.weekly_rows:
            lines.append(
                f"{row['week']} | {row['units_ordered']:.0f} | {row['skus']} | "
                f"{row['total_cost']:.2f} | {row['baseline_cost']:.2f} | "
                f"{row['cost_delta']:+.2f}"
            )
        lines.append("")
        lines.append("## Cumulative")
        total_cost = sum(r["total_cost"] for r in self.weekly_rows)
        total_baseline = sum(r["baseline_cost"] for r in self.weekly_rows)
        delta = total_cost - total_baseline
        lines.append(f"- Total cost: €{total_cost:,.2f}")
        lines.append(f"- Baseline cost: €{total_baseline:,.2f}")
        lines.append(f"- Improvement vs baseline: €{delta:+,.2f}")
        summary_md.write_text("\n".join(lines))

    def _write_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "spec_path": str(self.spec_path_copy),
            "weeks": self.weeks,
            "output_dir": str(self.run_dir),
            "paths": {k: str(v) for k, v in self.paths.items()},
            "cli_args": {
                "run_id": self.cli_args.run_id,
                "tag": self.cli_args.tag,
                "dry_run": self.cli_args.dry_run,
                "weeks_override": self.cli_args.weeks,
            },
            "strategy": self.strategy_cfg,
            "costs": {"holding": self.costs.holding, "shortage": self.costs.shortage},
            "metrics_csv": str(self.run_dir / "run_metrics.csv"),
            "weekly_summary_csv": str(self.run_dir / "weekly_summary.csv"),
            "summary_md": str(self.run_dir / "run_summary.md"),
        }
        with open(self.metadata_path, "w") as fp:
            json.dump(metadata, fp, indent=2)


def parse_args() -> HarnessArgs:
    parser = argparse.ArgumentParser(
        description="Sequential backtest harness (L=2 lead time)"
    )
    parser.add_argument("--run-spec", type=Path, required=True,
                        help="Path to YAML run specification")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override run_id from spec")
    parser.add_argument("--tag", type=str, default=None,
                        help="Additional tag appended to run_id")
    parser.add_argument("--weeks", nargs="+", type=int, default=None,
                        help="Override weeks list (space separated)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load configuration only (no simulation)")
    args = parser.parse_args()
    return HarnessArgs(
        run_spec=args.run_spec,
        run_id=args.run_id,
        tag=args.tag,
        weeks=args.weeks,
        dry_run=args.dry_run
    )


def main():
    cli_args = parse_args()
    spec_path = cli_args.run_spec.resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec not found: {spec_path}")
    with open(spec_path, "r") as fp:
        spec = yaml.safe_load(fp)
    harness = BacktestHarness(spec, cli_args)
    harness.run()


if __name__ == "__main__":
    main()

