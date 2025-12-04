#!/usr/bin/env python3
"""
Powell-style lookahead policy vs. local harness decisions (Weeks 1–6).

For each SKU-week we evaluate candidate order adjustments via Monte Carlo rollouts
and keep the action with the best expected total cost. Future weeks revert to the baseline
policy, approximating the stochastic lookahead ideas from Powell.
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from run_backtest_harness import BacktestHarness, HarnessArgs, QUANTILE_LEVELS
from vn2.analyze.sequential_planner import Costs, choose_order_L2
from vn2.analyze.sequential_backtest import quantiles_to_pmf

log = logging.getLogger("powell_policy")


@dataclass
class SKUState:
    on_hand: int
    intransit_1: int
    intransit_2: int

    def copy(self) -> "SKUState":
        return SKUState(self.on_hand, self.intransit_1, self.intransit_2)


class ForecastProvider:
    """Serve PMFs for (store, product, decision week)."""

    def __init__(self, harness: BacktestHarness):
        self.harness = harness
        self.selector = harness._get_selector_lookup()
        self.cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def get(self, store: int, product: int, week: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (store, product, week)
        if key in self.cache:
            return self.cache[key]
        model = self.selector.get((store, product))
        if not model:
            raise KeyError(f"No model for SKU ({store}, {product})")
        fold_idx = max(0, week - 1 + self.harness.fold_offset)
        qdf = self.harness._load_quantiles(model, store, product, fold_idx)
        if qdf is None or 1 not in qdf.index or 2 not in qdf.index:
            raise ValueError(f"Missing quantiles for SKU ({store},{product}) week {week}")
        h1 = quantiles_to_pmf(qdf.loc[1].values, QUANTILE_LEVELS, self.harness.sip_grain)
        h2 = quantiles_to_pmf(qdf.loc[2].values, QUANTILE_LEVELS, self.harness.sip_grain)
        self.cache[key] = (h1, h2)
        return self.cache[key]


def sample_from_pmf(pmf: np.ndarray, rng: random.Random) -> int:
    support = np.arange(len(pmf))
    return int(rng.choices(population=support, weights=pmf, k=1)[0])


def step_state(state: SKUState, demand: int, order_qty: int, costs: Costs) -> Tuple[SKUState, float]:
    available = state.on_hand + state.intransit_1
    sold = min(available, demand)
    lost = max(0, demand - available)
    ending_on_hand = available - sold
    next_state = SKUState(
        on_hand=ending_on_hand,
        intransit_1=state.intransit_2,
        intransit_2=order_qty
    )
    cost = costs.holding * ending_on_hand + costs.shortage * lost
    return next_state, cost


class PowellLookaheadPolicy:
    def __init__(
        self,
        forecast_provider: ForecastProvider,
        costs: Costs,
        deltas: Optional[List[int]] = None,
        scenario_count: int = 25,
        horizon_weeks: int = 6,
        rng_seed: int = 42,
    ):
        self.forecast_provider = forecast_provider
        self.costs = costs
        self.deltas = deltas or [-30, 0, 30]
        self.scenario_count = scenario_count
        self.horizon_weeks = horizon_weeks
        self.rng = random.Random(rng_seed)

    def choose(self, store: int, product: int, week: int, base_order: int, state: SKUState) -> int:
        best = base_order
        best_cost = float("inf")
        for delta in self.deltas:
            candidate = max(0, base_order + delta)
            est = self._estimate_cost(store, product, week, state, candidate)
            if est < best_cost:
                best_cost = est
                best = candidate
        return best

    def _estimate_cost(self, store: int, product: int, week: int, state: SKUState, first_order: int) -> float:
        total = 0.0
        for _ in range(self.scenario_count):
            total += self._simulate_path(store, product, week, state, first_order)
        return total / self.scenario_count

    def _simulate_path(self, store: int, product: int, week: int, state: SKUState, first_order: int) -> float:
        state_copy = state.copy()
        total_cost = 0.0
        for w in range(week, self.horizon_weeks + 1):
            h1, h2 = self.forecast_provider.get(store, product, w)
            demand = sample_from_pmf(h1, self.rng)
            order = first_order if w == week else self._base_policy(store, product, w, state_copy, h1, h2)
            state_copy, cost = step_state(state_copy, demand, order, self.costs)
            total_cost += cost
        return total_cost

    def _base_policy(
        self,
        store: int,
        product: int,
        week: int,
        state: SKUState,
        h1_pmf: np.ndarray,
        h2_pmf: np.ndarray
    ) -> int:
        q, _ = choose_order_L2(
            h1_pmf, h2_pmf,
            state.on_hand,
            state.intransit_1,
            state.intransit_2,
            self.costs,
            micro_refine=True
        )
        return max(0, int(q))


class PowellRunner:
    def __init__(
        self,
        harness: BacktestHarness,
        baseline_metrics: Path,
        scenario_count: int,
        deltas: List[int],
        horizon: int,
        max_skus: Optional[int],
        rng_seed: int
    ):
        self.harness = harness
        self.policy = PowellLookaheadPolicy(
            ForecastProvider(harness),
            harness.costs,
            deltas=deltas,
            scenario_count=scenario_count,
            horizon_weeks=horizon,
            rng_seed=rng_seed
        )
        self.baseline_metrics_path = baseline_metrics
        self.max_skus = max_skus
        self.output_dir = harness.run_dir

    def run(self):
        state = self.harness._load_initial_state().fillna(0).astype(float)
        powell_costs = []

        for week in self.harness.weeks:
            demand_df, _ = self.harness._load_week_actuals(week)
            state = self.harness._ensure_state_index(state, demand_df.index).fillna(0)
            base_orders, _ = self.harness._generate_orders_for_week(week, state)
            powell_orders = base_orders.copy()

            targets = self._select_targets(base_orders, demand_df)
            for store, product in targets:
                inv = state.loc[(store, product)]
                sku_state = SKUState(int(inv["on_hand"]), int(inv["intransit_1"]), int(inv["intransit_2"]))
                base_order = int(base_orders.loc[(store, product), "order_qty"])
                adjusted = self.policy.choose(store, product, week, base_order, sku_state)
                powell_orders.at[(store, product), "order_qty"] = adjusted

            next_state, cost_detail = self.harness._simulate_week(
                state,
                demand_df["demand"],
                powell_orders["order_qty"]
            )
            powell_costs.append(float(cost_detail["total_cost"].sum()))
            state = next_state
            powell_orders.reset_index().to_csv(self.output_dir / f"powell_orders_week{week}.csv", index=False)

        self._write_summary(powell_costs)

    def _select_targets(self, base_orders: pd.DataFrame, demand_df: pd.DataFrame):
        if self.max_skus is None:
            return base_orders.index
        # Rank by baseline shortage proxy (demand > available pipeline)
        demand = demand_df["demand"]
        supply = base_orders["order_qty"]
        shortage_flag = (demand > supply).astype(int)
        ranked = shortage_flag.sort_values(ascending=False)
        return ranked.head(self.max_skus).index

    def _write_summary(self, powell_costs: List[float]):
        baseline = pd.read_csv(self.baseline_metrics_path)
        baseline_total = float(baseline["total_cost"].sum())
        powell_total = float(sum(powell_costs))
        summary_path = self.output_dir / "powell_summary.md"
        summary_path.write_text(
            "\n".join([
                "# Powell Global Policy",
                f"- Baseline cost: €{baseline_total:,.2f}",
                f"- Powell cost: €{powell_total:,.2f}",
                f"- Delta: €{powell_total - baseline_total:+,.2f}"
            ])
        )
        print(summary_path.read_text())


def parse_args():
    parser = argparse.ArgumentParser(description="Powell global policy comparison (Weeks 1-6)")
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--scenario-count", type=int, default=25)
    parser.add_argument("--deltas", type=int, nargs="+", default=[-30, 0, 30])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-skus", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("reports/global_policy"))
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    spec = yaml.safe_load(open(args.run_spec))
    spec.setdefault("paths", {})
    spec["paths"]["output_root"] = str(args.output_root)

    cli_args = HarnessArgs(
        run_spec=args.run_spec,
        run_id=spec.get("run_id", "powell_global"),
        tag="powell",
        weeks=spec.get("weeks"),
        dry_run=True
    )
    harness = BacktestHarness(spec, cli_args)

    runner = PowellRunner(
        harness=harness,
        baseline_metrics=args.baseline_metrics,
        scenario_count=args.scenario_count,
        deltas=args.deltas,
        horizon=args.horizon,
        max_skus=args.max_skus,
        rng_seed=args.seed
    )
    runner.run()


if __name__ == "__main__":
    main()
