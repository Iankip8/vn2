#!/usr/bin/env python
"""
Test bias correction and 3-week aggregation on a small subset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rich import print as rprint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.analyze.sequential_eval import run_sequential_evaluation, SequentialConfig

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    checkpoints_dir = base_dir / 'models' / 'checkpoints'
    demand_path = base_dir / 'data' / 'processed' / 'demand_imputed.parquet'
    state_path = base_dir / 'data' / 'processed' / 'initial_state.parquet'
    output_dir = base_dir / 'models' / 'results'
    
    # Config with 3-week aggregation (no bias correction)
    config = SequentialConfig(
        checkpoints_dir=checkpoints_dir,
        demand_path=demand_path,
        state_path=state_path,
        output_dir=output_dir,
        run_tag='3week_aggregation',
        n_jobs=4,  # Parallel over 4 cores
        holdout_weeks=12,
        sip_grain=500,
        bias_correction=1.0  # No bias correction, just 3-week aggregation
    )
    
    rprint("[bold]Testing 3-week aggregation (h=3+4+5)[/bold]")
    rprint(f"Using choose_order_3week with critical fractile 0.833")
    rprint(f"Output: {output_dir / f'sequential_results_{config.run_tag}.parquet'}")
    
    # Run evaluation (will auto-discover models)
    results_df = run_sequential_evaluation(config)
    
    # Display results
    rprint("\n[bold green]Results:[/bold green]")
    model_costs = results_df.groupby('model_name')['total_cost'].sum()
    for model, cost in model_costs.items():
        rprint(f"  {model}: €{cost:,.0f}")
    
    # Compare to baseline (h=3,4,5 without aggregation = €9,950)
    total_cost = model_costs.sum()
    baseline_cost = 9950  # 2-period optimization with h=3,4
    improvement = (baseline_cost - total_cost) / baseline_cost * 100
    
    rprint(f"\n[bold]Total cost: €{total_cost:,.0f}[/bold]")
    rprint(f"Baseline (2-period, h=3,4): €{baseline_cost:,.0f}")
    rprint(f"Improvement: {improvement:+.1f}%")
    rprint(f"\nVN2 benchmark: €5,248")
    rprint(f"Gap to benchmark: {(total_cost / 5248 - 1) * 100:+.1f}%")

if __name__ == '__main__':
    main()
