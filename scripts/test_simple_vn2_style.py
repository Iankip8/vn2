#!/usr/bin/env python
"""
Test VN2-style simple forecasting approach.

VN2 benchmark uses: order_up_to = 4 weeks of forecast (simple moving average)
This achieved €5,248 cost.

Let's implement this exact approach using our probabilistic forecasts.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vn2.analyze.sequential_eval import run_sequential_evaluation, SequentialConfig


def main():
    base_dir = Path(__file__).parent.parent
    checkpoints_dir = base_dir / 'models' / 'checkpoints'
    demand_path = base_dir / 'data' / 'processed' / 'demand_imputed.parquet'
    state_path = base_dir / 'data' / 'processed' / 'initial_state.parquet'
    output_dir = base_dir / 'models' / 'results'
    
    # Test simple VN2-style approach with 4-week coverage
    config = SequentialConfig(
        checkpoints_dir=checkpoints_dir,
        demand_path=demand_path,
        state_path=state_path,
        output_dir=output_dir,
        run_tag='vn2_4week',
        n_jobs=4,
        holdout_weeks=12,
        sip_grain=500,
        bias_correction=1.0
    )
    
    rprint("[bold cyan]Testing VN2-style with 4-week coverage[/bold cyan]")
    rprint("Approach: order_up_to = 4 weeks × median(h3, h4, h5)")
    rprint(f"Previous results:")
    rprint(f"  - 2-period newsvendor: €9,950")
    rprint(f"  - 3-week simple: €9,543")
    rprint(f"  - VN2 benchmark: €5,248")
    rprint()
    
    # Run evaluation
    results_df = run_sequential_evaluation(config)
    
    # Save detailed results
    output_file = output_dir / f'sequential_results_{config.run_tag}.parquet'
    results_df.to_parquet(output_file, index=False)
    rprint(f"[green]✅ Saved detailed results to: {output_file}[/green]")
    
    # Display summary
    rprint("\n[bold]Results by model:[/bold]")
    model_costs = results_df.groupby('model_name')['total_cost'].sum().sort_values()
    for model, cost in model_costs.items():
        rprint(f"  {model:25s}: €{cost:,.0f}")
    
    best_cost = model_costs.min()
    rprint(f"\n[bold]Best model cost: €{best_cost:,.0f}[/bold]")
    rprint(f"VN2 benchmark: €5,248")
    rprint(f"Gap: {(best_cost / 5248 - 1) * 100:+.1f}%")


if __name__ == '__main__':
    main()
