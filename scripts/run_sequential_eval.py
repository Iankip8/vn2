#!/usr/bin/env python
"""Run sequential evaluation matching VN2 competition structure."""

from pathlib import Path
from vn2.analyze.sequential_eval import SequentialConfig, run_full_sequential_eval

def main():
    config = SequentialConfig(
        checkpoints_dir=Path("models/checkpoints"),
        demand_path=Path("data/processed/demand_imputed.parquet"),
        state_path=Path("data/processed/initial_state.parquet"),
        output_dir=Path("models/results"),
        run_tag="seq12_h12",
        n_jobs=4,
        holding_cost=0.2,
        shortage_cost=1.0,
        sip_grain=500,
        holdout_weeks=12
    )
    
    print("="*80)
    print("🔄 Running Sequential Evaluation (VN2 Competition Structure)")
    print("="*80)
    print(f"Checkpoints: {config.checkpoints_dir}")
    print(f"Demand data: {config.demand_path}")
    print(f"State data: {config.state_path}")
    print(f"Output: {config.output_dir}")
    print(f"Workers: {config.n_jobs}")
    print("="*80)
    
    run_full_sequential_eval(config)

if __name__ == "__main__":
    main()
