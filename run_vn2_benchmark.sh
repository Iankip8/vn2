#!/bin/bash
# Evaluate VN2 Benchmark Approach
# Target: €5,248

set -e

echo "========================================================================"
echo "VN2 BENCHMARK EVALUATION"
echo "========================================================================"
echo ""
echo "Approach: Seasonal 13-week MA + 4-week coverage order-up-to"
echo "Target cost: €5,248"
echo ""
echo "This validates our framework can reproduce the benchmark result."
echo ""

# Activate environment
source activate.sh

# Run evaluation
python -m vn2.analyze.sequential_eval_vn2_benchmark \
  --demand data/processed/demand_imputed.parquet \
  --state "Play VN2/Data/Week 0 - 2024-04-08 - Initial State.csv" \
  --output models/results/vn2_benchmark_results.parquet \
  --n-jobs 11 \
  --holdout 12

echo ""
echo "========================================================================"
echo "Evaluation complete"
echo "========================================================================"
