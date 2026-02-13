#!/bin/bash
# Run sequential evaluation using Patrick's corrected policy

set -e

echo "=========================================="
echo "Sequential Evaluation: Patrick's Approach"
echo "=========================================="
echo ""
echo "This uses Patrick's 3 critical fixes:"
echo "  1. Protection period = 3 weeks (h=3,4,5)"
echo "  2. Critical fractile = 0.833 explicit"
echo "  3. MC aggregation (10,000 samples)"
echo ""

# Activate environment
source activate.sh

# Run evaluation
python scripts/sequential_eval_patrick.py

echo ""
echo "Evaluation complete!"
echo "Check models/results/sequential_eval_patrick.csv for detailed results"
