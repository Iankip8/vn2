#!/bin/bash
#
# Complete workflow for paper Section 6.1 Challenger Benchmark Plan
#
# Implements:
# 1. Data preparation (winsorized imputed data)
# 2. Model training (SLURP + challengers)
# 3. Sequential evaluation with Patrick's corrected policy
# 4. Jensen gap analysis
# 5. Cohort analysis
# 6. Results for paper
#
# Usage:
#   ./run_full_challenger_study.sh
#   ./run_full_challenger_study.sh --quick  # Subset for testing
#

set -e  # Exit on error

QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "=== QUICK MODE: Subset of models for testing ==="
fi

LOG_DIR="logs"
RESULTS_DIR="models/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "CHALLENGER BENCHMARK STUDY - FULL PIPELINE"
echo "================================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Quick mode: $QUICK_MODE"
echo ""

# ============================================================================
# PHASE 0: Prerequisites Check
# ============================================================================
echo "PHASE 0: Checking prerequisites..."

if [ ! -f "data/processed/demand_long.parquet" ]; then
    echo "ERROR: demand_long.parquet not found!"
    echo "Run data preprocessing first."
    exit 1
fi

echo "✓ demand_long.parquet exists"

# ============================================================================
# PHASE 1: Data Preparation
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 1: Data Preparation"
echo "================================================================================"

if [ ! -f "data/processed/demand_imputed_winsor.parquet" ]; then
    echo "Creating winsorized imputed data for non-SLURP models..."
    uv run python scripts/create_winsorized_data.py \
        2>&1 | tee "$LOG_DIR/winsorize_${TIMESTAMP}.log"
else
    echo "✓ demand_imputed_winsor.parquet already exists"
fi

# ============================================================================
# PHASE 2: Model Training
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 2: Model Training"
echo "================================================================================"

if $QUICK_MODE; then
    MODELS_TO_TRAIN="slurp_bootstrap lightgbm_quantile"
    echo "Quick mode: Training only $MODELS_TO_TRAIN"
else
    MODELS_TO_TRAIN="all"
    echo "Full mode: Training all challenger models"
fi

echo "Starting training at $(date)..."
uv run python scripts/train_challenger_suite.py \
    --models $MODELS_TO_TRAIN \
    --horizons 3 4 5 \
    --folds 5 \
    2>&1 | tee "$LOG_DIR/train_${TIMESTAMP}.log"

echo "Training complete at $(date)"

# ============================================================================
# PHASE 3: Sequential Evaluation with Patrick's Corrected Policy
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 3: Sequential Evaluation (Patrick's corrected policy)"
echo "================================================================================"

echo "Protection period: 3 weeks (h=3,4,5)"
echo "Critical fractile: τ = 0.833"
echo "Aggregation: Monte Carlo (10,000 samples)"
echo ""

uv run python scripts/eval_all_models_patrick.py \
    --output "$RESULTS_DIR/eval_patrick_v${TIMESTAMP}.parquet" \
    2>&1 | tee "$LOG_DIR/eval_${TIMESTAMP}.log"

# ============================================================================
# PHASE 4: Jensen Gap Analysis
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 4: Jensen Gap Analysis"
echo "================================================================================"

echo "Computing Jensen deltas: Δ = cost(point policy) - cost(SIP)..."
echo "Positive values = density-aware SIP wins"
echo ""

# The eval script already computes this, just summarize
python3 << EOF
import pandas as pd
from pathlib import Path

# Find latest results
results_dir = Path('$RESULTS_DIR')
jensen_files = sorted(results_dir.glob('*_jensen.parquet'))
if not jensen_files:
    print("No Jensen gap results found!")
    exit(1)

latest = jensen_files[-1]
print(f"Analyzing: {latest}")

df = pd.read_parquet(latest)

print("\n" + "="*80)
print("JENSEN GAP SUMMARY (by model)")
print("="*80)

summary = df.groupby('model').agg({
    'jensen_gap': ['sum', 'mean', 'median', 'std', 'min', 'max'],
    'cost_sip': 'sum',
    'cost_point': 'sum'
}).round(2)

summary.columns = ['gap_total', 'gap_mean', 'gap_median', 'gap_std', 'gap_min', 'gap_max', 'cost_sip_total', 'cost_point_total']
summary['improvement_pct'] = 100 * summary['gap_total'] / summary['cost_point_total']
summary = summary.sort_values('gap_total', ascending=False)

print(summary)

print("\n" + "="*80)
print("H1 (Jensen Gap Hypothesis) Results:")
print("="*80)
print(f"Models with positive Jensen gap: {(summary['gap_total'] > 0).sum()}/{len(summary)}")
print(f"Mean improvement from SIP vs point: {summary['improvement_pct'].mean():.2f}%")
print(f"Max improvement: {summary['improvement_pct'].max():.2f}% ({summary['improvement_pct'].idxmax()})")
EOF

# ============================================================================
# PHASE 5: Cohort Analysis
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 5: Cohort Analysis"
echo "================================================================================"

echo "Analyzing Jensen gaps by:"
echo "  - Demand rate (sparse vs dense)"
echo "  - Zero ratio (intermittent vs continuous)"
echo "  - Stockout rate (censored vs observed)"
echo "  - CV (dispersion)"
echo ""

# TODO: Implement cohort analysis script
echo "NOTE: Create scripts/analyze_cohorts.py for detailed cohort breakdowns"

# ============================================================================
# PHASE 6: Results Summary for Paper
# ============================================================================
echo ""
echo "================================================================================"
echo "PHASE 6: Results Summary for Paper"
echo "================================================================================"

echo "Generating summary for paper Section 7..."

python3 << 'EOF'
import pandas as pd
from pathlib import Path

results_dir = Path('models/results')

# Latest evaluation
eval_files = sorted(results_dir.glob('eval_patrick_v*.parquet'))
if not eval_files:
    print("No evaluation results found!")
    exit(1)

latest_eval = eval_files[-1]
df = pd.read_parquet(latest_eval)

print("\n" + "="*80)
print("PAPER SECTION 7.1 - JENSEN EFFECT (Decision Gap)")
print("="*80)

# Cost by policy
print("\nTotal costs by policy:")
sip_cost = df[df['policy'] == 'density_sip'].groupby('model')['cost'].sum()
point_cost = df[df['policy'] == 'point_service'].groupby('model')['cost'].sum()

comparison = pd.DataFrame({
    'SIP_cost': sip_cost,
    'Point_cost': point_cost,
    'Improvement': point_cost - sip_cost,
    'Improvement_pct': 100 * (point_cost - sip_cost) / point_cost
}).round(2)

comparison = comparison.sort_values('Improvement', ascending=False)
print(comparison)

print("\n" + "="*80)
print("KEY FINDINGS FOR PAPER:")
print("="*80)
print(f"1. Models evaluated: {len(comparison)}")
print(f"2. Best density-aware model: {sip_cost.idxmin()} (€{sip_cost.min():.2f})")
print(f"3. Largest Jensen gap: {comparison['Improvement'].max():.2f} ({comparison['Improvement'].idxmax()})")
print(f"4. Mean improvement from SIP: {comparison['Improvement_pct'].mean():.2f}%")

print("\n" + "="*80)
print("COMPARISON TO BENCHMARKS:")
print("="*80)
print("VN2 Benchmark: €5,248")
print(f"Best model (SIP): €{sip_cost.min():.2f} ({100*(sip_cost.min()-5248)/5248:.1f}% gap)")
print(f"Patrick's baseline: €6,266 (from eval_patrick_integrated.py)")

EOF

# ============================================================================
# DONE
# ============================================================================
echo ""
echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
echo "Results saved in: $RESULTS_DIR/"
echo "Logs saved in: $LOG_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review leaderboards in logs/eval_${TIMESTAMP}.log"
echo "  2. Update paper Section 7 with results"
echo "  3. Create visualizations (Jensen gap plots, cohort analyses)"
echo "  4. Test hypotheses H2-H4 (stockout-awareness, SURD effect, etc.)"
echo ""
echo "================================================================================"
