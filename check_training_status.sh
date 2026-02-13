#!/bin/bash
# Quick status check for model training

echo "================================================================================"
echo "VN2 MODEL TRAINING STATUS"
echo "================================================================================"
echo ""

# Check if process is running
PROC_COUNT=$(ps aux | grep 'vn2.cli forecast' | grep -v grep | wc -l)
if [ "$PROC_COUNT" -gt 0 ]; then
    echo "✅ Process status: RUNNING"
    PID=$(ps aux | grep 'vn2.cli forecast' | grep -v grep | awk '{print $2}')
    echo "   PID: $PID"
else
    echo "⏸️  Process status: STOPPED/COMPLETED"
fi

echo ""

# Check checkpoints
CHECKPOINT_COUNT=$(find models/checkpoints -name '*.pkl' 2>/dev/null | wc -l)
echo "📦 Checkpoints created: $CHECKPOINT_COUNT"

# Latest activity
if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
    LATEST=$(ls -lt models/checkpoints/*.pkl 2>/dev/null | head -1 | awk '{print $6,$7,$8,$9}')
    echo "   Latest: $LATEST"
fi

echo ""

# Progress from logs
LOG_FILE=$(ls -t logs/full_training_*.log 2>/dev/null | head -1)
if [ -f "$LOG_FILE" ]; then
    echo "📋 Latest log: $LOG_FILE"
    echo ""
    echo "Recent activity:"
    tail -30 "$LOG_FILE" | grep -E "(✓|Success|Failed|complete|Starting model)" | tail -10
fi

echo ""
echo "================================================================================"

# Estimated progress
TOTAL_TASKS=79000  # ~11 models × 599 SKUs × 12 folds
PROGRESS_PCT=$(echo "scale=1; ($CHECKPOINT_COUNT / $TOTAL_TASKS) * 100" | bc)
echo "📊 Estimated progress: ${PROGRESS_PCT}% ($CHECKPOINT_COUNT / $TOTAL_TASKS tasks)"
echo ""

# Expected completion
if [ "$PROC_COUNT" -gt 0 ]; then
    echo "⏰ Still running..."
    echo "   Monitor: tail -f $LOG_FILE"
    echo "   Stop: kill $PID"
else
    echo "🏁 Training complete or stopped"
    echo "   Check results: models/results/training_results.parquet"
fi

echo "================================================================================"
