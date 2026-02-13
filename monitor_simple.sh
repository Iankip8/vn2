#!/bin/bash
# Simple evaluation monitor

LOG="/home/ian/vn2/logs/eval_corrected_fresh.log"
PID=135967

while true; do
    if ps -p $PID > /dev/null 2>&1; then
        BATCH=$(grep "📦 Batch" "$LOG" | tail -1)
        PROGRESS=$(grep "Done.*tasks" "$LOG" | tail -1)
        echo "$(date +%H:%M:%S) - $BATCH"
        echo "  Latest: $PROGRESS"
        echo ""
        sleep 180  # Check every 3 minutes
    else
        echo "Process completed!"
        echo ""
        echo "=== FINAL RESULTS ==="
        tail -50 "$LOG" | grep -A 20 "Leaderboard"
        break
    fi
done
