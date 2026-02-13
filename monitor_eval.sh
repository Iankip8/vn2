#!/bin/bash
# Monitor the evaluation process

PID=133045
LOG_FILE="logs/eval_corrected_policy.log"

echo "=== Monitoring Evaluation Process ==="
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo ""

while true; do
    clear
    echo "=== Process Status ($(date)) ==="
    if ps -p $PID > /dev/null 2>&1; then
        ps -p $PID -o pid,etime,%cpu,%mem,cmd --no-headers
        echo ""
        echo "=== Latest Log Output ==="
        tail -20 "$LOG_FILE" 2>/dev/null || echo "No log output yet"
    else
        echo "Process completed or stopped"
        echo ""
        echo "=== Final Log Output ==="
        tail -50 "$LOG_FILE" 2>/dev/null || echo "No log file"
        break
    fi
    sleep 10
done
