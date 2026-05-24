#!/bin/bash
# Monitor training: check status, restart if crashed, maintain GPU utilization
set -e

TRAIN_DIR="/root/paddlejob/workspace/mem1/MEM1/Mem1/train"
LOG_FILE="/root/paddlejob/workspace/mem1/MEM1/logs/grpo_full.log"
GPU_TOOLS="/root/paddlejob/workspace/env_run/gpu_tools"
RUN_SCRIPT="$TRAIN_DIR/run_fix_test.sh"
MIN_GPU_UTIL=60

cd "$TRAIN_DIR"

# Check if training process is alive
TRAIN_PID=$(pgrep -f "verl.trainer.main_ppo" | head -1 || true)

if [ -z "$TRAIN_PID" ]; then
    echo "[$(date)] Training NOT running. Checking last error..."

    # Show last error from log
    if [ -f "$LOG_FILE" ]; then
        echo "--- Last 20 lines of log ---"
        tail -20 "$LOG_FILE"
        echo "---"
    fi

    # Kill any stale ray processes
    ray stop --force 2>/dev/null || true
    sleep 3

    # Restart training
    echo "[$(date)] Restarting training..."
    nohup bash "$RUN_SCRIPT" >> "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo "[$(date)] Training restarted with PID=$NEW_PID"
else
    echo "[$(date)] Training running (PID=$TRAIN_PID)"

    # Check for recent errors
    RECENT_ERR=$(tail -100 "$LOG_FILE" 2>/dev/null | grep -i "Error\|OOM\|CUDA\|Traceback" | tail -3 || true)
    if [ -n "$RECENT_ERR" ]; then
        echo "[$(date)] WARNING: Recent errors detected:"
        echo "$RECENT_ERR"
    fi

    # Show latest progress
    LATEST_STEP=$(grep -oP "step:\s*\d+" "$LOG_FILE" 2>/dev/null | tail -1 || true)
    echo "[$(date)] Latest progress: $LATEST_STEP"
fi

# Check GPU utilization
echo ""
echo "[$(date)] GPU Utilization:"
GPU_UTILS=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
echo "$GPU_UTILS"

# Calculate average utilization for GPUs 0-5 (training GPUs)
AVG_UTIL=$(echo "$GPU_UTILS" | head -6 | awk -F',' '{sum+=$2; n++} END {print int(sum/n)}')
echo "[$(date)] Average GPU util (GPU 0-5): ${AVG_UTIL}%"

if [ "$AVG_UTIL" -lt "$MIN_GPU_UTIL" ]; then
    echo "[$(date)] WARNING: GPU util ${AVG_UTIL}% < ${MIN_GPU_UTIL}%. Starting GPU keepalive..."
    # Check if gpu keepalive is already running
    if ! pgrep -f "gpu_tools/gg" > /dev/null 2>&1; then
        cd "$GPU_TOOLS"
        chmod +x gg
        for i in 0 1 2 3 4 5; do
            ./gg 5 100000000 $i &
        done
        echo "[$(date)] GPU keepalive started on GPUs 0-5"
    else
        echo "[$(date)] GPU keepalive already running"
    fi
else
    # If training is running well with high util, kill any keepalive processes
    # (they'd compete for GPU resources)
    if pgrep -f "gpu_tools/gg" > /dev/null 2>&1; then
        pkill -f "gpu_tools/gg" || true
        echo "[$(date)] Killed GPU keepalive (training util is sufficient)"
    fi
fi
