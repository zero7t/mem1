#!/bin/bash
LOG=/root/paddlejob/workspace/mem1/MEM1/logs/gpu_memory.csv
echo "timestamp,gpu0_used_mb,gpu1_used_mb,gpu2_used_mb,gpu3_used_mb,gpu4_used_mb,gpu5_used_mb" > $LOG
while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4,5 | tr '\n' ',' | sed 's/,$//')
  echo "$TS,$MEM" >> $LOG
  sleep 10
done
