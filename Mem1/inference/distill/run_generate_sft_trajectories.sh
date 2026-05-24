#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/root/paddlejob/workspace/hf-cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MEM1_ROOT=/root/paddlejob/workspace/mem1/MEM1
PYTHON=/root/paddlejob/workspace/miniforge3/envs/mem1/bin/python

DATA_FILE=${DATA_FILE:-$MEM1_ROOT/Mem1/train/data/nq_hotpotqa_train_multi_2/train.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/outputs}
RAW_OUTPUT=${RAW_OUTPUT:-$OUTPUT_DIR/sft_trajectories_raw.jsonl}
SFT_OUTPUT=${SFT_OUTPUT:-$OUTPUT_DIR/sft_train.json}
API_BASE=${API_BASE:-http://127.0.0.1:8014/v1}
MODEL=${MODEL:-$MEM1_ROOT/assets/models/Mem-Lab__Qwen2.5-7B-RL-RAG-Q2-EM-Release}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MEM1_ROOT/assets/models/Qwen__Qwen2.5-7B}
SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:8013/retrieve}
LIMIT=${LIMIT:-1000}
OFFSET=${OFFSET:-0}
WORKERS=${WORKERS:-4}
MAX_TURNS=${MAX_TURNS:-6}
TOPK=${TOPK:-3}
TEMPERATURE=${TEMPERATURE:-0.01}
MIN_EXACT_MATCH=${MIN_EXACT_MATCH:--1}

mkdir -p "$OUTPUT_DIR"

cd "$MEM1_ROOT/Mem1/inference"
"$PYTHON" "$SCRIPT_DIR/generate_sft_trajectories.py" \
  --data_file "$DATA_FILE" \
  --output_jsonl "$RAW_OUTPUT" \
  --sft_output "$SFT_OUTPUT" \
  --api_base "$API_BASE" \
  --model "$MODEL" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --search_url "$SEARCH_URL" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --workers "$WORKERS" \
  --max_turns "$MAX_TURNS" \
  --topk "$TOPK" \
  --temperature "$TEMPERATURE" \
  --min_exact_match "$MIN_EXACT_MATCH" \
  "$@"
