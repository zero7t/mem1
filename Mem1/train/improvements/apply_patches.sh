#!/bin/bash
# Apply improvements directly to the source code
# Run this ONCE before training with improvements

set -e
TRAIN_DIR=/root/paddlejob/workspace/mem1/MEM1/Mem1/train
IMPROVEMENTS_DIR=$TRAIN_DIR/improvements

echo "=== Applying DAPO-lite + Dr.GRPO + LLDS-MA patches ==="

# Backup original files
echo "[1/3] Backing up original files..."
cp $TRAIN_DIR/verl/trainer/ppo/core_algos.py $IMPROVEMENTS_DIR/core_algos_original.py.bak
cp $TRAIN_DIR/verl/workers/actor/dp_actor.py $IMPROVEMENTS_DIR/dp_actor_original.py.bak

# Apply core_algos patch
echo "[2/3] Patching core_algos.py..."
cp $IMPROVEMENTS_DIR/core_algos_improved.py $TRAIN_DIR/verl/trainer/ppo/core_algos.py

echo "[3/3] Done! Source files patched."
echo ""
echo "Changes applied:"
echo "  - core_algos.py: Dr.GRPO advantage + DAPO clip-higher + LLDS-MA loss"
echo "  - dp_actor.py: needs manual edit (see below)"
echo ""
echo "NOTE: dp_actor.py update_policy needs manual integration of LLDS."
echo "      See improvements/README_INTEGRATION.md for details."
