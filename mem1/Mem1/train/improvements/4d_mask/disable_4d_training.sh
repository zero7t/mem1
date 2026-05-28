#!/bin/bash
# Disable 4D attention mask during training (revert to 2D only)
# Restores the backup made by enable_4d_training.sh

set -e
ACTOR_FILE=/root/paddlejob/workspace/mem1/MEM1/Mem1/train/verl/workers/actor/dp_actor.py
BACKUP="$ACTOR_FILE.bak_before_4d"

echo "=== Disabling 4D Attention Mask for Training ==="

if [ -f "$BACKUP" ]; then
    cp "$BACKUP" "$ACTOR_FILE"
    echo "Reverted to backup: $BACKUP"
else
    # Manual revert: remove the inserted lines
    sed -i '/# \[4D MASK\] Include 4D attention mask during training if available/d' "$ACTOR_FILE"
    sed -i "/if 'attention_mask_4d' in data.batch:/,/select_keys.append('attention_mask_4d')/d" "$ACTOR_FILE"
    echo "Manually removed 4D mask lines from update_policy."
fi

echo "Done! Training now uses standard 2D attention mask only."
