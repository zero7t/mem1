#!/bin/bash
# Enable 4D attention mask during training (update_policy)
# This adds attention_mask_4d to the select_keys in update_policy

set -e
ACTOR_FILE=/root/paddlejob/workspace/mem1/MEM1/Mem1/train/verl/workers/actor/dp_actor.py

echo "=== Enabling 4D Attention Mask for Training ==="

# Check if already applied
if grep -q "attention_mask_4d.*in data.batch" "$ACTOR_FILE" 2>/dev/null; then
    # Check if it's in the update_policy context (not just compute_log_prob)
    if grep -A2 "'old_log_probs', 'advantages'" "$ACTOR_FILE" | grep -q "attention_mask_4d"; then
        echo "Already enabled. Nothing to do."
        exit 0
    fi
fi

# Backup
cp "$ACTOR_FILE" "$ACTOR_FILE.bak_before_4d"

# Apply patch: add attention_mask_4d to select_keys in update_policy
# The line to modify is:
#   select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
#   if self.config.state_masking:
# We insert a check for attention_mask_4d between these two lines

sed -i "/select_keys = \['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages'\]/a\\
        # [4D MASK] Include 4D attention mask during training if available\\
        if 'attention_mask_4d' in data.batch:\\
            select_keys.append('attention_mask_4d')" "$ACTOR_FILE"

echo "Done! 4D attention mask enabled for training."
echo "  File modified: $ACTOR_FILE"
echo "  Backup: $ACTOR_FILE.bak_before_4d"
echo ""
echo "To verify: grep 'attention_mask_4d' $ACTOR_FILE"
