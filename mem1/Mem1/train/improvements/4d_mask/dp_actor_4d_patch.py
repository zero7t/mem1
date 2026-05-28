"""
4D Attention Mask Patch for dp_actor.py update_policy

This file shows the exact modifications needed to enable 4D attention mask
during training (update_policy). Apply via enable_4d_training.sh.

Changes:
1. Add 'attention_mask_4d' to select_keys in update_policy
2. The existing _forward_micro_batch already handles 4D mask when present

That's it — the rest of the pipeline (generation, ray_trainer pop/restore)
already correctly passes the 4D mask through to the batch.
"""

# ============================================================================
# CHANGE 1: In update_policy(), add 'attention_mask_4d' to select_keys
# ============================================================================
#
# BEFORE (line ~233 in dp_actor.py):
#   select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
#   if self.config.state_masking:
#       select_keys.append('loss_mask')
#
# AFTER:
#   select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
#   if 'attention_mask_4d' in data.batch:
#       select_keys.append('attention_mask_4d')
#   if self.config.state_masking:
#       select_keys.append('loss_mask')
#
# ============================================================================
# That's the ONLY change needed. _forward_micro_batch already checks for
# 'attention_mask_4d' in micro_batch and uses it if present (line ~70):
#
#   if 'attention_mask_4d' in micro_batch:
#       attention_mask_4d = micro_batch['attention_mask_4d'].to(torch.bfloat16)
#   else:
#       attention_mask_4d = None
#   ...
#   output = self.actor_module(
#       input_ids=input_ids,
#       attention_mask=attention_mask if attention_mask_4d is None else attention_mask_4d,
#       ...
#   )
# ============================================================================


# For reference, here's what the modified update_policy select_keys section looks like:
def update_policy_select_keys_example(self, data):
    """Example showing the modified select_keys logic."""
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']

    # [4D MASK] Include 4D attention mask if available in the batch
    if 'attention_mask_4d' in data.batch:
        select_keys.append('attention_mask_4d')

    if self.config.state_masking:
        select_keys.append('loss_mask')
    if self.config.use_kl_loss:
        select_keys.append('ref_log_prob')

    return select_keys
