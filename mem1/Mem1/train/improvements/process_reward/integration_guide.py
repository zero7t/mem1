"""
Integration Guide: How to plug StepReward into the existing training pipeline.

Minimal change: modify RewardManager._process_item() in main_ppo.py
"""

# =============================================================================
# INTEGRATION PATCH for main_ppo.py
# =============================================================================
#
# Location: verl/trainer/main_ppo.py, inside RewardManager._process_item()
#
# BEFORE (around line 77):
#   score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
#   if format_rewards:
#       score = score + format_rewards
#
# AFTER:
#   score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
#   if format_rewards:
#       score = score + format_rewards
#
#   # [PROCESS REWARD] Add step-level process reward
#   if self.use_process_reward:
#       from improvements.process_reward.step_reward import compute_process_reward
#       process_score = compute_process_reward(
#           trajectory_text=sequences_str,
#           ground_truth=ground_truth,
#           outcome_score=score,
#           config=self.process_reward_config,
#       )
#       score = score + process_score
#
# =============================================================================
# Also need to add in RewardManager.__init__():
#   self.use_process_reward = True  # or from config
#   self.process_reward_config = {
#       'lambda_process': 0.5,
#       'w_retrieval': 0.4,
#       'w_novelty': 0.3,
#       'w_progress': 0.3,
#       'max_turns': 6,
#       'format_penalty': True,
#       'efficiency_bonus': True,
#   }
# =============================================================================


# =============================================================================
# QUICK ENABLE SCRIPT
# =============================================================================

PATCH_INSTRUCTIONS = """
To enable process reward, add these lines to main_ppo.py:

1. At the top of the file, add import:
   import sys
   sys.path.insert(0, '/root/paddlejob/workspace/mem1/MEM1/Mem1/train')

2. In RewardManager.__init__(), add:
   self.use_process_reward = True
   self.process_reward_config = {
       'lambda_process': 0.5,
       'w_retrieval': 0.4,
       'w_novelty': 0.3,
       'w_progress': 0.3,
       'max_turns': 6,
   }

3. In RewardManager._process_item(), after line:
   score = compute_score_fn(...)

   Add:
   if self.use_process_reward:
       from improvements.process_reward.step_reward import compute_process_reward
       process_score = compute_process_reward(
           trajectory_text=sequences_str,
           ground_truth=ground_truth,
           outcome_score=score,
           config=self.process_reward_config,
       )
       score = score + process_score

That's it. No other files need to change.
The process reward is added to the same scalar score that gets placed
at the last token position, fully compatible with existing GRPO/Dr.GRPO/DAPO.
"""

if __name__ == "__main__":
    print(PATCH_INSTRUCTIONS)
