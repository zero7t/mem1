# Working Backup: n4_bs96_noJudge

## Date: 2026-05-24

## Config (verified working, ~338s/step at util=0.55)
- n_agent=4, train_batch_size=96, ppo_mini_batch_size=384
- gpu_memory_utilization=0.55 (optimal, 0.75 was slower due to KV cache squeezing update_actor)
- ppo_micro_batch_size=12, log_prob_micro_batch_size=12
- param_offload=true, grad_offload=false, optimizer_offload=false
- LLM Judge: DISABLED (both Outcome and Process Judge commented out)
- Rule Process Reward: ENABLED
- Algorithm: DAPO-lite + Dr.GRPO + LLDS-MA (llds_lambda=0.1, clip_higher=0.28)
- max_turns=6, temperature=1
- recompute_log_prob=False in generation_think.py (compute_log_prob done once at end)
- total_training_steps=883 (84807 samples / bs96 = 883)

## Timing Breakdown (per step, util=0.55)
- generation: 111s
- adv (compute_log_prob + rule reward): 0.3s
- update_actor: 182s
- total step: 338s (5.6 min)
- Estimated total: 883 * 338s = 83h

## Key Findings
- LLM Judge was 76% of step time (497s/649s) when enabled due to serial API calls
- gpu_memory_utilization 0.75 made update_actor 22% slower (223s vs 182s) due to KV cache occupying GPU during actor update
- generation time barely changes with utilization (111s vs 113s)

## Files
- train.sh: training launch script
- main_ppo.py: reward manager with LLM Judge disabled
- ray_trainer.py: trainer with Process Judge disabled
- generation_think.py: multi-turn generation loop
- fsdp_workers.py: FSDP worker with compute_log_prob
- llm_judge.py: LLM Judge client (OneAPI endpoint)
- rule_reward.py: Rule-based process reward (4 dimensions)
