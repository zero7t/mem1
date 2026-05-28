#!/bin/bash
# V2 + LLM Judge Training Script
# Aligned with FULL-n4-bs96-noJudge config (6 GPUs, fixed ppo_mini_batch_size)
# Rule process reward REDUCED (lambda_process=0.2), LLM Judge provides main process signal

cd /root/paddlejob/workspace/new_llm_judge_mem1/mem1/Mem1/train
export PATH=/root/paddlejob/workspace/miniforge3/envs/mem1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_TMPDIR=/tmp/ray-mem1-judge
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/paddlejob/workspace/hf-cache
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1

# Internal API - no proxy
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

exec python -m grpo_improved.main_ppo_v2 \
  data.train_files=/root/paddlejob/workspace/mem1/MEM1/Mem1/train/data/nq_hotpotqa_train_multi_2/train.parquet \
  data.val_files=/root/paddlejob/workspace/mem1/MEM1/Mem1/train/data/nq_hotpotqa_train_multi_2/test.parquet \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=96 \
  data.val_batch_size=96 \
  data.max_prompt_length=4096 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=500 \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=/root/paddlejob/workspace/mem1/MEM1/assets/models/Qwen__Qwen2.5-7B \
  actor_rollout_ref.actor.optim.lr=2e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size=384 \
  actor_rollout_ref.actor.ppo_micro_batch_size=12 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  +actor_rollout_ref.actor.llds_lambda=0.1 \
  +actor_rollout_ref.actor.clip_higher=0.28 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=12 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=12 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.n_agent=4 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.no_think_rl=false \
  trainer.critic_warmup=0 \
  "trainer.logger=['swanlab','console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=false \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=6 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=-1 \
  trainer.project_name=MEM1 \
  trainer.experiment_name=V2-judge-n4-bs96 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=883 \
  trainer.default_local_dir=verl_checkpoints/V2-judge-n4-bs96 \
  max_turns=6 \
  +algorithm.dapo_max_retries=2 \
  +algorithm.dapo_start_step=50 \
  +reward.lambda_process=0.2 \
  +reward.turn_weight_alpha=0.3 \
  +reward.turn_weight_clip=1.0 \
  +reward.turn_weight_enabled=true \
  retriever.url=http://127.0.0.1:8013/retrieve \
  retriever.topk=3
