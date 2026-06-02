export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mem1

KEEPER_DIR=/root/paddlejob/workspace/env_run

stop_keeper_0_5() {
  ps -eo pid,args | python3 -c '
import os, re, signal, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    pid_s, _, args = line.partition(" ")
    if not pid_s.isdigit():
        continue
    if re.search(r"(?:^|/)gpu\.sh [0-5](\s|$)", args) or re.search(r"(?:^|/)gg .* [0-5](\s|$)", args):
        try:
            os.kill(int(pid_s), signal.SIGTERM)
        except ProcessLookupError:
            pass
'
}

start_keeper_0_5() {
  cd "$KEEPER_DIR"
  chmod -R 777 ./gpu_tools/gg || true
  for gpu in 0 1 2 3 4 5; do
    sh ./gpu_tools/gpu.sh "$gpu" >/tmp/run_gpu_${gpu}.log 2>&1 &
  done
}

trap start_keeper_0_5 EXIT
stop_keeper_0_5
sleep 2

WAND_PROJECT='MEM1'

export DATA_DIR='data/nq_hotpotqa_train_multi_2'
export BASE_MODEL="/root/paddlejob/workspace/new_mem1/mem1/mem1/Mem1/train/verl_checkpoints/V3-grpo-n4-bs96-resume200/actor/global_step_220"
export EXPERIMENT_NAME=V3-grpo-n4-bs96-resume220
export PROGRAM_ENTRY=grpo_v3.main_ppo_v3
export MAX_TURNS=6

export VLLM_ATTENTION_BACKEND=XFORMERS

export RAY_TMPDIR=/tmp/ray-$USER
mkdir -p $RAY_TMPDIR
export RAY_memory_usage_threshold=0.9

PYTHONUNBUFFERED=1 python3 -m $PROGRAM_ENTRY \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
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
    +algorithm.dapo_max_retries=1 \
    +algorithm.dapo_start_step=0 \
    +algorithm.dapo_process_thresh=0.01 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.llds_lambda=0.1 \
    +actor_rollout_ref.actor.clip_higher=0.28 \
    actor_rollout_ref.actor.ppo_mini_batch_size=384 \
    actor_rollout_ref.actor.ppo_micro_batch_size=12 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=12 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=12 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['swanlab','console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=683 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=$MAX_TURNS \
    do_search=true \
    retriever.url="http://127.0.0.1:8013/retrieve" \
    retriever.topk=3 \
    +curriculum.init_phase=transition \
    +curriculum.warmup_format_thresh=0.9 \
    +curriculum.warmup_max_step=80 \
    +curriculum.convergence_em_thresh=0.15 \
    +curriculum.convergence_max_step=300 \
    +curriculum.transition_em_thresh=0.30 \
    +curriculum.transition_max_step=500 \
    +reward.turn_weight_alpha=0.3 \
    +reward.turn_weight_beta=0.3 \
    +reward.turn_weight_gamma=1.5 \
    2>&1 | tee -a $EXPERIMENT_NAME.log
