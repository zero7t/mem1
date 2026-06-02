# GRPO V3 多阶段课程学习训练启动指南

## 概述

GRPO V3 实现了基于课程学习（Curriculum Learning）的多阶段 GRPO 训练，包含 4 个自动递进的训练阶段，每个阶段使用不同的奖励信号组合。阶段切换基于 EMA 指标自动触发，无需手动干预。

## 快速启动

```bash
cd /root/paddlejob/workspace/new_llm_judge_mem1/mem1/Mem1/train
bash grpo_v3/scripts/train_v3.sh
```

## 前置条件

1. **GPU 资源**: 至少 6 张 GPU 用于训练（GPU 6-7 留给检索服务）
2. **检索服务**: 需提前在 `http://127.0.0.1:8013/retrieve` 启动检索服务
3. **模型权重**: Qwen2.5-7B 模型放置于指定路径
4. **训练数据**: parquet 格式数据文件
5. **环境**: conda 环境 `mem1`，含 verl、vllm、ray 等依赖

## 启动命令详解

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

python -m grpo_v3.main_ppo_v3 \
  data.train_files=<训练数据路径> \
  data.val_files=<验证数据路径> \
  data.train_batch_size=96 \
  actor_rollout_ref.model.path=<模型路径> \
  algorithm.adv_estimator=grpo \
  trainer.total_training_steps=883 \
  max_turns=6 \
  do_search=true
```

使用 Hydra 配置系统，所有参数均可通过命令行覆盖 `config/ppo_trainer.yaml` 中的默认值。

## 四阶段课程学习

| 阶段 | 目标 | 退出条件 | 关键信号 |
|------|------|----------|----------|
| **Warmup** | 学习输出格式 (`<think>`, `<search>`, `<answer>`) | format_acc EMA ≥ 0.9 或 step ≥ 80 | format_reward=1.0, 无 process reward |
| **Convergence** | 学习有效检索 | em_rate EMA ≥ 0.15 或 step ≥ 300 | lambda_process=0.4, DAPO 重采样开启 |
| **Transition** | 引入 LLM Judge 细粒度信号 | em_rate EMA ≥ 0.30 或 step ≥ 500 | pointwise judge (scale=0.2) |
| **Refinement** | 推高上限 | 训练结束 | pointwise + listwise judge |

阶段只会前进，不会回退。使用 EMA (alpha=0.05) 平滑指标防止抖动。

## 默认超参数

### 数据与序列

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_batch_size` | 96 | 每步训练样本数 |
| `max_prompt_length` | 4096 | 最大 prompt 长度 |
| `max_response_length` | 500 | 每轮最大生成长度 |
| `max_start_length` | 2048 | 起始上下文最大长度 |
| `max_obs_length` | 500 | 观测（检索结果）最大长度 |
| `max_turns` | 6 | 最大交互轮数 |

### Actor 模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `actor.optim.lr` | 2e-7 | Actor 学习率 |
| `actor.optim.lr_warmup_steps_ratio` | 0.02 | 学习率 warmup 比例 |
| `actor.ppo_mini_batch_size` | 384 | PPO mini-batch 大小 |
| `actor.ppo_micro_batch_size` | 12 | PPO micro-batch 大小 |
| `actor.clip_ratio` | 0.2 | PPO clip 比率 |
| `actor.entropy_coeff` | 0.001 | 熵正则系数 |
| `actor.state_masking` | true | 遮蔽 `<information>` 块的 loss |
| `actor.llds_lambda` | 0.1 | LLDS 正则系数 |
| `actor.clip_higher` | 0.28 | 上界 clip 阈值 |
| `actor.fsdp_config.param_offload` | true | 参数卸载到 CPU |

### Rollout (vLLM)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rollout.n_agent` | 4 | 每个 prompt 采样轨迹数 (GRPO) |
| `rollout.temperature` | 1.0 | 采样温度 |
| `rollout.top_p` | 0.95 | Top-p 采样 |
| `rollout.gpu_memory_utilization` | 0.4 | vLLM GPU 显存占用比 |
| `rollout.tensor_model_parallel_size` | 1 | 张量并行度 |

### 算法

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.adv_estimator` | grpo | 优势估计方法 |
| `algorithm.kl_ctrl.kl_coef` | 0.001 | KL 惩罚系数 |
| `algorithm.dapo_max_retries` | 2 | DAPO 重采样最大重试次数 |
| `algorithm.dapo_start_step` | 50 | DAPO 启动步数 |
| `algorithm.no_think_rl` | false | 是否对 think 部分不做 RL |

### Turn-Weighted Advantage

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `reward.turn_weight_alpha` | 0.3 | 调制强度 |
| `reward.turn_weight_beta` | 0.3 | 位置权重峰值振幅 (bell-curve) |
| `reward.turn_weight_gamma` | 1.5 | 质量分数敏感度 |

### 课程学习阈值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `curriculum.warmup_format_thresh` | 0.9 | Warmup 退出格式准确率阈值 |
| `curriculum.warmup_max_step` | 80 | Warmup 最大步数 |
| `curriculum.convergence_em_thresh` | 0.15 | Convergence 退出 EM 阈值 |
| `curriculum.convergence_max_step` | 300 | Convergence 最大步数 |
| `curriculum.transition_em_thresh` | 0.30 | Transition 退出 EM 阈值 |
| `curriculum.transition_max_step` | 500 | Transition 最大步数 |

### Trainer

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainer.n_gpus_per_node` | 6 | 训练使用 GPU 数 |
| `trainer.total_training_steps` | 883 | 总训练步数 |
| `trainer.save_freq` | 50 | checkpoint 保存频率 |
| `trainer.experiment_name` | V3-curriculum-n4-bs96 | 实验名 |
| `trainer.logger` | ['swanlab','console'] | 日志后端 |

### 检索

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `retriever.url` | http://127.0.0.1:8013/retrieve | 检索服务地址 |
| `retriever.topk` | 3 | 检索返回文档数 |

## 自定义启动示例

### 修改模型和数据

```bash
python -m grpo_v3.main_ppo_v3 \
  actor_rollout_ref.model.path=/path/to/your/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  trainer.experiment_name=my-experiment
```

### 调整课程阈值（更激进的阶段切换）

```bash
python -m grpo_v3.main_ppo_v3 \
  +curriculum.warmup_max_step=50 \
  +curriculum.convergence_em_thresh=0.10 \
  +curriculum.transition_em_thresh=0.25
```

### 减少 GPU 使用

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m grpo_v3.main_ppo_v3 \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

## 输出

- **Checkpoints**: `verl_checkpoints/<experiment_name>/`
- **日志**: SwanLab + 控制台输出
- **阶段切换日志**: `[Curriculum] Phase transition: warmup → convergence at step XX`
