# MEM1 项目代码地图

> 最后更新: 2026-05-24
> GitHub: https://github.com/zero7t/mem1.git

---

## 一、项目概述

MEM1 是基于 GRPO（Group Relative Policy Optimization）的多轮 RAG QA 训练系统。
模型（Qwen 2.5-7B）学习通过多轮搜索检索来回答复杂问题。

**核心流程**:
```
Question → [Think → Search → Retrieval] × N轮 → Answer
                                                   ↓
                                          Exact Match → 0/1 reward
                                                   ↓
                                     GRPO Advantage → Policy Gradient Update
```

**硬件**: 6× GPU (FSDP 分布式训练 + vLLM 推理)

---

## 二、目录结构总览

```
MEM1/
├── assets/
│   ├── models/Qwen__Qwen2.5-7B/     # 基座模型 (不上传GitHub)
│   └── wiki-18/                       # 检索知识库 (不上传GitHub)
├── logs/                              # 训练日志
│   ├── grpo_full.log                  # baseline 286步日志
│   └── grpo_improved.log             # 改进版日志 (当前在跑)
├── Mem1/
│   ├── inference/                     # 推理/评估
│   └── train/                         # ★ 核心训练代码 (见下方详细)
├── setup/
└── README.md
```

---

## 三、核心训练代码 (`Mem1/train/`)

### 3.1 训练入口与主循环

| 文件 | 作用 |
|------|------|
| `verl/trainer/main_ppo.py` | **训练入口** + RewardManager（计算reward） |
| `verl/trainer/ppo/ray_trainer.py` | **训练主循环**：rollout → reward → advantage → update |
| `verl/trainer/config/ppo_trainer.yaml` | 默认配置 |

### 3.2 核心算法 ★★★

| 文件 | 作用 | 改动状态 |
|------|------|---------|
| `verl/trainer/ppo/core_algos.py` | **GRPO核心**：advantage计算、policy loss、LLDS | ✅ 已修改 |
| `verl/workers/actor/dp_actor.py` | **Actor更新**：forward + backward + LLDS集成 | ✅ 已修改 |
| `verl/workers/fsdp_workers.py` | FSDP分布式worker | ✅ 已修改 (deadlock fix) |

### 3.3 生成/Rollout

| 文件 | 作用 |
|------|------|
| `rollout/llm_agent/generation_think.py` | **多轮生成主逻辑**：LLM loop、execute_predictions |
| `rollout/llm_agent/attn_mask_utils.py` | 4D attention mask 构造（Markov结构） |
| `rollout/llm_agent/tensor_helper.py` | 轨迹 padding/拼接工具 |
| `rollout/search/retrieval_server.py` | 检索服务（Flask, localhost:8013） |

### 3.4 Reward 计算

| 文件 | 作用 |
|------|------|
| `verl/utils/reward_score/qa_multiple.py` | **当前使用的reward**：exact_match |
| `verl/utils/reward_score/qa_em.py` | 单答案 EM |
| `verl/utils/reward_score/websearch.py` | F1 score (WebSearch任务用) |

### 3.5 模型加载 & vLLM

| 文件 | 作用 |
|------|------|
| `verl/third_party/vllm/vllm_v_0_6_3/llm.py` | vLLM推理引擎封装 |
| `verl/third_party/vllm/vllm_v_0_6_3/worker.py` | vLLM worker (修改: keep_generation_mode) |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | Rollout调度 |
| `verl/workers/sharding_manager/fsdp_vllm.py` | FSDP↔vLLM 权重同步 |

---

## 四、改进代码 (`Mem1/train/improvements/`) ★★★

### 4.1 已启用（当前在跑的训练已包含）

| 文件 | 内容 | 对应源码修改 |
|------|------|------------|
| `core_algos_improved.py` | 改进版参考实现 | → `verl/trainer/ppo/core_algos.py` |
| `dp_actor_patch.py` | Actor更新patch | → `verl/workers/actor/dp_actor.py` |
| `run_improved.sh` | **启动脚本** | 独立 |

**当前生效的改进**:
- Dr.GRPO: advantage = reward - group_mean (不除std)
- DAPO Dynamic Sampling: 全同group → skip
- DAPO Clip-Higher: clip_low=0.2, clip_high=0.28
- LLDS-MA: λ=0.1, 防止好轨迹collapse

### 4.2 待启用（已实现，下次重启时集成）

```
improvements/process_reward/
├── step_reward.py           # 规则过程奖励 (4维: 检索相关性/新颖性/覆盖度/效率)
├── llm_judge.py             # LLM Judge (outcome语义判断 + listwise排名)
└── integration_guide.py     # 集成到 main_ppo.py 的说明
```

### 4.3 备用（已实现，决定不启用）

```
improvements/4d_mask/
├── dp_actor_4d_patch.py     # 4D Markov attention mask 训练版
├── enable_4d_training.sh    # 启用脚本
└── disable_4d_training.sh   # 禁用脚本
└── README.md                # 分析: 为什么不用4D mask
```

---

## 五、如何使用

### 5.1 启动当前改进版训练

```bash
# 1. 启动检索服务
cd /root/paddlejob/workspace/mem1/MEM1/Mem1/train
bash retrieval_launch.sh

# 2. 启动训练
nohup bash improvements/run_improved.sh > /root/paddlejob/workspace/mem1/MEM1/logs/grpo_improved.log 2>&1 &
```

### 5.2 当前 `run_improved.sh` 关键配置

```bash
data.train_batch_size=96
actor_rollout_ref.rollout.n_agent=4           # [更新] 从2改为4
actor_rollout_ref.actor.ppo_mini_batch_size=384  # [更新] 96×4
actor_rollout_ref.actor.ppo_micro_batch_size=6
actor_rollout_ref.actor.optim.lr=2e-7
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02  # [更新] 从0.05改为0.02
+actor_rollout_ref.actor.llds_lambda=0.1
+actor_rollout_ref.actor.clip_higher=0.28
actor_rollout_ref.rollout.gpu_memory_utilization=0.55
trainer.save_freq=100
trainer.total_training_steps=1413
max_turns=6
```

### 5.3 查看训练状态

```bash
# 查进程
ps aux | grep main_ppo | grep -v grep

# 查最新日志
tail -20 /root/paddlejob/workspace/mem1/MEM1/logs/grpo_improved.log

# 查 GPU
nvidia-smi

# SwanLab 可视化
cd Mem1/train && swanlab watch swanlog/
```

### 5.4 下一步集成 Process Reward（不影响当前运行）

集成点: `verl/trainer/main_ppo.py` 的 `RewardManager._process_item()`

```python
# 在 score = compute_score_fn(...) 之后加入:
import sys
sys.path.insert(0, '/root/paddlejob/workspace/mem1/MEM1/Mem1/train')
from improvements.process_reward.step_reward import compute_process_reward

process_score = compute_process_reward(
    trajectory_text=sequences_str,
    ground_truth=ground_truth,
    outcome_score=score,
)
score = score + process_score
```

---

## 六、训练数据流 (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│ ray_trainer.py 主循环 (每个 training step)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. DataLoader → batch (96 prompts)                                 │
│       ↓                                                             │
│  2. batch.repeat(n_agent=4) → 384 trajectories                     │
│       ↓                                                             │
│  3. generation_think.py::run_llm_loop()                             │
│     ┌──────────────────────────────────┐                            │
│     │ for step in range(max_turns=6):  │                            │
│     │   vLLM generate → response       │                            │
│     │   extract search query           │                            │
│     │   call retriever → information   │                            │
│     │   append to context              │                            │
│     │   if <answer> → done             │                            │
│     └──────────────────────────────────┘                            │
│       ↓                                                             │
│  4. compose_final_output() → input_ids, attention_mask, info_mask   │
│       ↓                                                             │
│  5. compute_log_prob() → old_log_probs                              │
│  6. ref_log_prob() → ref_log_probs (for KL)                        │
│       ↓                                                             │
│  7. RewardManager.__call__()                                        │
│     extract_solution() → em_check() → score (0/1)                  │
│     → reward_tensor[i, last_token] = score                          │
│       ↓                                                             │
│  8. compute_grpo_outcome_advantage()                                │
│     [Dr.GRPO] advantage = score - group_mean                        │
│     [DAPO] skip groups where std < ε                                │
│       ↓                                                             │
│  9. update_actor() (dp_actor.py)                                    │
│     for micro_batch in mini_batches:                                │
│       forward → new_log_probs                                       │
│       [DAPO] pg_loss with clip_higher=0.28                          │
│       [LLDS] if good trajectory likelihood ↓ → penalty              │
│       loss.backward()                                               │
│     optimizer.step()                                                │
│       ↓                                                             │
│ 10. sync_weights FSDP → vLLM (for next step generation)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、关键修复记录

| 问题 | 修复 | 文件 |
|------|------|------|
| Deadlock (FSDP↔vLLM) | `keep_generation_mode=True` | `vllm_v_0_6_3/worker.py` |
| OOM in compute_log_prob | `compute_entropy=False` | `dp_actor.py` |
| Config normalization | `per_worker = global // num_workers * rollout.n` | `fsdp_workers.py` |
| SwanLab offline | `SWANLAB_MODE=disabled` + manual init | `tracking.py` |

---

## 八、环境

```
Python: miniforge3/envs/mem1
Model: Qwen 2.5-7B (assets/models/)
Retriever: localhost:8013 (wiki-18 index)
GPUs: 6× (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5)
Framework: verl (FSDP + vLLM + Ray)
```
