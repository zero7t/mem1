# MEM1 GRPO Training Algorithm Improvements

## Overview

本文档记录对 MEM1 多轮 GRPO 训练的算法层面优化，基于以下三项改进：
1. **Dr.GRPO** — 稳定 advantage 计算
2. **DAPO-lite** — 动态采样 + 非对称裁剪
3. **LLDS-MA** — 似然保持正则化

这些改进解决了标准 GRPO 在 binary/sparse reward 下的训练不稳定问题。

---

## 问题分析

### 标准 GRPO 的缺陷（binary reward 场景）

MEM1 使用 exact-match 作为奖励信号（reward ∈ {0, 1}），标准 GRPO 的 group std 归一化在此场景下有严重问题：

1. **Wild Amplification**: group [1, 0, 0, 0] → std=0.5, "1"样本的 advantage=(1-0.25)/0.5=1.5；而 group [1, 1, 0, 0] → std=0.577, "1"样本的 advantage=(1-0.5)/0.577=0.87。相同的正确回答在不同 group composition 下获得完全不同的 advantage 幅度。

2. **Gradient Waste**: 所有样本奖励相同的 group（如全0或全1）std→0，advantage→∞ 或需要特殊处理。

3. **Collapse Risk**: 长时间训练后，正确回答的 likelihood 持续下降（即使 advantage 为正），导致 ratio 膨胀，梯度爆炸，最终 collapse。

---

## 改进一：Dr.GRPO — 稳定 Advantage 归一化

**论文**: Dr.GRPO (Liu et al., 2024)

**核心修改**: 移除 group std 除法，仅使用 group mean 作为 baseline。

**修改文件**: `verl/trainer/ppo/core_algos.py` → `compute_grpo_outcome_advantage`

### 修改前
```python
scores[i] = (scores[i] - id2mean[idx]) / max(id2std[idx], epsilon)
```

### 修改后
```python
# [Dr.GRPO] Mean-only baseline, no std normalization
scores[i] = scores[i] - id2mean[idx]
```

### 原理
- Binary reward 下，advantage 的量纲本身就是 [0, 1] 范围的差值
- 不需要 std 归一化来"标准化"，因为 reward 尺度固定
- 消除了不同 group composition 导致的 advantage 幅度不一致问题

---

## 改进二：DAPO-lite — 动态采样 + Clip-Higher

**论文**: DAPO (Yu et al., 2025)

### 2.1 Dynamic Sampling（动态采样）

**修改文件**: `verl/trainer/ppo/core_algos.py` → `compute_grpo_outcome_advantage`

当一个 group 内所有样本奖励相同时（std < epsilon），将该 group 的 advantage 设为 0，等效于跳过该 group 的梯度更新。

```python
# [DAPO] Dynamic sampling: skip groups where all rewards are identical
if id2std[idx] < epsilon:
    scores[i] = 0.0
```

**效果**: 节省梯度预算，避免在无信息量的 group 上浪费计算。

### 2.2 Clip-Higher（非对称裁剪）

**修改文件**: `verl/trainer/ppo/core_algos.py` → `compute_policy_loss`

标准 PPO 使用对称裁剪 [1-ε, 1+ε]。DAPO 的 Clip-Higher 使用非对称裁剪：
- 下界: `1 - clip_low` (0.8)
- 上界: `1 + clip_high` (1.28)

```python
clip_low = cliprange      # 0.2
clip_high = clip_higher   # 0.28

pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
```

**原理**: 正 advantage 的样本可以更大幅度增加 ratio（即更快增加概率），加速学习好的轨迹。而负 advantage 的样本仍受严格裁剪约束，防止过度惩罚。

**配置参数**:
- `actor_rollout_ref.actor.clip_higher=0.28`

---

## 改进三：LLDS-MA — 似然保持正则化

**论文**: "On GRPO Collapse in Search-R1" (Deng et al., 2025)

### 问题：LLD Death Spiral

GRPO 训练中存在 Likelihood Displacement Death Spiral：
1. 正确回答的 log-likelihood 逐步下降（尽管 advantage>0）
2. 这导致 ratio = exp(new_lp - old_lp) 在后续 epoch 中膨胀
3. 膨胀的 ratio 使梯度变大，进一步 destabilize 其他样本
4. 最终所有样本的 log-likelihood 崩溃，模型输出变为垃圾

### LLDS-MA 公式

```
L_total = L_GRPO + λ * L_LLDS

L_LLDS = (1/N_tokens) * Σ_{y_i ∈ Y_pre}
          1[Σ_t(old_lp_t - new_lp_t) > 0]      # response-level gate
          * Σ_t max(0, old_lp_t - new_lp_t)     # token-level penalty
```

### 各组件解释

| 组件 | 含义 |
|------|------|
| **Y_pre** | 非负 advantage 的响应集合（正确+中性） |
| **Response-level gate** | 只有当整体响应的 likelihood 下降时才激活 |
| **Token-level penalty** | 仅惩罚那些 likelihood 下降的 token |
| **梯度方向** | push displaced tokens' likelihood 回升 |

### 实现细节

**修改文件**:
- `verl/trainer/ppo/core_algos.py` → 新增 `compute_llds_loss` 函数
- `verl/workers/actor/dp_actor.py` → `update_policy` 集成 LLDS

```python
def compute_llds_loss(old_log_probs, new_log_probs, eos_mask, advantages, ...):
    with torch.no_grad():
        # Y_pre: 非负 advantage 的响应
        response_advantages = advantages[:, 0]
        preserve_mask = (response_advantages >= 0).float()

        # Response-level gate（注意用 detach 计算 gate，梯度不流过 gate）
        token_disp_detached = (old_log_probs - new_log_probs.detach()) * reg_mask
        response_displacement = token_disp_detached.sum(dim=-1)
        response_gate = (response_displacement > 0).float()

        active_mask = preserve_mask * response_gate

    # Token-level penalty（梯度通过 new_log_probs 流过）
    token_displacement = (old_log_probs - new_log_probs) * reg_mask
    token_penalty = torch.clamp(token_displacement, min=0.0)

    # 归一化
    llds_loss = (response_penalty * active_mask).sum() / total_active_tokens
    return llds_loss, metrics
```

**关键实现注意**:
- Gate/mask 在 `torch.no_grad()` 内计算（纯选择作用）
- Token penalty 在 `no_grad` 外计算，梯度通过 `new_log_probs` 反传
- 使用 `new_log_probs.detach()` 计算 gate 避免二次梯度

**配置参数**:
- `actor_rollout_ref.actor.llds_lambda=0.1` (λ值，来自论文 7B 模型消融实验)

---

## 配置汇总

```bash
# 新增配置项（使用 + 前缀因为是 Hydra 新字段）
+actor_rollout_ref.actor.llds_lambda=0.1     # LLDS 正则化权重
+actor_rollout_ref.actor.clip_higher=0.28    # DAPO 上裁剪边界
```

完整训练脚本：`improvements/run_improved.sh`

---

## 修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `verl/trainer/ppo/core_algos.py` | Dr.GRPO advantage + DAPO dynamic sampling + Clip-Higher + LLDS loss 函数 |
| `verl/workers/actor/dp_actor.py` | update_policy 集成 LLDS loss + Clip-Higher 参数传递 |
| `improvements/run_improved.sh` | 训练脚本（bs=96, save_freq=100, 新参数） |

备份文件：`improvements/core_algos_original.py.bak`, `improvements/dp_actor_original.py.bak`

---

## 监控指标

训练日志中新增的指标：

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `llds/loss` | LLDS 正则化 loss | 0 ~ 0.1（0表示无位移，健康） |
| `llds/num_active` | 被正则化的响应数量 | 0 ~ batch_size |
| `critic/advantages/mean` | 平均 advantage | 接近 0（Dr.GRPO centered） |
| `actor/pg_clipfrac` | PPO clip 触发比例 | < 0.3 |

### 解读
- `llds/loss = 0`: 训练健康，无 likelihood displacement
- `llds/loss` 持续增长: 早期预警，LLDS 正在主动阻止 collapse
- `llds/num_active` 接近 batch_size: 大量响应被保护，可能需要降低 lr

---

## 实验配置

```
模型: Qwen2.5-7B
GPU: 6 × A800 80GB
batch_size: 96 (fallback 60 if OOM)
ppo_mini_batch_size: 192
ppo_micro_batch_size: 6
lr: 2e-7
warmup_ratio: 0.05
clip_ratio: 0.2 (lower), 0.28 (higher)
llds_lambda: 0.1
max_turns: 6
n_agent: 2
total_steps: 1413
save_freq: 100 steps
```

---

## 与其他优化的关系

本改进与以下已有优化正交、可叠加：
- **Deadlock Fix** (DEADLOCK_FIX.md): 解决 FSDP 死锁问题
- **Weight Sync 优化** (VERL_OPTIMIZATIONS.md): 减少多轮间冗余 weight sync
- **compute_entropy=False**: 节省 inference 阶段 13GB 显存
- **SwanLab offline logging**: 离线训练指标记录

---

## 首步验证结果 (2024-05-24)

Step 1 完成，关键指标：
```
llds/loss: 0.000              ← 健康（无位移）
llds/num_active: 0.000        ← 无样本被正则化（lr warmup 阶段）
actor/pg_loss: -0.010         ← 正常
actor/grad_norm: 0.904        ← 稳定
critic/score/mean: 0.021      ← 初始正确率 ~2%
timing_s/step: 218.3s         ← ~3.6min/step
timing_s/gen: 92.0s           ← 生成阶段
timing_s/update_actor: 102.5s ← 训练阶段（含 LLDS 开销）
```

无 OOM，bs=96 顺利通过。

---

## 改进四：中间 Reward 信号 Metrics 记录 (2026-05-24)

### 问题
Process reward、LLM Judge 升级数等关键中间信号只通过 print 输出到 stdout，未记录到 swanlab，无法追踪训练过程中 reward 信号的变化趋势。

### 修改
**文件**: `verl/trainer/main_ppo.py`, `verl/trainer/ppo/ray_trainer.py`

在 `RewardManager.__call__()` 中收集以下 metrics，通过 `reward_fn._step_metrics` 传递给 training loop 写入 swanlab：

| Metric | 含义 |
|--------|------|
| `reward/em_score_mean` | 纯 EM 匹配平均分 |
| `reward/em_score_nonzero_ratio` | EM>0 的样本比例 |
| `reward/process_reward_mean` | 过程奖励均值 |
| `reward/process_reward_max` | 过程奖励最大值 |
| `reward/process_reward_nonzero_ratio` | 有过程奖励的样本比例 |
| `judge/llm_judge_fired` | LLM Judge 发出的 API 调用数 |
| `judge/llm_judge_upgraded` | LLM Judge 升级的样本数 |
| `judge/gen_time_judge_fired` | 生成时 Judge 发出的调用数 |
| `judge/gen_time_judge_upgraded` | 生成时 Judge 升级的样本数 |
| `judge/total_upgraded` | 总升级数 |

---

## 改进五：update_actor 显存碎片化修复 (2026-05-24)

### 问题
`update_actor` 耗时逐步膨胀（step1=163s → step2=214s → step3=274s），MFU 持续下降（0.076→0.060→0.044）。

**根因**: `param_offload=true` + vLLM rollout 后 GPU 显存碎片化，导致 actor update 阶段 PyTorch 分配器效率持续下降。

### 修改
**文件**: `verl/trainer/ppo/ray_trainer.py`

在 `update_actor` 之前添加 `torch.cuda.empty_cache()`，将 PyTorch 缓存的空闲碎片块归还 CUDA driver，使后续分配能获得大块连续显存。

```python
# implement critic warmup
if self.config.trainer.critic_warmup <= self.global_steps:
    torch.cuda.empty_cache()  # 清理 vLLM 遗留的显存碎片
    with _timer('update_actor', timing_raw):
        ...
```

预期效果：`update_actor` 时间稳定在 ~160s，不再逐步膨胀。
