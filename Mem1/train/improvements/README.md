# Training Improvements

## 状态总览 (Status)

| 改进 | 状态 | 效果 |
|------|------|------|
| Dr.GRPO (mean-only advantage) | ✅ **已启用** | 梯度稳定性 |
| DAPO Dynamic Sampling | ✅ **已启用** | 跳过无效group |
| DAPO Clip-Higher (0.28) | ✅ **已启用** | 正样本强化 |
| LLDS-MA (λ=0.1) | ✅ **已启用** | 防collapse |
| Rule Process Reward | ⏳ **待集成** | 密集过程信号 |
| LLM Outcome Judge (语义EM) | ⏳ **待集成** | 同义词识别 |
| LLM Process Judge (listwise) | ⏳ **待集成** | tied group利用 |
| n_agent=4 | ⏳ **待启用** | 更好梯度估计 |

### 当前正在运行的配置

```
Reward:      纯 outcome = exact_match (0 or 1)
Advantage:   Dr.GRPO (reward - group_mean, no std division)
Skip:        DAPO dynamic sampling (全同group → advantage=0)
Clipping:    asymmetric (low=0.2, high=0.28)
Regularizer: LLDS-MA (λ=0.1, 防止好轨迹likelihood塌陷)
Batch:       train_batch_size=96, n_agent=2
PPO:         mini_batch=192, micro_batch=6
LR:          2e-7, warmup 5%
Save:        每100步
```

### Reward 详细流程（当前版本）

```
1. 生成阶段 (generation_think.py):
   - 模型生成 <think><search>query</search> → 获取 <information>
   - 重复最多 6 轮
   - 最终输出 <answer>...</answer>
   - 每步 batch_rewards = 0（格式惩罚已注释掉）

2. 结果评估 (main_ppo.py::RewardManager):
   - extract_solution(): 提取最后一个 <answer>...</answer> 内容
   - em_check(answer, ground_truth['target']): exact match
   - score = 0 (错) or 1 (对)
   - reward_tensor[i, last_valid_token_pos] = score

3. Advantage 计算 (core_algos.py):
   - 按 prompt index 分组 (n_agent=2 → 每组2条)
   - [DAPO] 如果组内 std < epsilon → advantage=0, 跳过
   - [Dr.GRPO] advantage = score - group_mean (不除std)
   - 广播到 response 所有 token

4. Policy Update (dp_actor.py):
   - [DAPO Clip-Higher] clip_low=0.2, clip_high=0.28
   - [LLDS-MA] 检测好轨迹的likelihood下降，施加惩罚推回
   - loss = pg_loss - entropy_coeff*entropy + llds_lambda*llds_loss
```

---

## Phase 1: GRPO 优化（已启用）

Three improvements applied directly to the source code:

### 1. Dr.GRPO: Stable Advantage Normalization
**File**: `verl/trainer/ppo/core_algos.py` → `compute_grpo_outcome_advantage`

- **Change**: Remove division by group std. Only subtract group mean as baseline.
- **Why**: Binary exact-match reward + group std normalization causes wild amplification
  of noisy groups. E.g., group with rewards [1, 0, 0, 0] gets std=0.5, so the "1" sample
  gets advantage = (1-0.25)/0.5 = 1.5, but group [1, 1, 0, 0] gets std=0.577 giving 0.87.
  This inconsistency destabilizes training.
- **New behavior**: advantage = reward - group_mean (raw centered advantage)

### 2. DAPO-lite: Dynamic Sampling + Clip-Higher
**Files**: `core_algos.py`

- **Dynamic Sampling**: Groups where all rewards are identical (std < epsilon) get
  advantage = 0, effectively skipped. Saves gradient budget.
- **Clip-Higher** (asymmetric): `clip_low=0.2`, `clip_high=0.28`
  - Positive advantage samples can increase their ratio more
  - Gives "good" trajectories stronger reinforcement without destabilizing "bad" ones

### 3. LLDS-MA: Likelihood-Preserving Regularization
**Files**: `core_algos.py` + `verl/workers/actor/dp_actor.py`

From paper: "On GRPO Collapse in Search-R1" (Deng et al., 2025)

Formula:
```
L_total = L_GRPO + λ * L_LLDS

L_LLDS = (1/N_tokens) * Σ_{y_i ∈ Y_pre}
          1[Σ_t(old_lp_t - new_lp_t) > 0]      # response-level gate
          * Σ_t max(0, old_lp_t - new_lp_t)    # token-level penalty
```

- **Y_pre**: Responses with non-negative advantage (correct + average)
- **Response-level gate**: Only activates when the OVERALL response likelihood decreased
- **Token-level selectivity**: Penalizes ONLY the tokens whose likelihood went down
- **Gradient**: pushes down-displaced tokens' likelihood back UP
- **λ = 0.1** (from paper's ablation on 7B model)

**Why it prevents collapse**: The LLD Death Spiral happens when correct responses'
likelihood steadily decreases, inflating ratios, causing gradient explosion. LLDS directly
prevents this displacement by penalizing any decrease in likelihood of "good" trajectories.

## Config Parameters

New config fields in `actor_rollout_ref.actor`:
```
actor_rollout_ref.actor.llds_lambda=0.1     # LLDS regularization weight
actor_rollout_ref.actor.clip_higher=0.28    # DAPO upper clip (lower is cliprange=0.2)
```

## Files Modified

1. `verl/trainer/ppo/core_algos.py`
   - `compute_grpo_outcome_advantage`: Dr.GRPO + DAPO dynamic sampling
   - `compute_policy_loss`: Added `clip_higher` parameter
   - `compute_llds_loss`: New function for LLDS-MA

2. `verl/workers/actor/dp_actor.py`
   - `update_policy`: Integrated LLDS loss + Clip-Higher

## Backups

Original files backed up in `improvements/`:
- `core_algos_original.py.bak`
- `dp_actor_original.py.bak`

## How to Run

```bash
bash improvements/run_improved.sh
```

## Monitoring

Watch these new metrics in the log:
- `llds/loss`: LLDS regularization loss (should be small, ~0.01-0.1)
- `llds/num_active`: Number of responses being regularized per micro-batch
- If `llds/loss` stays at 0: training is healthy (no likelihood displacement)
- If `llds/loss` grows: early warning of potential collapse (LLDS is actively preventing it)

---

# Phase 2: Process Reward + LLM Judge System

## Overview

在 Phase 1（DAPO + Dr.GRPO + LLDS）解决训练稳定性之后，Phase 2 解决**奖励信号稀疏**问题。

当前问题：
- Outcome reward = binary EM (0/1)，6轮交互只有最后一个信号
- Score ~2% 时，96% 的 GRPO group 被 DAPO skip（全是 [0,0]）
- 模型无法区分"差一步就对"和"完全跑偏"

Phase 2 包含三个独立组件，可单独启用：

| 组件 | 作用 | 成本 | 适用阶段 |
|------|------|------|---------|
| Rule-based Step Reward | 基于规则的每步过程奖励 | 免费 | 全程 |
| Outcome Semantic Judge | EM=0时调用LLM判断同义词 | ~¥3.5/全程 | 全程 |
| Process Listwise Judge | LLM对tied group做排名 | ~¥13-17/全程 | 前500步 or 全程 |

## Component 1: Rule-based Step Reward

**文件**: `improvements/process_reward/step_reward.py`

4维过程奖励 + 格式惩罚：

```
R_total = R_outcome + λ_process × R_process

R_process = Σ_t [w1×检索相关性 + w2×信息新颖性 + w3×渐进覆盖] + 效率奖励
```

| 维度 | 范围 | 信号来源 | Anti-hack 机制 |
|------|------|---------|--------------|
| Retrieval Relevance | [0, 0.3]/step | 检索结果 vs 答案实体 | 外部检索系统，模型只能间接影响 |
| Information Novelty | [-0.15, 0.15]/step | 本次vs历史检索n-gram去重 | 改写query但结果相同仍被惩罚 |
| Progressive Coverage | [0, 0.2]/step | think中答案实体增量出现 | 只奖励增量，一步猜完不给分 |
| Efficiency | [0, 0.2] | 用了几轮+是否答对 | 仅outcome=1时生效 |
| Format Penalty | [-0.3, 0] | action格式是否合规 | 越早犯错惩罚越重 |

量级保证：过程奖励总量 max ~0.25 << 结果奖励 1.0

**配置参数**:
```python
process_reward_config = {
    'lambda_process': 0.5,   # 过程奖励总缩放
    'w_retrieval': 0.4,      # 检索相关性权重
    'w_novelty': 0.3,        # 信息新颖性权重
    'w_progress': 0.3,       # 渐进覆盖权重
    'max_turns': 6,
    'format_penalty': True,
    'efficiency_bonus': True,
}
```

## Component 2: Outcome Semantic Judge (LLM)

**文件**: `improvements/process_reward/llm_judge.py`

当 EM=0 但模型有有效答案时，调用 DeepSeek V4 Flash 判断语义等价性。

**动机**：纯 EM 无法处理同义词/格式差异：
- "NYC" vs "New York City" → EM=0, LLM judge → correct → reward=0.8
- "Feb 14" vs "February 14th" → EM=0, LLM judge → correct → reward=0.8

**设计**：
- 每步、每个 EM=0 且有有效 answer 的样本都调用
- 判正 → reward=0.8（略低于 EM=1.0，保留 exact match 微弱优势）
- 全程启用（不只前500步）

**Token 消耗**: ~24M tokens / ~¥3.5（全训练周期）

## Component 3: Process Listwise Judge (LLM)

**文件**: `improvements/process_reward/llm_judge.py`

对 tied group（outcome相同的组）做 listwise 排名，把96%被浪费的样本变为有效梯度。

**工作方式**:
```
同一 prompt 的 n_agent 条轨迹 → LLM listwise ranking → ranking → advantage

例: n_agent=4, group=[traj_A, traj_B, traj_C, traj_D], 全部 outcome=0
LLM judge 排名: B > D > A > C
→ advantages: A=-0.1, B=+0.3, C=-0.3, D=+0.1
→ GRPO 用这些 advantage 计算梯度（原本这个 group 会被 skip！）
```

**Prompt 设计**:
- 不用细粒度 rubric（反而限制了 LLM 的判断能力）
- 直接让 LLM 基于 search strategy quality + reasoning progress 做整体排名
- 输出 JSON: `{"ranking": [2, 4, 1, 3]}`

**调度策略**:
- 只对 tied groups 调用（mixed group 已有 outcome 信号）
- 可配置：全程 / 前500步 / 每N步一次 / 衰减

**Token 消耗估算（DeepSeek V4 Flash）**:

| 配置 | Total Tokens | 成本 |
|------|-------------|------|
| n_agent=4, 全量 (1413步) | 120M | ¥17 |
| n_agent=4, 前500步 | 92M | ¥13 |
| n_agent=4, 前500步/每3步 | 47M | ¥7 |
| n_agent=8, 前500步/每3步 | 37M | ¥5 |

## 实验计划

### 实验 A: PRM 前500步 + 衰减
- Step 0-500: 全量 process judge (每步)
- Step 500: **保存 checkpoint**
- Step 500-1413: process judge 每5步一次（衰减）

### 实验 B: PRM 全程贯穿
- Step 0-1413: 全量 process judge (每步)
- 从 Step 500 checkpoint 分叉

### 对比方式
- 两个实验从 step 500 checkpoint 分叉
- A 继续减少 PRM，B 保持全量
- 对比最终 EM score 和收敛曲线

## n_agent 分析

当前 n_agent=2。增大 n_agent 对 GRPO 的影响：

| n_agent | Group size | P(有效group) @2%准确率 | Listwise信号丰富度 |
|---------|-----------|----------------------|------------------|
| 2 | 2 | 4% | 只有 A>B or B>A |
| 4 | 4 | 7.8% | 4! = 24种排列 |
| 8 | 8 | 14.9% | 8! = 40320种排列 |

**n_agent 与 step 数量的关系**:

保持 total trajectories/step 不变（=192）：
- n_agent=2, batch=96: 96 prompts/step → 1 epoch = 883 steps
- n_agent=4, batch=48: 48 prompts/step → 1 epoch = 1767 steps
- n_agent=8, batch=24: 24 prompts/step → 1 epoch = 3534 steps

保持 batch=96 不变（增加 total trajectories/step）：
- n_agent=2: 192 traj/step → 每步 ~220s
- n_agent=4: 384 traj/step → 每步 ~440s（rollout 翻倍）
- n_agent=8: 768 traj/step → 每步 ~880s（rollout ×4）

建议：n_agent=4, batch_size=48（保持 192 traj/step，step 时间不变，总步数增加到 ~1767）

## 集成说明

见 `improvements/process_reward/integration_guide.py`

修改点：
1. `main_ppo.py::RewardManager._process_item()` — 加入规则reward + outcome judge
2. `ray_trainer.py` — 在 advantage 计算前加入 process judge 的排名结果
3. `run_improved.sh` — 添加 process reward 相关配置

## 文件清单

```
improvements/
├── README.md                          # 本文档
├── run_improved.sh                    # 改进版训练脚本
├── core_algos_original.py.bak         # 原始备份
├── dp_actor_original.py.bak           # 原始备份
├── core_algos_improved.py             # 改进后的 core_algos（参考）
└── process_reward/                    # Phase 2: 过程奖励系统
    ├── README.md                      # 过程奖励详细设计文档
    ├── step_reward.py                 # 规则过程奖励实现
    ├── llm_judge.py                   # LLM judge 系统（outcome + listwise）
    └── integration_guide.py           # 集成指南
```
