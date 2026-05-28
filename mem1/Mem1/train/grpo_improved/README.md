# GRPO Improved - Multi-Turn RAG QA Training System

## 版本

- **V1** (当前运行): 标量过程奖励 + 异步LLM Judge
- **V2** (新增): Turn-level过程奖励 + Turn-weighted advantage + 生成时LLM Judge

## 文件结构

```
grpo_improved/
├── __init__.py
├── main_ppo_v2.py              # V2 训练入口（自包含，不改verl/）
├── core/
│   ├── __init__.py
│   ├── turn_weighted_advantage.py   # Turn-level advantage 调制
│   ├── generation_judge.py          # 生成时 LLM Judge（0s额外开销）
│   ├── actor_update.py              # 参考：LLDS-MA + DAPO (dp_actor.py副本)
│   └── grpo_algos.py               # 参考：Dr.GRPO + DAPO (core_algos.py副本)
├── reward/
│   ├── __init__.py
│   ├── rule_reward_v2.py           # V2: Per-turn过程奖励 + 利用度信号
│   ├── rule_reward.py              # V1: 标量过程奖励（4维度）
│   └── llm_judge.py               # LLM Outcome + Process Judge
├── scripts/
│   ├── train_v2.sh                 # V2 训练脚本
│   └── train.sh                    # V1 训练脚本
└── docs/
    ├── v2_design.md               # V2 完整设计文档
    ├── algorithm_design.md
    └── reward_design.md
```

## 快速使用

### V2 训练（推荐）
```bash
cd /root/paddlejob/workspace/mem1/MEM1/Mem1/train
bash grpo_improved/scripts/train_v2.sh
```

### V1 训练（当前）
```bash
bash grpo_improved/scripts/train.sh
```

## V2 核心改进

### 1. Per-Turn 过程奖励 (rule_reward_v2.py)

V1 返回标量，V2 返回每轮分数 `[r_0, r_1, ..., r_T]`：

| 维度 | 权重 | 含义 |
|------|------|------|
| Hit | 0.3 | 检索是否命中答案实体 |
| **Utilization** | **0.4** | 检索是否被后续推理/答案引用（**新增**） |
| Novelty | 0.2 | 是否带来非冗余信息 |
| Efficiency | 0.1 | 答对时用更少turn的奖励 |

**Utilization 解决的问题：**
- Turn 1 不再天然占优（搜到了但没用 → 低分）
- 散点覆盖 vs 系统覆盖可区分（用了的信息才算分）

### 2. Turn-Weighted Advantage (turn_weighted_advantage.py)

标准GRPO所有token同一个advantage。V2按turn质量调制：

```
w_t = 1 + 0.3 × sign(A_i) × normalize(turn_score_t)
```

效果：
- 好轨迹中好turn → 放大正梯度
- 好轨迹中差turn → 减弱正梯度
- 差轨迹中好turn → **保护**（减轻惩罚）
- 差轨迹中差turn → 加重惩罚

### 3. 生成时 LLM Judge (generation_judge.py)

轨迹完成时立即发API，不等generation结束：
- 大多数轨迹Turn 1-2完成（~20s），到generation结束（111s）已返回
- QPS=20，额外等待时间≈0s

## 与 verl 框架的关系

V2 设计为**最小侵入**：

| 需要改动的verl文件 | 改动内容 | 行数 |
|---|---|---|
| `ray_trainer.py` | 在 compute_advantage 后加一行 `apply_turn_weighting_to_batch` | +1行 |
| `generation_think.py` | 在 dones[i]=True 时调用 `judge.on_trajectory_complete()` | +5行 |

其余所有逻辑都在 `grpo_improved/` 内部。

## 算法栈（V2完整）

```
Dr.GRPO (mean-only baseline, no std normalization)
  + DAPO Dynamic Sampling (skip zero-variance groups)
  + DAPO Clip-Higher (asymmetric clip: 0.2/0.28)
  + LLDS-MA (protect good-trajectory token likelihood)
  + Turn-Weighted Advantage (per-turn credit assignment)  ← NEW
  + Rule Process Reward V2 (utilization signal)           ← NEW
  + LLM Outcome Judge (semantic EM, generation-time)      ← IMPROVED
```

## 测试

```bash
cd /root/paddlejob/workspace/mem1/MEM1/Mem1/train

# 测试 per-turn 过程奖励
/root/paddlejob/workspace/miniforge3/envs/mem1/bin/python -m grpo_improved.reward.rule_reward_v2

# 测试 turn-weighted advantage
/root/paddlejob/workspace/miniforge3/envs/mem1/bin/python -m grpo_improved.core.turn_weighted_advantage
```

## 详细设计文档

→ [docs/v2_design.md](docs/v2_design.md)
