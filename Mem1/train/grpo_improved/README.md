# GRPO 改进版训练系统

## 文件结构

```
grpo_improved/
├── README.md                    # 本文档
├── core/                        # 核心算法（改动的源码）
│   ├── grpo_algos.py           # GRPO 算法：advantage计算 + policy loss + LLDS
│   └── actor_update.py         # Actor 策略更新：forward/backward + LLDS集成
├── reward/                      # 奖励系统
│   ├── rule_reward.py          # 规则过程奖励（4维，免费）
│   └── llm_judge.py            # LLM Judge（outcome语义 + listwise排名）
├── scripts/                     # 训练脚本
│   └── train.sh                # 启动训练（最新配置）
└── docs/                        # 设计文档
    ├── algorithm_design.md     # 算法设计详解
    └── reward_design.md        # 奖励设计详解
```

## 与 verl 框架的关系

verl 框架用 `ppo/` 命名，但我们实际跑的是 **GRPO**（无 critic/value network）。

对应关系：

| 本目录 | verl 框架中的位置 | 说明 |
|--------|-------------------|------|
| `core/grpo_algos.py` | `verl/trainer/ppo/core_algos.py` | **实际运行的文件** |
| `core/actor_update.py` | `verl/workers/actor/dp_actor.py` | **实际运行的文件** |
| `reward/rule_reward.py` | 待集成到 `verl/trainer/main_ppo.py` | 待启用 |
| `reward/llm_judge.py` | 待集成到 `verl/trainer/main_ppo.py` | 待启用 |
| `scripts/train.sh` | 独立启动脚本 | 直接使用 |

> ⚠️ **注意**：`core/` 下的文件是从实际运行位置复制过来的**参考副本**。
> 真正被训练进程加载的仍然是 `verl/trainer/ppo/core_algos.py` 和 `verl/workers/actor/dp_actor.py`。
> 修改算法时，改这里做参考，确认后再同步到 verl 目录。

## 当前状态

### ✅ 已启用（正在跑的训练）

| 改进 | 位置 | 效果 |
|------|------|------|
| Dr.GRPO | `core/grpo_algos.py` → `compute_grpo_outcome_advantage()` | advantage = reward - mean，不除std |
| DAPO Dynamic Sampling | `core/grpo_algos.py` → `compute_grpo_outcome_advantage()` | std<ε的group跳过 |
| DAPO Clip-Higher | `core/grpo_algos.py` → `compute_policy_loss()` | 正advantage用clip=0.28 |
| LLDS-MA | `core/grpo_algos.py` → `compute_llds_loss()` | 防止好轨迹likelihood塌陷 |

### ⏳ 待启用（下次重启生效）

| 改进 | 位置 | 配置变更 |
|------|------|---------|
| n_agent=4 | `scripts/train.sh` | 每prompt生成4条轨迹（原来2条） |
| warmup=0.02 | `scripts/train.sh` | 28步warmup（原来70步） |
| Rule Process Reward | `reward/rule_reward.py` | 集成到main_ppo.py |
| LLM Outcome Judge | `reward/llm_judge.py` | EM=0时语义判断 |
| LLM Process Judge | `reward/llm_judge.py` | tied group listwise排名 |

## 快速使用

### 启动训练
```bash
cd /root/paddlejob/workspace/mem1/MEM1/Mem1/train
nohup bash grpo_improved/scripts/train.sh > /root/paddlejob/workspace/mem1/MEM1/logs/grpo_improved.log 2>&1 &
```

### 修改算法流程
```
1. 在 grpo_improved/core/ 中修改和测试
2. 确认无误后，复制到 verl/ 对应位置：
   cp grpo_improved/core/grpo_algos.py verl/trainer/ppo/core_algos.py
   cp grpo_improved/core/actor_update.py verl/workers/actor/dp_actor.py
3. 重启训练
```

### 集成新奖励
```
1. 在 grpo_improved/reward/ 中开发
2. 修改 verl/trainer/main_ppo.py 的 RewardManager 调用新奖励
3. 重启训练
```
