# GRPO Improved Training Scripts

## 目录结构

```
grpo_improved/
├── scripts/
│   ├── train.sh          # V1: 基础改进版（当前主力）
│   ├── train_v2.sh       # V2: Turn-Weighted Advantage（开发中）
│   └── README.md         # 本文档
├── reward/
│   ├── rule_reward.py    # 规则过程奖励（4维度）
│   ├── rule_reward_v2.py # V2过程奖励（含utilization signal）
│   └── llm_judge.py      # LLM Outcome Judge（语义EM）
├── core/
│   ├── turn_weighted_advantage.py  # Turn-Weighted Advantage
│   └── generation_judge.py         # 生成时LLM Judge
├── main_ppo_v2.py        # V2独立入口（开发中）
└── __init__.py
```

## V1 vs V2 对比

### V1 (train.sh) — 当前主力
- **入口**: `python -m verl.trainer.main_ppo`（复用 verl/ 原始框架）
- **算法**: Dr.GRPO + DAPO Clip-Higher(0.28) + LLDS-MA(0.1)
- **奖励**: EM + Rule Process Reward(4维) + Gen-Time LLM Judge
- **优势计算**: 标准 GRPO（scalar advantage tiled to all tokens）
- **状态**: 稳定运行中

### V2 (train_v2.sh) — 开发中
- **入口**: `python -m grpo_improved.main_ppo_v2`（独立入口）
- **算法**: 同V1
- **奖励**: 同V1 + Utilization Signal
- **优势计算**: Turn-Weighted Advantage（per-turn modulation）
- **额外参数**:
  - `reward.turn_weight_alpha=0.3` — 调制强度
  - `reward.turn_weight_clip=1.0` — 权重裁剪
  - `reward.lambda_process=0.5` — 过程奖励系数
- **状态**: 需要配置 hydra config 目录，暂未就绪

## 启动方式

### V1
```bash
cd /root/paddlejob/workspace/mem1/MEM1/Mem1/train
bash grpo_improved/scripts/train.sh
# 日志: logs/train_v1_fixed.log
```

### V2（待完善）
```bash
cd /root/paddlejob/workspace/mem1/MEM1/Mem1/train
bash grpo_improved/scripts/train_v2.sh
# 日志: logs/train_v2.log
```

## 隔离原则

**V1 和 V2 严格隔离，互不影响：**

1. **代码隔离**: V1 修改 `verl/trainer/main_ppo.py` 和 `verl/trainer/ppo/ray_trainer.py`；V2 使用独立的 `grpo_improved/main_ppo_v2.py`
2. **更新隔离**: 之后更新时分开更新，V1 的改动不影响 V2，反之亦然
3. **公用代码**: 如果 `grpo_improved/reward/` 或 `grpo_improved/core/` 中的公用代码需要修改，应该：
   - V1 用 `rule_reward.py`，V2 用 `rule_reward_v2.py`
   - 各自开副本，不修改对方依赖的文件
4. **实验复现**: 每个版本的配置完全自包含在对应的 `.sh` 脚本中，可严格复现

## 关键指标说明

| 指标 | 含义 |
|------|------|
| `reward/em_score_mean` | Exact Match 平均分 |
| `reward/process_reward_mean` | 规则过程奖励平均值 |
| `reward/process_reward_nonzero_ratio` | 获得非零过程奖励的轨迹比例 |
| `judge/gen_time_judge_fired` | 生成时LLM Judge发出的请求数 |
| `judge/gen_time_judge_upgraded` | 被LLM Judge升级的轨迹数 |
| `critic/score/mean` | 最终奖励均值（EM + process + judge upgrade） |

## 过程奖励维度（V1）

1. **Retrieval Relevance (0.4)**: 检索结果是否包含答案相关信息
2. **Information Novelty (0.3)**: 新搜索是否带来非冗余信息
3. **Progressive Coverage (0.3)**: think中是否逐步覆盖更多答案要素
4. **Efficiency Bonus**: 正确答案用更少turn完成时的奖励
5. **Format Penalty**: 无效action的惩罚

## DAPO 改进

- **Clip-Higher**: `clip_higher=0.28`（标准PPO=0.2），正advantage样本有更大ratio空间
- **Dynamic Sampling**: 跳过组内方差为0的group（全对/全错无梯度信号）
- **Dr.GRPO**: Mean-only baseline，不除以std（避免低方差group被过度放大）

## LLM Judge 系统

### 架构概览

LLM Judge 分为三个独立组件，可分别启用/禁用：

```
┌─────────────────────────────────────────────────────────┐
│ 1. Gen-Time Outcome Judge（生成时触发，0额外等待）         │
│    - 轨迹完成时立即发 API，与生成重叠                      │
│    - 判断 EM=0 的样本是否语义正确，升级 reward 到 0.8      │
│    - 代码: grpo_improved/core/generation_judge.py         │
│    - 集成点: ray_trainer.py (generation loop)             │
│    - 状态: ✅ 已启用                                      │
├─────────────────────────────────────────────────────────┤
│ 2. Pre-Fire Outcome Judge（compute_log_prob 前触发）      │
│    - Gen-Time Judge 的 fallback                          │
│    - 如果 Gen-Time Judge 已激活则自动跳过                  │
│    - 代码: verl/trainer/main_ppo.py (pre_fire_llm_judge) │
│    - 状态: ✅ 已启用（自动降级为 fallback）                │
├─────────────────────────────────────────────────────────┤
│ 3. Process Judge（Listwise Ranking，advantage 后触发）    │
│    - 对同 question 的 4 条轨迹做整体排序                   │
│    - 排名转为 [-0.3, +0.3] 的 advantage 加成              │
│    - 代码: grpo_improved/reward/llm_judge.py             │
│    - 集成点: ray_trainer.py (after compute_advantage)    │
│    - 状态: ⚠️ 暂时禁用（额外 160s/step，待优化）          │
│    - 备份: grpo_improved/llm_judge_backup/               │
└─────────────────────────────────────────────────────────┘
```

### Gen-Time Outcome Judge 实现细节

**核心思路**: 在轨迹生成阶段，每条轨迹完成时（`dones[i]=True`）立即发起 LLM API 调用判断语义正确性。由于大部分轨迹在 1-2 turn 内完成（~20s），到生成结束时（~110s）API 结果已经返回，实现 0 额外等待。

**关键修改**:
- `verl/trainer/ppo/ray_trainer.py`: 生成前创建 `GenerationTimeJudge`，生成后收集结果
- `rollout/llm_agent/generation_think.py`: 在 `dones[i]=True` 时调用 `_gen_judge.on_trajectory_complete()`
- `verl/trainer/main_ppo.py`: `RewardManager.__call__()` 中应用 upgraded indices

**性能**: fired=370-375/step, elapsed=127-148s（完全与生成重叠），0s 额外等待

### Process Judge (Listwise Ranking) 实现细节

**核心思路**: 对 GRPO 中同一 question 的 n_agent=4 条轨迹，让 LLM 做整体排序，排名转为 zero-mean advantage 叠加到原始 advantage 上。

**评估维度**（prompt 中指定）:
1. Search Strategy: query 是否有针对性
2. Information Usage: 是否利用了检索到的信息
3. Progress toward Answer: 是否逐步接近正确答案
4. Efficiency: 是否避免冗余搜索

**LLM 输出**: `{"ranking": [3, 1, 2, 4]}`
**Advantage 映射**: 排名第1→+0.3, 第2→+0.1, 第3→-0.1, 第4→-0.3

**关键实现细节**:
- `_balance_batch()` 会打乱 batch 顺序，需要用 `uid` 字段恢复 group 结构
- `uid = np.arange(batch_size) // n_agent` 在 repeat 前设置，确保同 question 的轨迹有相同 uid
- `response_mask = attention_mask[:, -response_length:]` 用于 advantage 加成的 mask
- API 并发 20，96 groups 约需 ~160s（主要瓶颈）

**暂时禁用原因**: 每步额外 160s（+32%），需要优化（异步化/降频/只对 tied groups 做）

### 启用/禁用 Process Judge

禁用（当前状态）: 在 `ray_trainer.py` 中将 Process Judge 代码块注释掉或设置条件跳过
启用: 恢复 `grpo_improved/llm_judge_backup/ray_trainer_with_pjudge.py` 中的相关代码段

### 已修复的 Bug

1. **Process Reward numpy 格式**: `ground_truth.target` 是嵌套 numpy array，需要展平为 `List[str]`
2. **uid 分组**: `non_tensor_batch['index']` 全为 0（数据中无 index 字段），改用 `np.arange // n_agent`
3. **dtype=object**: `non_tensor_batch` 要求所有值为 `dtype=object` 的 numpy array
4. **response_mask**: advantages shape 是 `(bs, response_length)`，不能用 `attention_mask`（含 prompt）
5. **numpy truth value**: `if reward_info` 对 numpy array 报错，改为 `if reward_info is not None`

