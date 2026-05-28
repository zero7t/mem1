# 奖励系统设计

## 1. 当前状态：纯 Outcome Reward

```
轨迹 → extract <answer> → exact_match(answer, ground_truth) → 0 or 1
                                    ↓
                    reward_tensor[last_token] = score
                                    ↓
                    GRPO advantage = score - group_mean
```

问题：
- 6轮交互只有最后一个信号
- Score ~2% 时，96% 的 group 全是 [0,0,0,0] → DAPO skip → 无梯度
- 无法区分"差一步就对"和"完全跑偏"

## 2. 设计目标

```
Total Reward = Outcome + Rule Process + LLM Judge
                 ↑           ↑              ↑
               当前在用      待启用         待启用
```

三层叠加，各自独立，可逐步开启。

## 3. Layer 1: Rule-based Process Reward（免费）

**文件**: `reward/rule_reward.py`

### 4个维度

#### 3.1 Retrieval Relevance（检索相关性）
```
检索结果是否包含答案实体？

输入: retrieved_text, answer_targets (ground truth)
计算: 答案实体在检索文本中的覆盖率
输出: [0, 0.3]

示例:
  question: "谁发明了电话？什么时候？"
  targets: ["Alexander Bell", "1876"]
  retrieved: "...Bell invented the telephone in 1876..."
  coverage = 2/2 = 1.0 → reward = 0.3
```

#### 3.2 Information Novelty（信息新颖性）
```
本次搜索是否带来新信息？

输入: current_query, current_retrieval, previous_queries, previous_retrievals
计算:
  - query 与历史 query 的 token overlap (>0.7 = 重复)
  - 检索结果与历史结果的 3-gram 去重率
输出: [-0.15, +0.15]

惩罚: 重复搜索同样的内容
奖励: 搜到了之前没有的新信息
```

#### 3.3 Progressive Coverage（渐进覆盖）
```
模型推理是否逐步接近答案？

输入: think_at_t, think_at_t_minus_1, answer_targets
计算: 当前think中覆盖的答案实体数 - 上一步覆盖数（增量）
输出: [0, 0.2]

关键: 只奖励增量，防止模型第一步就猜完所有答案
```

#### 3.4 Efficiency（效率）
```
正确答案用了几轮？

输入: num_turns_used, max_turns, is_correct
计算: 只有 outcome=1 时才给效率奖励
输出: [0, 0.2]

关键: 答错不给效率分（防止模型为了效率乱猜）
```

### 量级控制

```
R_process = λ × Σ [w1×relevance + w2×novelty + w3×coverage] + efficiency
         = 0.5 × (0.4×0.3 + 0.3×0.15 + 0.3×0.2) × ~3步 + 0.2
         ≈ max 0.25

Outcome reward = 1.0

→ 过程奖励永远 < 结果奖励
→ 完美 hack 过程奖励(+0.25) + 答错(0) = 0.25 < 答对(1.0)
→ 结果导向不会被破坏
```

## 4. Layer 2: LLM Outcome Judge（语义EM）

**文件**: `reward/llm_judge.py`

### 解决什么
```
Ground truth: "New York City"
Model answer: "NYC"           → EM=0, 但语义正确
Model answer: "纽约市"        → EM=0, 但语义正确
```

### 方法
```
对所有 EM=0 且有有效 answer 的样本:
  → 调用 DeepSeek V4 Flash
  → "Is 'NYC' semantically equivalent to 'New York City'?"
  → "correct" → reward = 0.8 (略低于 EM=1.0)
  → "incorrect" → reward = 0
```

### 全程启用
每步、每个样本都判（只要EM=0且有答案）。
成本: ~¥3.5/全训练周期

## 5. Layer 3: LLM Process Judge（Listwise 排名）

**文件**: `reward/llm_judge.py`

### 解决什么
```
Group [0, 0, 0, 0]: outcome reward 无法区分
  - Traj A: 搜了 "telephone inventor" → 找到相关文档 → 差一步就对
  - Traj B: 搜了 "hello world" → 完全无关 → 完全跑偏

当前: DAPO skip，两者都 advantage=0
改进: LLM 排名 A>B → advantage_A=+0.3, advantage_B=-0.3
```

### 方法
```
对同一 prompt 的 n_agent 条轨迹 (listwise):
  → 发送给 LLM: "排名这4条搜索轨迹，哪条搜索策略更好"
  → LLM 输出: {"ranking": [3, 1, 4, 2]}
  → 转化为 advantages: best=+0.3, worst=-0.3

最终 advantage = (1-α) × outcome_advantage + α × llm_rank_advantage
  α = 0.3-0.5
```

### 调用策略
- 所有 group 都调（不只 tied group），成本 ~¥40/全程
- 或只对 tied group 调，成本 ~¥17/全程

## 6. Anti-Hack 设计总结

| 可能的hack | 防御 |
|---|---|
| 重复搜索同一关键词 | novelty 维度惩罚 |
| 在 think 中直接猜答案 | coverage 只奖励增量 |
| 第一步就回答 | efficiency 只有 outcome=1 才给 |
| 搜无意义的内容凑 novelty | LLM judge 会给低排名 |
| 学会"看起来好"但没用的搜索 | retrieval relevance 用外部系统验证 |
| 过程满分但答错 | 过程奖励 max=0.25 < outcome=1.0 |

## 7. 集成到训练的改动

只需修改 `verl/trainer/main_ppo.py` 的 `RewardManager._process_item()`:

```python
# 现有代码
score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

# 新增: Rule Process Reward
from grpo_improved.reward.rule_reward import compute_process_reward
process_score = compute_process_reward(sequences_str, ground_truth, score)
score = score + process_score

# 新增: LLM Outcome Judge (当EM=0时)
if score == 0 and answer is not None:
    from grpo_improved.reward.llm_judge import outcome_judge
    if outcome_judge(question, ground_truth, answer) == "correct":
        score = 0.8
```

LLM Process Judge 集成点不同——需要在 `ray_trainer.py` 的 advantage 计算阶段插入。

## 8. 成本预算（DeepSeek V4 Flash）

| 组件 | Tokens | 成本 | 启用范围 |
|------|--------|------|---------|
| Rule Process Reward | 0 | ¥0 | 全程 |
| LLM Outcome Judge | ~24M | ~¥3.5 | 全程 |
| LLM Process Judge (tied only) | ~93M | ~¥13 | 前500步 or 全程 |
| LLM Process Judge (全量) | ~284M | ~¥40 | 全程 |
| **推荐组合** | ~120M | **~¥17** | 全程 |
