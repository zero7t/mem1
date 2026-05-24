# 多轮 RAG 场景的过程奖励设计

## 1. 当前系统分析

### 1.1 Reward Pipeline 现状

```
generation_think.py::run_llm_loop()
  → execute_predictions() → rewards.append(0)  # 每步都是0
  → batch_rewards 累加 → meta_info['batch_rewards']

main_ppo.py::RewardManager.__call__()
  → compute_score_em() → exact_match(extracted_answer, ground_truth)
  → score + batch_rewards → reward_tensor[i, last_valid_token] = score
  → 单一标量放在最后一个 token 上
```

**问题**：6 轮多步交互，只有终端一个 0/1 信号。

### 1.2 轨迹结构

```
Turn 0: <think>推理</think><search>query_0</search>  →  <information>检索结果_0</information>
Turn 1: <think>推理</think><search>query_1</search>  →  <information>检索结果_1</information>
Turn 2: <think>推理</think><search>query_2</search>  →  <information>检索结果_2</information>
...
Turn N: <think>推理</think><answer>final_answer</answer>
```

### 1.3 可用信号

训练时可获取的信息：
- `ground_truth['target']`: 标准答案列表（如 `["New York City", "1977"]`）
- 每一步的 search query（从 response 中提取）
- 每一步的检索结果（`<information>` 内容）
- 每一步的 think 内容
- 轨迹终止方式（正常 answer / 格式错误 / 超时）

---

## 2. 设计原则

### 2.1 Anti-Hack 原则

| 原则 | 含义 | 实例 |
|------|------|------|
| **必要不充分** | 过程奖励应是成功的必要条件，但单独不足以得分 | "检索到相关文档"必要但不充分（还需正确推理） |
| **不可捷径** | 模型不能通过简单策略 hack 过程奖励 | 如果奖励"query长度>3"，模型会乱凑长query |
| **单调性** | 过程奖励方向应与最终成功正相关 | 好的过程不应反而降低最终成功率 |
| **可验证** | 基于客观可计算的信号，非主观判断 | overlap 计算 > LLM 打分 |

### 2.2 计算约束

- 不引入额外 LLM 调用（太慢、太贵）
- 不引入额外模型（如单独的 PRM 模型）
- 基于规则和文本匹配计算
- 对训练速度影响 < 1%

---

## 3. 多维过程奖励设计

### 总公式

```
R_total = R_outcome + λ_process * R_process

R_process = Σ_{t=0}^{N} [ w1*R_retrieval(t) + w2*R_novelty(t) + w3*R_progress(t) ] + R_efficiency
```

建议超参：
- `λ_process = 0.5`（过程奖励总权重，不超过结果奖励）
- `w1 = 0.4, w2 = 0.3, w3 = 0.3`（各维度相对权重）

---

### 3.1 维度一：Retrieval Relevance（检索相关性）

**直觉**：好的 search query 应该检索到包含答案相关信息的文档。

**计算方法**：

```python
def retrieval_relevance_reward(retrieved_text: str, answer_entities: List[str]) -> float:
    """
    检查检索结果是否包含答案相关实体。

    Args:
        retrieved_text: 本轮检索返回的文档内容
        answer_entities: ground truth 答案中的关键实体

    Returns:
        reward in [0, 0.3]
    """
    if not retrieved_text or not answer_entities:
        return 0.0

    retrieved_lower = normalize_answer(retrieved_text)

    # 计算实体覆盖率
    covered = 0
    for entity in answer_entities:
        entity_normalized = normalize_answer(entity)
        if entity_normalized in retrieved_lower:
            covered += 1

    coverage = covered / len(answer_entities)
    return 0.3 * coverage
```

**为什么不易 hack**：
- 检索是由外部系统（retriever）执行的，模型无法直接控制返回结果
- 模型唯一能做的是写出好的 query 来获取相关文档
- 随机/无意义的 query 不会检索到答案相关实体

**局限**：
- 需要从 ground_truth 提取实体（对我们的数据集，answer 本身就是实体）
- 某些情况下，间接相关的文档也很有价值但不包含答案实体

---

### 3.2 维度二：Information Novelty（信息新颖性）

**直觉**：每次搜索应该带来新信息，重复搜索是浪费。

**计算方法**：

```python
def information_novelty_reward(
    current_retrieval: str,
    previous_retrievals: List[str],
    current_query: str,
    previous_queries: List[str]
) -> float:
    """
    奖励新颖的搜索，惩罚重复的搜索。

    Returns:
        reward in [-0.15, 0.15]
    """
    if not current_retrieval:
        return 0.0

    # 1. Query 多样性检查
    if previous_queries:
        max_query_overlap = max(
            compute_token_overlap(current_query, prev_q)
            for prev_q in previous_queries
        )
        if max_query_overlap > 0.7:  # 与之前某个 query 高度重复
            return -0.15  # 惩罚重复搜索

    # 2. 检索结果新颖性
    if previous_retrievals:
        # 计算新检索结果与已有信息的去重率
        current_ngrams = set(get_ngrams(current_retrieval, n=3))
        previous_ngrams = set()
        for prev in previous_retrievals:
            previous_ngrams.update(get_ngrams(prev, n=3))

        if not current_ngrams:
            return 0.0

        novelty = len(current_ngrams - previous_ngrams) / len(current_ngrams)

        if novelty < 0.2:  # 80%+ 内容重复
            return -0.1
        elif novelty > 0.5:  # 50%+ 是新内容
            return 0.15
        else:
            return 0.05

    return 0.1  # 第一次搜索默认有新颖性
```

**为什么不易 hack**：
- 惩罚重复 = 模型不能通过反复搜索同样的内容来获取奖励
- 新颖性基于 n-gram overlap，不是 query 表面形式
  （模型改写 query 但检索到同样内容仍会被惩罚）

---

### 3.3 维度三：Progressive Coverage（渐进覆盖）

**直觉**：对于多跳题目（需要多个 fact），每一步应该增加已覆盖的事实数量。

**计算方法**：

```python
def progressive_coverage_reward(
    think_content_at_t: str,
    think_content_at_t_minus_1: str,
    answer_targets: List[str]
) -> float:
    """
    检查模型的推理是否在逐步覆盖答案的各个部分。

    对于多答案题目 (如 "answer1; answer2")，检查模型在推理中
    是否逐步提到了各个答案的相关内容。

    Returns:
        reward in [0, 0.2]
    """
    if not think_content_at_t or not answer_targets:
        return 0.0

    # 当前步覆盖的答案数
    covered_now = sum(
        1 for target in answer_targets
        if normalize_answer(target) in normalize_answer(think_content_at_t)
        or any(word in normalize_answer(think_content_at_t)
               for word in normalize_answer(target).split() if len(word) > 3)
    )

    # 上一步覆盖的答案数
    covered_prev = 0
    if think_content_at_t_minus_1:
        covered_prev = sum(
            1 for target in answer_targets
            if normalize_answer(target) in normalize_answer(think_content_at_t_minus_1)
            or any(word in normalize_answer(think_content_at_t_minus_1)
                   for word in normalize_answer(target).split() if len(word) > 3)
        )

    # 只奖励增量覆盖
    delta = max(0, covered_now - covered_prev)
    return 0.2 * (delta / len(answer_targets))
```

**为什么不易 hack**：
- 只奖励**增量**，不奖励绝对值
  → 模型不能在第一步就把所有答案猜出来（那样 delta=0 之后每步）
- 基于 think 内容中的实体出现，不是最终答案格式
  → 模型需要真正在推理中提到答案相关信息
- 需要配合检索（think 中的信息来自检索结果）
  → 形成 query→retrieval→think 的闭环

---

### 3.4 维度四：Efficiency Bonus（效率奖励）

**直觉**：用更少的搜索步骤正确回答 = 更好。

```python
def efficiency_reward(num_turns_used: int, max_turns: int, is_correct: bool) -> float:
    """
    正确答案 + 更少步数 = bonus。错误答案不给效率奖励。

    Returns:
        reward in [0, 0.2]
    """
    if not is_correct:
        return 0.0  # 绝不奖励"快速错误回答"

    # 正确 + 高效
    saved_turns = max_turns - num_turns_used
    return 0.2 * (saved_turns / max_turns)
```

**为什么不易 hack**：
- 只有在 outcome reward = 1 时才给效率奖励
- 模型不能通过"立即回答"来获取效率奖励（因为大概率答错）
- 自然鼓励模型"够了就停"

---

### 3.5 格式惩罚（现有机制增强）

当前 `execute_predictions` 里的格式惩罚被注释掉了。建议启用轻量版：

```python
def format_penalty(is_valid_action: bool, cur_step: int) -> float:
    """
    格式错误惩罚。轻量级，只惩罚明显的格式违规。

    Returns:
        reward in [-0.3, 0]
    """
    if is_valid_action:
        return 0.0

    # 越早犯格式错误惩罚越重（浪费整条轨迹）
    if cur_step == 0:
        return -0.3
    elif cur_step <= 2:
        return -0.2
    else:
        return -0.1
```

---

## 4. 奖励放置策略

### 4.1 当前方式（Outcome Only）

```
Token positions:  [...response tokens...][LAST]
Reward:           [0, 0, 0, ..., 0, 0,    R]     # 只在最后
```

### 4.2 改进方式（Step-Level Placement）

每轮的过程奖励放在该轮 response 的最后一个 token 上：

```
Turn 0: [think][search][EOS_turn0]  [info_0]
         0 ... 0  ... R_process_0    0...0

Turn 1: [think][search][EOS_turn1]  [info_1]
         0 ... 0  ... R_process_1    0...0

Turn N: [think][answer][EOS_final]
         0 ... 0  ...  R_outcome + R_process_N + R_efficiency
```

**关键**：过程奖励放在每轮 response 的末尾（`</search>` 或 `</answer>` 位置），
info tokens 位置仍然是 0（因为 info 不参与 loss）。

### 4.3 与 GRPO 的兼容性

GRPO 的 advantage 计算使用 `token_level_rewards.sum()` 作为 per-sample score。
步骤级奖励不改变这个 sum（各步奖励累加 = 总过程奖励），
但放置位置会影响 token-level 梯度权重。

两种策略：
- **A) 仍然 sum 到最后**：过程奖励也累加到 last token（与现有代码完全兼容）
- **B) 分散放置**：每步奖励放在该步末尾（需要修改 reward_tensor 构造）

**建议从 A 开始**（最小改动），验证有效后再试 B。

---

## 5. 实现接口

### 5.1 核心函数签名

```python
class StepRewardComputer:
    """多轮 RAG 步进级过程奖励计算器"""

    def __init__(self, config: dict):
        self.lambda_process = config.get('lambda_process', 0.5)
        self.w_retrieval = config.get('w_retrieval', 0.4)
        self.w_novelty = config.get('w_novelty', 0.3)
        self.w_progress = config.get('w_progress', 0.3)

    def compute_trajectory_reward(
        self,
        trajectory: dict,       # 轨迹信息（queries, retrievals, thinks, answer）
        ground_truth: dict,     # 标准答案
        outcome_score: float,   # EM score (0 or 1)
    ) -> float:
        """
        计算单条轨迹的总 reward = outcome + process。

        Returns:
            total_reward: float
        """
        ...
```

### 5.2 集成点

修改 `main_ppo.py::RewardManager._process_item()`：

```python
def _process_item(self, i, data_item, ...):
    # ... 现有的 EM score 计算 ...
    outcome_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

    # [NEW] 过程奖励
    trajectory_info = extract_trajectory_info(sequences_str)  # 提取每步的 query/retrieval/think
    process_score = self.step_reward_computer.compute_trajectory_reward(
        trajectory=trajectory_info,
        ground_truth=ground_truth,
        outcome_score=outcome_score,
    )

    score = outcome_score + process_score
    return i, valid_response_length - 1, score
```

---

## 6. Anti-Hack 分析

### 6.1 可能的 Hack 行为及防御

| 可能的 Hack | 防御机制 |
|---|---|
| 生成很多无意义搜索以获取 novelty 奖励 | novelty 只有 0.15/步，且需要真正检索到新内容 |
| 在 think 中直接猜测答案以获取 coverage 奖励 | coverage 只奖励增量；且没有检索支撑的猜测大概率答错，outcome=0 |
| 第一步就直接回答以获取 efficiency 奖励 | efficiency 仅在 outcome=1 时生效；直接猜大概率错 |
| 重复搜索同一关键词 | novelty 维度惩罚 query 重复和结果重复 |
| 搜索答案原文以 hack retrieval relevance | 这恰好是正确行为！如果模型能搜到答案，说明 query 质量高 |

### 6.2 奖励量级设计

```
Outcome reward:   0 或 1（对于多答案题可能是 0/1/2）
Process reward:   最多 ~0.5 per trajectory
  - retrieval:    最多 0.3/step × ~3 steps = ~0.9, weighted ×0.4 = 0.36
  - novelty:      最多 0.15/step × ~3 steps = 0.45, weighted ×0.3 = 0.135
  - progress:     最多 0.2/step × ~3 steps = 0.6, weighted ×0.3 = 0.18
  - efficiency:   最多 0.2
  Total max: ~0.5 × λ_process(0.5) = ~0.25 additional reward
```

**关键设计**：过程奖励总量 << 结果奖励。
即使模型完美 hack 了所有过程奖励（~0.25），如果最终答错（outcome=0），
它的总 reward 仍然低于一个答对但过程奖励为 0 的轨迹（outcome=1）。

**这确保了结果导向不被破坏。**

---

## 7. 预期效果

### 7.1 训练初期（score < 5%）

- **当前**：98% 轨迹 reward=0，几乎无梯度信号
- **改进后**：即使答错，好的搜索策略也能获得 ~0.1-0.2 的过程奖励
- **效果**：GRPO 的有效 group 大幅增加（不再是 4% 的稀疏信号）

### 7.2 训练中期（score 5-30%）

- **当前**：梯度信号逐步增多，但不区分"差一步就对"和"完全跑偏"
- **改进后**：接近正确的轨迹获得更高总 reward，梯度方向更精确

### 7.3 训练后期（score > 30%）

- **当前**：收敛但可能陷入 pattern（总是搜 N 次然后答）
- **改进后**：efficiency 奖励鼓励"够了就停"，减少冗余搜索

---

## 8. 分阶段实施计划

### Phase 1: Minimal（最小改动，立即可跑）

只加 `format_penalty` + `efficiency_reward`。
修改 `execute_predictions` 启用格式惩罚 + `_process_item` 加效率奖励。
不需要解析轨迹细节。

### Phase 2: Core（核心过程奖励）

加入 `retrieval_relevance` + `information_novelty`。
需要在 `_process_item` 中解析出每步的 query 和 retrieval 文本。

### Phase 3: Full（完整框架）

加入 `progressive_coverage`。
需要解析每步 think 内容，计算答案覆盖率增量。

---

## 9. 配置参数

```bash
# 新配置项
+process_reward.enabled=true
+process_reward.lambda_process=0.5
+process_reward.w_retrieval=0.4
+process_reward.w_novelty=0.3
+process_reward.w_progress=0.3
+process_reward.format_penalty=true
+process_reward.efficiency_bonus=true
```

---

## 10. 对 ECHO-Mem 的借鉴与取舍

| ECHO-Mem 概念 | 本设计的对应/取舍 |
|---|---|
| Step-Level PRM | ✅ 我们的多维过程奖励（规则基，无需额外模型） |
| 未折叠惩罚 | ✅ 对应 efficiency_reward（鼓励及时停止） |
| 越界/幻觉惩罚 | ✅ 对应 format_penalty + novelty（惩罚无效行为） |
| 熵驱动认知奖励 | ⚠️ 简化为 information_novelty（新信息 = 高熵） |
| 有效回调奖励 | ⚠️ 简化为 retrieval_relevance（检索成功 = 有效回调） |
| 自适应记忆加载 | ❌ 不需要（6轮，上下文不长） |
| Context Folding | ❌ 不需要（总 token < 5k） |
| SUPO 轨迹分割 | ❌ 不需要（轨迹不超长） |

**核心取舍**：ECHO-Mem 的 Step-Level PRM 理念很好，但我们用规则替代模型，
用文本 overlap 替代 LLM 打分，保持计算零开销。
