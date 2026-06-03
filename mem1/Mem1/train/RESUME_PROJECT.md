# MEM1：基于强化学习的多轮 RAG 推理大模型训练系统

## 项目背景

基于 Search-R1/veRL 框架，使用 GRPO 算法在 6×A800 80GB 上训练 Qwen2.5-7B，使其通过 RL 自主习得多轮"思考→检索→推理→回答"能力（最多 6 轮交互）。原始框架设计面向单轮 PPO 场景，直接用于多轮多 GPU GRPO 训练时存在大量工程和算法问题。

技术栈：PyTorch FSDP / vLLM 0.6.3 / Ray / NCCL / Hydra / SwanLab

---

## 第一部分：分布式训练系统优化

---

### 1.1 FSDP 多轮生成死锁修复

**Motivation**：veRL 的多轮生成需要在每轮调用 `generate_sequences` 时通过 `rollout_sharding_manager` 进出 FSDP sharding 状态。多轮场景使用 `keep_generation_mode=True` 标志让 sharding manager 在轮间保持打开状态，避免反复 sync。但当某些 worker 的 batch 大小不对齐时，会走 GPU padding 路径创建新的 `padded_active_batch`——这个新 batch **丢失了** `keep_generation_mode` 标志。

**问题链路**：
```
Worker 0: batch_size=5, 需要 padding 到 6
→ 创建 padded_active_batch（meta_info 为空）
→ keep_generation_mode 丢失
→ generate 结束后 sharding_manager.__exit__() 被调用
→ 触发 FSDP state_dict() all-gather 集合通信

Worker 1-5: batch_size=6, 不需要 padding
→ keep_generation_mode=True 保留
→ sharding_manager 保持打开，不参与 all-gather

→ NCCL 死锁：Worker 0 在等其他人一起做 all-gather，其他人不知道要做
```

**修复**：在 padding 路径中将 `keep_generation_mode=True` 显式写入 padded batch 的 meta_info。

**效果**：6 GPU 多轮生成不再因 worker 间 sharding 状态不一致而死锁。

---

### 1.2 Batch Size 整除截断导致 ZeroDivisionError

**Motivation**：veRL 的 FSDP worker 初始化时会将全局 batch size 除以 GPU 数来计算每个 worker 的本地 batch size：

```python
self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] // ulysses_sp_size)
```

**问题**：如果用户设置 `ppo_micro_batch_size=2`，在 6 GPU 下：`2 // 6 = 0`。后续 actor update 中用这个值做 batch 切分时触发除零错误。这不是显而易见的 bug——用户设置的是"全局"batch size，框架内部的整除逻辑是隐式的。

**修复**：确保所有 `*_batch_size` 配置值 ≥ n_gpus_per_node，建立约束文档。

---

### 1.3 compute_log_prob 路径 OOM（跳过冗余 Entropy 计算）

**Motivation**：`compute_log_prob` 只需要 log_probs，不需要 entropy。但底层 `_forward_micro_batch` 函数无条件计算 entropy：

```python
entropy = entropy_from_logits(logits)  # softmax over vocab_size=152064
```

多轮生成后序列可达 ~6000 tokens，单个样本的 entropy softmax 需要分配：
```
float32 × (1, 6000, 152064) = 13.31 GB
```

而 vLLM 已经预分配了 55% GPU 显存给 KV cache，剩余空间不足以容纳这个临时张量。

**修复**：给 `_forward_micro_batch` 添加 `compute_entropy` 参数，在 `compute_log_prob` 调用路径传入 `False`，跳过 softmax。训练路径（需要 entropy bonus loss）仍正常计算。

**效果**：inference 阶段节省 ~13GB/micro-batch，消除 OOM。

---

### 1.4 多轮冗余 Weight Sync 消除（核心性能优化）

**Motivation**：这是对训练吞吐量影响最大的优化。原始流程中，每轮 `generate_sequences` 的完整生命周期为：

```
load_fsdp_param_and_grad()           # CPU→GPU 搬 FSDP 参数（~14GB）
rollout_sharding_manager.__enter__() # FSDP all-gather + sync 到 vLLM
vLLM generate                        # 实际推理（唯一有用的 GPU 工作）
rollout_sharding_manager.__exit__()  # offload vLLM 到 CPU + empty_cache
compute_log_prob()                   # FSDP forward pass（多轮时结果被丢弃）
offload_fsdp_param_and_grad()        # GPU→CPU 搬回 FSDP 参数
```

max_turns=6 时，步骤 1/2/4/5/6 **每轮重复**，每次 1-2 分钟，总计约 10-12 分钟 GPU 空闲。但同一 step 内模型权重并未改变，这些操作完全冗余。

**修复**：设计 `enter_generation_mode` / `exit_generation_mode` 生命周期 API：

```
enter_generation_mode()          # 一次性 sync（步开始）
  for turn in range(max_turns):
    generate(skip_weight_sync=True)  # 直接用已在 GPU 上的模型生成
exit_generation_mode()           # 一次性 offload（步结束）
compute_log_prob()               # 只在最终轨迹上算一次
```

同时在 batch 的 meta_info 中设置 `skip_weight_sync=True` 和 `recompute_log_prob=False`，让 `generate_sequences` 内部跳过冗余操作。

**效果**：6 次 weight sync 降为 1 次，单步节省 ~10-12 分钟 GPU 空闲时间。

---

### 1.5 vLLM 模型常驻 GPU（Keep-on-GPU）

**Motivation**：即使经过 1.4 的优化，每步仍有一次完整的 FSDP→vLLM weight sync（~1-2 分钟），其中大部分时间花在 CPU→GPU 的 14GB 模型参数搬运上。

**思路**：让 vLLM 模型副本永远留在 GPU 上，sync 时只需 GPU-to-GPU 拷贝（几秒级别）。代价是需要降低 `gpu_memory_utilization`（KV cache 比例），为两份模型（FSDP 训练 + vLLM 推理）共存腾出空间。

**修改**：
- vLLM worker 添加 `keep_on_gpu` 标志，跳过 `offload_model_weights()`
- sharding manager 初始化时传入配置，条件性跳过 offload
- `gpu_memory_utilization` 从 0.55 降至 0.25-0.4

**效果**：weight sync 从 ~1-2 分钟降至 ~10 秒（纯 GPU 内存拷贝）。

---

### 1.6 显存碎片化导致 update_actor 耗时膨胀

**Motivation**：观察到 `update_actor` 耗时逐步增长：step1=163s → step2=214s → step3=274s，MFU 持续下降。

**根因分析**：`param_offload=true` 模式下，FSDP 参数在训练前从 CPU 加载到 GPU，训练后卸载回 CPU。而 vLLM rollout 阶段会分配和释放大量不规则大小的 KV cache 块。这些操作交替进行后，PyTorch CUDA allocator 的 free list 中充满了碎片化的小块，导致后续 actor update 的大张量分配需要反复 compact，效率持续下降。

**修复**：在 `update_actor` 前执行 `torch.cuda.empty_cache()`，将 allocator 缓存的碎片块归还 CUDA driver，使后续分配能获得大块连续显存。

**效果**：`update_actor` 耗时稳定在 ~160s，不再逐步膨胀。

---

### 1.7 其他修复

| 问题 | 修复 |
|------|------|
| `torch.compile` + FSDP ref forward 死锁 | 禁用 entropy 计算的 compile |
| `use_kl_loss=false` 时仍创建 RefPolicy worker | 条件化 worker 注册，节省一份 7B 模型内存 |
| 训练集群无外网，SwanLab 登录超时 | 添加 `mode='offline'` 后端适配 |
| generation mode 结束后 compute_log_prob 状态残留 | 在 compute_log_prob 入口检查并退出 generation mode |

---

### 系统优化总效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 能否完成 1 step | 不能（死锁/OOM/崩溃） | 稳定运行 1400+ steps |
| 单步 GPU 空闲时间（6轮） | ~12 min | ~10s |
| 单步总时间 | ~15 min | ~3.5 min |
| update_actor 耗时趋势 | 逐步膨胀 163→274s | 稳定 ~160s |
| compute_log_prob 显存 | OOM（13GB 冗余） | 正常 |

---

## 第二部分：GRPO 算法改进

---

### 2.1 Dr.GRPO：移除 Group Std 归一化

**Motivation**：标准 GRPO 的 advantage 计算为 `(reward - mean) / std`。在 binary reward（EM 0/1）+ 小 group（n_agent=2~4）场景下，std 归一化引入严重问题：

```
Group [1, 0, 0, 0]: mean=0.25, std=0.5  → "1"的 advantage = (1-0.25)/0.5 = 1.5
Group [1, 1, 0, 0]: mean=0.5,  std=0.577 → "1"的 advantage = (1-0.5)/0.577 = 0.87
```

相同的"正确回答"在不同 group 组成下获得完全不同的 advantage 幅度。这导致梯度方向受 group 随机组成影响，训练不稳定。

**修复**：移除 std 除法，advantage = reward - group_mean。Binary reward 本身就是 [0,1] 范围，不需要额外标准化。

**效果**：advantage 幅度只取决于该样本与组均值的偏差，不再受 group 组成的随机性干扰。

---

### 2.2 DAPO Dynamic Sampling + 重采样

**Motivation**：当 group 内所有样本 reward 相同时（如全错 [0,0,0,0]），advantage 全为 0，这些样本不贡献梯度。在 EM≈2% 的训练初期，约 96% 的 group 是全 0，绝大多数计算被浪费。

**两层解决方案**：

1. **重采样**（ray_trainer 层）：生成后快速计算 EM，检测 all-same 的 group，对其重新生成（最多 N 次），尽量让 group 出现 diverse 的结果
2. **兜底 skip**（core_algos 层）：重采样后仍为 all-same 的 group，将 advantage 显式设为 0，避免浮点误差产生噪音梯度

**为什么不只做重采样**：即使重试多次，EM=2% 时 P(至少一个对) ≈ 1-(0.98)^4 ≈ 7.8%，大量 group 仍然无法变 diverse。兜底 skip 确保这些 group 不引入噪音。

---

### 2.3 DAPO Clip-Higher：非对称裁剪

**Motivation**：标准 PPO 使用对称 clip [1-ε, 1+ε]，对"好轨迹"和"差轨迹"施加相同约束。但在稀疏奖励 + EM≈2% 场景下，难得出现的正确轨迹应该被**更强地强化**，而不应和错误轨迹受相同的 clip 约束。

**修复**：
- 下界（惩罚方向）：`1 - 0.2 = 0.8`（保持严格，防止过度惩罚）
- 上界（强化方向）：`1 + 0.28 = 1.28`（放宽，让好轨迹的 ratio 可以更大幅增长）

**效果**：正确轨迹的概率增长速度更快，加速从稀疏信号中学习，同时错误轨迹仍受严格约束不会被过度惩罚。

---

### 2.4 LLDS-MA：防止 Likelihood Displacement Collapse

**Motivation**：GRPO 训练中存在一个隐蔽但致命的问题——LLD Death Spiral：

```
Step 1: 正确轨迹 A 的 log_prob = -2.0, advantage > 0 → 应该被强化
Step 2: 但梯度同时影响了其他 token → A 的 log_prob 下降到 -2.5
Step 3: ratio = exp(-2.5 - (-2.0)) = exp(-0.5) = 0.6（ratio < 1，反而在减弱！）
        或：old_log_prob 更新后，下一轮 ratio 膨胀 → 梯度爆炸
Step N: 所有样本的 log_prob 持续下降 → 模型输出退化为垃圾
```

核心矛盾：positive advantage 的轨迹的 likelihood 反而在下降（displacement），这导致 ratio 不可控，最终 collapse。

**LLDS-MA 公式**：

```
L_total = L_GRPO + λ × L_LLDS

L_LLDS = (1/N) × Σ_{好轨迹}
          × 1[整体 response likelihood 确实下降了]     ← response-level gate
          × Σ_t max(0, old_lp_t - new_lp_t)          ← token-level penalty
```

**设计细节**：
- **保护对象**（Y_pre）：advantage ≥ 0 的轨迹（正确的 + 中性的）
- **Response-level gate**：只有当整条 response 的总 likelihood 确实下降时才激活（避免对正常 token 重分布的过度干预）
- **Token-level selectivity**：只惩罚那些 likelihood 下降的具体 token（精准定位 displacement 发生的位置）
- **梯度方向**：推动 displaced tokens 的 likelihood 回升
- **Gate 在 `no_grad()` 内计算**：纯选择作用，梯度只通过 token penalty 项流过

**λ = 0.1**（来自论文 7B 模型消融实验的最优值）

**效果**：训练过程中 `llds/loss` 作为早期预警指标，当 displacement 发生时 LLDS 主动介入阻止 collapse，grad_norm 保持稳定。

---

## 第三部分：多层奖励系统

---

### 3.0 问题定义

MEM1 的 6 轮交互只有**终端一个 0/1 信号**。训练初期 EM≈2% 时：
- 98% 轨迹 reward=0，几乎无梯度信号
- DAPO skip 后有效 group 仅 ~4%
- 模型无法区分"差一步就对"和"完全跑偏"

需要在中间步骤引入**方向正确的弱信号**，同时确保信号不会被模型 hack（学会"看起来好"但实际无用的行为）。

---

### 3.1 Layer 1：规则过程奖励（零计算开销）

#### 设计约束

1. **必要不充分**：过程奖励反映的应是成功的必要条件，单独满足不等于最终答对
2. **不可捷径**：模型不能通过简单策略 hack 获得高过程分
3. **量级压制**：过程奖励总量 max ≈ 0.25 << 结果奖励 1.0，结果导向不会被覆盖

#### 维度一：Retrieval Relevance（检索相关性）[0, 0.3]

**信号来源**：检索结果是否包含 ground truth 答案实体

**为什么设置**：多轮 RAG 的核心能力是"写好 query → 检索到有用信息"。如果检索结果包含答案实体，说明模型的 query 质量高。这是最终答对的**必要条件**（但不充分——检索到了还需要正确推理）。

**为什么有效/不可 hack**：
- 信号来自**外部检索系统**，模型只能通过写好 query 间接影响
- 随机/无意义 query → 外部系统检索不到答案相关内容 → reward=0
- 模型唯一的"hack"方式是写出精准 query → 这恰好就是我们希望它学会的行为

#### 维度二：Information Novelty（信息新颖性）[-0.15, +0.15]

**信号来源**：本次检索结果与历史检索结果的 3-gram 去重率

**为什么设置**：多轮交互的价值在于逐步从不同角度收集信息。如果模型每轮搜同一个词，6 轮等于 1 轮，完全浪费了多轮的优势。需要一个信号逼迫模型**换角度思考、分解子问题**。

**为什么有效/不可 hack**：
- 基于检索**结果**的 n-gram 去重，不是 query 的表面形式
- 模型改写 query 措辞但检索到同样内容 → 结果 n-gram 高度重复 → 仍被惩罚
- 搜无意义内容凑新颖性？→ retrieval relevance 维度给 0 分 + 最终答错 outcome=0，净收益为负

#### 维度三：Progressive Coverage（渐进覆盖）[0, 0.2]

**信号来源**：think 内容中覆盖的答案实体数的**增量**

**为什么设置**：检索到有用信息 ≠ 用了它。模型可能检索到答案相关文档但在 think 中完全没有引用。渐进覆盖检查模型是否真的把检索到的信息**融入了推理过程**，形成 search→retrieve→think 的闭环。

**为什么有效/不可 hack**：
- **只奖励增量**（delta），不奖励绝对值 → 第一步就把所有答案猜完后，后续步 delta=0，拿不到分
- 需要信息出现在 think 推理文本中（不是 answer 格式），模型需要真的在"用"检索结果
- 凭空在 think 中编造答案实体？→ 没有检索支撑大概率最终答错 → outcome=0

#### 维度四：Efficiency（效率奖励）[0, 0.2]

**信号来源**：正确回答时使用的轮数

**为什么设置**：训练后期模型已有一定准确率，但可能形成"总是搜满 6 轮再答"的固定 pattern。效率奖励鼓励"信息够了就停"，减少冗余搜索。实际应用中少搜一轮 = 少一次 API 调用 + 更快响应。

**为什么有效/不可 hack**：
- **仅在 outcome=1 时生效**，答错时效率分为 0
- 模型不能通过"第一步直接猜"来 hack（EM≈2% 时猜对概率极低，答错则效率分=0）
- 只有真正有能力"少搜几轮就答对"才能获得这个分数

#### 量级控制（最关键设计）

```
过程奖励 max = 0.5 × (0.4×0.3 + 0.3×0.15 + 0.3×0.2) × 3步 + 0.2 ≈ 0.25
结果奖励 = 1.0

→ 完美 hack 所有过程维度(+0.25) + 答错(0) = 0.25
  < 过程分为 0 但答对(1.0) = 1.0

→ 结果导向永远不会被过程信号覆盖
```

---

### 3.2 Layer 2：LLM Outcome Judge（语义 EM）

**Motivation**：纯 exact match 存在 false negative：

```
Ground truth: "New York City"
Model answer: "NYC"         → EM=0, 但语义正确
Model answer: "纽约市"      → EM=0, 但语义正确
Model answer: "Feb 14"     → EM=0, GT="February 14th", 语义正确
```

这些 false negative 导致模型明明答对了却得到 reward=0，是对正确行为的错误惩罚。

**方案**：对所有 EM=0 且有有效 answer 的样本，调用 DeepSeek V4 Flash 判断语义等价性。判正则 reward=0.8（略低于 EM=1.0，保留 exact match 的微弱优势鼓励精确输出）。

**成本**：~24M tokens / ~¥3.5（全训练周期），可忽略。

---

### 3.3 Layer 3：LLM Process Judge（Listwise Ranking）

**Motivation**：即使有了规则过程奖励，DAPO skip 的 all-same group 仍然 advantage=0。但这些 group 内部其实有质量差异：

```
Group [0, 0, 0, 0]（全部答错）:
  - Traj A: 搜了 "telephone inventor" → 找到相关文档 → 差一步就对
  - Traj B: 搜了 "hello world" → 完全无关 → 完全跑偏

当前：两者都 advantage=0，模型无法区分
改进：LLM 排名 A>B → advantage_A=+0.3, advantage_B=-0.3
```

**方案**：对 tied group（outcome 相同）的 n_agent 条轨迹做 listwise ranking，让 LLM 基于搜索策略质量给出排名，转化为 advantage 信号。

**异步调用优化**：轨迹在生成阶段逐个完成（大多数 Turn 1-2 就完成），完成时立即异步发送 Judge API 调用，不等 generation 结束。到 generation 结束时（~111s），早期发出的调用早已返回，实现 ~0s 额外等待。

**效果**：将被 DAPO skip 的 ~96% 无效样本转化为有效梯度信号，有效样本利用率从 ~4% 提升至 ~60%。

---

## 第四部分：课程学习训练策略

---

### Motivation

三层奖励信号的强度和适用时机不同：
- 训练初期（EM≈2%）：模型连格式都不会，此时引入复杂 reward 信号是噪音
- 训练中期（EM 5-15%）：需要过程奖励提供方向引导
- 训练后期（EM>30%）：需要 LLM Judge 的细粒度信号推高上限

### 四阶段自动课程

| 阶段 | 目标 | 奖励信号 | 退出条件 |
|------|------|----------|----------|
| **Warmup** | 学会输出格式 | 仅格式奖励（输出合法的 `<think><search><answer>` 结构即得分） | format_acc EMA ≥ 0.9 或 step ≥ 80 |
| **Convergence** | 学会有效检索 | 规则过程奖励 + DAPO 重采样 | EM EMA ≥ 0.15 或 step ≥ 300 |
| **Transition** | 引入细粒度信号 | + LLM Pointwise Judge | EM EMA ≥ 0.30 或 step ≥ 500 |
| **Refinement** | 推高上限 | + LLM Listwise Judge | 训练结束 |

阶段只前进不回退，使用 EMA (alpha=0.05) 平滑指标防止抖动误触发。

### Turn-Weighted Advantage

**Motivation**：标准 GRPO 给一条轨迹中所有 token 相同的 advantage。但 6 轮交互中，有的轮搜索质量高、有的低，应该给好的轮更强的梯度信号。

```
w_t = 1 + α × sign(A_i) × normalize(turn_score_t)

好轨迹中好 turn → 放大正梯度（多学好的搜索策略）
好轨迹中差 turn → 减弱正梯度（不强化差的搜索步骤）
差轨迹中好 turn → 保护（减轻惩罚，这步搜得好不该被连坐）
差轨迹中差 turn → 加重惩罚（这步搜得差，加强惩罚）
```

实现 turn 级别的 credit assignment，而非轨迹级别的粗粒度奖惩。

---

## 总效果汇总

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 训练稳定性 | 无法完成 1 step | 1400+ steps 零中断 |
| 单步训练时间 | ~15 min | ~3.5 min（4x 加速） |
| 有效梯度样本比例 | ~4% | ~60% |
| Collapse 风险 | 长程训练后 likelihood 崩溃 | LLDS-MA 持续防护 |
| 奖励信号密度 | 终端 1 个 0/1 | 每轮多维连续信号 + 语义判断 + 排名信号 |



Pointwise Judge（逐条评分）
做了什么
对每条轨迹独立调用 LLM，按 1-5 分 rubric 评估其搜索策略质量：


输入: question + ground_truth_answer + 单条 trajectory 文本
输出: {"score": 1-5, "reason": "..."}
Rubric（评分标准）
分数	含义
5	每个 query 语义相关，每步都带来新信息，推理正确综合了检索结果，有清晰的全局计划
4	query 相关，信息增益好，推理基本正确，有轻微冗余但整体递进
3	query 相关但部分冗余（重复信息），推理引用了检索但有逻辑缺口
2	部分 query 跑题或严重冗余（同一角度搜两次），推理与检索结果矛盾
1	query 完全无关或全部重复搜同一个东西，没有连贯推理链
分数 → Advantage 转换

# score=3 为中性（advantage=0），线性映射到 [-scale, +scale]
advantage = scale × (score - 3) / 2.0

# scale=0.2 时：
# score=5 → +0.2, score=4 → +0.1, score=3 → 0, score=2 → -0.1, score=1 → -0.2
触发条件
Transition 阶段（EM ≥ 15%）开始启用
只对有信号差异的 group 调用（group_std > 0 或 group 平均 EM > 0.5）
Listwise Judge（组内排名）
做了什么
把同一 prompt 的 n_agent 条轨迹一起发给 LLM，让它做排名：


输入: question + answer + n 条 trajectory 并排展示
输出: {"ranking": [best_idx, ..., worst_idx], "confidence": "high/medium/low"}
Margin-Based Gating（门控机制）
核心设计：不是所有排名结果都转化为 advantage。只有在 pointwise 分数确认质量差异时才生效：


# 只奖励 rank-1，且其 pointwise ≥ 4（确实好）
if pointwise_scores[best_idx] >= 4:
    advantages[best_idx] = +scale

# 只惩罚 rank-last，且其 pointwise ≤ 2（确实差）
if pointwise_scores[worst_idx] <= 2:
    advantages[worst_idx] = -scale

# 其余位置（中间排名）：advantage = 0
为什么这样设计：

防止"矮子里拔将军"：全是垃圾轨迹中排第一的也不该被奖励（pointwise < 4 → 不奖）
防止"优等生里罚末位"：全是好轨迹中排最后的不该被重罚（pointwise > 2 → 不罚）
触发条件（两层门控）
阶段门控：只在 Refinement 阶段（EM ≥ 30%）启用
差异门控：只有当 group 内 pointwise 分数 max - min ≥ 1 时才调用 listwise（分数太接近说明 LLM 自己都分不清，排名不可信）
两者的关系

┌─────────────────────────────────────────────────┐
│ Transition 阶段 (EM 15%~30%)                     │
│                                                   │
│  只用 Pointwise：                                 │
│  每条轨迹独立打分 → 直接转 advantage              │
│  advantage = 0.2 × (score - 3) / 2              │
└─────────────────────────────────────────────────┘
           │ EM ≥ 30%
           ▼
┌─────────────────────────────────────────────────┐
│ Refinement 阶段 (EM > 30%)                       │
│                                                   │
│  Pointwise + Listwise 叠加：                     │
│  advantage = pointwise_adv + listwise_adv        │
│                                                   │
│  Listwise 有双重门控：                            │
│  1. pointwise max-min ≥ 1 才调用                 │
│  2. 只有 rank-1 且 pw≥4 才奖，rank-last 且 pw≤2 才罚 │
└─────────────────────────────────────────────────┘
Advantage 分布到 Turn
Judge advantage 不是均匀加到整个 response，而是按 positional weight 分布到各 turn：


# 中间 turn（策略分歧点）获得更大权重
pos_w = compute_positional_weights(n_turns)  # bell-curve 形状
for t, (start, end) in enumerate(turn_boundaries):
    advantages[idx, start:end] += judge_adv × (pos_w[t] / total_w × n_turns)
这让 judge 信号集中作用在"关键决策步"（通常是中间轮），而不是平铺到格式化的首尾。