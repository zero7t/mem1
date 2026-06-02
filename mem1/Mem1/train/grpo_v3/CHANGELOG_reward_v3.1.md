# Reward V3.1 改进日志

日期: 2026-06-01

## 改动总结

### Fix #1: Judge advantage 按 turn 分布（非均匀）
- **文件**: `main_ppo_v3.py` apply_process_judge
- **问题**: judge pointwise 分数均匀加到整个 response 的 advantage 上，无法区分好/差 turn
- **修复**: judge advantage 按 positional_weight 分布到各 turn，中间 turn（策略分歧点）获得更大权重
- **效果**: judge 信号与 turn-weight 机制协同，强化对关键搜索步骤的奖惩

### Fix #3: Outcome judge 升级保留过程信号
- **文件**: `main_ppo_v3.py` reward computation
- **问题**: LLM outcome judge 判定语义正确时，直接覆盖 reward 为 0.8，丢失所有 process/efficiency 信号
- **修复**: 只升级 outcome 分量（0→0.8 的 delta），保留原有 process/format/efficiency 奖励
- **效果**: 语义正确但搜索策略差的 trajectory 不再与策略好的获得相同 reward

### Fix #4: Listwise 门控阈值降低
- **文件**: `main_ppo_v3.py` apply_process_judge
- **问题**: `max(scores) - min(scores) >= 2` 太保守，分数差 1 时不触发 listwise
- **修复**: 阈值从 2 降为 1
- **效果**: 更多 group 能获得 listwise ranking 信号，refinement 阶段信号更密集

### Fix #6: 检索质量奖励改为 query 质量评估
- **文件**: `reward/rule_reward_v3.py` retrieval_quality_reward
- **问题**: 原实现检查 retrieval 内容是否包含答案（模型不可控），混淆 credit assignment
- **修复**: 改为评估 query 与 answer target 的 F1 overlap（模型可控）
- **效果**: 奖励信号直接反映模型的 query 生成质量，而非 retriever 的随机性

### Fix #7: Turn boundary 映射改进
- **文件**: `core/__init__.py` find_turn_boundaries
- **问题**: 线性 char→token 映射对变长 token（中文、代码）误差大
- **修复**: 使用 segment-proportional 映射，每个 turn 按其字符长度比例分配 token 数
- **效果**: turn boundary 更准确，turn-weight 作用在正确的 token 范围上

### Bugfix: LLM Judge score=None 崩溃
- **文件**: `reward/llm_judge_v3.py:156`
- **问题**: API 返回 `{"score": null}` 时 `int(None)` 报 TypeError
- **修复**: `data.get("score") or 3`
