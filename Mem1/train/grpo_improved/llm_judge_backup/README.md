# LLM Judge Backup

本目录包含 LLM Process Judge (Listwise Ranking) 的完整工作代码备份。

## 文件说明

| 文件 | 说明 |
|------|------|
| `ray_trainer_with_pjudge.py` | 包含 Process Judge 完整集成的 ray_trainer.py |
| `main_ppo_with_judge.py` | 包含 Gen-Time Judge + Rule Process Reward 的 main_ppo.py |
| `llm_judge.py` | LLM Judge 客户端（Outcome + Process Judge） |
| `generation_judge.py` | 生成时 LLM Judge（与生成重叠，0额外等待） |

## 验证结果

Process Judge 已验证正常工作（logs/train_v1_final.log, step 1）：
- groups=96, upgraded=360
- timing_s/adv=159.7s（主要是 API 调用时间）
- advantages 范围: [-1.060, +1.282]（含 process judge 加成）

## 如何重新启用

1. 将 `ray_trainer_with_pjudge.py` 中 `# === LLM Process Judge` 代码块
   复制回 `verl/trainer/ppo/ray_trainer.py` 的 `compute_advantage()` 之后
2. 确保 `from grpo_improved.reward.llm_judge import GRPOJudgeOrchestrator` import 存在
3. 确保 uid 设置正确: `batch.non_tensor_batch['uid'] = (np.arange(batch_size) // n_agent).astype(object)`

## 优化建议（减少 160s/step 开销）

1. **降频**: `process_judge_interval=3`（每 3 步跑一次）
2. **只对 tied groups**: `process_judge_only_tied=True`（只对 outcome 相同的组做排序）
3. **异步化**: 把 API 调用放到 actor_update 阶段并行执行（需要改架构）
4. **缩短 prompt**: 截断轨迹到 1500 字符，减少 API 响应时间
