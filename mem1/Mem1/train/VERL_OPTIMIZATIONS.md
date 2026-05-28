# verl 源代码优化记录

## 问题描述

MEM1 的 GRPO 训练中，每个 training step 的多轮生成循环（max_turns=6）会导致 GPU 利用率长时间为0%。
原因是每轮调用 `generate_sequences` 都会触发完整的 weight sync + offload 周期，但同一 step 内模型权重并未改变，这些操作完全冗余。

### 原始流程（每轮重复）
```
每轮 generate_sequences 调用：
1. load_fsdp_param_and_grad()       → CPU→GPU 搬 FSDP 参数
2. rollout_sharding_manager.__enter__()  → FSDP state_dict all-gather + sync到vLLM
3. vLLM generate                    → 实际推理（唯一有用的GPU工作）
4. rollout_sharding_manager.__exit__()   → offload vLLM 到CPU + empty_cache
5. compute_log_prob()               → FSDP forward pass（结果被丢弃，完全浪费）
6. offload_fsdp_param_and_grad()    → GPU→CPU 搬回 FSDP 参数
```

max_turns=6 时，步骤 1/2/4/5/6 重复6次，每次1-2分钟，总计约10-12分钟GPU空闲。

---

## 优化方案一：消除多轮冗余 weight sync（本次主要改动）

### 核心思路
将 weight sync/offload 的生命周期从"每轮"提升到"每步"——整个多轮循环只做一次 sync 和一次 offload。

### 修改的文件

#### 1. `verl/workers/fsdp_workers.py`

**改动1：新增 `enter_generation_mode` 方法**
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def enter_generation_mode(self):
    """在多轮循环开始前调用一次：加载FSDP参数 + sync权重到vLLM"""
    if self._is_offload_param:
        load_fsdp_param_and_grad(...)
    self.rollout_sharding_manager.__enter__()
    self._in_generation_mode = True
```

**改动2：新增 `exit_generation_mode` 方法**
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def exit_generation_mode(self):
    """在多轮循环结束后调用一次：offload vLLM + 卸载FSDP参数"""
    self.rollout_sharding_manager.__exit__(None, None, None)
    if self._is_offload_param:
        offload_fsdp_param_and_grad(...)
    torch.cuda.empty_cache()
    self._in_generation_mode = False
```

**改动3：`generate_sequences` 支持 `skip_weight_sync` 标志**
```python
def generate_sequences(self, prompts: DataProto):
    skip_weight_sync = prompts.meta_info.get('skip_weight_sync', False)

    if not skip_weight_sync:
        # 原始逻辑：load params + enter sharding manager
        ...

    if skip_weight_sync:
        # 已在 generation mode 中，直接 generate（不进出 sharding manager）
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)
    else:
        with self.rollout_sharding_manager:
            # 原始逻辑
            ...

    # recompute_log_prob 也通过 meta_info 控制，多轮时设为 False
    if self._is_actor and recompute_log_prob:
        ...

    if not skip_weight_sync:
        # 原始逻辑：offload params
        ...
```

**改动4：`init_model` 中初始化 `_in_generation_mode = False`**

#### 2. `rollout/llm_agent/generation_think.py`

**改动1：多轮循环前后加 enter/exit**
```python
# 循环前：一次性 sync
self.actor_rollout_wg.enter_generation_mode()

for step in range(self.config.max_turns):
    ...
    gen_output = self._generate_with_gpu_padding(rollings_active)
    ...

# 循环后：一次性 offload
self.actor_rollout_wg.exit_generation_mode()
```

**改动2：`_generate_with_gpu_padding` 设置 skip 标志**
```python
def _generate_with_gpu_padding(self, active_batch):
    # 跳过冗余 weight sync 和 log_prob 计算
    active_batch.meta_info['skip_weight_sync'] = True
    active_batch.meta_info['recompute_log_prob'] = False
    ...
```

### 效果
- max_turns=6 时：从6次 weight sync 降为1次，节省约 10-12分钟/step
- max_turns=2 时：从2次降为1次，节省约 2-3分钟/step
- log_prob 只在最终轨迹上计算一次（由 trainer 层在 rollout 后统一做）

---

## 优化方案二：keep_on_gpu（之前的改动）

### 核心思路
vLLM 模型永远留在GPU上，避免每步的 CPU↔GPU 14GB 传输。通过降低 `gpu_memory_utilization` 为两份模型（FSDP + vLLM）共存腾空间。

### 修改的文件

#### 1. `verl/third_party/vllm/vllm_v_0_6_3/worker.py`
- 添加 `self.keep_on_gpu = False` 标志
- `offload_model_weights()` 中检查 `keep_on_gpu`，若为 True 则跳过 offload

#### 2. `verl/third_party/vllm/vllm_v_0_6_3/llm.py`
- 添加 `set_keep_on_gpu(bool)` 方法传递标志到 worker

#### 3. `verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py`
- `load_dtensor_weights()` 中条件性跳过 `.cuda()`（如果模型已在GPU上）

#### 4. `verl/workers/sharding_manager/fsdp_vllm.py`
- `FSDPVLLMShardingManager.__init__` 接受 `keep_vllm_on_gpu` 参数
- 初始化时调用 `set_keep_on_gpu(True)`

#### 5. `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- 条件性跳过初始 `offload_model_weights()`

#### 6. `verl/workers/fsdp_workers.py`
- 传递 `keep_vllm_on_gpu=self.config.rollout.get('keep_on_gpu', False)` 给 sharding manager

### 配置
```bash
# train_grpo.sh 中启用
actor_rollout_ref.rollout.keep_on_gpu=true
actor_rollout_ref.rollout.gpu_memory_utilization=0.25  # 降低KV cache给模型腾空间
```

---

## 优化效果对比（预估）

| 场景 | 原始代码 | 方案一 | 方案一+二 |
|------|---------|--------|-----------|
| 每步 weight sync 次数 | max_turns 次 | 1次 | 1次 |
| 每步 log_prob 冗余计算 | max_turns 次 | 0次 | 0次 |
| weight sync 耗时 | ~1-2分钟 | ~1-2分钟 | ~10秒（GPU-to-GPU） |
| 每步总空闲时间(6轮) | ~12分钟 | ~2分钟 | ~10秒 |

---

## 注意事项

1. `enter_generation_mode` 使用了 `Dispatch.ONE_TO_ALL` 模式，所有 worker 同时执行
2. `break` 提前退出循环后，`exit_generation_mode()` 仍会正确执行（在 for 循环外）
3. 方案二需要降低 `gpu_memory_utilization` 避免 OOM（两份模型共存GPU）
4. validation 路径不走这个优化（validation 走 ray_trainer 里单独的逻辑）
