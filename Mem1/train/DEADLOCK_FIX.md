# MEM1 GRPO Multi-Turn Training Deadlock Fix

## Problem

Multi-turn GRPO training with FSDP on 6 GPUs would deadlock/hang at various stages:
1. Multi-turn generation phase (sharding manager exit/enter between turns)
2. `compute_ref_log_prob` FSDP forward all-gather deadlock
3. `update_actor` ZeroDivisionError (ppo_micro_batch_size=0 after normalization)

## Root Causes & Fixes

### Fix 1: `keep_generation_mode` lost in GPU padding path

**File**: `rollout/llm_agent/generation_think.py`

**Problem**: In `_generate_with_gpu_padding`, when padding is needed to align batch sizes across workers, a new `padded_active_batch` is created. The `keep_generation_mode=True` meta_info was not copied to the padded batch, causing the FSDP sharding manager to prematurely exit between multi-turn generation calls. This triggered NCCL all-gather deadlocks because workers would attempt FSDP state_dict() collectives out of sync.

**Fix**: Add `keep_generation_mode` to the padded batch meta_info:
```python
padded_active_batch.meta_info['keep_generation_mode'] = True
```

### Fix 2: Batch size integer division truncation

**File**: `verl/workers/fsdp_workers.py` (lines 97-101)

**Problem**: The config normalization divides global batch sizes by the number of FSDP workers:
```python
self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
```
With `ppo_micro_batch_size=2` and 6 GPUs: `2 // 6 = 0`, causing ZeroDivisionError in `dp_actor.py`.

**Fix**: All `*_batch_size` config values must be >= `n_gpus_per_node` to survive integer division:
```
ppo_mini_batch_size=6      (6 // 6 = 1 per worker)
ppo_micro_batch_size=6     (6 // 6 = 1 per worker)
log_prob_micro_batch_size=6 (6 // 6 = 1 per worker)
```

Note: `rollout.n` (not `n_agent`) is the multiplier applied after division. Default is 1.

### Fix 3: Disable RefPolicy when use_kl_loss=false

**File**: `verl/trainer/main_ppo.py`

**Problem**: RefPolicy workers were always created even when `use_kl_loss=false`, causing unnecessary FSDP collective operations that could deadlock.

**Fix**: Conditionally create RefPolicy:
```python
if config.actor_rollout_ref.actor.use_kl_loss:
    role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    mapping[Role.RefPolicy] = global_pool_id
```

### Fix 4: Disable torch.compile for FSDP actor

**File**: `verl/workers/actor/dp_actor.py`

**Problem**: `torch.compile` on entropy computation caused hangs during FSDP ref model forward pass (all-gather deadlock).

**Fix**: Use plain function instead of compiled version:
```python
# Disable torch.compile - causes hangs with FSDP ref model forward
self.compute_entropy_from_logits = verl_F.entropy_from_logits
```

### Fix 5: Exit generation mode in compute_log_prob

**File**: `verl/workers/fsdp_workers.py` (`compute_log_prob` method)

**Problem**: After multi-turn generation with `keep_generation_mode=True`, the sharding manager stays open. When `compute_log_prob` is called next, it needs to properly exit generation mode first.

**Fix**: Check and exit generation mode at the start of `compute_log_prob`:
```python
if self._in_generation_mode:
    self.rollout_sharding_manager.__exit__(None, None, None)
    self._in_generation_mode = False
elif self._is_offload_param:
    load_fsdp_param_and_grad(...)
```

### Fix 6: OOM in compute_log_prob — skip unnecessary entropy computation

**File**: `verl/workers/actor/dp_actor.py`

**Problem**: `compute_log_prob` only needs `log_probs`, not entropy. But `_forward_micro_batch` always computed entropy via `entropy_from_logits` (softmax over vocab_size=152064). With max_turns=6, sequences can reach ~6000 tokens. A single sample's entropy softmax requires ~13 GB (float32 `(1, 6000, 152064)`) — exceeding GPU memory after vLLM pre-allocates 55% for KV cache.

Error: `torch.cuda.OutOfMemoryError: Tried to allocate 13.31 GiB`

**Fix**: Add `compute_entropy` parameter to `_forward_micro_batch`, default `True`. In `compute_log_prob` path, pass `compute_entropy=False` to skip the wasteful softmax:
```python
def _forward_micro_batch(self, micro_batch, temperature, compute_entropy=True):
    # ...
    entropy = verl_F.entropy_from_logits(logits) if compute_entropy else None
    # ...

def compute_log_prob(self, data):
    # ...
    _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, compute_entropy=False)
```

This saves ~13 GB per micro-batch during inference without affecting training (which still computes entropy for the entropy bonus loss).

### Fix 7: SwanLab offline logging integration

**File**: `verl/utils/tracking.py`

**Problem**: Training cluster has no external network access, causing SwanLab login to timeout when trying to reach `api.swanlab.cn`.

**Fix**: Add SwanLab as a supported backend with `mode='offline'` (no network required):
```python
supported_backend = ['wandb', 'mlflow', 'console', 'swanlab']

if 'swanlab' in default_backend:
    import swanlab
    swanlab.init(project=project_name, experiment_name=experiment_name, config=config, mode='offline')
    self.logger['swanlab'] = _SwanLabLoggingAdapter()

class _SwanLabLoggingAdapter:
    def log(self, data, step):
        import swanlab
        swanlab.log(data=data, step=step)
```

Usage in config: `trainer.logger=['swanlab','console']`

## Verification

Successfully ran 2 training steps end-to-end with:
- 6 GPUs, batch_size=6, n_agent=2, max_turns=3
- Full pipeline: generation -> compute_log_prob -> update_actor
- Log: `/tmp/train_fix.log`
- Step 1 metrics: timing_s/gen=40.5s, timing_s/update_actor=5.0s, timing_s/step=53.0s

## Config Notes

When setting batch sizes, ensure:
```
ppo_mini_batch_size >= n_gpus_per_node
ppo_micro_batch_size >= n_gpus_per_node
log_prob_micro_batch_size >= n_gpus_per_node
```

The normalization formula is:
```
per_worker_value = global_value // num_dp_workers * rollout.n
```
Where `num_dp_workers = n_gpus_per_node // ulysses_sequence_parallel_size` and `rollout.n` defaults to 1.

---

## Related Documentation

- **Algorithm Improvements (DAPO-lite + Dr.GRPO + LLDS-MA)**: See `TRAINING_IMPROVEMENTS.md`
- **Weight Sync / Multi-turn Optimization**: See `VERL_OPTIMIZATIONS.md`
