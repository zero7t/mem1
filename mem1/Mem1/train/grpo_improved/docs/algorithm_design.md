# GRPO 算法改进设计

## 1. 原始 GRPO（baseline）

```
对每个 prompt 生成 n_agent 条轨迹（group）
advantage_i = (reward_i - mean(group)) / std(group)
policy_loss = -advantage × min(ratio, clip(ratio, 1-ε, 1+ε))
```

问题：
- std 归一化在 binary reward + 小 group 时极不稳定
- 全同 group（都对或都错）仍计算无意义的梯度
- 好轨迹 likelihood 持续下降导致 collapse

## 2. Dr.GRPO：去除 std 归一化

```python
# 位置: core/grpo_algos.py → compute_grpo_outcome_advantage()

advantage_i = reward_i - mean(group)   # 不除以 std
```

**为什么**：Binary reward {0,1} + n_agent=2：
- Group [1,0]: std=0.5 → advantage = ±1.0（归一化后）
- Group [1,1,0,0]: std=0.577 → advantage = ±0.87
- 不同 group size/组成导致 advantage scale 不一致

去掉 std 后：advantage 直接反映与组均值的偏差，scale 一致。

## 3. DAPO Dynamic Sampling：跳过无信息 group

```python
# 位置: core/grpo_algos.py → compute_grpo_outcome_advantage()

if id2std[idx] < epsilon:  # group 内所有 reward 相同
    scores[i] = 0.0        # advantage = 0, 不贡献梯度
```

**为什么**：Group [0,0,0,0] → mean=0, 所有 advantage=0。
本来就不应该有梯度，显式 skip 避免浮点误差带来噪音。

## 4. DAPO Clip-Higher：非对称 clipping

```python
# 位置: core/grpo_algos.py → compute_policy_loss()

clip_low = cliprange       # 0.2 (负advantage方向)
clip_high = clip_higher    # 0.28 (正advantage方向)

pg_losses2 = -advantages * clamp(ratio, 1-clip_low, 1+clip_high)
```

**为什么**：正 advantage（好轨迹）允许更大的 ratio 变化 → 更强的强化。
负 advantage（差轨迹）保持较紧的 clip → 防止过度惩罚。

## 5. LLDS-MA：Likelihood Displacement 防护

```python
# 位置: core/grpo_algos.py → compute_llds_loss()

# 保护对象：advantage ≥ 0 的轨迹（好的/平均的）
preserve_mask = (response_advantages >= 0)

# 触发条件：整体 likelihood 确实下降了
response_displacement = Σ_t(old_lp_t - new_lp_t)
response_gate = (response_displacement > 0)

# 组合
active_mask = preserve_mask AND response_gate

# Token-level 惩罚（有梯度）
token_penalty = max(0, old_lp_t - new_lp_t)  # 只惩罚下降的 token
llds_loss = sum(token_penalty * active_mask) / N_active_tokens
```

**整合到总 loss**:
```
total_loss = policy_loss - entropy_coeff × entropy + llds_lambda × llds_loss
```

参数: `llds_lambda = 0.1`

## 6. 完整 update_policy 流程

```python
# 位置: core/actor_update.py → update_policy()

for micro_batch in split(mini_batch, micro_batch_size=6):
    # Forward
    entropy, log_prob = forward(micro_batch, temperature)

    # Policy Loss (DAPO Clip-Higher)
    pg_loss, clipfrac, kl = compute_policy_loss(
        old_log_prob, log_prob, advantages, response_mask,
        cliprange=0.2, clip_higher=0.28
    )

    # Entropy bonus
    entropy_loss = masked_mean(entropy, response_mask)
    policy_loss = pg_loss - entropy_coeff * entropy_loss

    # LLDS regularization
    if llds_lambda > 0:
        llds_loss, llds_metrics = compute_llds_loss(
            old_log_probs, log_prob, response_mask, advantages
        )
        policy_loss = policy_loss + llds_lambda * llds_loss

    # Backward
    loss = policy_loss / gradient_accumulation
    loss.backward()

optimizer.step()
```

## 7. 配置参数汇总

```yaml
# GRPO 核心
algorithm.adv_estimator: grpo
actor_rollout_ref.rollout.n_agent: 4        # group size
actor_rollout_ref.actor.ppo_mini_batch_size: 384  # = batch_size × n_agent

# Dr.GRPO + DAPO
actor_rollout_ref.actor.clip_higher: 0.28   # 正advantage clip上界

# LLDS
actor_rollout_ref.actor.llds_lambda: 0.1    # LLDS 正则化权重

# 优化器
actor_rollout_ref.actor.optim.lr: 2e-7
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio: 0.02

# State masking (info tokens 不参与 loss)
actor_rollout_ref.actor.state_masking: true
```
