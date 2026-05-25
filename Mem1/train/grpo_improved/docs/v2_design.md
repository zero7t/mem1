# GRPO Improved V2 - Turn-Level Process Reward System

## Overview

V2 introduces **turn-level credit assignment** for multi-turn RAG QA training:
- Per-turn process rewards (not just a scalar at the end)
- Turn-weighted advantage modulation (good turns amplified, bad turns protected)
- Generation-time LLM Judge (fire API as trajectories complete, 0s overhead)

## Directory Structure

```
grpo_improved/
├── __init__.py
├── main_ppo_v2.py              # Training entry point (replaces verl/trainer/main_ppo.py)
├── core/
│   ├── __init__.py
│   ├── turn_weighted_advantage.py   # Turn-level advantage modulation
│   ├── generation_judge.py          # Fire LLM Judge during generation
│   ├── actor_update.py              # Reference: LLDS-MA + DAPO (copy of dp_actor.py)
│   └── grpo_algos.py               # Reference: Dr.GRPO + DAPO (copy of core_algos.py)
├── reward/
│   ├── __init__.py
│   ├── rule_reward_v2.py           # Per-turn process reward with utilization signal
│   ├── rule_reward.py              # V1 (deprecated, kept for reference)
│   └── llm_judge.py               # LLM Outcome + Process Judge client
├── scripts/
│   ├── train_v2.sh                 # V2 training script
│   └── train.sh                    # V1 training script
└── docs/
    ├── algorithm_design.md
    └── reward_design.md
```

## Quick Start

```bash
cd /root/paddlejob/workspace/mem1/MEM1/Mem1/train
bash grpo_improved/scripts/train_v2.sh
```

## Key Changes from V1

### 1. Per-Turn Process Reward (rule_reward_v2.py)

V1: Single scalar `process_reward` added to outcome score at last token.
V2: Returns `List[float]` — one score per turn.

**New dimension: Utilization Signal**

```
Turn t's reward = weighted sum of:
  - Hit (0.3):     Does retrieval contain answer entities?
  - Utilization (0.4): Is retrieval USED in later reasoning/answer?  ← NEW
  - Novelty (0.2): Is this non-redundant information?
```

The utilization signal solves two problems:
- **Turn 1 bias**: Turn 1 always has max "incremental info" in V1. In V2, Turn 1 only scores high if its retrieval is actually used later.
- **Scattered vs systematic**: A trajectory that retrieves info but never uses it scores low, even if the retrieval "hit" the answer.

### 2. Turn-Weighted Advantage (turn_weighted_advantage.py)

Standard GRPO gives all tokens the same advantage. V2 modulates per-turn:

```
w_t = 1 + α × sign(A_i) × normalize(process_score_t)

Effective advantage for tokens in turn t = A_i × w_t
```

| Trajectory | Turn Quality | Effect |
|---|---|---|
| Winner (A>0) | Good turn | Amplified positive (learn more from good search) |
| Winner (A>0) | Bad turn | Dampened positive (don't reinforce bad search) |
| Loser (A<0) | Good turn | Protected (less negative, don't punish good search) |
| Loser (A<0) | Bad turn | Amplified negative (punish bad search more) |

**Hyperparameters:**
- `turn_weight_alpha=0.3`: w ∈ [0.7, 1.3]. Conservative start.
- `turn_weight_clip=1.0`: Prevents extreme weights.

### 3. Generation-Time LLM Judge (generation_judge.py)

V1: Fire all judge calls after generation → 139s wait time.
V2: Fire as each trajectory completes during generation → ~0s additional time.

```
Timeline:
  t=0s    Generation starts (384 trajectories)
  t=20s   ~300 trajectories done (turns 1-2) → 300 judge calls fired
  t=40s   ~350 done → 50 more calls fired
  t=80s   ~370 done → 20 more calls fired
  t=111s  Generation ends. Turn-1 calls (fired at t=20s) returned 90s ago!
  t=111s  Collect remaining results → ~0s wait
```

**QPS: 20 concurrent** (user confirmed safe for internal API).

## Integration with verl/

The V2 system requires **one line** added to `verl/trainer/ppo/ray_trainer.py`:

```python
# After line 859 (after compute_advantage):
batch = self.reward_fn.apply_turn_weighting_to_batch(batch)
```

And for generation-time judge, in `generation_think.py` at line 382:

```python
# After: if dones[i] and "num_rounds" not in reconstruction_list[i]:
if hasattr(self, '_gen_judge') and self._gen_judge is not None:
    # Reconstruct trajectory text for this sample
    traj_text = self._reconstruct_text(reconstruction_list[i], step)
    self._gen_judge.on_trajectory_complete(i, traj_text)
```

## Reward Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Generation Phase (111s)                                      │
│                                                              │
│  Turn 1: generate → done[i]=True → fire_judge(i) ─────┐    │
│  Turn 2: generate → done[j]=True → fire_judge(j) ───┐ │    │
│  ...                                                  │ │    │
│  Turn 6: last trajectories complete                   │ │    │
│                                                       ▼ ▼    │
│                                          [API calls running] │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Reward Phase                                                 │
│                                                              │
│  1. EM Score (instant)                                       │
│  2. Per-turn process reward (instant, rule_reward_v2)        │
│  3. Collect judge results (already returned!) → upgrade 0→0.8│
│                                                              │
│  Output: reward_tensor + turn_scores_per_sample              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Advantage Phase                                              │
│                                                              │
│  1. Standard GRPO: A_i = score_i - mean(group)              │
│  2. Turn weighting: A_i × w_t per turn                       │
│                                                              │
│  Output: per-token advantages (turn-differentiated)          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Actor Update (unchanged)                                     │
│                                                              │
│  PPO loss with DAPO Clip-Higher + LLDS-MA                   │
│  Now: good-turn tokens get stronger gradient signal          │
│       bad-turn tokens in losing trajectories get punished    │
│       good-turn tokens in losing trajectories get protected  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

All V2 config is passed via hydra overrides in train_v2.sh:

```yaml
# Process reward
reward.lambda_process: 0.5      # Overall process reward scale
reward.turn_weight_alpha: 0.3   # Turn weighting strength
reward.turn_weight_clip: 1.0    # Max normalized score
reward.turn_weight_enabled: true

# Generation-time judge
reward.gen_judge_enabled: true
reward.gen_judge_qps: 20
```

## Relationship to LLDS

| Mechanism | Signal | Protects | Granularity |
|---|---|---|---|
| LLDS-MA | Token probability (model confidence) | Already-learned tokens | Per-token |
| Turn Weighting | Process quality (search strategy) | Good-search turns | Per-turn |

They are **orthogonal and multiplicative**:
```
effective_gradient = llds_weight × turn_weight × advantage × ratio
```

## Testing

```bash
# Test rule_reward_v2
python -m grpo_improved.reward.rule_reward_v2

# Test turn_weighted_advantage
python -m grpo_improved.core.turn_weighted_advantage
```
