# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
        # advantages[advantages > 3 | advantages < -3] = 0
        
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# [IMPROVED] Dr.GRPO + DAPO dynamic sampling
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO with Dr.GRPO + DAPO improvements.

    Dr.GRPO fix: Do NOT divide by group std for binary/sparse rewards.
      - Only subtract group mean as baseline
      - Groups with zero variance get zero advantage (DAPO dynamic sampling)
      - This avoids amplifying noisy groups and wasting gradient on uninformative groups

    Args:
        token_level_rewards: shape (bs, response_length)
        eos_mask: shape (bs, response_length)
        index: group index for each sample
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i].item())
        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = 0.0
                id2std[idx] = 1.0
            elif len(group_scores) > 1:
                id2mean[idx] = float(np.mean(group_scores))
                id2std[idx] = float(np.std(group_scores))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            idx = index[i]
            # [DAPO] Dynamic sampling: skip groups where all rewards are identical
            if id2std[idx] < epsilon:
                scores[i] = 0.0
            else:
                # [Dr.GRPO] Mean-only baseline, no std normalization
                # This avoids amplifying binary reward groups
                scores[i] = scores[i] - id2mean[idx]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


# [IMPROVED] DAPO Clip-Higher: asymmetric clipping
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, clip_higher=None):
    """PPO-style policy loss with DAPO Clip-Higher support.

    Args:
        old_log_prob: (bs, response_length)
        log_prob: (bs, response_length)
        advantages: (bs, response_length)
        eos_mask: (bs, response_length)
        cliprange: lower clip range (e.g., 0.2)
        clip_higher: upper clip range (e.g., 0.28). If None, uses cliprange (symmetric).

    DAPO Clip-Higher: positive advantage samples get more room to increase ratio.
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    clip_low = cliprange
    clip_high = clip_higher if clip_higher is not None else cliprange

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


# ============================================================================
# [LLDS-MA] Likelihood-preserving regularization
# From: "On GRPO Collapse in Search-R1" (Deng et al., 2025)
# ============================================================================

def compute_llds_loss(old_log_probs: torch.Tensor,
                      new_log_probs: torch.Tensor,
                      eos_mask: torch.Tensor,
                      advantages: torch.Tensor,
                      loss_mask: torch.Tensor = None,
                      answer_mask: torch.Tensor = None,
                      mask_answer: bool = True):
    """
    Compute LLDS with Action-Level Gating (paper default).

    L_LLDS = (1/N_active_tokens) * sum_{i in Y_pre} sum_{t=0}^{T_i}
              1[sum_{k in action_t} (old_lp_k - new_lp_k) > 0]   # action-level gate
              * sum_{k in action_t} max(0, old_lp_k - new_lp_k)  # token-level penalty

    Action boundaries are derived from loss_mask: consecutive 1-segments (separated
    by information blocks where loss_mask=0) each form one action.
    If loss_mask is not provided, falls back to response-level gating.

    Y_pre: responses with non-negative advantage (correct + untrained)

    Args:
        old_log_probs: (bs, response_length) - log probs from old policy
        new_log_probs: (bs, response_length) - log probs from current policy
        eos_mask: (bs, response_length) - valid token mask
        advantages: (bs, response_length) - per-token advantages
        loss_mask: (bs, response_length) - 1 for action tokens (think+response),
                   0 for information tokens. Used to detect action boundaries.
        answer_mask: (bs, response_length) - 1 for answer tokens, 0 otherwise
        mask_answer: whether to exclude answer tokens (LLDS-MA mode)

    Returns:
        llds_loss: scalar tensor
        metrics: dict
    """
    bs, seq_len = old_log_probs.shape

    with torch.no_grad():
        # Y_pre: responses with non-negative advantage
        response_advantages = advantages[:, 0]  # (bs,)
        preserve_mask = (response_advantages >= 0).float()  # (bs,)

        # Regularization mask: valid tokens only (ensure float)
        reg_mask = eos_mask.float()
        if mask_answer and answer_mask is not None:
            reg_mask = reg_mask * (1.0 - answer_mask.float())

        # Token displacement (detached for gate computation)
        token_disp_detached = (old_log_probs - new_log_probs.detach()) * reg_mask

        # Build action-level gate mask (per-token)
        if loss_mask is not None:
            # Detect action segment IDs from loss_mask transitions
            # Each contiguous block of 1s in loss_mask is one action
            # Transitions: 0->1 marks start of new action
            loss_mask_f = loss_mask.float()
            padded = torch.cat([torch.zeros(bs, 1, device=loss_mask.device), loss_mask_f], dim=1)
            starts = (padded[:, 1:] - padded[:, :-1]) > 0  # (bs, seq_len) True at action starts
            action_ids = starts.long().cumsum(dim=1) * loss_mask.long()  # 0 for info, 1,2,3... for actions

            # For each action segment, compute sum of displacement
            max_actions = action_ids.max().item() + 1
            # action_gate_mask: (bs, seq_len) - 1.0 if this token's action has displacement > 0
            action_gate_mask = torch.zeros(bs, seq_len, device=reg_mask.device, dtype=torch.float32)

            num_active_actions = 0
            total_actions = 0

            for aid in range(1, max_actions):
                # Mask for this action across all batch items
                seg_mask = (action_ids == aid).float() * reg_mask  # (bs, seq_len)
                seg_disp = (token_disp_detached * seg_mask).sum(dim=-1)  # (bs,)
                seg_active = (seg_disp > 0).float()  # (bs,) - gate per response for this action
                # Broadcast gate back to tokens
                action_gate_mask += seg_mask * seg_active.unsqueeze(-1)
                # Stats
                seg_exists = (seg_mask.sum(dim=-1) > 0).float()  # which batch items have this action
                total_actions += (seg_exists * preserve_mask).sum().item()
                num_active_actions += (seg_active * seg_exists * preserve_mask).sum().item()

            # Combined: preserve_mask (response-level) * action_gate_mask (action-level per-token)
            active_token_mask = reg_mask * action_gate_mask * preserve_mask.unsqueeze(-1)
            total_active_tokens = active_token_mask.sum()
        else:
            # Fallback: response-level gate (original behavior)
            response_displacement = token_disp_detached.sum(dim=-1)  # (bs,)
            response_gate = (response_displacement > 0).float()  # (bs,)
            active_mask = preserve_mask * response_gate  # (bs,)
            active_token_mask = reg_mask * active_mask.unsqueeze(-1)
            total_active_tokens = active_token_mask.sum()
            num_active_actions = active_mask.sum().item()
            total_actions = preserve_mask.sum().item()

    # Token-level penalty (WITH gradient through new_log_probs)
    token_displacement = (old_log_probs - new_log_probs) * reg_mask
    token_penalty = torch.clamp(token_displacement, min=0.0)

    # Apply action-level gate mask and normalize
    masked_penalty = token_penalty * active_token_mask
    if total_active_tokens > 0:
        llds_loss = masked_penalty.sum() / total_active_tokens
    else:
        llds_loss = (token_penalty * 0.0).sum()  # zero but keeps computation graph

    metrics = {
        'llds/num_active_actions': num_active_actions,
        'llds/total_actions': total_actions,
        'llds/num_preserved': preserve_mask.sum().item(),
        'llds/mean_token_disp': token_disp_detached.sum(dim=-1).mean().item(),
        'llds/loss': llds_loss.item(),
    }

    return llds_loss, metrics
