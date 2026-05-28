"""
Improved GRPO core algorithms:
1. DAPO-lite: Dynamic Sampling + Clip-Higher
2. Dr.GRPO: Stable advantage normalization (no std division for binary rewards)
3. LLDS-MA: Likelihood-preserving regularization from the paper

Drop-in replacement for verl/trainer/ppo/core_algos.py
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


# ============================================================================
# Original unchanged utilities
# ============================================================================

class AdaptiveKLController:
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
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam):
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
    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_entropy_loss(logits, eos_mask):
    entropy = verl_F.entropy_from_logits(logits)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "kl":
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    raise NotImplementedError


# ============================================================================
# [Dr.GRPO] Improved advantage computation
# - No division by group std for binary/sparse rewards (avoids amplification)
# - Adds epsilon floor to std to prevent explosion
# - Dynamic sampling: groups with zero variance get zero advantage
# ============================================================================

def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   # Dr.GRPO options
                                   use_std_normalization: bool = False,
                                   std_floor: float = 0.1,
                                   # DAPO dynamic sampling
                                   skip_zero_variance: bool = True):
    """
    Improved GRPO advantage computation with Dr.GRPO fixes.

    Changes from vanilla:
    1. By default, does NOT divide by group std (Dr.GRPO style for binary rewards)
       - Only subtracts group mean as baseline
       - This avoids amplifying noisy groups with small variance
    2. Groups where all rewards are identical get zero advantage (dynamic sampling)
       - Saves gradient budget on uninformative groups
    3. Optional: if use_std_normalization=True, uses std with a floor
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # per-sample scalar reward

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
            else:
                id2mean[idx] = np.mean(group_scores)
                id2std[idx] = np.std(group_scores)

        for i in range(bsz):
            idx = index[i]
            group_std = id2std[idx]

            # [DAPO] Dynamic sampling: skip groups with zero variance
            if skip_zero_variance and group_std < epsilon:
                scores[i] = 0.0
            else:
                # [Dr.GRPO] Subtract mean, optionally normalize by std
                advantage = scores[i].item() - id2mean[idx]
                if use_std_normalization:
                    # Use std with a floor to prevent explosion
                    advantage = advantage / max(group_std, std_floor)
                scores[i] = advantage

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


# ============================================================================
# [DAPO] Clip-Higher: Asymmetric clipping for policy loss
# ============================================================================

def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange,
                        clip_higher: float = None):
    """
    PPO-style policy loss with optional DAPO Clip-Higher.

    Args:
        cliprange: lower clip range (e.g., 0.2)
        clip_higher: upper clip range (e.g., 0.28). If None, uses cliprange (symmetric).

    DAPO Clip-Higher:
        - Positive advantage samples get more room to increase likelihood ratio
        - clip_low = cliprange (e.g., 0.2)
        - clip_high = clip_higher (e.g., 0.28)
        - This allows "good" trajectories to get stronger reinforcement
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


# ============================================================================
# [LLDS-MA] Likelihood-preserving regularization
# From: "On GRPO Collapse in Search-R1" (Deng et al., 2025)
#
# L_LLDS = (1/sum|y_i|) * sum_{y_i in Y_pre}
#           1[sum_token(old_lp - new_lp) > 0]   <-- response-level gate
#           * sum_token max(0, old_lp - new_lp)  <-- token-level penalty
#
# LLDS-MA variant: exclude answer tokens from the regularization
# ============================================================================

def compute_llds_loss(old_log_probs: torch.Tensor,
                      new_log_probs: torch.Tensor,
                      eos_mask: torch.Tensor,
                      advantages: torch.Tensor,
                      answer_mask: torch.Tensor = None,
                      mask_answer: bool = True):
    """
    Compute LLDS(-MA) regularization loss.

    Args:
        old_log_probs: (bs, response_length) - log probs from old policy (before update)
        new_log_probs: (bs, response_length) - log probs from current policy
        eos_mask: (bs, response_length) - valid token mask
        advantages: (bs, response_length) - per-token advantages (used to determine Y_pre)
        answer_mask: (bs, response_length) - mask where answer tokens are 1, others 0
                     If None and mask_answer=True, no masking is applied
        mask_answer: whether to exclude answer tokens (LLDS-MA mode)

    Returns:
        llds_loss: scalar tensor
        llds_metrics: dict with monitoring metrics
    """
    with torch.no_grad():
        # Determine preserving set Y_pre: responses with non-negative advantage
        # For GRPO, all tokens in same response share the same advantage value
        # Use the first valid token's advantage to determine response-level advantage
        response_advantages = advantages[:, 0]  # (bs,) - same for all tokens in response
        preserve_mask = (response_advantages >= 0).float()  # (bs,)

    # Token-level likelihood displacement: old_lp - new_lp
    # Positive means likelihood decreased (bad)
    token_displacement = old_log_probs - new_log_probs  # (bs, response_length)

    # Apply eos_mask
    token_displacement = token_displacement * eos_mask

    # [LLDS-MA] Optionally mask out answer tokens
    reg_mask = eos_mask.clone()
    if mask_answer and answer_mask is not None:
        reg_mask = reg_mask * (1.0 - answer_mask)  # exclude answer tokens

    # Response-level gate: activate only when total likelihood decreased
    # sum of (old - new) > 0 means overall likelihood went down
    response_displacement = (token_displacement * reg_mask).sum(dim=-1)  # (bs,)
    response_gate = (response_displacement > 0).float()  # (bs,)

    # Combine with preserve mask
    active_mask = preserve_mask * response_gate  # (bs,)

    # Token-level penalty: max(0, old_lp - new_lp) for likelihood-reducing tokens
    token_penalty = torch.clamp(token_displacement, min=0.0)  # (bs, response_length)
    token_penalty = token_penalty * reg_mask  # apply answer masking

    # Per-response penalty sum
    response_penalty = (token_penalty).sum(dim=-1)  # (bs,)

    # Apply active mask and normalize
    total_tokens = (reg_mask * active_mask.unsqueeze(-1)).sum()
    if total_tokens > 0:
        llds_loss = (response_penalty * active_mask).sum() / total_tokens
    else:
        llds_loss = torch.tensor(0.0, device=old_log_probs.device)

    # Metrics for monitoring
    metrics = {
        'llds/num_active_responses': active_mask.sum().item(),
        'llds/total_responses': preserve_mask.sum().item(),
        'llds/mean_displacement': response_displacement.mean().item(),
        'llds/loss': llds_loss.item(),
    }

    return llds_loss, metrics
