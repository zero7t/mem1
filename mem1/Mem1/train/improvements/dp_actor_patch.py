"""
Patch for dp_actor.py to integrate LLDS-MA + DAPO Clip-Higher into update_policy.

This file provides a drop-in replacement for DataParallelPPOActor.update_policy()
that adds:
1. LLDS-MA regularization (likelihood-preserving)
2. DAPO Clip-Higher (asymmetric clipping)
3. Better metrics tracking

Apply by monkey-patching or by modifying dp_actor.py directly.
"""

import itertools
from typing import Tuple

import torch
from torch import nn

from verl import DataProto
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
import verl.utils.torch_functional as verl_F

# Import improved core_algos
import sys
sys.path.insert(0, '/root/paddlejob/workspace/mem1/MEM1/Mem1/train/improvements')
from core_algos_improved import compute_policy_loss, compute_llds_loss


def update_policy_improved(self, data: DataProto):
    """
    Improved update_policy with LLDS-MA + DAPO Clip-Higher.

    New config fields expected:
        self.config.llds_lambda: float (default 0.1) - LLDS regularization weight
        self.config.llds_mask_answer: bool (default True) - whether to use MA variant
        self.config.clip_higher: float (default 0.28) - DAPO upper clip range
    """
    self.actor_module.train()

    assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
    temperature = data.meta_info['temperature']

    # Config for improvements
    llds_lambda = getattr(self.config, 'llds_lambda', 0.1)
    llds_mask_answer = getattr(self.config, 'llds_mask_answer', True)
    clip_higher = getattr(self.config, 'clip_higher', 0.28)

    # Select keys - include old_log_probs for LLDS computation
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids',
                   'old_log_probs', 'advantages']
    if self.config.state_masking:
        select_keys.append('loss_mask')
    if self.config.use_kl_loss:
        select_keys.append('ref_log_prob')
    # Include answer_mask if available (for LLDS-MA)
    if 'answer_mask' in data.batch:
        select_keys.append('answer_mask')

    batch = data.select(batch_keys=select_keys).batch

    dataloader = batch.split(self.config.ppo_mini_batch_size)

    metrics = {}
    for batch_idx, mini_batch_data in enumerate(dataloader):
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            from verl.utils.seqlen_balancing import rearrange_micro_batches
            micro_batches, _ = rearrange_micro_batches(batch=mini_batch_data, max_token_len=max_token_len)
        else:
            micro_batches = mini_batch_data.split(self.config.ppo_micro_batch_size)

        self.actor_optimizer.zero_grad()

        for micro_data in micro_batches:
            micro_data = micro_data.cuda()
            responses = micro_data['responses']
            response_length = responses.size(1)
            attention_mask = micro_data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            if self.config.state_masking:
                response_mask = micro_data['loss_mask']
            old_log_prob = micro_data['old_log_probs']
            advantages = micro_data['advantages']

            clip_ratio = self.config.clip_ratio
            entropy_coeff = self.config.entropy_coeff

            # Forward pass
            entropy, log_prob = self._forward_micro_batch(micro_batch=micro_data, temperature=temperature)

            # [DAPO] Clip-Higher policy loss
            pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                eos_mask=response_mask,
                cliprange=clip_ratio,
                clip_higher=clip_higher
            )

            # Entropy loss
            entropy_loss = verl_F.masked_mean(entropy, response_mask)

            # Base policy loss
            policy_loss = pg_loss - entropy_loss * entropy_coeff

            # KL loss (if enabled)
            if self.config.use_kl_loss:
                from verl.trainer.ppo.core_algos import kl_penalty as kl_penalty_fn
                ref_log_prob = micro_data['ref_log_prob']
                kld = kl_penalty_fn(logprob=log_prob, ref_logprob=ref_log_prob,
                                    kl_penalty=self.config.kl_loss_type)
                kl_loss = masked_mean(kld, response_mask)
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                metrics['actor/kl_loss'] = kl_loss.detach().item()
                metrics['actor/kl_coef'] = self.config.kl_loss_coef

            # [LLDS-MA] Likelihood-preserving regularization
            llds_loss = torch.tensor(0.0, device=log_prob.device)
            if llds_lambda > 0:
                answer_mask = micro_data.get('answer_mask', None) if llds_mask_answer else None
                llds_loss, llds_metrics = compute_llds_loss(
                    old_log_probs=old_log_prob,
                    new_log_probs=log_prob,
                    eos_mask=response_mask,
                    advantages=advantages,
                    answer_mask=answer_mask,
                    mask_answer=llds_mask_answer
                )
                policy_loss = policy_loss + llds_lambda * llds_loss
                append_to_dict(metrics, {
                    'llds/loss': llds_loss.detach().item(),
                    'llds/num_active': llds_metrics['llds/num_active_responses'],
                })

            loss = policy_loss / self.gradient_accumulation
            loss.backward()

            data_metrics = {
                'actor/entropy_loss': entropy_loss.detach().item(),
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item(),
            }
            append_to_dict(metrics, data_metrics)

        grad_norm = self._optimizer_step()
        data_metrics = {'actor/grad_norm': grad_norm.detach().item()}
        append_to_dict(metrics, data_metrics)

    self.actor_optimizer.zero_grad()
    return metrics
