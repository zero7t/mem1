# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        # Disable torch.compile - causes hangs with FSDP ref model forward
        self.compute_entropy_from_logits = verl_F.entropy_from_logits

    def _forward_micro_batch(self, micro_batch, temperature, compute_entropy=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len) or None if compute_entropy=False
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            if 'attention_mask_4d' in micro_batch:
                attention_mask_4d = micro_batch['attention_mask_4d'].to(torch.bfloat16)
            else:
                attention_mask_4d = None
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad) if compute_entropy else None

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if entropy_rmpad is not None:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if entropy_rmpad is not None:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                else:
                    entropy = None
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask if attention_mask_4d is None else attention_mask_4d,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits) if compute_entropy else None

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        if 'attention_mask_4d' in data.batch:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'attention_mask_4d', 'position_ids']
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)
        log_probs_lst = []
        for i, micro_batch in enumerate(micro_batches):
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, compute_entropy=False)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        # Log peak GPU memory during compute_log_prob
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[GPU_MEM compute_log_prob] peak={peak_mem_gb:.2f}GB")

        return log_probs

    def update_policy(self, data: DataProto):
        """
        [IMPROVED] update_policy with LLDS-MA + DAPO Clip-Higher.

        New config fields (optional, with defaults):
            self.config.llds_lambda: float (default 0.1)
            self.config.clip_higher: float (default 0.28)
        """
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        # [LLDS] Config
        llds_lambda = self.config.get('llds_lambda', 0.1)
        clip_higher = self.config.get('clip_higher', 0.28)

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff_base = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                # [DAPO] Clip-Higher: asymmetric clipping
                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio,
                                                                              clip_higher=clip_higher)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # [Adaptive Entropy] Scale up entropy_coeff when entropy drops below threshold
                entropy_target = getattr(self.config, 'entropy_target', 0.6)
                entropy_coeff_max = getattr(self.config, 'entropy_coeff_max', 0.05)
                entropy_val = entropy_loss.detach().item()
                if entropy_val < entropy_target:
                    # Linear ramp: coeff increases as entropy drops further below target
                    # At entropy=0, coeff = entropy_coeff_max; at entropy=target, coeff = base
                    ratio = max(0.0, (entropy_target - entropy_val) / entropy_target)
                    entropy_coeff = entropy_coeff_base + ratio * (entropy_coeff_max - entropy_coeff_base)
                else:
                    entropy_coeff = entropy_coeff_base

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                # [LLDS Action-Level] Likelihood-preserving regularization
                if llds_lambda > 0:
                    # Pass loss_mask for action-level gating (identifies action boundaries)
                    action_loss_mask = data['loss_mask'] if self.config.state_masking else None
                    llds_loss, llds_metrics = core_algos.compute_llds_loss(
                        old_log_probs=old_log_prob,
                        new_log_probs=log_prob,
                        eos_mask=response_mask,
                        advantages=advantages,
                        loss_mask=action_loss_mask,
                        answer_mask=None,
                        mask_answer=False
                    )
                    policy_loss = policy_loss + llds_lambda * llds_loss
                    append_to_dict(metrics, {
                        'llds/loss': llds_metrics['llds/loss'],
                        'llds/num_active_actions': llds_metrics['llds/num_active_actions'],
                        'llds/total_actions': llds_metrics['llds/total_actions'],
                    })

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                # [LLD Monitor] Log likelihood statistics for detecting LLD
                with torch.no_grad():
                    mean_new_log_prob = (log_prob * response_mask).sum() / response_mask.sum()
                    mean_old_log_prob = (old_log_prob * response_mask).sum() / response_mask.sum()
                    # Per-response displacement: positive means likelihood decreased
                    resp_disp = ((old_log_prob - log_prob) * response_mask).sum(dim=-1)
                    num_displaced = (resp_disp > 0).sum().item()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/entropy_coeff': entropy_coeff,
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                    'actor/mean_log_prob': mean_new_log_prob.item(),
                    'actor/mean_old_log_prob': mean_old_log_prob.item(),
                    'actor/lld_displaced_responses': num_displaced,
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()

        # Log peak GPU memory
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            current_mem_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_mem_gb = torch.cuda.memory_reserved() / (1024**3)
            append_to_dict(metrics, {
                'actor/peak_memory_gb': peak_mem_gb,
                'actor/current_memory_gb': current_mem_gb,
                'actor/reserved_memory_gb': reserved_mem_gb,
            })
            print(f"[GPU_MEM] peak={peak_mem_gb:.2f}GB, current={current_mem_gb:.2f}GB, reserved={reserved_mem_gb:.2f}GB")
            torch.cuda.reset_peak_memory_stats()

        return metrics
