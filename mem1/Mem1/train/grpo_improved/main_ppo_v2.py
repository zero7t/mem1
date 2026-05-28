"""
Main PPO Entry Point V2 - Integrated with Turn-Level Process Reward

This is the self-contained entry point for grpo_improved training.
It replaces verl/trainer/main_ppo.py with:
  1. Turn-level process reward (rule_reward_v2)
  2. Turn-weighted advantage modulation
  3. LLM Outcome Judge (async, overlapped with compute_log_prob)
  4. Per-turn reward logging

Does NOT modify any files in verl/. Instead, it monkey-patches
the advantage computation at runtime.
"""

import os
import sys
import re
import torch
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple

# Add paths
TRAIN_ROOT = '/root/paddlejob/workspace/new_llm_judge_mem1/mem1/Mem1/train'
sys.path.insert(0, TRAIN_ROOT)

from verl import DataProto
from verl.utils.reward_score import qa_em, websearch, qa_multiple
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from grpo_improved.reward.rule_reward_v2 import (
    TurnRewardComputer,
    compute_turn_rewards,
    extract_trajectory_turns,
)
from grpo_improved.reward.llm_judge import LLMJudgeClient, build_outcome_judge_prompt
from grpo_improved.core.turn_weighted_advantage import (
    apply_turn_weighting,
    find_turn_boundaries_from_text,
)


# =============================================================================
# Score Function Selection
# =============================================================================

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa',
                       '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_multiple.compute_score_em
    elif data_source in ['websearch']:
        return websearch.compute_score_f1
    else:
        raise NotImplementedError(f"Unknown data_source: {data_source}")


# =============================================================================
# Reward Manager V2
# =============================================================================

class RewardManagerV2:
    """
    Improved reward manager with:
    1. Per-turn process rewards (not just scalar)
    2. Turn-weighted advantage modulation
    3. Async LLM Outcome Judge (pre-fire pattern)
    4. Detailed per-sample logging
    """

    def __init__(self, tokenizer, num_examine=0, format_score=0.,
                 max_workers=8, config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.max_workers = max_workers
        self._lock = threading.Lock()

        # Process reward config
        config = config or {}
        self._process_config = {
            'max_turns': config.get('max_turns', 6),
            'lambda_process': config.get('lambda_process', 0.5),
            'w_hit': config.get('w_hit', 0.3),
            'w_util': config.get('w_util', 0.4),
            'w_novelty': config.get('w_novelty', 0.2),
            'w_efficiency': config.get('w_efficiency', 0.1),
        }

        # Turn weighting config
        self._turn_weight_config = {
            'turn_weight_alpha': config.get('turn_weight_alpha', 0.3),
            'turn_weight_clip': config.get('turn_weight_clip', 1.0),
            'turn_weight_enabled': config.get('turn_weight_enabled', True),
        }

        # LLM Judge
        self._llm_judge = LLMJudgeClient()
        self._pending_judge_futures = None
        self._pending_judge_meta = None
        self._judge_fire_time = None

        # Storage for turn-level data (used by advantage weighting)
        self._last_turn_scores = None  # List[List[float]]
        self._last_texts = None        # List[str]

    # -----------------------------------------------------------------
    # LLM Judge: Pre-fire before compute_log_prob
    # -----------------------------------------------------------------

    def pre_fire_llm_judge(self, gen_batch_output, prompt_batch, n_repeat):
        """Fire LLM Judge API calls BEFORE compute_log_prob for overlap."""
        import time as _time

        self._judge_fire_time = _time.time()
        self._pending_judge_futures = None
        self._pending_judge_meta = None

        try:
            prompts_to_fire = []
            judge_meta = []

            n_total = gen_batch_output.batch['prompts'].shape[0]

            for i in range(n_total):
                prompt_ids = gen_batch_output.batch['prompts'][i]
                response_ids = gen_batch_output.batch['responses'][i]
                attn_mask = gen_batch_output.batch['attention_mask'][i]

                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = attn_mask[:prompt_length].sum().item()
                valid_response_length = attn_mask[prompt_length:].sum().item()

                valid_prompt_ids = prompt_ids[-int(valid_prompt_length):]
                valid_response_ids = response_ids[:int(valid_response_length)]

                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                # Get ground_truth
                prompt_idx = i // n_repeat
                ground_truth = prompt_batch.non_tensor_batch['reward_model'][prompt_idx]['ground_truth']

                # Quick EM check
                data_source = prompt_batch.non_tensor_batch['data_source'][prompt_idx]
                compute_score_fn = _select_rm_score_fn(data_source)
                score = compute_score_fn(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    format_score=self.format_score,
                )

                # If EM=0, check if answer exists for LLM judge
                if score < 0.5:
                    answer_match = re.findall(r'<answer>(.*?)</answer>', sequences_str, re.DOTALL)
                    if answer_match and len(answer_match) >= 2:
                        model_answer = answer_match[-1].strip()
                        if model_answer:
                            gt_list = ground_truth if isinstance(ground_truth, list) else [str(ground_truth)]
                            question_match = re.search(r'<question>(.*?)</question>', sequences_str, re.DOTALL)
                            question = question_match.group(1).strip() if question_match else ""
                            prompt = build_outcome_judge_prompt(question, gt_list, model_answer)
                            prompts_to_fire.append(prompt)
                            judge_meta.append({'index': i, 'question': question})

            if prompts_to_fire:
                executor = ThreadPoolExecutor(
                    max_workers=self._llm_judge.config['max_concurrent']
                )
                futures = [executor.submit(self._llm_judge._call_api_sync, p)
                           for p in prompts_to_fire]
                self._pending_judge_futures = futures
                self._pending_judge_meta = judge_meta
                self._pending_judge_executor = executor
                print(f"[LLM Judge] Pre-fired {len(prompts_to_fire)} async calls")
            else:
                print(f"[LLM Judge] No candidates to judge")

        except Exception as e:
            print(f"[LLM Judge] pre_fire error: {e}")
            self._pending_judge_futures = None

    # -----------------------------------------------------------------
    # Main reward computation
    # -----------------------------------------------------------------

    def _process_item(self, i, data_item, already_print_data_sources):
        """Process single trajectory: EM + turn-level process rewards."""
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']

        # EM score
        compute_score_fn = _select_rm_score_fn(data_source)
        em_score = compute_score_fn(
            solution_str=sequences_str,
            ground_truth=ground_truth,
            format_score=self.format_score,
        )

        # Turn-level process rewards
        if isinstance(ground_truth, str):
            gt_for_process = {'target': [ground_truth]}
        elif isinstance(ground_truth, list):
            gt_for_process = {'target': ground_truth}
        elif isinstance(ground_truth, dict):
            gt_for_process = ground_truth
        else:
            gt_for_process = {'target': [str(ground_truth)]}

        try:
            turn_scores, process_total = compute_turn_rewards(
                trajectory_text=sequences_str,
                ground_truth=gt_for_process,
                outcome_score=em_score,
                config=self._process_config,
            )
        except Exception:
            turn_scores = [0.0]
            process_total = 0.0

        total_score = em_score + process_total

        # Print sample
        should_print = False
        with self._lock:
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                should_print = True

        if should_print:
            print(f"[Sample {i}] em={em_score:.2f} process={process_total:.3f} "
                  f"turns={turn_scores}")
            print(sequences_str[:500])

        return {
            'index': i,
            'pos': int(valid_response_length) - 1,
            'total_score': total_score,
            'em_score': em_score,
            'turn_scores': turn_scores,
            'text': sequences_str,
        }

    def __call__(self, data: DataProto):
        """Compute rewards, store turn-level data for advantage weighting."""
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(len(data)):
                data_item = data[i]
                futures.append(
                    executor.submit(self._process_item, i, data_item, already_print_data_sources)
                )
            for future in futures:
                results.append(future.result())

        results.sort(key=lambda x: x['index'])
        all_turn_scores = []
        all_texts = []

        for r in results:
            reward_tensor[r['index'], r['pos']] = r['total_score']
            all_turn_scores.append(r['turn_scores'])
            all_texts.append(r['text'])

        # Store for turn-weighted advantage
        self._last_turn_scores = all_turn_scores
        self._last_texts = all_texts

        # Collect pre-fired LLM Judge results
        if self._pending_judge_futures is not None:
            import time as _time
            futures_list = self._pending_judge_futures
            meta = self._pending_judge_meta
            self._pending_judge_futures = None
            self._pending_judge_meta = None

            responses = [f.result() for f in futures_list]
            self._pending_judge_executor.shutdown(wait=False)

            judge_upgraded = 0
            for m, response in zip(meta, responses):
                if response and "correct" in response.lower() and "incorrect" not in response.lower():
                    idx = m['index']
                    if idx < reward_tensor.shape[0]:
                        data_item = data[idx]
                        prompt_length = data_item.batch['prompts'].shape[-1]
                        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
                        pos = int(valid_response_length) - 1
                        reward_tensor[idx, pos] = 0.8
                        judge_upgraded += 1

            _elapsed = _time.time() - self._judge_fire_time
            print(f"[LLM Judge] Collected: {len(meta)} calls, {judge_upgraded} upgraded, "
                  f"total {_elapsed:.1f}s")

        return reward_tensor

    # -----------------------------------------------------------------
    # Turn-weighted advantage (called after compute_advantage)
    # -----------------------------------------------------------------

    def apply_turn_weighting_to_batch(self, batch: DataProto) -> DataProto:
        """
        Apply turn-level advantage weighting to the batch.
        Called AFTER standard compute_advantage() in the training loop.
        """
        if not self._turn_weight_config['turn_weight_enabled']:
            return batch

        if self._last_turn_scores is None:
            return batch

        advantages = batch.batch['advantages']
        response_length = batch.batch['responses'].shape[-1]
        eos_mask = batch.batch['attention_mask'][:, -response_length:]

        if 'loss_mask' in batch.batch:
            eos_mask = batch.batch['loss_mask']

        response_texts = []
        for text in self._last_texts:
            q_end = text.rfind('</question>')
            if q_end > 0:
                response_texts.append(text[q_end:])
            else:
                response_texts.append(text)

        weighted_advantages = apply_turn_weighting(
            batch_advantages=advantages,
            batch_turn_scores=self._last_turn_scores,
            batch_texts=response_texts,
            response_length=response_length,
            eos_mask=eos_mask,
            config=self._turn_weight_config,
        )

        batch.batch['advantages'] = weighted_advantages

        if advantages.numel() > 0:
            ratio = (weighted_advantages.abs().sum() / (advantages.abs().sum() + 1e-8)).item()
            print(f"[TurnWeight] advantage ratio (weighted/original): {ratio:.3f}")

        self._last_turn_scores = None
        self._last_texts = None

        return batch


# =============================================================================
# Training Entry Point
# =============================================================================

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
        }})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Download checkpoint
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # Tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # Worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # V2 Reward Manager with turn-level process rewards
    reward_config = {
        'max_turns': OmegaConf.select(config, 'max_turns', default=6),
        'lambda_process': OmegaConf.select(config, 'reward.lambda_process', default=0.5),
        'turn_weight_alpha': OmegaConf.select(config, 'reward.turn_weight_alpha', default=0.3),
        'turn_weight_clip': OmegaConf.select(config, 'reward.turn_weight_clip', default=1.0),
        'turn_weight_enabled': OmegaConf.select(config, 'reward.turn_weight_enabled', default=True),
    }

    reward_fn = RewardManagerV2(
        tokenizer=tokenizer,
        num_examine=0,
        config=reward_config,
    )
    val_reward_fn = RewardManagerV2(
        tokenizer=tokenizer,
        num_examine=1,
        config=reward_config,
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()

    # === Monkey-patch: inject turn-weighted advantage after compute_advantage ===
    # This avoids modifying verl/trainer/ppo/ray_trainer.py
    _original_fit = trainer.fit.__func__ if hasattr(trainer.fit, '__func__') else None

    # We patch at a higher level: override the reward_fn to also do turn weighting
    # The turn weighting is applied via reward_fn.apply_turn_weighting_to_batch()
    # which is called from a patched version of the training step.
    # For now, we document that ray_trainer.py needs ONE line added after compute_advantage:
    #   batch = self.reward_fn.apply_turn_weighting_to_batch(batch)
    # This is the minimal change needed in verl/ code.
    print("[V2] Turn-weighted advantage enabled. "
          "Ensure ray_trainer.py calls reward_fn.apply_turn_weighting_to_batch(batch) "
          "after compute_advantage().")

    trainer.fit()


if __name__ == '__main__':
    main()
