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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, websearch, qa_multiple
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import sys

# Add grpo_improved to path for reward modules
sys.path.insert(0, '/root/paddlejob/workspace/mem1/MEM1/Mem1/train')
from grpo_improved.reward.rule_reward import compute_process_reward
from grpo_improved.reward.llm_judge import LLMJudgeClient, build_outcome_judge_prompt

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        # return qa_em.compute_score_em
        return qa_multiple.compute_score_em
        # return qa_multiple.model_estimated_match_score
        # return qa_em.model_estimated_match_score
        # return websearch.compute_score_f1
    elif data_source in ['websearch']:
        return websearch.compute_score_f1
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., max_workers=8) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.max_workers = 8  # concurrent workers for reward computation
        self._lock = threading.Lock()
        # LLM Outcome Judge client
        self._llm_judge = LLMJudgeClient()
        # Process reward config
        self._process_reward_config = {
            'lambda_process': 0.5,
            'max_turns': 6,
        }
        # Async LLM Judge state
        self._pending_judge_futures = None
        self._pending_judge_meta = None
        self._judge_fire_time = None
        # Metrics for logging
        self._step_metrics = {}

    def pre_fire_llm_judge(self, gen_batch_output, prompt_batch, n_repeat):
        """Pre-fire LLM Judge API calls BEFORE compute_log_prob.
        Extracts text from generated trajectories and fires async API calls.
        Results are collected later in __call__ via collect_llm_judge().

        This overlaps LLM Judge latency (~50s) with compute_log_prob (~40s).
        """
        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        self._judge_fire_time = _time.time()
        self._pending_judge_futures = None
        self._pending_judge_meta = None

        try:
            # Decode all trajectories to find those needing LLM judge
            prompts_to_fire = []
            judge_meta = []

            # gen_batch_output has shape [batch_size * n_repeat, seq_len]
            # prompt_batch has ground_truth for [batch_size] items
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

                # Get ground_truth (prompt_batch is pre-repeat, so index = i // n_repeat)
                prompt_idx = i // n_repeat
                ground_truth = prompt_batch.non_tensor_batch['reward_model'][prompt_idx]['ground_truth']

                # Quick EM check
                data_source = prompt_batch.non_tensor_batch['data_source'][prompt_idx]
                compute_score_fn = _select_rm_score_fn(data_source)
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

                # If EM=0, check if answer exists for LLM judge
                if score < 0.5:
                    answer_match = re.findall(r'<answer>(.*?)</answer>', sequences_str, re.DOTALL)
                    if answer_match and len(answer_match) >= 2:
                        model_answer = answer_match[-1].strip()
                        if model_answer:
                            if isinstance(ground_truth, str):
                                gt_list = [ground_truth]
                            elif isinstance(ground_truth, list):
                                gt_list = ground_truth
                            else:
                                gt_list = [str(ground_truth)]
                            question_match = re.search(r'<question>(.*?)</question>', sequences_str, re.DOTALL)
                            question = question_match.group(1).strip() if question_match else ""
                            prompt = build_outcome_judge_prompt(question, gt_list, model_answer)
                            prompts_to_fire.append(prompt)
                            judge_meta.append({'index': i, 'question': question})

            if prompts_to_fire:
                # Fire all API calls in background (non-blocking)
                executor = ThreadPoolExecutor(max_workers=self._llm_judge.config['max_concurrent'])
                futures = [executor.submit(self._llm_judge._call_api_sync, p) for p in prompts_to_fire]
                self._pending_judge_futures = futures
                self._pending_judge_meta = judge_meta
                self._pending_judge_executor = executor
                print(f"[LLM Judge] Pre-fired {len(prompts_to_fire)} async calls before compute_log_prob")
            else:
                print(f"[LLM Judge] No candidates to judge (all EM=1 or no answers)")

        except Exception as e:
            print(f"[LLM Judge] pre_fire error: {e}")
            self._pending_judge_futures = None

    def collect_llm_judge(self, reward_tensor):
        """Collect pre-fired LLM Judge results and update reward_tensor.
        Called after compute_log_prob completes.
        """
        if self._pending_judge_futures is None:
            return reward_tensor

        import time as _time

        futures = self._pending_judge_futures
        meta = self._pending_judge_meta
        self._pending_judge_futures = None
        self._pending_judge_meta = None

        # Wait for all futures
        responses = [f.result() for f in futures]
        self._pending_judge_executor.shutdown(wait=False)

        # Apply results
        judge_upgraded = 0
        upgraded_indices = []
        for m, response in zip(meta, responses):
            if response and "correct" in response.lower() and "incorrect" not in response.lower():
                judge_upgraded += 1
                upgraded_indices.append(m['index'])

        _elapsed = _time.time() - self._judge_fire_time
        print(f"[LLM Judge] Collected: {len(meta)} calls, {judge_upgraded} upgraded, "
              f"total {_elapsed:.1f}s (overlapped with compute_log_prob)")

        # Return upgraded indices so __call__ can apply them
        self._judge_upgraded_indices = set(upgraded_indices)
        return reward_tensor

    def _process_item(self, i, data_item, already_print_data_sources):
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

        # select rm_score
        data_source = data_item.non_tensor_batch['data_source']
        compute_score_fn = _select_rm_score_fn(data_source)
        try:
            format_rewards = data_item.batch['batch_rewards']
        except:
            format_rewards = None

        score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
        if format_rewards:
            score = score + format_rewards

        # === Rule Process Reward ===
        process_score = 0.0
        try:
            # ground_truth can be str or list; wrap in dict for rule_reward
            if isinstance(ground_truth, str):
                gt_for_process = {'target': [ground_truth]}
            elif isinstance(ground_truth, list):
                gt_for_process = {'target': ground_truth}
            elif isinstance(ground_truth, dict):
                gt_for_process = ground_truth
            else:
                gt_for_process = {'target': [str(ground_truth)]}

            process_score = compute_process_reward(
                trajectory_text=sequences_str,
                ground_truth=gt_for_process,
                outcome_score=score,
                config=self._process_reward_config,
            )
            score = score + process_score
        except Exception as e:
            if not getattr(self, '_process_reward_error_logged', False):
                print(f"[ProcessReward] ERROR (first occurrence): {type(e).__name__}: {e}")
                self._process_reward_error_logged = True

        # === LLM Outcome Judge: collect candidates for batch processing ===
        # Collect info for batch LLM judge (done after all items processed)
        llm_judge_info = None
        if score < 0.5:
            try:
                answer_match = re.findall(r'<answer>(.*?)</answer>', sequences_str, re.DOTALL)
                if answer_match and len(answer_match) >= 2:
                    model_answer = answer_match[-1].strip()
                    if model_answer:
                        if isinstance(ground_truth, str):
                            gt_list = [ground_truth]
                        elif isinstance(ground_truth, list):
                            gt_list = ground_truth
                        else:
                            gt_list = [str(ground_truth)]
                        question_match = re.search(r'<question>(.*?)</question>', sequences_str, re.DOTALL)
                        question = question_match.group(1).strip() if question_match else ""
                        llm_judge_info = {
                            'question': question,
                            'gt_list': gt_list,
                            'model_answer': model_answer,
                            'process_score': process_score if 'process_score' in dir() else 0.0,
                        }
            except Exception:
                pass

        should_print = False
        with self._lock:
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                should_print = True

        if should_print:
            print(sequences_str)

        em_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=0.)
        return i, valid_response_length - 1, score, llm_judge_info, process_score, em_score

    def __call__(self, data: DataProto):
        """Compute EM + Rule rewards, then collect pre-fired LLM Judge results.
        LLM Judge was already fired in pre_fire_llm_judge() before compute_log_prob.
        """

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        # Metrics accumulators
        all_process_scores = []
        all_em_scores = []

        # Compute EM + Rule Rewards (parallel, fast, no API calls)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(len(data)):
                data_item = data[i]
                futures.append(executor.submit(self._process_item, i, data_item, already_print_data_sources))

            for future in futures:
                i, pos, score, _, process_score, em_score = future.result()
                reward_tensor[i, pos] = score
                all_process_scores.append(process_score)
                all_em_scores.append(em_score)

        # Collect pre-fired LLM Judge results (fired before compute_log_prob)
        judge_fired = 0
        judge_upgraded = 0
        if self._pending_judge_futures is not None:
            import time as _time
            futures_list = self._pending_judge_futures
            meta = self._pending_judge_meta
            self._pending_judge_futures = None
            self._pending_judge_meta = None

            # Wait for remaining futures (most should already be done)
            responses = [f.result() for f in futures_list]
            self._pending_judge_executor.shutdown(wait=False)

            judge_fired = len(meta)
            # Apply LLM judge: upgrade score for samples judged "correct"
            for m, response in zip(meta, responses):
                if response and "correct" in response.lower() and "incorrect" not in response.lower():
                    idx = m['index']
                    if idx < reward_tensor.shape[0]:
                        # Find position where score was placed
                        data_item = data[idx]
                        prompt_length = data_item.batch['prompts'].shape[-1]
                        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
                        pos = int(valid_response_length) - 1
                        reward_tensor[idx, pos] = 0.8
                        judge_upgraded += 1

            _elapsed = _time.time() - self._judge_fire_time
            print(f"[LLM Judge] Collected: {len(meta)} calls, {judge_upgraded} upgraded, "
                  f"total {_elapsed:.1f}s (overlapped with compute_log_prob)")

        # Apply generation-time judge results (fired DURING generation, already returned)
        gen_judge_fired = 0
        gen_judge_upgraded = 0
        gen_judge_stats = getattr(self, '_gen_judge_stats', {})
        gen_judge_candidates = getattr(self, '_gen_judge_upgraded', None)
        if gen_judge_candidates:
            gen_judge_fired = gen_judge_stats.get('fired', len(gen_judge_candidates))
            gen_upgraded_count = 0
            for idx in gen_judge_candidates:
                if idx < reward_tensor.shape[0]:
                    data_item = data[idx]
                    prompt_length = data_item.batch['prompts'].shape[-1]
                    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
                    pos = int(valid_response_length) - 1
                    # Only upgrade if current score < 0.5 (don't downgrade EM=1)
                    if reward_tensor[idx, pos] < 0.5:
                        reward_tensor[idx, pos] = 0.8
                        gen_upgraded_count += 1
            if gen_upgraded_count > 0:
                print(f"[GenTimeJudge] Applied {gen_upgraded_count} upgrades from generation-time judge")
            gen_judge_upgraded = gen_upgraded_count
            self._gen_judge_upgraded = set()
        else:
            gen_judge_fired = gen_judge_stats.get('fired', 0)
            gen_judge_upgraded = 0

        # Store metrics for logging
        self._step_metrics = {
            'reward/em_score_mean': float(np.mean(all_em_scores)) if all_em_scores else 0.0,
            'reward/em_score_nonzero_ratio': float(np.mean([1.0 if s > 0 else 0.0 for s in all_em_scores])) if all_em_scores else 0.0,
            'reward/process_reward_mean': float(np.mean(all_process_scores)) if all_process_scores else 0.0,
            'reward/process_reward_max': float(np.max(all_process_scores)) if all_process_scores else 0.0,
            'reward/process_reward_nonzero_ratio': float(np.mean([1.0 if s != 0 else 0.0 for s in all_process_scores])) if all_process_scores else 0.0,
            'judge/llm_judge_fired': judge_fired,
            'judge/llm_judge_upgraded': judge_upgraded,
            'judge/gen_time_judge_fired': gen_judge_fired,
            'judge/gen_time_judge_upgraded': gen_judge_upgraded,
            'judge/total_upgraded': judge_upgraded + gen_judge_upgraded,
        }

        return reward_tensor

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
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

    # Only create ref policy if KL loss is enabled
    if config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
