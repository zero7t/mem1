"""
Main PPO Entry Point V3 - Curriculum GRPO with Turn-Weighted Advantage + LLM Judge

Integrates:
1. CurriculumController: auto phase transitions based on EM/format metrics
2. Turn-weighted advantage: smooth positional + quality weighting
3. Rule rewards V3: format, efficiency, retrieval quality, per-turn process
4. LLM Judge V3: pointwise (rubric 1-5) + listwise (margin-gated)
5. DAPO dynamic sampling (in ray_trainer.py, controlled via config)

Does NOT modify verl/ code. Monkey-patches reward_fn and advantage computation.
"""

import os
import sys
import re
import torch
import numpy as np
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

TRAIN_ROOT = '/root/paddlejob/workspace/new_llm_judge_mem1/mem1/Mem1/train'
sys.path.insert(0, TRAIN_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl import DataProto
from verl.utils.reward_score import qa_em, websearch, qa_multiple
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from core import apply_turn_weighted_advantage, find_turn_boundaries
from core.curriculum import CurriculumController, PHASES
from reward.rule_reward_v3 import compute_reward_v3, format_reward
from reward.llm_judge_v3 import LLMJudgeV3
from reward.judge_advantage import compute_judge_advantages


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa',
                       '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_multiple.compute_score_em
    elif data_source in ['websearch']:
        return websearch.compute_score_f1
    else:
        return qa_multiple.compute_score_em


# =============================================================================
# Reward Manager V3
# =============================================================================

class RewardManagerV3:
    """
    Curriculum-aware reward manager.

    Behavior changes based on current phase:
    - Warmup: format reward only + outcome
    - Convergence: rule process + outcome + DAPO
    - Transition: + pointwise LLM judge
    - Refinement: + listwise LLM judge (margin-gated)
    """

    def __init__(self, tokenizer, config: Dict = None):
        self.tokenizer = tokenizer
        config = config or {}

        self.curriculum = CurriculumController(config.get('curriculum', {}))
        self.judge = LLMJudgeV3(config.get('judge', {}))
        self.max_turns = config.get('max_turns', 6)
        self.n_agent = config.get('n_agent', 4)

        # Turn weighting params
        self.tw_alpha = config.get('turn_weight_alpha', 0.3)
        self.tw_beta = config.get('turn_weight_beta', 0.3)
        self.tw_gamma = config.get('turn_weight_gamma', 1.5)

        # Storage for turn-level data
        self._last_turn_scores = None
        self._last_texts = None
        self._last_em_scores = None

        # Async judge state
        self._pending_judge = None
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    # Pre-fire: Outcome Judge (async, overlapped with compute_log_prob)
    # -----------------------------------------------------------------

    def pre_fire_outcome_judge(self, gen_batch_output, prompt_batch, n_repeat):
        """Fire outcome judge API calls before compute_log_prob for overlap."""
        self._pending_judge = None
        try:
            items_to_judge = []
            n_total = gen_batch_output.batch['prompts'].shape[0]

            for i in range(n_total):
                attn = gen_batch_output.batch['attention_mask'][i]
                p_len = gen_batch_output.batch['prompts'].shape[-1]
                vp = int(attn[:p_len].sum())
                vr = int(attn[p_len:].sum())
                p_ids = gen_batch_output.batch['prompts'][i][-vp:]
                r_ids = gen_batch_output.batch['responses'][i][:vr]
                text = self.tokenizer.decode(torch.cat([p_ids, r_ids]))

                prompt_idx = i // n_repeat
                gt = prompt_batch.non_tensor_batch['reward_model'][prompt_idx]['ground_truth']
                ds = prompt_batch.non_tensor_batch['data_source'][prompt_idx]

                score_fn = _select_rm_score_fn(ds)
                em = score_fn(solution_str=text, ground_truth=gt, format_score=0.)

                if em < 0.5:
                    answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
                    if answers and len(answers) >= 2:
                        model_ans = answers[-1].strip()
                        if model_ans:
                            gt_list = gt if isinstance(gt, list) else [str(gt)]
                            q_match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
                            question = q_match.group(1).strip() if q_match else ""
                            items_to_judge.append({
                                'index': i, 'question': question,
                                'expected': ', '.join(gt_list),
                                'model_answer': model_ans,
                            })

            if items_to_judge:
                executor = ThreadPoolExecutor(max_workers=30)
                futures = [
                    executor.submit(
                        self.judge.outcome_judge,
                        it['question'], it['expected'], it['model_answer']
                    ) for it in items_to_judge
                ]
                self._pending_judge = {
                    'futures': futures, 'meta': items_to_judge, 'executor': executor,
                    'fire_time': _time.time(),
                }
                print(f"[V3 Judge] Pre-fired {len(items_to_judge)} outcome calls")
        except Exception as e:
            print(f"[V3 Judge] pre_fire error: {e}")

    # -----------------------------------------------------------------
    # Main reward computation
    # -----------------------------------------------------------------

    def __call__(self, data: DataProto):
        """Compute rewards based on current curriculum phase."""
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        phase = self.curriculum.phase
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_turn_scores = []
        all_texts = []
        all_em = []
        format_correct = 0

        n_total = len(data)
        for i in range(n_total):
            item = data[i]
            p_ids = item.batch['prompts']
            p_len = p_ids.shape[-1]
            vp = int(item.batch['attention_mask'][:p_len].sum())
            vr = int(item.batch['attention_mask'][p_len:].sum())
            valid_p = p_ids[-vp:]
            valid_r = item.batch['responses'][:vr]
            text = self.tokenizer.decode(torch.cat([valid_p, valid_r]))

            gt = item.non_tensor_batch['reward_model']['ground_truth']
            ds = item.non_tensor_batch['data_source']
            score_fn = _select_rm_score_fn(ds)
            em = score_fn(solution_str=text, ground_truth=gt, format_score=0.)

            # Ensure ground_truth is a dict with string list
            if isinstance(gt, dict):
                gt_dict = {k: [str(x) for x in v] if hasattr(v, '__iter__') and not isinstance(v, str) else v for k, v in gt.items()}
            else:
                gt_list = list(gt) if hasattr(gt, '__iter__') and not isinstance(gt, str) else [gt]
                gt_dict = {'target': [str(x) for x in gt_list]}

            result = compute_reward_v3(
                text=text, ground_truth=gt_dict, outcome_score=em,
                phase_config=phase, max_turns=self.max_turns,
            )

            pos = max(0, vr - 1)
            reward_tensor[i, pos] = result['total_reward']
            all_turn_scores.append(result['turn_scores'])
            all_texts.append(text)
            all_em.append(em)
            if result['format_score'] > 0:
                format_correct += 1

        # Collect pre-fired outcome judge results
        if self._pending_judge is not None:
            pj = self._pending_judge
            self._pending_judge = None
            results = [f.result() for f in pj['futures']]
            pj['executor'].shutdown(wait=False)
            upgraded = 0
            for meta, is_correct in zip(pj['meta'], results):
                if is_correct:
                    idx = meta['index']
                    if idx < n_total:
                        item = data[idx]
                        p_len = item.batch['prompts'].shape[-1]
                        vr = int(item.batch['attention_mask'][p_len:].sum())
                        pos = max(0, vr - 1)
                        reward_tensor[idx, pos] = 0.8
                        all_em[idx] = 0.8
                        upgraded += 1
            elapsed = _time.time() - pj['fire_time']
            print(f"[V3 Judge] Outcome: {upgraded} upgraded, {elapsed:.1f}s")

        # Store for turn weighting and curriculum update
        self._last_turn_scores = all_turn_scores
        self._last_texts = all_texts
        self._last_em_scores = all_em

        # Pre-fire process judge (async, overlaps with compute_log_prob/ref/advantage)
        self._pending_process_judge = None
        if phase.pointwise_judge:
            self._pre_fire_process_judge(all_texts, all_em)

        # Update curriculum metrics
        em_rate = sum(1 for e in all_em if e > 0.5) / max(len(all_em), 1)
        fmt_rate = format_correct / max(n_total, 1)
        self.curriculum.update_metrics(
            step=getattr(self, '_current_step', 0),
            em_rate=em_rate, format_acc=fmt_rate,
        )

        # Compute process reward stats for logging
        avg_turn_scores = []
        for ts in all_turn_scores:
            if ts:
                avg_turn_scores.append(np.mean(ts))
        mean_process = np.mean(avg_turn_scores) if avg_turn_scores else 0.0
        mean_reward = reward_tensor.sum(-1).mean().item()

        self._step_metrics = {
            'v3/em_rate': em_rate,
            'v3/format_rate': fmt_rate,
            'v3/mean_reward': mean_reward,
            'v3/mean_process_score': mean_process,
            'v3/curriculum_phase': float(list(PHASES.keys()).index(self.curriculum.current_phase)),
            'v3/num_turns_avg': np.mean([len(ts) for ts in all_turn_scores]) if all_turn_scores else 0.0,
        }

        return reward_tensor

    # -----------------------------------------------------------------
    # Pre-fire process judge (async)
    # -----------------------------------------------------------------

    def _pre_fire_process_judge(self, all_texts, all_em):
        """Fire pointwise judge API calls async to overlap with GPU compute."""
        n_total = len(all_em)
        n_groups = n_total // self.n_agent
        em_arr = np.array(all_em)
        em_grouped = em_arr.reshape(n_groups, self.n_agent)
        group_std = em_grouped.std(axis=1)

        groups_to_judge = [g for g in range(n_groups)
                          if group_std[g] > 0 or em_grouped[g].mean() > 0.5]
        if not groups_to_judge:
            return

        items = []
        item_map = []
        for g in groups_to_judge:
            for j in range(self.n_agent):
                idx = g * self.n_agent + j
                if idx >= len(all_texts):
                    continue
                text = all_texts[idx]
                q_match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
                question = q_match.group(1).strip() if q_match else ""
                a_match = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
                answer = a_match[-1].strip() if a_match else ""
                items.append({'question': question, 'answer': answer, 'trajectory': text})
                item_map.append((g, j, idx))

        if not items:
            return

        executor = ThreadPoolExecutor(max_workers=30)
        futures = [executor.submit(self.judge.pointwise_score,
                                   it['question'], it['answer'], it['trajectory'])
                   for it in items]
        self._pending_process_judge = {
            'futures': futures, 'item_map': item_map,
            'groups_to_judge': groups_to_judge, 'executor': executor,
            'fire_time': _time.time(),
        }
        print(f"[V3 Judge] Pre-fired {len(items)} pointwise calls")

    # -----------------------------------------------------------------
    # Turn-weighted advantage (called after compute_advantage)
    # -----------------------------------------------------------------

    def apply_turn_weighting_to_batch(self, batch: DataProto) -> DataProto:
        """Apply turn-level advantage weighting."""
        if self._last_turn_scores is None:
            return batch

        advantages = batch.batch['advantages']
        resp_len = batch.batch['responses'].shape[-1]
        eos_mask = batch.batch['attention_mask'][:, -resp_len:]
        if 'loss_mask' in batch.batch:
            eos_mask = batch.batch['loss_mask']

        # Extract response-only texts
        response_texts = []
        for text in (self._last_texts or []):
            q_end = text.rfind('</question>')
            response_texts.append(text[q_end:] if q_end > 0 else text)

        weighted = apply_turn_weighted_advantage(
            advantages=advantages,
            turn_scores_batch=self._last_turn_scores,
            response_texts=response_texts,
            eos_mask=eos_mask,
            alpha=self.tw_alpha,
            beta=self.tw_beta,
            gamma=self.tw_gamma,
        )
        batch.batch['advantages'] = weighted
        self._last_turn_scores = None
        self._last_texts = None
        return batch

    # -----------------------------------------------------------------
    # Process Judge (pointwise + listwise, called after advantage)
    # -----------------------------------------------------------------

    def apply_process_judge(self, batch: DataProto, step: int) -> DataProto:
        """
        Collect pre-fired pointwise judge results and apply to advantages.
        Only runs in transition/refinement phases.
        """
        phase = self.curriculum.phase
        if not phase.pointwise_judge or self._pending_process_judge is None:
            self._last_em_scores = None
            return batch

        pj = self._pending_process_judge
        self._pending_process_judge = None

        # Collect futures (should already be done since they overlapped with GPU work)
        pw_results = [f.result() for f in pj['futures']]
        pj['executor'].shutdown(wait=False)
        elapsed = _time.time() - pj['fire_time']
        print(f"[V3 Judge] Pointwise collected: {len(pw_results)} items, {elapsed:.1f}s (async)")

        item_map = pj['item_map']
        texts = self._last_texts or []

        # Organize scores by group
        group_scores = {}
        for (g, j, idx), (score, reason) in zip(item_map, pw_results):
            if g not in group_scores:
                group_scores[g] = []
            group_scores[g].append((j, idx, score))

        # Compute advantages per group
        advantages = batch.batch['advantages']
        resp_len = advantages.shape[-1]

        for g, entries in group_scores.items():
            scores = [0] * self.n_agent
            for j, idx, s in entries:
                scores[j] = s

            # Listwise (only in refinement phase)
            ranking = None
            if phase.listwise_judge and len(entries) == self.n_agent:
                # Only run listwise if pointwise shows differentiation
                if max(scores) - min(scores) >= 2:
                    g_texts = [texts[g * self.n_agent + j] for j in range(self.n_agent)]
                    q_match = re.search(r'<question>(.*?)</question>', g_texts[0], re.DOTALL)
                    question = q_match.group(1).strip() if q_match else ""
                    ranking = self.judge.listwise_rank(question, "", g_texts)

            judge_adv = compute_judge_advantages(
                group_pointwise_scores=scores,
                group_ranking=ranking,
                pointwise_scale=phase.pointwise_scale,
                listwise_scale=phase.listwise_scale,
                use_listwise=phase.listwise_judge,
                reward_threshold=phase.listwise_threshold,
                punish_threshold=phase.listwise_punish_threshold,
            )

            # Apply to advantages (broadcast to all tokens)
            for j, idx, _ in entries:
                if idx < advantages.shape[0]:
                    advantages[idx] += judge_adv[j]

        batch.batch['advantages'] = advantages
        self._last_em_scores = None

        # Log judge metrics
        all_pw = [s for entries in group_scores.values() for _, _, s in entries]
        if all_pw:
            self._step_metrics.update({
                'v3/judge_pointwise_mean': np.mean(all_pw),
                'v3/judge_pointwise_max': float(max(all_pw)),
                'v3/judge_pointwise_min': float(min(all_pw)),
                'v3/judge_groups_scored': len(group_scores),
            })

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

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # Worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
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
        from verl.workers.fsdp_workers import RewardModelWorker
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # V3 Reward Manager
    reward_config = {
        'max_turns': OmegaConf.select(config, 'max_turns', default=6),
        'n_agent': config.actor_rollout_ref.rollout.n_agent,
        'turn_weight_alpha': OmegaConf.select(config, 'reward.turn_weight_alpha', default=0.3),
        'turn_weight_beta': OmegaConf.select(config, 'reward.turn_weight_beta', default=0.3),
        'turn_weight_gamma': OmegaConf.select(config, 'reward.turn_weight_gamma', default=1.5),
        'curriculum': {
            'warmup_format_thresh': OmegaConf.select(config, 'curriculum.warmup_format_thresh', default=0.9),
            'convergence_em_thresh': OmegaConf.select(config, 'curriculum.convergence_em_thresh', default=0.15),
            'transition_em_thresh': OmegaConf.select(config, 'curriculum.transition_em_thresh', default=0.30),
            'warmup_max_step': OmegaConf.select(config, 'curriculum.warmup_max_step', default=80),
            'convergence_max_step': OmegaConf.select(config, 'curriculum.convergence_max_step', default=300),
            'transition_max_step': OmegaConf.select(config, 'curriculum.transition_max_step', default=500),
        },
    }

    reward_fn = RewardManagerV3(tokenizer=tokenizer, config=reward_config)
    val_reward_fn = RewardManagerV3(tokenizer=tokenizer, config=reward_config)

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

    print("[V3] Curriculum GRPO initialized.")
    print(f"[V3] Turn-weighted advantage: alpha={reward_config['turn_weight_alpha']}, "
          f"beta={reward_config['turn_weight_beta']}, gamma={reward_config['turn_weight_gamma']}")
    print(f"[V3] Phase transitions: warmup→convergence@{reward_config['curriculum']['warmup_max_step']}, "
          f"convergence→transition@EM>{reward_config['curriculum']['convergence_em_thresh']}")
    print("[V3] NOTE: ray_trainer.py must call:")
    print("       batch = self.reward_fn.apply_turn_weighting_to_batch(batch)")
    print("       batch = self.reward_fn.apply_process_judge(batch, self.global_steps)")
    print("     after compute_advantage().")

    trainer.fit()


if __name__ == '__main__':
    main()