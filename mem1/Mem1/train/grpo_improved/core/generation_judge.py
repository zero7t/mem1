"""
Generation-Time LLM Judge: Fire Outcome Judge AS trajectories complete.

Instead of waiting until all generation is done, this module fires
LLM Judge API calls the moment each trajectory finishes (outputs <answer>
or exhausts max_turns). Since most trajectories finish in turns 1-2
(within ~20s of the 111s generation phase), their judge results are
already back by the time generation ends.

Integration: Injected into generation_think.py's main loop at line 382
where `dones[i]=True`.

QPS target: 20 concurrent calls.
"""

import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Optional, Tuple
from collections import deque

from grpo_improved.reward.llm_judge import LLMJudgeClient, build_outcome_judge_prompt


class GenerationTimeJudge:
    """
    Fires LLM Outcome Judge calls during trajectory generation.

    Usage:
        judge = GenerationTimeJudge(tokenizer, ground_truths, questions)

        # In generation loop, when trajectory i finishes:
        judge.on_trajectory_complete(i, trajectory_text)

        # After generation ends:
        results = judge.collect_all()
        # results[i] = 'correct' | 'incorrect' | None
    """

    def __init__(
        self,
        tokenizer,
        ground_truths: List,
        questions: List[str],
        n_repeat: int = 4,
        max_concurrent: int = 20,
        compute_score_fn=None,
    ):
        """
        Args:
            tokenizer: HF tokenizer for decoding
            ground_truths: ground truth per prompt (len = batch_size / n_repeat)
            questions: question text per prompt
            n_repeat: n_agent (trajectories per prompt)
            max_concurrent: QPS limit
            compute_score_fn: EM score function (to skip judge if EM=1)
        """
        self.tokenizer = tokenizer
        self.ground_truths = ground_truths
        self.questions = questions
        self.n_repeat = n_repeat
        self.compute_score_fn = compute_score_fn

        self._client = LLMJudgeClient(config={'max_concurrent': max_concurrent})
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._futures: Dict[int, Future] = {}  # index -> future
        self._results: Dict[int, Optional[str]] = {}  # index -> result
        self._lock = threading.Lock()
        self._fire_count = 0
        self._skip_count = 0
        self._start_time = time.time()

    def on_trajectory_complete(self, index: int, trajectory_text: str):
        """
        Called when trajectory `index` finishes generation.
        Fires an async LLM Judge call if needed.

        Args:
            index: trajectory index in the batch (0..batch_size-1)
            trajectory_text: full decoded trajectory text
        """
        # Extract answer
        answer_match = re.findall(r'<answer>(.*?)</answer>', trajectory_text, re.DOTALL)
        if not answer_match or len(answer_match) < 2:
            self._skip_count += 1
            return  # No valid answer format

        model_answer = answer_match[-1].strip()
        if not model_answer:
            self._skip_count += 1
            return

        # Get ground truth for this trajectory
        prompt_idx = index // self.n_repeat
        if prompt_idx >= len(self.ground_truths):
            return

        ground_truth = self.ground_truths[prompt_idx]

        # Quick EM check - skip judge if already correct
        # Note: compute_score_fn may not work perfectly on raw text
        # (it expects specific format). If it returns >0.5, definitely skip.
        # If it returns 0, we still fire (might be format issue, not wrong answer).
        if self.compute_score_fn is not None:
            try:
                em_score = self.compute_score_fn(
                    solution_str=trajectory_text,
                    ground_truth=ground_truth,
                    format_score=0.,
                )
                if em_score >= 0.5:
                    self._skip_count += 1
                    return  # Already correct by EM, no need for judge
            except Exception:
                pass

        # Build judge prompt
        gt_list = ground_truth if isinstance(ground_truth, list) else [str(ground_truth)]
        question = self.questions[prompt_idx] if prompt_idx < len(self.questions) else ""
        prompt = build_outcome_judge_prompt(question, gt_list, model_answer)

        # Fire async
        future = self._executor.submit(self._client._call_api_sync, prompt)
        with self._lock:
            self._futures[index] = future
            self._fire_count += 1

    def on_trajectory_complete_from_tokens(
        self,
        index: int,
        prompt_ids: 'torch.Tensor',
        response_ids: 'torch.Tensor',
    ):
        """
        Alternative: decode from token IDs and fire judge.
        Used when text isn't readily available.
        """
        import torch
        # Decode
        valid_prompt = prompt_ids[prompt_ids != self.tokenizer.pad_token_id]
        valid_response = response_ids[response_ids != self.tokenizer.pad_token_id]
        sequences = torch.cat((valid_prompt, valid_response))
        text = self.tokenizer.decode(sequences, skip_special_tokens=True)
        self.on_trajectory_complete(index, text)

    def collect_all(self, timeout: float = 30.0) -> Dict[int, Optional[str]]:
        """
        Wait for all pending futures and return results.
        Called after generation completes.

        Returns:
            Dict[index, response_text] for all fired calls.
        """
        results = {}
        with self._lock:
            futures_copy = dict(self._futures)

        for idx, future in futures_copy.items():
            try:
                result = future.result(timeout=timeout)
                results[idx] = result
            except Exception:
                results[idx] = None

        elapsed = time.time() - self._start_time
        upgraded = sum(
            1 for r in results.values()
            if r and "correct" in r.lower() and "incorrect" not in r.lower()
        )
        print(f"[GenTimeJudge] fired={self._fire_count}, skipped={self._skip_count}, "
              f"upgraded={upgraded}, elapsed={elapsed:.1f}s")

        self._executor.shutdown(wait=False)
        return results

    def get_upgraded_indices(self, timeout: float = 30.0) -> set:
        """
        Convenience: collect and return set of indices judged 'correct'.
        """
        results = self.collect_all(timeout=timeout)
        upgraded = set()
        for idx, response in results.items():
            if response and "correct" in response.lower() and "incorrect" not in response.lower():
                upgraded.add(idx)
        return upgraded


# =============================================================================
# Integration Helper for generation_think.py
# =============================================================================

def create_generation_judge(
    tokenizer,
    gen_batch,
    n_repeat: int = 4,
    max_concurrent: int = 20,
    compute_score_fn=None,
) -> Optional[GenerationTimeJudge]:
    """
    Create a GenerationTimeJudge from the gen_batch DataProto.

    Call this BEFORE the generation loop starts.
    Then call judge.on_trajectory_complete() inside the loop.

    Args:
        tokenizer: HF tokenizer
        gen_batch: DataProto with non_tensor_batch containing reward_model info
        n_repeat: n_agent
        max_concurrent: QPS limit
        compute_score_fn: EM function

    Returns:
        GenerationTimeJudge instance, or None if data doesn't support it
    """
    try:
        # Extract ground truths and questions from gen_batch
        # gen_batch.non_tensor_batch['reward_model'] is array of dicts
        reward_info = gen_batch.non_tensor_batch.get('reward_model')
        if reward_info is None:
            return None

        batch_size = len(reward_info)
        n_prompts = batch_size  # before repeat

        ground_truths = []
        questions = []

        for i in range(n_prompts):
            gt = reward_info[i].get('ground_truth', '')
            ground_truths.append(gt)

            # Try to extract question from the prompt text
            # (will be filled in by the caller if available)
            questions.append("")

        # Try to get questions from input_ids
        if 'input_ids' in gen_batch.batch:
            for i in range(min(n_prompts, gen_batch.batch['input_ids'].shape[0])):
                input_text = tokenizer.decode(
                    gen_batch.batch['input_ids'][i],
                    skip_special_tokens=True,
                )
                q_match = re.search(r'<question>(.*?)</question>', input_text, re.DOTALL)
                if q_match:
                    questions[i] = q_match.group(1).strip()

        return GenerationTimeJudge(
            tokenizer=tokenizer,
            ground_truths=ground_truths,
            questions=questions,
            n_repeat=n_repeat,
            max_concurrent=max_concurrent,
            compute_score_fn=compute_score_fn,
        )

    except Exception as e:
        print(f"[GenTimeJudge] Failed to create: {e}")
        return None
