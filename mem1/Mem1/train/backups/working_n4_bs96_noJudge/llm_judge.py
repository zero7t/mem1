"""
LLM Judge System for Multi-Turn RAG GRPO Training

Two components:
1. Outcome Judge: When EM=0, call LLM to check semantic equivalence
2. Process Judge: Listwise ranking of trajectories within GRPO groups

Uses DeepSeek V4 Flash for both.
"""

import re
import json
import time
import asyncio
try:
    import aiohttp
except ImportError:
    aiohttp = None  # Optional, sync mode used instead
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # API settings - Baidu internal OneAPI (NO proxy needed)
    "api_base": "https://oneapi-comate.baidu-int.com/v1",
    "api_key": "sk-lFsuhIBP98RfWqKI4b68E447E46f433eBdF70b88Ea9eBbA9",
    "model": "DeepSeek-V4-Flash",
    "temperature": 0.1,  # Low temp for judging consistency
    "max_tokens": 150,
    "timeout": 30,  # seconds per request
    "max_concurrent": 6,  # concurrent API calls (rate limit ~8, keep margin)

    # Outcome judge
    "outcome_judge_enabled": True,

    # Process judge (listwise ranking)
    "process_judge_enabled": True,
    "process_judge_interval": 1,  # every N steps (1=every step)
    "process_judge_start_step": 0,
    "process_judge_end_step": 1413,  # run for full training
    "process_judge_decay_after": 1413,  # no decay
    "process_judge_only_tied": False,  # judge ALL groups (user request)

    # Reward scaling
    "outcome_judge_reward": 0.8,  # reward for LLM-judged correct (less than EM=1.0)
    "process_reward_scale": 0.3,  # scale for process rank rewards
}


# =============================================================================
# Outcome Judge: Semantic Equivalence Check
# =============================================================================

OUTCOME_JUDGE_PROMPT = """You are evaluating whether a model's answer is semantically correct.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

The model's answer may use different wording, abbreviations, or formats but still be correct.
Examples of matches:
- "NYC" = "New York City"
- "February 14" = "Feb 14" = "Valentine's Day (February 14)"
- "Albert Einstein" = "Einstein"
- "2.5 million" = "2,500,000"

Is the model's answer semantically equivalent to the ground truth?
Output ONLY: "correct" or "incorrect"
"""


def build_outcome_judge_prompt(question: str, ground_truth: List[str], model_answer: str) -> str:
    """Build prompt for outcome semantic judge."""
    gt_str = " OR ".join(ground_truth) if isinstance(ground_truth, list) else ground_truth
    return OUTCOME_JUDGE_PROMPT.format(
        question=question,
        ground_truth=gt_str,
        model_answer=model_answer,
    )


# =============================================================================
# Process Judge: Listwise Ranking
# =============================================================================

PROCESS_JUDGE_PROMPT_TEMPLATE = """You are ranking {n} search trajectories for the same question.
Your goal: rank them from best to worst based on search strategy quality and reasoning progress.

Question: {question}
Correct Answer: {ground_truth}

{trajectories_text}

Evaluation criteria (in order of importance):
1. Search Strategy: Are queries targeted, specific, and likely to find relevant information?
2. Information Usage: Does the reasoning incorporate and build upon retrieved information?
3. Progress toward Answer: Does the trajectory get progressively closer to the correct answer?
4. Efficiency: Does it avoid redundant searches and unnecessary steps?

Output ONLY a JSON ranking like: {{"ranking": [3, 1, 2, 4]}}
where the first number is the BEST trajectory and last is the WORST.
If trajectories are roughly equal, still provide a ranking (break ties by efficiency).
"""


def build_process_judge_prompt(
    question: str,
    ground_truth: str,
    trajectories: List[str],
) -> str:
    """Build prompt for listwise process ranking."""
    n = len(trajectories)
    trajectories_text = ""
    for i, traj in enumerate(trajectories):
        # Truncate very long trajectories to save tokens
        if len(traj) > 3000:
            # Keep first 1500 and last 1500 chars
            traj = traj[:1500] + "\n... [truncated] ...\n" + traj[-1500:]
        trajectories_text += f"\n=== Trajectory {i+1} ===\n{traj}\n"

    gt_str = " OR ".join(ground_truth) if isinstance(ground_truth, list) else ground_truth

    return PROCESS_JUDGE_PROMPT_TEMPLATE.format(
        n=n,
        question=question,
        ground_truth=gt_str,
        trajectories_text=trajectories_text,
    )


def parse_ranking_response(response: str, n: int) -> Optional[List[int]]:
    """Parse LLM ranking output into a list of indices (0-indexed)."""
    try:
        # Try JSON parse
        json_match = re.search(r'\{.*?"ranking".*?\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            ranking = data.get("ranking", [])
            if len(ranking) == n and set(ranking) == set(range(1, n+1)):
                return [r - 1 for r in ranking]  # convert to 0-indexed

        # Fallback: try to extract numbers
        numbers = re.findall(r'\d+', response)
        numbers = [int(x) for x in numbers if 1 <= int(x) <= n]
        if len(numbers) >= n:
            ranking = numbers[:n]
            if set(ranking) == set(range(1, n+1)):
                return [r - 1 for r in ranking]

    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    return None  # Failed to parse


def ranking_to_advantages(ranking: List[int], n: int, scale: float = 0.3) -> List[float]:
    """
    Convert ranking to advantage scores for GRPO.

    Args:
        ranking: List of trajectory indices ordered from best to worst
        n: Number of trajectories
        scale: Maximum advantage magnitude

    Returns:
        List of advantage scores (one per trajectory), mean=0
    """
    advantages = [0.0] * n
    for rank_pos, traj_idx in enumerate(ranking):
        # rank_pos 0 = best, n-1 = worst
        # Map to [-scale, +scale]
        advantages[traj_idx] = scale * (1.0 - 2.0 * rank_pos / (n - 1))

    # Ensure mean=0 (should already be by construction)
    mean_adv = sum(advantages) / n
    advantages = [a - mean_adv for a in advantages]

    return advantages


# =============================================================================
# API Client
# =============================================================================

class LLMJudgeClient:
    """Async LLM judge client for DeepSeek V4 Flash."""

    def __init__(self, config: dict = None):
        import os
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.api_key = self.config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_base = self.config["api_base"]
        self.model = self.config["model"]
        self._lock = threading.Lock()
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _call_api_sync(self, prompt: str) -> Optional[str]:
        """Synchronous API call with retry (for use in thread pool)."""
        import requests
        import os

        # IMPORTANT: Internal API - must NOT use proxy
        session = requests.Session()
        session.trust_env = False  # Ignore env proxy settings
        # Also explicitly unset for this thread
        no_proxy_env = {
            'http_proxy': '', 'https_proxy': '',
            'HTTP_PROXY': '', 'HTTPS_PROXY': '',
        }
        old_env = {}
        for k, v in no_proxy_env.items():
            old_env[k] = os.environ.get(k)
            if k in os.environ:
                del os.environ[k]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.config["timeout"],
                )
                if resp.status_code == 429:
                    # Rate limited - wait and retry
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()

                # Track token usage
                usage = data.get("usage", {})
                with self._lock:
                    self._call_count += 1
                    self._total_input_tokens += usage.get("prompt_tokens", 0)
                    self._total_output_tokens += usage.get("completion_tokens", 0)

                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"[LLM Judge] API error after {max_retries} retries: {e}")
                return None
            finally:
                # Restore env
                for k, v in old_env.items():
                    if v is not None:
                        os.environ[k] = v

        return None  # All retries exhausted

    def batch_judge(self, prompts: List[str]) -> List[Optional[str]]:
        """Run multiple judge calls in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.config["max_concurrent"]) as executor:
            results = list(executor.map(self._call_api_sync, prompts))
        return results

    def get_stats(self) -> Dict:
        """Return usage statistics."""
        return {
            "total_calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": (
                self._total_input_tokens * 0.00000015 +
                self._total_output_tokens * 0.0000006
            ),
        }


# =============================================================================
# Main Judge Orchestrator
# =============================================================================

class GRPOJudgeOrchestrator:
    """
    Orchestrates LLM judge calls within the GRPO training loop.

    Integration point: After outcome reward computation in ray_trainer.py,
    before advantage computation.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.client = LLMJudgeClient(config)
        self.step_count = 0

    def should_run_process_judge(self, step: int) -> bool:
        """Determine if process judge should run at this step."""
        if not self.config["process_judge_enabled"]:
            return False

        # Decay phase: reduce frequency
        if step > self.config["process_judge_decay_after"]:
            interval = self.config["process_judge_interval"] * 5  # 5x less frequent
            return step % interval == 0

        return step % self.config["process_judge_interval"] == 0

    def judge_outcome_batch(
        self,
        questions: List[str],
        ground_truths: List[List[str]],
        model_answers: List[str],
        em_scores: List[float],
    ) -> List[float]:
        """
        For samples where EM=0, call LLM to check semantic equivalence.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answer lists
            model_answers: List of model's extracted answers
            em_scores: List of EM scores (0 or 1)

        Returns:
            Updated scores (EM score OR LLM judge score)
        """
        if not self.config["outcome_judge_enabled"]:
            return em_scores

        updated_scores = list(em_scores)
        prompts_to_judge = []
        indices_to_judge = []

        for i, (q, gt, ans, em) in enumerate(zip(questions, ground_truths, model_answers, em_scores)):
            if em == 0 and ans is not None and ans.strip():
                # EM failed but answer exists -> worth checking semantically
                prompt = build_outcome_judge_prompt(q, gt, ans)
                prompts_to_judge.append(prompt)
                indices_to_judge.append(i)

        if not prompts_to_judge:
            return updated_scores

        # Batch API calls
        responses = self.client.batch_judge(prompts_to_judge)

        for idx, response in zip(indices_to_judge, responses):
            if response and "correct" in response.lower() and "incorrect" not in response.lower():
                updated_scores[idx] = self.config["outcome_judge_reward"]

        return updated_scores

    def judge_process_listwise(
        self,
        questions: List[str],
        ground_truths: List[List[str]],
        trajectory_groups: List[List[str]],
        outcome_scores: List[List[float]],
    ) -> List[List[float]]:
        """
        Listwise ranking of trajectories within each GRPO group.

        Args:
            questions: List of questions (one per group)
            ground_truths: List of ground truth answers (one per group)
            trajectory_groups: List of trajectory lists (one list per group)
            outcome_scores: List of score lists (one list per group)

        Returns:
            List of process advantage lists (one list per group)
        """
        n_groups = len(questions)
        all_advantages = [[0.0] * len(tg) for tg in trajectory_groups]
        prompts_to_judge = []
        group_indices = []

        for g in range(n_groups):
            n = len(trajectory_groups[g])
            scores = outcome_scores[g]

            # Only judge tied groups (all same outcome)
            if self.config["process_judge_only_tied"]:
                if len(set([int(s * 10) for s in scores])) > 1:
                    continue  # Mixed group, outcome reward is enough

            prompt = build_process_judge_prompt(
                question=questions[g],
                ground_truth=ground_truths[g],
                trajectories=trajectory_groups[g],
            )
            prompts_to_judge.append(prompt)
            group_indices.append(g)

        if not prompts_to_judge:
            return all_advantages

        # Batch API calls
        responses = self.client.batch_judge(prompts_to_judge)

        for g_idx, response in zip(group_indices, responses):
            if response is None:
                continue

            n = len(trajectory_groups[g_idx])
            ranking = parse_ranking_response(response, n)

            if ranking is not None:
                advantages = ranking_to_advantages(
                    ranking, n, scale=self.config["process_reward_scale"]
                )
                all_advantages[g_idx] = advantages

        return all_advantages

    def compute_judge_rewards(
        self,
        step: int,
        questions: List[str],
        ground_truths: List[List[str]],
        model_answers: List[str],
        trajectories: List[str],
        em_scores: List[float],
        group_indices: List[int],  # which prompt each trajectory belongs to
        n_agent: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Main entry point: compute both outcome judge and process judge rewards.

        Args:
            step: Current training step
            questions: List of questions (batch_size,)
            ground_truths: Ground truth answers (batch_size,)
            model_answers: Extracted model answers (batch_size,)
            trajectories: Full trajectory texts (batch_size,)
            em_scores: EM scores (batch_size,)
            group_indices: Group membership indices (batch_size,)
            n_agent: Number of trajectories per prompt

        Returns:
            outcome_scores: Updated outcome scores (after LLM semantic check)
            process_advantages: Process advantages from listwise ranking
        """
        batch_size = len(questions)

        # 1. Outcome judge (always runs if enabled)
        outcome_scores = self.judge_outcome_batch(
            questions, ground_truths, model_answers, em_scores
        )

        # 2. Process judge (runs conditionally)
        process_advantages = [0.0] * batch_size

        if self.should_run_process_judge(step):
            # Organize into groups
            groups = {}
            for i in range(batch_size):
                gid = group_indices[i]
                if gid not in groups:
                    groups[gid] = {"indices": [], "trajectories": [], "scores": []}
                groups[gid]["indices"].append(i)
                groups[gid]["trajectories"].append(trajectories[i])
                groups[gid]["scores"].append(outcome_scores[i])

            # Prepare for listwise ranking
            group_questions = []
            group_gts = []
            group_trajs = []
            group_scores = []
            group_ids = []

            for gid, group_data in groups.items():
                if len(group_data["indices"]) >= 2:  # need at least 2 for ranking
                    group_questions.append(questions[group_data["indices"][0]])
                    group_gts.append(ground_truths[group_data["indices"][0]])
                    group_trajs.append(group_data["trajectories"])
                    group_scores.append(group_data["scores"])
                    group_ids.append(gid)

            # Get rankings
            group_advantages = self.judge_process_listwise(
                group_questions, group_gts, group_trajs, group_scores
            )

            # Map back to flat batch
            for g, gid in enumerate(group_ids):
                for local_idx, global_idx in enumerate(groups[gid]["indices"]):
                    process_advantages[global_idx] = group_advantages[g][local_idx]

        self.step_count = step
        return outcome_scores, process_advantages


# =============================================================================
# Token Estimation Utility
# =============================================================================

def estimate_token_consumption(
    total_steps: int = 1413,
    batch_size: int = 96,
    n_agent: int = 4,
    avg_trajectory_tokens: int = 2000,
    score_progression: List[Tuple[int, float]] = None,
    process_judge_interval: int = 1,
    process_judge_end_step: int = 500,
) -> Dict:
    """
    Estimate total token consumption for the full training run.

    Args:
        total_steps: Total training steps
        batch_size: Batch size
        n_agent: Trajectories per prompt
        avg_trajectory_tokens: Average trajectory length in tokens
        score_progression: [(step, accuracy), ...] estimated accuracy over time
        process_judge_interval: Judge every N steps
        process_judge_end_step: Stop process judging after this step

    Returns:
        Dict with token estimates
    """
    if score_progression is None:
        # Estimated accuracy progression
        score_progression = [
            (0, 0.02), (100, 0.05), (200, 0.10), (300, 0.15),
            (500, 0.25), (700, 0.30), (1000, 0.35), (1413, 0.40)
        ]

    n_prompts = batch_size // n_agent

    # Interpolate accuracy at each step
    def get_accuracy_at_step(step):
        for i in range(len(score_progression) - 1):
            s0, a0 = score_progression[i]
            s1, a1 = score_progression[i+1]
            if s0 <= step <= s1:
                t = (step - s0) / (s1 - s0)
                return a0 + t * (a1 - a0)
        return score_progression[-1][1]

    # === Outcome Judge ===
    # Called for every sample where EM=0 and answer exists
    outcome_prompt_tokens = 200  # template
    outcome_input_per_call = outcome_prompt_tokens + 50 + 100  # prompt + question + answer
    outcome_output_per_call = 10

    total_outcome_input = 0
    total_outcome_output = 0
    total_outcome_calls = 0

    for step in range(total_steps):
        acc = get_accuracy_at_step(step)
        # Samples with EM=0 but valid answer: ~(1-acc) * batch_size * 0.7 (some have no answer)
        n_to_judge = int((1 - acc) * batch_size * 0.7)
        total_outcome_input += n_to_judge * outcome_input_per_call
        total_outcome_output += n_to_judge * outcome_output_per_call
        total_outcome_calls += n_to_judge

    # === Process Judge ===
    # Listwise ranking for tied groups
    process_prompt_tokens = 300  # template
    process_input_per_call = process_prompt_tokens + 70 + n_agent * min(avg_trajectory_tokens, 3000)
    process_output_per_call = 80 + n_agent * 10

    total_process_input = 0
    total_process_output = 0
    total_process_calls = 0

    for step in range(min(total_steps, process_judge_end_step)):
        if step % process_judge_interval != 0:
            continue

        acc = get_accuracy_at_step(step)
        # Tied groups: P(all same outcome in group of n_agent)
        # P(all 0) = (1-acc)^n_agent, P(all 1) = acc^n_agent
        p_tied = (1 - acc) ** n_agent + acc ** n_agent
        n_tied_groups = int(p_tied * n_prompts)

        total_process_input += n_tied_groups * process_input_per_call
        total_process_output += n_tied_groups * process_output_per_call
        total_process_calls += n_tied_groups

    # After process_judge_end_step, reduced frequency (every 5 steps)
    for step in range(process_judge_end_step, total_steps):
        if step % (process_judge_interval * 5) != 0:
            continue
        acc = get_accuracy_at_step(step)
        p_tied = (1 - acc) ** n_agent + acc ** n_agent
        n_tied_groups = int(p_tied * n_prompts)
        total_process_input += n_tied_groups * process_input_per_call
        total_process_output += n_tied_groups * process_output_per_call
        total_process_calls += n_tied_groups

    total_input = total_outcome_input + total_process_input
    total_output = total_outcome_output + total_process_output

    return {
        "outcome_judge": {
            "total_calls": total_outcome_calls,
            "total_input_tokens": total_outcome_input,
            "total_output_tokens": total_outcome_output,
        },
        "process_judge": {
            "total_calls": total_process_calls,
            "total_input_tokens": total_process_input,
            "total_output_tokens": total_process_output,
        },
        "combined": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
        },
        "estimated_cost": {
            "deepseek_v4_flash": {
                "input_cost": total_input * 0.14 / 1_000_000,  # ¥0.14/M (DeepSeek)
                "output_cost": total_output * 0.28 / 1_000_000,  # ¥0.28/M
                "total_rmb": total_input * 0.14 / 1_000_000 + total_output * 0.28 / 1_000_000,
            }
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TOKEN CONSUMPTION ESTIMATES")
    print("=" * 60)

    configs = [
        ("n_agent=2, 全量 process judge", {"n_agent": 2, "process_judge_interval": 1, "process_judge_end_step": 1413}),
        ("n_agent=4, 全量 process judge", {"n_agent": 4, "process_judge_interval": 1, "process_judge_end_step": 1413}),
        ("n_agent=4, 前500步 process judge", {"n_agent": 4, "process_judge_interval": 1, "process_judge_end_step": 500}),
        ("n_agent=4, 前500步/每3步", {"n_agent": 4, "process_judge_interval": 3, "process_judge_end_step": 500}),
        ("n_agent=8, 前500步/每3步", {"n_agent": 8, "process_judge_interval": 3, "process_judge_end_step": 500}),
    ]

    for name, cfg in configs:
        result = estimate_token_consumption(**cfg)
        print(f"\n--- {name} ---")
        print(f"  Outcome Judge: {result['outcome_judge']['total_calls']:,} calls, "
              f"{result['outcome_judge']['total_input_tokens']/1e6:.1f}M input tokens")
        print(f"  Process Judge: {result['process_judge']['total_calls']:,} calls, "
              f"{result['process_judge']['total_input_tokens']/1e6:.1f}M input tokens")
        print(f"  Total: {result['combined']['total_tokens']/1e6:.1f}M tokens")
        print(f"  Cost (DeepSeek V4 Flash): ¥{result['estimated_cost']['deepseek_v4_flash']['total_rmb']:.1f}")
