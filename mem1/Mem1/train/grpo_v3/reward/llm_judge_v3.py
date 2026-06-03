"""
LLM Judge V3: Pointwise + Listwise with Margin-Based Gating

Pointwise: Score each trajectory independently (1-5 rubric)
Listwise: Rank trajectories within group (only in refinement phase)

Key design:
- Pointwise is the primary signal (rubric-based, absolute quality)
- Listwise only activates when pointwise scores show clear differentiation
- Margin-based: only reward rank-1 if clearly better, only punish rank-last if clearly worse
"""

import os
import re
import json
import time
import requests
import threading
import numpy as np
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


DEFAULT_CONFIG = {
    "api_base": "https://oneapi-comate.baidu-int.com/v1",
    "api_key": "sk-lFsuhIBP98RfWqKI4b68E447E46f433eBdF70b88Ea9eBbA9",
    "model": "DeepSeek-V4-Flash",
    "temperature": 0.1,
    "max_tokens": 200,
    "timeout": 30,
    "max_concurrent": 30,
}

# Pointwise rubric prompt
POINTWISE_PROMPT = """You are evaluating a search trajectory for a question-answering task.

Question: {question}
Correct Answer: {answer}

Trajectory (the model's search process):
{trajectory}

Score this trajectory's SEARCH STRATEGY on a 1-5 scale:

5: Every query is semantically relevant and each adds NEW information not covered before. Reasoning correctly synthesizes retrieved info. Clear global plan covering all sub-questions.
4: Queries are relevant with good information gain. Reasoning is mostly sound. Minor redundancy but overall progressive.
3: Queries are relevant but some are partially redundant (overlapping info). Reasoning references retrieval but with logical gaps or missed key details.
2: Some queries are off-topic OR significantly redundant (same aspect searched twice with no new angle). Reasoning contradicts or ignores retrieved content.
1: Queries are random/completely irrelevant OR all searches target the same thing repeatedly. No coherent reasoning chain.

Evaluate these dimensions:
1. Query semantic relevance: Is each query targeting information needed for the answer? (not just keyword overlap — "Steve Jobs career" is relevant to "Who founded Apple")
2. Information gain: Does each query seek NEW knowledge vs repeating previous searches? Consider both the query intent and the actual retrieval novelty.
3. Reasoning coherence: Does the <think> block correctly use retrieved information? Are conclusions logically supported?
4. Global strategy: Are all necessary sub-questions addressed? Is the search order logical?

Output ONLY a JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

LISTWISE_PROMPT = """You are ranking {n} search trajectories for the same question.

Question: {question}
Correct Answer: {answer}

{trajectories_text}

Rank these trajectories from BEST to WORST search strategy.
Consider: query relevance, logical progression, information utilization, efficiency.

Output ONLY a JSON: {{"ranking": [best_idx, ..., worst_idx], "confidence": "high"/"medium"/"low"}}"""

OUTCOME_JUDGE_PROMPT = """Question: {question}
Expected answer: {expected}
Model's answer: {model_answer}

Is the model's answer semantically equivalent to the expected answer? Consider that different phrasings, abbreviations, or partial matches may still be correct.

Output ONLY: "correct" or "incorrect"."""

PER_TURN_PROMPT = """You are evaluating each search turn in a multi-turn question-answering trajectory.

Question: {question}
Correct Answer: {answer}

Trajectory:
{trajectory}

Score EACH SEARCH TURN (1-5) on these dimensions:
1. Query quality: Is the query well-formed and targeting useful information for answering the question?
2. Progressive refinement: Does this query build on what was learned from previous turns? Are later queries refined using entities or facts discovered in prior retrieval results?
3. Reasoning quality: Does the <think> block correctly synthesize prior information and guide the next search?
4. Information utilization: Does the query leverage newly retrieved facts rather than repeating the same approach?
5. Synthesis (final turn only): Does the final reasoning correctly integrate all retrieved evidence to form the answer?

Output ONLY a JSON: {{"turn_scores": [s1, s2, ...], "reason": "<one sentence>"}}
Each score is 1-5. Number of scores MUST equal the number of <search> tags in the trajectory."""


class LLMJudgeV3:
    """Unified LLM Judge with pointwise scoring and listwise ranking."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._lock = threading.Lock()
        self._call_count = 0
        self._total_tokens = 0

    def _call_api(self, prompt: str) -> Optional[str]:
        """Single API call with retry."""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
        }

        old_env = {}
        for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            old_env[k] = os.environ.pop(k, None)

        try:
            for attempt in range(3):
                try:
                    resp = requests.post(
                        f"{self.config['api_base']}/chat/completions",
                        headers=headers, json=payload,
                        timeout=self.config["timeout"],
                    )
                    if resp.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    with self._lock:
                        self._call_count += 1
                        self._total_tokens += data.get("usage", {}).get("total_tokens", 0)
                    return data["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    return None
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v

        return None

    # -----------------------------------------------------------------
    # Pointwise Judge
    # -----------------------------------------------------------------

    def pointwise_score(self, question: str, answer: str, trajectory: str) -> Tuple[int, str]:
        """Score a single trajectory 1-5."""
        if len(trajectory) > 3000:
            trajectory = trajectory[:1500] + "\n...[truncated]...\n" + trajectory[-1500:]

        prompt = POINTWISE_PROMPT.format(
            question=question, answer=answer, trajectory=trajectory
        )
        response = self._call_api(prompt)
        if not response:
            return 3, "api_error"

        try:
            # Try JSON parse
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = int(data.get("score") or 3)
                reason = data.get("reason", "")
                return max(1, min(5, score)), reason
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract number
        nums = re.findall(r'[1-5]', response)
        if nums:
            return int(nums[0]), response[:50]
        return 3, "parse_error"

    def batch_pointwise(self, items: List[Dict]) -> List[Tuple[int, str]]:
        """
        Score multiple trajectories in parallel.
        items: [{'question': str, 'answer': str, 'trajectory': str}, ...]
        """
        def _score_one(item):
            return self.pointwise_score(item['question'], item['answer'], item['trajectory'])

        with ThreadPoolExecutor(max_workers=self.config["max_concurrent"]) as executor:
            results = list(executor.map(_score_one, items))
        return results

    # -----------------------------------------------------------------
    # Per-Turn Judge (replaces pointwise)
    # -----------------------------------------------------------------

    def per_turn_score(self, question: str, answer: str,
                       trajectory: str, num_turns: int) -> List[int]:
        """Score each turn 1-5. Returns list of scores matching num_turns."""
        if len(trajectory) > 3000:
            trajectory = trajectory[:1500] + "\n...[truncated]...\n" + trajectory[-1500:]

        prompt = PER_TURN_PROMPT.format(
            question=question, answer=answer, trajectory=trajectory
        )
        response = self._call_api(prompt)
        if not response:
            return [3] * num_turns

        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                scores = data.get("turn_scores", [])
                if isinstance(scores, list) and len(scores) > 0:
                    scores = [max(1, min(5, int(s))) for s in scores]
                    # Pad or truncate to match num_turns
                    if len(scores) >= num_turns:
                        return scores[:num_turns]
                    else:
                        return scores + [3] * (num_turns - len(scores))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to extract numbers
        nums = re.findall(r'[1-5]', response)
        if len(nums) >= num_turns:
            return [int(n) for n in nums[:num_turns]]
        return [3] * num_turns

    # -----------------------------------------------------------------
    # Listwise Judge (margin-based)
    # -----------------------------------------------------------------

    def listwise_rank(self, question: str, answer: str,
                      trajectories: List[str]) -> Optional[List[int]]:
        """Rank trajectories, return ordering [best_idx, ..., worst_idx]."""
        n = len(trajectories)
        if n <= 1:
            return None

        traj_text = ""
        for i, traj in enumerate(trajectories):
            t = traj[:2000] if len(traj) > 2000 else traj
            traj_text += f"\n=== Trajectory {i+1} ===\n{t}\n"

        prompt = LISTWISE_PROMPT.format(
            n=n, question=question, answer=answer, trajectories_text=traj_text
        )
        response = self._call_api(prompt)
        if not response:
            return None

        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                ranking = data.get("ranking", [])
                # Validate: must be permutation of 1..n
                if sorted(ranking) == list(range(1, n + 1)):
                    return [r - 1 for r in ranking]  # 0-indexed
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # -----------------------------------------------------------------
    # Outcome Judge
    # -----------------------------------------------------------------

    def outcome_judge(self, question: str, expected: str, model_answer: str) -> bool:
        """Check if model answer is semantically correct."""
        prompt = OUTCOME_JUDGE_PROMPT.format(
            question=question, expected=expected, model_answer=model_answer
        )
        response = self._call_api(prompt)
        if response and "correct" in response.lower() and "incorrect" not in response.lower():
            return True
        return False

    def batch_outcome(self, items: List[Dict]) -> List[bool]:
        """Batch outcome judging."""
        def _judge_one(item):
            return self.outcome_judge(item['question'], item['expected'], item['model_answer'])

        with ThreadPoolExecutor(max_workers=self.config["max_concurrent"]) as executor:
            results = list(executor.map(_judge_one, items))
        return results
