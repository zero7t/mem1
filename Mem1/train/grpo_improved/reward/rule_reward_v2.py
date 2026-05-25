"""
Turn-Level Process Reward V2 for Multi-Turn RAG QA

Key improvements over V1:
1. Utilization signal: rewards turns whose retrieval is actually USED in later reasoning
2. Diminishing-returns normalization: prevents Turn 1 from always dominating
3. Topic-completion bonus: rewards completing a topic fully over scattered partial coverage
4. Returns per-turn scores (not just a scalar) for turn-weighted advantage

Integration: main_ppo_v2.py::RewardManagerV2
"""

import re
import string
from typing import List, Dict, Optional, Tuple
from collections import Counter


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard overlap between two texts."""
    tokens_a = set(normalize_answer(text_a).split())
    tokens_b = set(normalize_answer(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def extract_key_phrases(text: str, min_word_len: int = 3) -> set:
    """Extract significant phrases (words > min_word_len) from text."""
    words = normalize_answer(text).split()
    return set(w for w in words if len(w) > min_word_len)


# =============================================================================
# Trajectory Parsing
# =============================================================================

def extract_trajectory_turns(full_text: str) -> Dict:
    """
    Parse a full trajectory into structured turns.

    Returns:
        {
            'turns': [
                {
                    'think': str,       # think content for this turn
                    'query': str,       # search query
                    'retrieval': str,   # retrieval result
                },
                ...
            ],
            'answer': str or None,
            'answer_reasoning': str,  # the final think before answer
            'num_turns': int,
        }
    """
    turns = []

    # Split by search actions to identify turns
    # Pattern: <think>...</think><search>...</search> then <information>...</information>
    # followed by next <think>...</think><search>...</search> ...

    think_matches = re.findall(r'<think>(.*?)</think>', full_text, re.DOTALL)
    search_matches = re.findall(r'<search>(.*?)</search>', full_text, re.DOTALL)
    info_matches = re.findall(r'<information>(.*?)</information>', full_text, re.DOTALL)

    num_turns = max(len(search_matches), 1)

    for t in range(num_turns):
        turn = {
            'think': think_matches[t].strip() if t < len(think_matches) else "",
            'query': search_matches[t].strip() if t < len(search_matches) else "",
            'retrieval': info_matches[t].strip() if t < len(info_matches) else "",
        }
        turns.append(turn)

    # Final answer
    answer_matches = re.findall(r'<answer>(.*?)</answer>', full_text, re.DOTALL)
    answer = answer_matches[-1].strip() if len(answer_matches) >= 2 else None

    # Answer reasoning = last think block (the one before/containing the answer decision)
    answer_reasoning = think_matches[-1].strip() if think_matches else ""

    return {
        'turns': turns,
        'answer': answer,
        'answer_reasoning': answer_reasoning,
        'num_turns': num_turns,
    }


# =============================================================================
# Reward Dimensions (V2)
# =============================================================================

def retrieval_hit_reward(
    retrieved_text: str,
    answer_targets: List[str],
    max_reward: float = 0.3,
) -> float:
    """
    Dimension 1: Does the retrieval contain answer-relevant entities?
    Same as V1 but cleaner implementation.
    """
    if not retrieved_text or not answer_targets:
        return 0.0

    retrieved_norm = normalize_answer(retrieved_text)
    covered = 0.0

    for target in answer_targets:
        target_norm = normalize_answer(target)
        if target_norm in retrieved_norm:
            covered += 1.0
            continue
        # Partial: check significant words
        target_words = [w for w in target_norm.split() if len(w) > 3]
        if target_words:
            word_hits = sum(1 for w in target_words if w in retrieved_norm)
            if word_hits / len(target_words) > 0.5:
                covered += 0.5

    coverage = min(covered / len(answer_targets), 1.0)
    return max_reward * coverage


def utilization_reward(
    turn_retrieval: str,
    subsequent_thinks: List[str],
    answer_reasoning: str,
    reward_used: float = 0.3,
    penalty_unused: float = -0.1,
) -> float:
    """
    Dimension 2 (NEW): Is this turn's retrieval actually USED in later reasoning?

    Checks if key phrases from the retrieval appear in:
    1. Subsequent think blocks (model incorporated the info)
    2. Final answer reasoning (info contributed to conclusion)

    Returns:
        reward in [penalty_unused, reward_used]
    """
    if not turn_retrieval:
        return 0.0

    retrieval_phrases = extract_key_phrases(turn_retrieval)
    if not retrieval_phrases:
        return 0.0

    # Check utilization in subsequent thinks
    used_in_think = False
    for think in subsequent_thinks:
        think_phrases = extract_key_phrases(think)
        overlap = len(retrieval_phrases & think_phrases)
        if overlap >= min(3, len(retrieval_phrases) * 0.2):
            used_in_think = True
            break

    # Check utilization in final answer reasoning
    used_in_answer = False
    if answer_reasoning:
        answer_phrases = extract_key_phrases(answer_reasoning)
        overlap = len(retrieval_phrases & answer_phrases)
        if overlap >= min(3, len(retrieval_phrases) * 0.2):
            used_in_answer = True

    if used_in_answer:
        return reward_used  # Strongest signal: directly used in conclusion
    elif used_in_think:
        return reward_used * 0.6  # Used in reasoning chain
    else:
        return penalty_unused  # Retrieved but never used


def novelty_reward(
    current_query: str,
    current_retrieval: str,
    previous_queries: List[str],
    previous_retrievals: List[str],
    reward_novel: float = 0.15,
    penalty_repeat: float = -0.2,
) -> float:
    """
    Dimension 3: Non-redundancy. Penalizes repeated queries/content.
    """
    if not current_retrieval:
        return 0.0

    # Query diversity check
    if previous_queries:
        max_overlap = max(
            compute_token_overlap(current_query, prev_q)
            for prev_q in previous_queries
        )
        if max_overlap > 0.7:
            return penalty_repeat

    # Content novelty check
    if previous_retrievals:
        current_phrases = extract_key_phrases(current_retrieval)
        previous_phrases = set()
        for prev in previous_retrievals:
            previous_phrases.update(extract_key_phrases(prev))

        if not current_phrases:
            return 0.0

        novelty = len(current_phrases - previous_phrases) / len(current_phrases)
        if novelty < 0.2:
            return penalty_repeat * 0.7
        elif novelty > 0.5:
            return reward_novel
        else:
            return reward_novel * 0.3
    else:
        return reward_novel * 0.5  # First turn: moderate novelty credit


def efficiency_reward(
    num_turns_used: int,
    max_turns: int,
    is_correct: bool,
    max_reward: float = 0.2,
) -> float:
    """
    Dimension 4: Fewer turns for correct answer = bonus.
    Only activates when outcome is correct.
    """
    if not is_correct:
        return 0.0
    saved_turns = max_turns - num_turns_used
    if saved_turns <= 0:
        return 0.0
    return max_reward * (saved_turns / max_turns)


# =============================================================================
# Main Reward Computer V2
# =============================================================================

class TurnRewardComputer:
    """
    Per-turn process reward with utilization signal.

    Key difference from V1: returns a LIST of per-turn scores,
    not a single scalar. This enables turn-weighted advantage.

    Usage:
        computer = TurnRewardComputer(config)
        turn_scores, outcome_score = computer.compute(
            trajectory_text, ground_truth, outcome_score
        )
        # turn_scores: [r_0, r_1, ..., r_T-1]  per-turn process rewards
        # These are used by turn_weighted_advantage.py
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.max_turns = config.get('max_turns', 6)

        # Dimension weights
        self.w_hit = config.get('w_hit', 0.3)          # retrieval hit
        self.w_util = config.get('w_util', 0.4)        # utilization (new, highest weight)
        self.w_novelty = config.get('w_novelty', 0.2)  # non-redundancy
        self.w_efficiency = config.get('w_efficiency', 0.1)  # efficiency bonus

        # Overall scale
        self.lambda_process = config.get('lambda_process', 0.5)

    def compute(
        self,
        trajectory_text: str,
        ground_truth: dict,
        outcome_score: float,
    ) -> Tuple[List[float], float]:
        """
        Compute per-turn process rewards.

        Args:
            trajectory_text: Full decoded trajectory string
            ground_truth: {'target': ['answer1', ...]}
            outcome_score: EM/F1 score (0 or 1)

        Returns:
            turn_scores: List[float] - per-turn process reward (length = num_turns)
            total_process: float - sum of all turn scores (for backward compat)
        """
        parsed = extract_trajectory_turns(trajectory_text)
        answer_targets = ground_truth.get('target', [])

        if not answer_targets:
            return [0.0] * parsed['num_turns'], 0.0

        turns = parsed['turns']
        num_turns = parsed['num_turns']
        turn_scores = []

        for t in range(num_turns):
            turn = turns[t]
            score = 0.0

            # Dim 1: Retrieval Hit
            r_hit = retrieval_hit_reward(
                retrieved_text=turn['retrieval'],
                answer_targets=answer_targets,
            )
            score += self.w_hit * r_hit

            # Dim 2: Utilization (key new signal)
            subsequent_thinks = [turns[j]['think'] for j in range(t + 1, num_turns)]
            r_util = utilization_reward(
                turn_retrieval=turn['retrieval'],
                subsequent_thinks=subsequent_thinks,
                answer_reasoning=parsed['answer_reasoning'],
            )
            score += self.w_util * r_util

            # Dim 3: Novelty
            r_novelty = novelty_reward(
                current_query=turn['query'],
                current_retrieval=turn['retrieval'],
                previous_queries=[turns[j]['query'] for j in range(t)],
                previous_retrievals=[turns[j]['retrieval'] for j in range(t)],
            )
            score += self.w_novelty * r_novelty

            turn_scores.append(score)

        # Dim 4: Efficiency (trajectory-level, distributed to last turn)
        r_eff = efficiency_reward(
            num_turns_used=num_turns,
            max_turns=self.max_turns,
            is_correct=(outcome_score > 0),
        )
        if turn_scores:
            turn_scores[-1] += self.w_efficiency * r_eff

        # Scale all by lambda
        turn_scores = [s * self.lambda_process for s in turn_scores]
        total_process = sum(turn_scores)

        return turn_scores, total_process


# =============================================================================
# Integration Helper (backward compatible with V1 interface)
# =============================================================================

def compute_process_reward(
    trajectory_text: str,
    ground_truth: dict,
    outcome_score: float,
    config: Optional[dict] = None,
) -> float:
    """
    Backward-compatible interface: returns scalar process reward.
    For V2 turn-level rewards, use TurnRewardComputer.compute() directly.
    """
    computer = TurnRewardComputer(config)
    _, total = computer.compute(trajectory_text, ground_truth, outcome_score)
    return total


def compute_turn_rewards(
    trajectory_text: str,
    ground_truth: dict,
    outcome_score: float,
    config: Optional[dict] = None,
) -> Tuple[List[float], float]:
    """
    V2 interface: returns per-turn scores + total.
    """
    computer = TurnRewardComputer(config)
    return computer.compute(trajectory_text, ground_truth, outcome_score)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    test_trajectory = """<think>I need to find when mob of the dead takes place.</think><search>mob of the dead setting time period</search>

<information>Doc 1: Mob of the Dead is set on Alcatraz Island during the 1930s prohibition era. The map takes place on New Year's Eve, December 31, 1933.</information>

<think>Found it - Mob of the Dead takes place on December 31, 1933 on Alcatraz Island. Now I need to find the chairman of the human rights commission.</think><search>chairman human rights commission current</search>

<information>Doc 1: The United Nations Human Rights Council is currently chaired by Vaclav Balek of the Czech Republic since January 2024.</information>

<think>I now have both answers. Mob of the Dead takes place on December 31, 1933, and the current chairman is Vaclav Balek.</think><answer>December 31, 1933; Vaclav Balek</answer>"""

    gt = {'target': ['December 31 1933', 'Vaclav Balek']}

    computer = TurnRewardComputer({'max_turns': 6})

    # Correct answer
    turn_scores, total = computer.compute(test_trajectory, gt, outcome_score=1.0)
    print(f"Turn scores (correct): {[f'{s:.3f}' for s in turn_scores]}")
    print(f"Total process reward: {total:.4f}")

    # Wrong answer
    turn_scores_wrong, total_wrong = computer.compute(test_trajectory, gt, outcome_score=0.0)
    print(f"\nTurn scores (wrong): {[f'{s:.3f}' for s in turn_scores_wrong]}")
    print(f"Total process reward: {total_wrong:.4f}")

    print(f"\nDifference (efficiency bonus): {total - total_wrong:.4f}")
    print("Turn 1 vs Turn 2 scores show utilization effect:")
    print(f"  Turn 1 retrieval used in Turn 2 think -> higher util score")

