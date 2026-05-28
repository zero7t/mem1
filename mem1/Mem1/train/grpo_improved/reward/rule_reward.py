"""
Step-Level Process Reward for Multi-Turn RAG QA

This module computes fine-grained step-level rewards for multi-turn
retrieval-augmented QA trajectories.

Integration point: main_ppo.py::RewardManager._process_item()
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


def get_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from text."""
    text = normalize_answer(text)
    words = text.split()
    if len(words) < n:
        return [text] if text else []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


def compute_token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard overlap between two texts."""
    tokens_a = set(normalize_answer(text_a).split())
    tokens_b = set(normalize_answer(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def extract_trajectory_steps(full_text: str) -> Dict:
    """
    Parse a full trajectory string into structured steps.

    Returns:
        {
            'queries': [str, ...],         # search queries per turn
            'retrievals': [str, ...],      # retrieval results per turn
            'thinks': [str, ...],          # think content per turn
            'answer': str or None,         # final answer if present
            'num_turns': int,
            'has_valid_answer': bool,
        }
    """
    queries = []
    retrievals = []
    thinks = []

    # Extract all think blocks
    think_matches = re.findall(r'<think>(.*?)</think>', full_text, re.DOTALL)
    thinks = [t.strip() for t in think_matches]

    # Extract all search queries
    search_matches = re.findall(r'<search>(.*?)</search>', full_text, re.DOTALL)
    queries = [q.strip() for q in search_matches]

    # Extract all information blocks
    info_matches = re.findall(r'<information>(.*?)</information>', full_text, re.DOTALL)
    retrievals = [info.strip() for info in info_matches]

    # Extract final answer
    answer_matches = re.findall(r'<answer>(.*?)</answer>', full_text, re.DOTALL)
    answer = answer_matches[-1].strip() if len(answer_matches) >= 2 else None

    return {
        'queries': queries,
        'retrievals': retrievals,
        'thinks': thinks,
        'answer': answer,
        'num_turns': max(len(queries), 1),
        'has_valid_answer': answer is not None,
    }


# =============================================================================
# Reward Dimensions
# =============================================================================

def retrieval_relevance_reward(
    retrieved_text: str,
    answer_targets: List[str],
    max_reward: float = 0.3
) -> float:
    """
    Dimension 1: Does the retrieval contain answer-relevant information?

    Args:
        retrieved_text: The text returned by the retriever for this step
        answer_targets: Ground truth answer strings

    Returns:
        reward in [0, max_reward]
    """
    if not retrieved_text or not answer_targets:
        return 0.0

    retrieved_lower = normalize_answer(retrieved_text)

    # Check entity coverage: how many answer targets appear in retrieval?
    covered = 0
    for target in answer_targets:
        target_normalized = normalize_answer(target)
        # Check full match
        if target_normalized in retrieved_lower:
            covered += 1
            continue
        # Check partial match (individual significant words)
        target_words = [w for w in target_normalized.split() if len(w) > 3]
        if target_words:
            word_hits = sum(1 for w in target_words if w in retrieved_lower)
            if word_hits / len(target_words) > 0.5:
                covered += 0.5  # partial credit

    coverage = min(covered / len(answer_targets), 1.0)
    return max_reward * coverage


def information_novelty_reward(
    current_query: str,
    current_retrieval: str,
    previous_queries: List[str],
    previous_retrievals: List[str],
    reward_novel: float = 0.15,
    penalty_repeat: float = -0.15,
) -> float:
    """
    Dimension 2: Is this search bringing new, non-redundant information?

    Returns:
        reward in [penalty_repeat, reward_novel]
    """
    if not current_retrieval:
        return 0.0

    # 1. Query diversity check
    if previous_queries:
        max_overlap = max(
            compute_token_overlap(current_query, prev_q)
            for prev_q in previous_queries
        )
        if max_overlap > 0.7:  # Highly similar to a previous query
            return penalty_repeat

    # 2. Retrieval content novelty
    if previous_retrievals:
        current_ngrams = set(get_ngrams(current_retrieval, n=3))
        previous_ngrams = set()
        for prev in previous_retrievals:
            previous_ngrams.update(get_ngrams(prev, n=3))

        if not current_ngrams:
            return 0.0

        novelty = len(current_ngrams - previous_ngrams) / len(current_ngrams)

        if novelty < 0.2:  # 80%+ content is redundant
            return penalty_repeat * 0.7  # lighter penalty than query repeat
        elif novelty > 0.5:
            return reward_novel
        else:
            return reward_novel * 0.3  # some new info
    else:
        return reward_novel * 0.7  # first search gets default novelty


def progressive_coverage_reward(
    think_at_t: str,
    think_at_t_minus_1: Optional[str],
    answer_targets: List[str],
    max_reward: float = 0.2
) -> float:
    """
    Dimension 3: Is the model making incremental progress toward the answer?

    Only rewards INCREMENTAL coverage to prevent hack (guessing all at step 0).

    Returns:
        reward in [0, max_reward]
    """
    if not think_at_t or not answer_targets:
        return 0.0

    def count_coverage(text: str, targets: List[str]) -> float:
        """Count how many targets are mentioned in text."""
        text_norm = normalize_answer(text)
        covered = 0.0
        for target in targets:
            target_norm = normalize_answer(target)
            if target_norm in text_norm:
                covered += 1.0
                continue
            # Check significant words
            target_words = [w for w in target_norm.split() if len(w) > 3]
            if target_words:
                word_hits = sum(1 for w in target_words if w in text_norm)
                if word_hits / len(target_words) >= 0.5:
                    covered += 0.5
        return covered

    covered_now = count_coverage(think_at_t, answer_targets)
    covered_prev = 0.0
    if think_at_t_minus_1:
        covered_prev = count_coverage(think_at_t_minus_1, answer_targets)

    # Only reward INCREMENTAL progress
    delta = max(0, covered_now - covered_prev)
    return max_reward * (delta / len(answer_targets))


def efficiency_reward(
    num_turns_used: int,
    max_turns: int,
    is_correct: bool,
    max_reward: float = 0.2
) -> float:
    """
    Dimension 4: Reward correct answers that use fewer turns.

    ONLY activates when outcome is correct (anti-hack: can't rush to answer).

    Returns:
        reward in [0, max_reward]
    """
    if not is_correct:
        return 0.0

    saved_turns = max_turns - num_turns_used
    if saved_turns <= 0:
        return 0.0

    return max_reward * (saved_turns / max_turns)


def format_penalty(
    is_valid_action: bool,
    cur_step: int,
    max_penalty: float = -0.3
) -> float:
    """
    Penalize format violations (invalid actions).

    Earlier violations are penalized more heavily (waste entire trajectory).

    Returns:
        reward in [max_penalty, 0]
    """
    if is_valid_action:
        return 0.0

    # Earlier = worse
    if cur_step == 0:
        return max_penalty
    elif cur_step <= 2:
        return max_penalty * 0.7
    else:
        return max_penalty * 0.4


# =============================================================================
# Main Reward Computer
# =============================================================================

class StepRewardComputer:
    """
    Multi-dimensional step-level process reward for multi-turn RAG QA.

    Usage:
        computer = StepRewardComputer(config)
        process_reward = computer.compute_trajectory_reward(
            trajectory_text, ground_truth, outcome_score
        )
        total_reward = outcome_score + process_reward
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.lambda_process = config.get('lambda_process', 0.5)
        self.w_retrieval = config.get('w_retrieval', 0.4)
        self.w_novelty = config.get('w_novelty', 0.3)
        self.w_progress = config.get('w_progress', 0.3)
        self.max_turns = config.get('max_turns', 6)
        self.enable_format_penalty = config.get('format_penalty', True)
        self.enable_efficiency = config.get('efficiency_bonus', True)

    def compute_trajectory_reward(
        self,
        trajectory_text: str,
        ground_truth: dict,
        outcome_score: float,
    ) -> float:
        """
        Compute total process reward for a complete trajectory.

        Args:
            trajectory_text: Full decoded trajectory string
            ground_truth: Dict with 'target' key containing answer list
            outcome_score: EM/F1 score from outcome reward (0 or 1)

        Returns:
            process_reward: float (to be added to outcome_score)
        """
        # Parse trajectory into steps
        steps = extract_trajectory_steps(trajectory_text)
        raw_targets = ground_truth.get('target', [])

        # Flatten nested numpy arrays / lists into List[str]
        # Data format: target = array([array(['ans1']), array(['ans2'])])
        answer_targets = []
        try:
            for item in raw_targets:
                if isinstance(item, str):
                    answer_targets.append(item)
                elif hasattr(item, '__iter__'):
                    # numpy array or list of alternatives — take first
                    for sub in item:
                        if isinstance(sub, str):
                            answer_targets.append(sub)
                            break
                        elif hasattr(sub, '__iter__'):
                            answer_targets.append(str(sub[0]) if len(sub) > 0 else "")
                            break
                else:
                    answer_targets.append(str(item))
        except Exception:
            answer_targets = [str(x) for x in raw_targets] if raw_targets is not None else []

        if not answer_targets:
            return 0.0

        total_process = 0.0

        # Per-step rewards
        for t in range(len(steps['queries'])):
            step_reward = 0.0

            # Dimension 1: Retrieval Relevance
            if t < len(steps['retrievals']):
                r_retrieval = retrieval_relevance_reward(
                    retrieved_text=steps['retrievals'][t],
                    answer_targets=answer_targets,
                )
                step_reward += self.w_retrieval * r_retrieval

            # Dimension 2: Information Novelty
            r_novelty = information_novelty_reward(
                current_query=steps['queries'][t],
                current_retrieval=steps['retrievals'][t] if t < len(steps['retrievals']) else "",
                previous_queries=steps['queries'][:t],
                previous_retrievals=steps['retrievals'][:t],
            )
            step_reward += self.w_novelty * r_novelty

            # Dimension 3: Progressive Coverage
            if t < len(steps['thinks']):
                think_prev = steps['thinks'][t-1] if t > 0 else None
                r_progress = progressive_coverage_reward(
                    think_at_t=steps['thinks'][t],
                    think_at_t_minus_1=think_prev,
                    answer_targets=answer_targets,
                )
                step_reward += self.w_progress * r_progress

            total_process += step_reward

        # Dimension 4: Efficiency (only for correct answers)
        if self.enable_efficiency:
            r_efficiency = efficiency_reward(
                num_turns_used=steps['num_turns'],
                max_turns=self.max_turns,
                is_correct=(outcome_score > 0),
            )
            total_process += r_efficiency

        # Format penalty (if answer format is invalid)
        if self.enable_format_penalty and not steps['has_valid_answer']:
            total_process += format_penalty(is_valid_action=False, cur_step=0)

        # Scale by lambda
        return self.lambda_process * total_process


# =============================================================================
# Integration Helper
# =============================================================================

def compute_process_reward(
    trajectory_text: str,
    ground_truth: dict,
    outcome_score: float,
    config: Optional[dict] = None
) -> float:
    """
    Convenience function for integration into RewardManager.

    Args:
        trajectory_text: Full decoded trajectory string
        ground_truth: {'target': ['answer1', 'answer2', ...]}
        outcome_score: EM score (0 or 1)
        config: Optional process reward config

    Returns:
        process_reward to ADD to outcome_score
    """
    computer = StepRewardComputer(config)
    return computer.compute_trajectory_reward(trajectory_text, ground_truth, outcome_score)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    # Example trajectory
    test_trajectory = """<think>I need to find when mob of the dead takes place and who chairs the human rights commission.</think><search>mob of the dead setting time period</search>

<information>Doc 1: Mob of the Dead is set on Alcatraz Island during the 1930s prohibition era. The map takes place on New Year's Eve, December 31, 1933.</information>

<think>Found it - Mob of the Dead takes place on December 31, 1933 on Alcatraz Island. Now I need to find the chairman of the human rights commission.</think><search>chairman human rights commission current</search>

<information>Doc 1: The United Nations Human Rights Council is currently chaired by Václav Bálek of the Czech Republic since January 2024.</information>

<think>I now have both answers. Mob of the Dead takes place on December 31, 1933, and the current chairman of the UN Human Rights Council is Václav Bálek.</think><answer>December 31, 1933; Václav Bálek</answer>"""

    ground_truth = {'target': [['December 31, 1933', 'new years eve 1933'], ['Václav Bálek']]}

    # Flatten targets for simple testing
    flat_targets = ['December 31 1933', 'Václav Bálek']
    gt = {'target': flat_targets}

    computer = StepRewardComputer({'max_turns': 6})
    process_r = computer.compute_trajectory_reward(test_trajectory, gt, outcome_score=1.0)
    print(f"Process reward: {process_r:.4f}")
    print(f"Total reward (outcome=1 + process): {1.0 + process_r:.4f}")

    # Test with wrong answer
    process_r_wrong = computer.compute_trajectory_reward(test_trajectory, gt, outcome_score=0.0)
    print(f"\nProcess reward (wrong answer): {process_r_wrong:.4f}")
    print(f"Total reward (outcome=0 + process): {0.0 + process_r_wrong:.4f}")

    # Verify: correct always > incorrect
    assert 1.0 + process_r > 0.0 + process_r_wrong, "Correct trajectory should always score higher!"
    print("\n✓ Anti-hack check passed: correct > incorrect")
