"""
Rule-Based Rewards V3

Three core rule dimensions:
1. Format: Correct use of <think>, <search>, <answer> tags
2. Efficiency: Fewer turns for correct answer
3. Retrieval Quality: Answer appears in retrieved documents (hindsight)

Plus per-turn process scores for turn-weighted advantage.
"""

import re
import math
import string
from typing import List, Dict, Optional, Tuple


def normalize_answer(s) -> str:
    import numpy as np
    if isinstance(s, np.ndarray):
        s = str(s.item()) if s.ndim == 0 else str(s[0]) if len(s) == 1 else " ".join(str(x) for x in s)
    if not isinstance(s, str):
        s = str(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def extract_key_phrases(text: str) -> set:
    return set(w for w in normalize_answer(text).split() if len(w) > 3)


def compute_token_overlap(a: str, b: str) -> float:
    ta, tb = set(normalize_answer(a).split()), set(normalize_answer(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _retrieval_hit(retrieval_text: str, answer_targets: List[str]) -> float:
    """
    Per-turn retrieval hit. Each answer target matched as a complete entity.
    Substring match → 1.0, else token recall > 0.6 → [0.1, 0.5].
    """
    if not retrieval_text or not answer_targets:
        return 0.0
    norm_ret = normalize_answer(retrieval_text)
    ret_tokens = set(norm_ret.split())
    best_score = 0.0
    for target in answer_targets:
        norm_target = normalize_answer(target)
        if not norm_target:
            continue
        if norm_target in norm_ret:
            return 1.0
        target_tokens = set(norm_target.split())
        if not target_tokens or not ret_tokens:
            continue
        recall = len(target_tokens & ret_tokens) / len(target_tokens)
        if recall > 0.6:
            score = 0.1 + 0.4 * (recall - 0.6) / 0.4
            best_score = max(best_score, score)
    return best_score


# =============================================================================
# Format Reward
# =============================================================================

def format_reward(text: str, strict: bool = True) -> float:
    """
    Check format correctness. Returns 0.0 or 1.0 (binary).

    Required structure:
    - At least one <think>...</think> block
    - At least one <search>...</search> block
    - Exactly one final <answer>...</answer> (the last one counts)
    - No nested/duplicate answer tags (anti-hack)

    strict=True (warmup): penalize malformed output (-0.5)
    strict=False (later): just 0 for bad format
    """
    thinks = re.findall(r'<think>.*?</think>', text, re.DOTALL)
    searches = re.findall(r'<search>.*?</search>', text, re.DOTALL)
    answers = re.findall(r'<answer>.*?</answer>', text, re.DOTALL)

    # Must have at least 1 think, 1 search, and answer(s)
    if not thinks or not searches:
        return -0.5 if strict else 0.0

    # Answer validation: need exactly the right pattern
    # Valid: multiple <answer> is ok (intermediate + final), but content must exist
    if len(answers) < 2:
        # Need at least 2: one intermediate "I need to search" and one final
        # Actually for single-turn correct, 1 answer is fine
        if len(answers) < 1:
            return -0.5 if strict else 0.0

    # Anti-hack: if there are way too many answer tags, it's gaming
    if len(answers) > 8:
        return -0.5 if strict else 0.0

    # Check final answer has content
    last_answer = answers[-1]
    content = re.search(r'<answer>(.*?)</answer>', last_answer, re.DOTALL)
    if not content or not content.group(1).strip():
        return -0.3 if strict else 0.0

    return 1.0


def format_penalty_tokens(
    text: str,
    response_length: int,
    tau: float = 30.0,
    strength: float = 0.5,
) -> Optional[List[float]]:
    """
    Penalty-only format reward: returns token-level penalty vector when format is wrong.
    Returns None if format is correct (no reward, no penalty).

    Penalty uses exponential decay (recency effect): tokens closer to the error
    position receive stronger penalty. penalty[t] = -strength * exp(-d/tau)
    where d = error_token_pos - t.
    """
    thinks = re.findall(r'<think>.*?</think>', text, re.DOTALL)
    searches = re.findall(r'<search>.*?</search>', text, re.DOTALL)
    answers = re.findall(r'<answer>.*?</answer>', text, re.DOTALL)

    # Determine error type and character position
    error_char_pos = None

    if not thinks or not searches:
        # Missing basic structure — error at the start
        error_char_pos = 0
    elif len(answers) < 1:
        # No answer tag — error at the end
        error_char_pos = len(text) - 1
    elif len(answers) > 8:
        # Too many answer tags — find the 9th one
        positions = [m.start() for m in re.finditer(r'<answer>', text)]
        error_char_pos = positions[8] if len(positions) > 8 else len(text) - 1
    else:
        # Check final answer content
        last_answer = answers[-1]
        content = re.search(r'<answer>(.*?)</answer>', last_answer, re.DOTALL)
        if not content or not content.group(1).strip():
            # Empty answer — error at the last <answer> tag
            last_pos = text.rfind('<answer>')
            error_char_pos = last_pos if last_pos >= 0 else len(text) - 1

    if error_char_pos is None:
        # Format is correct — no penalty, no reward
        return None

    # Map character position to token position (linear approximation)
    text_len = max(len(text), 1)
    error_token_pos = min(int(error_char_pos / text_len * response_length), response_length - 1)

    # Build exponential decay penalty vector
    penalties = [0.0] * response_length
    for t in range(error_token_pos + 1):
        d = error_token_pos - t
        penalties[t] = -strength * math.exp(-d / tau)

    return penalties


# =============================================================================
# Efficiency Reward
# =============================================================================

def efficiency_reward(num_turns: int, max_turns: int, is_correct: bool) -> float:
    """
    Reward fewer turns for correct answers; penalize fast-wrong (too few turns when incorrect).

    Correct: reward = 0.1 * (max_turns - num_turns) / max_turns
    Incorrect + too few turns: penalty = -0.1 * (max_turns - num_turns) / max_turns
    """
    if num_turns <= 0:
        return 0.0
    saved = max_turns - num_turns
    if saved <= 0:
        return 0.0
    if is_correct:
        # Reward for being efficient when correct
        return 0.1 * saved / max_turns
    else:
        # Penalize for giving up too quickly (fast-wrong)
        return -0.1 * saved / max_turns


# =============================================================================
# Retrieval Quality (Hindsight)
# =============================================================================

def retrieval_quality_reward(
    retrievals: List[str],
    answer_targets: List[str],
    queries: Optional[List[str]] = None,
) -> float:
    """
    Average retrieval hit across turns × 0.3.
    """
    if not retrievals or not answer_targets:
        return 0.0
    total = sum(_retrieval_hit(r, answer_targets) for r in retrievals)
    return 0.3 * total / len(retrievals)

# =============================================================================
# Per-Turn Process Scores (for turn-weighted advantage)
# =============================================================================

def compute_per_turn_scores(
    text: str,
    answer_targets: List[str],
    outcome_score: float,
) -> Tuple[List[float], Dict]:
    """
    Compute per-turn process scores based on incremental retrieval hit.
    Only gives credit for newly covered answer targets (no repeat credit).
    """
    infos = re.findall(r'<information>(.*?)</information>', text, re.DOTALL)
    searches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    num_turns = max(len(searches), 1)

    turn_scores = []
    total_hit = 0.0
    covered_targets = set()  # track which targets already found

    for t in range(num_turns):
        retrieval = infos[t].strip() if t < len(infos) else ""
        score = 0.0
        if retrieval and answer_targets:
            norm_ret = normalize_answer(retrieval)
            ret_tokens = set(norm_ret.split())
            for i, target in enumerate(answer_targets):
                if i in covered_targets:
                    continue
                norm_target = normalize_answer(target)
                if not norm_target:
                    continue
                # Substring match
                if norm_target in norm_ret:
                    score = 1.0
                    covered_targets.add(i)
                    break
                # Recall fallback
                target_tokens = set(norm_target.split())
                if target_tokens and ret_tokens:
                    recall = len(target_tokens & ret_tokens) / len(target_tokens)
                    if recall > 0.6:
                        s = 0.1 + 0.4 * (recall - 0.6) / 0.4
                        if s > score:
                            score = s
                            covered_targets.add(i)
        turn_scores.append(score)
        total_hit += score

    return turn_scores, {
        'num_turns': num_turns,
        'retrieval_hit': total_hit,
    }


# =============================================================================
# Combined Reward Function
# =============================================================================

def compute_reward_v3(
    text: str,
    ground_truth: Dict,
    outcome_score: float,
    phase_config,
    max_turns: int = 6,
) -> Dict:
    """
    Compute all reward components for a single trajectory.

    Returns dict with:
        - total_reward: float (scalar for token_level_rewards)
        - turn_scores: List[float] (for turn-weighted advantage)
        - format_score: float
        - efficiency_score: float
        - retrieval_score: float
        - process_total: float
    """
    answer_targets = ground_truth.get('target', [])
    if isinstance(answer_targets, str):
        answer_targets = [answer_targets]

    # Format
    strict_format = phase_config.format_reward_weight >= 0.5
    fmt_score = format_reward(text, strict=strict_format)

    # Efficiency
    searches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    eff_score = efficiency_reward(len(searches), max_turns, outcome_score > 0.5)

    # Retrieval quality (query-based)
    infos = re.findall(r'<information>(.*?)</information>', text, re.DOTALL)
    ret_score = retrieval_quality_reward(infos, answer_targets, queries=searches)

    # Per-turn process scores
    turn_scores, turn_meta = compute_per_turn_scores(text, answer_targets, outcome_score)

    # Outcome-gated process reward
    process_total = sum(turn_scores) * phase_config.lambda_process
    if outcome_score < 0.5:
        process_total *= phase_config.outcome_gate  # Dampen for wrong answers

    # Total reward
    total = outcome_score
    total += phase_config.format_reward_weight * fmt_score
    total += process_total
    total += eff_score  # Only non-zero when correct

    # Hindsight relabeling: found answer in retrieval but got wrong
    if outcome_score < 0.5 and ret_score > 0:
        total += ret_score * 0.5  # Partial credit

    return {
        'total_reward': total,
        'turn_scores': turn_scores,
        'format_score': fmt_score,
        'efficiency_score': eff_score,
        'retrieval_score': ret_score,
        'process_total': process_total,
        'process_breakdown': turn_meta,
    }
