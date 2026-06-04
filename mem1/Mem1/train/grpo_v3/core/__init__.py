"""
Turn-Weighted Advantage V3

Mathematical design for turn weight distribution:

Given T turns with process scores [s_0, s_1, ..., s_{T-1}], we want weights w_t such that:
1. First turn has slightly higher influence (it sets the search direction)
2. Weights are smooth and don't create extreme gradients
3. The total advantage is preserved (weights sum to T, i.e. mean=1)

Formula:
    raw_t = positional_weight(t) * quality_score(s_t)

    positional_weight(t) = 1 + beta * exp(-t / tau)
        - Exponential decay from first turn
        - beta=0.3: first turn gets 1.3x, decays to ~1.0 by turn 3
        - tau = T/2: half-life at midpoint

    quality_score(s_t) = tanh(gamma * (s_t - s_mean) / (s_std + eps))
        - Smooth normalization via tanh (bounded [-1, 1])
        - gamma=1.5: moderate sensitivity
        - Centered on trajectory mean (relative quality)

    final_weight(t) = 1 + alpha * positional_weight(t) * quality_score(s_t)
        - alpha controls overall modulation strength
        - Result: w_t in [1-alpha*1.3, 1+alpha*1.3] ≈ [0.61, 1.39] for alpha=0.3

The advantage for token at position p in turn t becomes:
    A'(p) = A_trajectory * final_weight(t)

Sign-alignment:
    - Winning trajectory (A>0) + good turn → amplified positive
    - Winning trajectory (A>0) + bad turn → dampened positive
    - Losing trajectory (A<0) + good turn → PROTECTED (less negative)
    - Losing trajectory (A<0) + bad turn → amplified negative
"""

import re
import math
import torch
import numpy as np
from typing import List, Tuple, Optional


def compute_positional_weights(num_turns: int, beta: float = 0.3) -> np.ndarray:
    """
    Bell-curve positional prior: peak at turn 2-3 (the strategic decision point).

    w_pos(t) = 1 + beta * exp(-(t - t_peak)^2 / (2*sigma^2))

    - t_peak = (T-1) * 0.35: peak at ~1/3 of trajectory (turn 2 for T=6)
    - sigma = T/3: covers the middle segment
    - First turn: moderate (initial query is formulaic)
    - Middle turns: highest (strategy divergence happens here)
    - Last turn: lowest (outcome reward already covers final answer)
    """
    if num_turns <= 1:
        return np.array([1.0])
    T = num_turns
    t_peak = (T - 1) * 0.35
    sigma = max(T / 3.0, 1.0)
    t = np.arange(T, dtype=np.float64)
    return 1.0 + beta * np.exp(-(t - t_peak)**2 / (2 * sigma**2))


def compute_turn_weights(
    turn_scores: List[float],
    alpha: float = 0.3,
    gamma: float = 1.5,
    sign_a: float = 1.0,
) -> np.ndarray:
    """
    Proportional turn weight with concave (sin-shaped) mapping.

    Steps:
      1. deviation(t) = (s_t - mean) / (range + eps)  → ∈ [-1, 1]
      2. mapped(t) = sin(π/2 * deviation)             → ∈ [-1, 1], concave
      3. w(t) = 1 + alpha * sign(A) * mapped(t)       → ∈ [1-alpha, 1+alpha]

    The sin mapping (concave for x>0, convex for x<0):
    - Small deviations → weight stays close to 1.0 (slope=π/2≈1.57 at origin, but curvature pulls back)
    - Large deviations → approaches ±alpha but with diminishing returns
    - Net effect: compresses weight variance, only clearly different turns get meaningful weight shift
    """
    n = len(turn_scores)
    if n <= 1:
        return np.array([1.0])

    scores = np.array(turn_scores, dtype=np.float64)
    mean = scores.mean()
    score_range = scores.max() - scores.min()

    if score_range < 1e-6:
        return np.ones(n)

    # Normalize to [-1, 1]
    deviation = (scores - mean) / score_range
    deviation = np.clip(deviation, -1.0, 1.0)

    # Concave mapping: sin(π/2 * x)
    mapped = np.sin(np.pi / 2.0 * deviation)

    return 1.0 + alpha * sign_a * mapped


def find_turn_boundaries(response_text: str, response_length: int) -> List[Tuple[int, int]]:
    """
    Map turn boundaries from text to token positions.
    Uses segment-length-proportional mapping (weighted by char count per segment)
    which is more accurate than uniform linear mapping for variable-density text.
    """
    total_chars = len(response_text) if response_text else 1
    info_ends = [m.end() for m in re.finditer(r'</information>', response_text)]

    if not info_ends:
        return [(0, response_length)]

    # Compute char lengths per segment
    segments = []
    prev = 0
    for end in info_ends:
        segments.append((prev, end))
        prev = end
    if prev < total_chars:
        segments.append((prev, total_chars))

    # Map proportionally: each segment gets tokens proportional to its char length
    char_lengths = [e - s for s, e in segments]
    total_char_len = sum(char_lengths)
    if total_char_len == 0:
        return [(0, response_length)]

    boundaries = []
    tok_cursor = 0
    for i, clen in enumerate(char_lengths):
        tok_span = int(round(clen / total_char_len * response_length))
        if i == len(char_lengths) - 1:
            tok_span = response_length - tok_cursor  # last segment gets remainder
        boundaries.append((tok_cursor, min(tok_cursor + tok_span, response_length)))
        tok_cursor += tok_span

    return boundaries if boundaries else [(0, response_length)]


def apply_turn_weighted_advantage(
    advantages: torch.Tensor,
    turn_scores_batch: List[List[float]],
    response_texts: List[str],
    eos_mask: torch.Tensor,
    alpha: float = 0.3,
    gamma: float = 1.5,
) -> torch.Tensor:
    """
    Apply turn-level weighting to GRPO advantages.

    Args:
        advantages: (bs, response_length) uniform per-trajectory
        turn_scores_batch: per-sample list of per-turn process scores
        response_texts: decoded response texts
        eos_mask: (bs, response_length)
        alpha, gamma: weight parameters

    Returns:
        weighted_advantages: (bs, response_length)
    """
    bs, resp_len = advantages.shape
    weighted = advantages.clone()

    for i in range(bs):
        traj_adv_sum = advantages[i].sum().item()
        if abs(traj_adv_sum) < 1e-8:
            continue  # DAPO-filtered, skip

        scores = turn_scores_batch[i] if i < len(turn_scores_batch) else [0.0]
        if not scores or len(scores) < 2:
            continue

        sign_a = 1.0 if traj_adv_sum > 0 else -1.0
        text = response_texts[i] if i < len(response_texts) else ""
        boundaries = find_turn_boundaries(text, resp_len)
        weights = compute_turn_weights(scores, alpha, gamma, sign_a)

        # V3 original: multiplicative weighting (no length bias)
        # Each token keeps its original advantage, scaled by turn weight
        for t, (start, end) in enumerate(boundaries):
            if t >= len(weights):
                break
            end = min(end, resp_len)
            if start >= end:
                continue
            weighted[i, start:end] = advantages[i, start:end] * weights[t]

    return weighted * eos_mask
