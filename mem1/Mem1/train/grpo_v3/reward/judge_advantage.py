"""
Judge-to-Advantage Conversion V3

Converts pointwise scores and listwise rankings into advantage adjustments.

Design principles:
1. Pointwise: absolute quality → direct reward/penalty
2. Listwise: relative ranking → only with margin (confidence gating)
3. Combined: listwise only activates when pointwise shows differentiation
"""

import numpy as np
from typing import List, Optional, Tuple


def pointwise_to_advantage(
    scores: List[int],
    scale: float = 0.2,
    neutral_score: int = 3,
) -> List[float]:
    """
    Convert pointwise scores (1-5) to advantage adjustments.

    Score 3 = neutral (no adjustment)
    Score 5 = +scale, Score 1 = -scale
    Linear mapping: advantage = scale * (score - neutral) / (5 - neutral)

    Args:
        scores: list of 1-5 scores for each trajectory in group
        scale: max advantage magnitude
        neutral_score: score that maps to 0 advantage
    """
    advantages = []
    for s in scores:
        # Map [1,5] → [-scale, +scale] with 3 as center
        adj = scale * (s - neutral_score) / 2.0
        advantages.append(adj)
    return advantages


def listwise_to_advantage_margin(
    ranking: List[int],
    pointwise_scores: List[int],
    scale: float = 0.2,
    reward_threshold: int = 4,
    punish_threshold: int = 2,
) -> List[float]:
    """
    Margin-based listwise advantage: only reward/punish when pointwise confirms.

    Rules:
    - Rank 1 gets +reward ONLY if its pointwise >= reward_threshold
    - Rank last gets -penalty ONLY if its pointwise <= punish_threshold
    - All others get 0 (no signal from ranking alone)

    This prevents:
    - Rewarding "best of garbage" (all bad but one ranked first)
    - Punishing "worst of excellence" (all good but one ranked last)

    Args:
        ranking: [best_idx, ..., worst_idx] (0-indexed)
        pointwise_scores: [score_0, score_1, ...] per trajectory
        scale: reward/penalty magnitude
        reward_threshold: min pointwise score to receive ranking reward
        punish_threshold: max pointwise score to receive ranking penalty
    """
    n = len(ranking)
    advantages = [0.0] * n

    if n < 2:
        return advantages

    best_idx = ranking[0]
    worst_idx = ranking[-1]

    # Only reward best if pointwise confirms quality
    if pointwise_scores[best_idx] >= reward_threshold:
        advantages[best_idx] = +scale

    # Only punish worst if pointwise confirms poor quality
    if pointwise_scores[worst_idx] <= punish_threshold:
        advantages[worst_idx] = -scale

    return advantages


def compute_judge_advantages(
    group_pointwise_scores: List[int],
    group_ranking: Optional[List[int]],
    pointwise_scale: float = 0.2,
    listwise_scale: float = 0.2,
    use_listwise: bool = False,
    reward_threshold: int = 4,
    punish_threshold: int = 2,
) -> List[float]:
    """
    Combined judge advantage for a single group.

    Phase-dependent behavior:
    - Transition phase: pointwise only
    - Refinement phase: pointwise + listwise (margin-gated)
    """
    n = len(group_pointwise_scores)

    # Pointwise contribution (always active when judge is on)
    pw_adv = pointwise_to_advantage(group_pointwise_scores, pointwise_scale)

    if not use_listwise or group_ranking is None:
        return pw_adv

    # Listwise contribution (margin-gated by pointwise)
    lw_adv = listwise_to_advantage_margin(
        ranking=group_ranking,
        pointwise_scores=group_pointwise_scores,
        scale=listwise_scale,
        reward_threshold=reward_threshold,
        punish_threshold=punish_threshold,
    )

    # Sum both signals
    return [p + l for p, l in zip(pw_adv, lw_adv)]
