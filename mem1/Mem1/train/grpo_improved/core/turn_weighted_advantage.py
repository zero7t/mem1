"""
Turn-Weighted Advantage for GRPO

Modulates per-token advantages based on turn-level process quality.
Works WITH existing GRPO advantage (not replacing it).

Key idea:
  - Standard GRPO: all tokens in a trajectory get the same advantage
  - Turn-weighted: tokens in high-quality turns get amplified advantage,
    tokens in low-quality turns get dampened advantage
  - Sign-aligned: good turns in winning trajectories get MORE positive signal,
    good turns in losing trajectories get PROTECTED from negative signal

Integration point: After compute_advantage() in ray_trainer.py,
before passing to actor update.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple


# =============================================================================
# Turn Boundary Detection
# =============================================================================

def find_turn_boundaries_from_tokens(
    response_ids: torch.Tensor,
    tokenizer,
    max_turns: int = 6,
) -> List[Tuple[int, int]]:
    """
    Find token-level boundaries for each turn in a response.

    A "turn" is defined as: <think>...</think><search>...</search>
    followed by <information>...</information> (environment response).

    Returns:
        List of (start_pos, end_pos) for each turn in token space.
        Positions are relative to response start (0-indexed).
    """
    # Decode to find turn boundaries in text, then map back to tokens
    # This is more robust than searching for specific token IDs
    text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Find each <search>...</search> as turn delimiter
    turn_boundaries = []
    search_pattern = re.compile(r'<search>.*?</search>', re.DOTALL)

    # Find character positions of each search block end
    # Each turn ends at the end of its <information> block
    info_pattern = re.compile(r'</information>', re.DOTALL)
    info_ends = [m.end() for m in info_pattern.finditer(text)]

    # Turn boundaries: [0, info_end_1], [info_end_1, info_end_2], ...
    prev_end = 0
    for info_end in info_ends[:max_turns]:
        turn_boundaries.append((prev_end, info_end))
        prev_end = info_end

    # Remaining text (final think + answer) = last segment
    if prev_end < len(text):
        turn_boundaries.append((prev_end, len(text)))

    # Convert character positions to approximate token positions
    # Use a simple heuristic: chars_per_token ratio
    total_chars = len(text)
    total_tokens = len(response_ids)

    if total_chars == 0:
        return [(0, total_tokens)]

    token_boundaries = []
    for char_start, char_end in turn_boundaries:
        tok_start = int(char_start / total_chars * total_tokens)
        tok_end = int(char_end / total_chars * total_tokens)
        tok_end = min(tok_end, total_tokens)
        token_boundaries.append((tok_start, tok_end))

    return token_boundaries


def find_turn_boundaries_from_text(
    full_text: str,
    response_length: int,
    prompt_length_chars: int = 0,
) -> List[Tuple[int, int]]:
    """
    Simpler version: find turn boundaries as fraction of response length.
    Used when we don't have access to tokenizer during advantage computation.

    Returns token-position boundaries (approximate).
    """
    # Only look at response portion
    response_text = full_text[prompt_length_chars:]
    total_chars = len(response_text) if response_text else 1

    # Find </information> positions as turn delimiters
    info_pattern = re.compile(r'</information>', re.DOTALL)
    info_ends = [m.end() for m in info_pattern.finditer(response_text)]

    boundaries = []
    prev_end = 0
    for info_end in info_ends:
        tok_start = int(prev_end / total_chars * response_length)
        tok_end = int(info_end / total_chars * response_length)
        boundaries.append((tok_start, min(tok_end, response_length)))
        prev_end = info_end

    # Final segment
    if prev_end < total_chars:
        tok_start = int(prev_end / total_chars * response_length)
        boundaries.append((tok_start, response_length))

    return boundaries if boundaries else [(0, response_length)]


# =============================================================================
# Turn-Weighted Advantage Computation
# =============================================================================

def compute_turn_weighted_advantages(
    advantages: torch.Tensor,
    turn_scores_batch: List[List[float]],
    turn_boundaries_batch: List[List[Tuple[int, int]]],
    eos_mask: torch.Tensor,
    alpha: float = 0.3,
    clip_range: float = 1.0,
) -> torch.Tensor:
    """
    Modulate per-token advantages based on turn-level process quality.

    Formula:
        w_t = 1 + alpha * sign(A_i) * clip(normalize(p_t), -clip_range, clip_range)
        token_advantage[i, pos] = A_i * w_t   (for pos in turn t)

    Args:
        advantages: (bs, response_length) - standard GRPO advantages (uniform per trajectory)
        turn_scores_batch: List[List[float]] - per-turn process scores for each sample
        turn_boundaries_batch: List[List[Tuple[int,int]]] - token boundaries per turn per sample
        eos_mask: (bs, response_length) - valid token mask
        alpha: float - weighting strength (0.3 = w in [0.7, 1.3])
        clip_range: float - max normalized score magnitude

    Returns:
        weighted_advantages: (bs, response_length) - modulated advantages
    """
    bs, response_length = advantages.shape
    weighted_adv = advantages.clone()

    for i in range(bs):
        turn_scores = turn_scores_batch[i]
        boundaries = turn_boundaries_batch[i]

        if not turn_scores or len(turn_scores) < 2:
            continue  # Single turn or no scores: no weighting needed

        # Get trajectory-level advantage sign
        # (all tokens have same advantage in standard GRPO)
        traj_adv = advantages[i, :].sum().item()
        if abs(traj_adv) < 1e-8:
            continue  # Zero advantage (DAPO filtered): skip

        sign_a = 1.0 if traj_adv > 0 else -1.0

        # Normalize turn scores within this trajectory
        scores_arr = np.array(turn_scores, dtype=np.float32)
        mean_s = scores_arr.mean()
        std_s = scores_arr.std()
        if std_s < 0.01:
            continue  # All turns roughly equal: no differentiation needed

        normalized = (scores_arr - mean_s) / (std_s + 1e-6)
        normalized = np.clip(normalized, -clip_range, clip_range)

        # Apply weights to each turn's tokens
        for t, (start, end) in enumerate(boundaries):
            if t >= len(normalized):
                break
            end = min(end, response_length)
            if start >= end:
                continue

            w_t = 1.0 + alpha * sign_a * normalized[t]
            weighted_adv[i, start:end] = advantages[i, start:end] * w_t

    # Re-apply eos_mask
    weighted_adv = weighted_adv * eos_mask

    return weighted_adv


# =============================================================================
# Batch Processing Helper
# =============================================================================

def apply_turn_weighting(
    batch_advantages: torch.Tensor,
    batch_turn_scores: List[List[float]],
    batch_texts: List[str],
    response_length: int,
    eos_mask: torch.Tensor,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    High-level helper: apply turn weighting to a batch of advantages.

    Args:
        batch_advantages: (bs, response_length)
        batch_turn_scores: per-sample list of per-turn scores
        batch_texts: decoded trajectory texts (response portion only)
        response_length: int
        eos_mask: (bs, response_length)
        config: optional config dict

    Returns:
        weighted_advantages: (bs, response_length)
    """
    config = config or {}
    alpha = config.get('turn_weight_alpha', 0.3)
    clip_range = config.get('turn_weight_clip', 1.0)

    bs = batch_advantages.shape[0]

    # Compute turn boundaries for each sample
    turn_boundaries_batch = []
    for i in range(bs):
        if i < len(batch_texts) and batch_texts[i]:
            boundaries = find_turn_boundaries_from_text(
                full_text=batch_texts[i],
                response_length=response_length,
                prompt_length_chars=0,  # texts should be response-only
            )
        else:
            boundaries = [(0, response_length)]
        turn_boundaries_batch.append(boundaries)

    # Pad turn_scores to match batch size
    padded_scores = []
    for i in range(bs):
        if i < len(batch_turn_scores) and batch_turn_scores[i]:
            padded_scores.append(batch_turn_scores[i])
        else:
            padded_scores.append([0.0])

    return compute_turn_weighted_advantages(
        advantages=batch_advantages,
        turn_scores_batch=padded_scores,
        turn_boundaries_batch=turn_boundaries_batch,
        eos_mask=eos_mask,
        alpha=alpha,
        clip_range=clip_range,
    )


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=== Turn-Weighted Advantage Test ===\n")

    # Simulate: 2 trajectories, 3 turns each, response_length=30
    bs, resp_len = 2, 30
    eos_mask = torch.ones(bs, resp_len)

    # Trajectory 0: positive advantage (good trajectory)
    # Trajectory 1: negative advantage (bad trajectory)
    advantages = torch.zeros(bs, resp_len)
    advantages[0, :] = 0.5   # winning
    advantages[1, :] = -0.3  # losing

    # Turn scores: [turn0, turn1, turn2]
    # Traj 0: turn0=bad search, turn1=great search, turn2=ok
    # Traj 1: turn0=great search, turn1=bad, turn2=bad
    turn_scores = [
        [-0.1, 0.4, 0.1],   # traj 0
        [0.3, -0.2, -0.1],  # traj 1
    ]

    # Boundaries: each turn = 10 tokens
    boundaries = [
        [(0, 10), (10, 20), (20, 30)],
        [(0, 10), (10, 20), (20, 30)],
    ]

    weighted = compute_turn_weighted_advantages(
        advantages=advantages,
        turn_scores_batch=turn_scores,
        turn_boundaries_batch=boundaries,
        eos_mask=eos_mask,
        alpha=0.3,
    )

    print("Trajectory 0 (winning, A=+0.5):")
    print(f"  Turn 0 (bad search):  adv {advantages[0,0]:.3f} -> {weighted[0,0]:.3f}")
    print(f"  Turn 1 (great search): adv {advantages[0,10]:.3f} -> {weighted[0,10]:.3f}")
    print(f"  Turn 2 (ok search):   adv {advantages[0,20]:.3f} -> {weighted[0,20]:.3f}")

    print(f"\nTrajectory 1 (losing, A=-0.3):")
    print(f"  Turn 0 (great search): adv {advantages[1,0]:.3f} -> {weighted[1,0]:.3f}")
    print(f"  Turn 1 (bad search):  adv {advantages[1,10]:.3f} -> {weighted[1,10]:.3f}")
    print(f"  Turn 2 (bad search):  adv {advantages[1,20]:.3f} -> {weighted[1,20]:.3f}")

    print("\nExpected behavior:")
    print("  Traj 0 Turn 1 (great in winner): amplified positive")
    print("  Traj 0 Turn 0 (bad in winner): dampened positive")
    print("  Traj 1 Turn 0 (great in loser): PROTECTED (less negative)")
    print("  Traj 1 Turn 1 (bad in loser): amplified negative")
