"""
Integration instructions for DAPO-lite + Dr.GRPO + LLDS-MA improvements.

To apply these improvements to the existing MEM1 training code:

1. Replace core_algos.py functions
2. Patch dp_actor.py update_policy
3. Add config parameters

This file applies the patches when imported.
"""

import sys
import os

# Add improvements to path
IMPROVEMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, IMPROVEMENTS_DIR)


def patch_core_algos():
    """Replace core_algos functions with improved versions."""
    from verl.trainer.ppo import core_algos
    from core_algos_improved import (
        compute_grpo_outcome_advantage,
        compute_policy_loss,
        compute_llds_loss,
    )

    # Patch advantage computation
    core_algos.compute_grpo_outcome_advantage = compute_grpo_outcome_advantage
    # Patch policy loss (with clip_higher support)
    core_algos.compute_policy_loss = compute_policy_loss
    # Add LLDS loss to module
    core_algos.compute_llds_loss = compute_llds_loss

    print("[IMPROVEMENTS] Patched core_algos with Dr.GRPO + DAPO + LLDS")


def patch_dp_actor():
    """Patch DataParallelPPOActor.update_policy with LLDS-MA + Clip-Higher."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from dp_actor_patch import update_policy_improved

    DataParallelPPOActor.update_policy = update_policy_improved
    print("[IMPROVEMENTS] Patched DataParallelPPOActor.update_policy with LLDS-MA + Clip-Higher")


def apply_all_patches():
    """Apply all improvements."""
    patch_core_algos()
    patch_dp_actor()
    print("[IMPROVEMENTS] All patches applied successfully")
    print("  - Dr.GRPO: mean-only advantage normalization (no std division)")
    print("  - DAPO: dynamic sampling (skip zero-variance groups)")
    print("  - DAPO: clip-higher=0.28 (asymmetric clipping)")
    print("  - LLDS-MA: likelihood-preserving regularization (lambda=0.1)")


if __name__ == "__main__":
    apply_all_patches()
