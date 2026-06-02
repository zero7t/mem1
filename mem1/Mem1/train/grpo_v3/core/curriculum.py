"""
Curriculum Learning Controller V3

Auto-transitions between training phases based on EM rate and F1 score,
not fixed step counts. Each phase has different reward signals and strengths.

Phases:
1. Warmup: Learn format (strong format reward, no process)
2. Convergence: Learn to search (rule process reward + DAPO resample)
3. Transition: Introduce LLM pointwise judge
4. Refinement: Pointwise + listwise for ceiling

Transition triggers:
- Warmup → Convergence: format_accuracy > 0.9 (or step > 80)
- Convergence → Transition: em_rate > 0.15 (or step > 300)
- Transition → Refinement: em_rate > 0.30 (or step > 500)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import deque


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    lambda_process: float          # rule process reward scale
    outcome_gate: float            # gate multiplier for process reward
    format_reward_weight: float    # format correctness reward
    dapo_resample: bool            # enable DAPO dynamic sampling
    pointwise_judge: bool          # enable pointwise LLM judge
    listwise_judge: bool           # enable listwise ranking judge
    pointwise_scale: float = 0.0   # scale for pointwise reward
    listwise_scale: float = 0.0    # scale for listwise reward
    listwise_threshold: int = 4    # only reward if pointwise >= this
    listwise_punish_threshold: int = 2  # only punish if pointwise <= this
    format_penalty_only: bool = False  # True = no format reward at all (learned already)


# Default phase configurations
PHASES = {
    'warmup': PhaseConfig(
        name='warmup',
        lambda_process=0.0,
        outcome_gate=0.0,
        format_reward_weight=1.0,
        dapo_resample=False,
        pointwise_judge=False,
        listwise_judge=False,
    ),
    'convergence': PhaseConfig(
        name='convergence',
        lambda_process=0.4,
        outcome_gate=0.5,
        format_reward_weight=0.3,
        dapo_resample=True,
        pointwise_judge=False,
        listwise_judge=False,
        format_penalty_only=True,
    ),
    'transition': PhaseConfig(
        name='transition',
        lambda_process=0.3,
        outcome_gate=0.6,
        format_reward_weight=0.1,
        dapo_resample=True,
        pointwise_judge=True,
        listwise_judge=False,
        pointwise_scale=0.2,
        format_penalty_only=True,
    ),
    'refinement': PhaseConfig(
        name='refinement',
        lambda_process=0.15,
        outcome_gate=0.8,
        format_reward_weight=0.05,
        dapo_resample=True,
        pointwise_judge=True,
        listwise_judge=True,
        pointwise_scale=0.15,
        listwise_scale=0.2,
        listwise_threshold=4,
        listwise_punish_threshold=2,
        format_penalty_only=True,
    ),
}

class CurriculumController:
    """
    Manages phase transitions based on running metrics.

    Uses exponential moving average of EM rate and format accuracy
    to decide when to advance phases. Also has hard step-based fallbacks.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        init_phase = config.get('init_phase', 'warmup')
        self.current_phase = init_phase
        self.phase_history: List[tuple] = []  # (step, phase_name)
        if init_phase != 'warmup':
            print(f"[Curriculum] Starting from phase: {init_phase}")

        # Transition thresholds (metric-based)
        self.warmup_to_convergence_format = config.get('warmup_format_thresh', 0.9)
        self.convergence_to_transition_em = config.get('convergence_em_thresh', 0.15)
        self.transition_to_refinement_em = config.get('transition_em_thresh', 0.30)

        # Hard step fallbacks (if metrics never reach threshold)
        self.warmup_max_step = config.get('warmup_max_step', 80)
        self.warmup_min_step = config.get('warmup_min_step', 10)
        self.convergence_max_step = config.get('convergence_max_step', 300)
        self.transition_max_step = config.get('transition_max_step', 500)

        # EMA tracking
        self._ema_alpha = config.get('ema_alpha', 0.05)
        self._ema_em = 0.0
        self._ema_format = 0.0
        self._ema_f1 = 0.0
        self._step_count = 0

        # Recent history for stability (don't flip-flop)
        self._recent_em = deque(maxlen=20)
        self._recent_format = deque(maxlen=20)

    @property
    def phase(self) -> PhaseConfig:
        return PHASES[self.current_phase]

    def update_metrics(self, step: int, em_rate: float, format_acc: float, f1: float = 0.0):
        """Update running metrics and check for phase transition."""
        self._step_count = step
        self._ema_em = self._ema_alpha * em_rate + (1 - self._ema_alpha) * self._ema_em
        self._ema_format = self._ema_alpha * format_acc + (1 - self._ema_alpha) * self._ema_format
        self._ema_f1 = self._ema_alpha * f1 + (1 - self._ema_alpha) * self._ema_f1
        self._recent_em.append(em_rate)
        self._recent_format.append(format_acc)

        new_phase = self._check_transition(step)
        if new_phase != self.current_phase:
            print(f"[Curriculum] Phase transition: {self.current_phase} → {new_phase} "
                  f"at step {step} (EM={self._ema_em:.3f}, format={self._ema_format:.3f})")
            self.current_phase = new_phase
            self.phase_history.append((step, new_phase))

    def _check_transition(self, step: int) -> str:
        """Determine if phase should advance. Never goes backward."""
        if self.current_phase == 'warmup':
            # Use recent 10-step window average, with min_step floor and max_step ceiling
            if step >= self.warmup_max_step:
                return 'convergence'
            if step >= self.warmup_min_step and len(self._recent_format) >= 10:
                recent_avg = sum(list(self._recent_format)[-10:]) / 10
                if recent_avg >= self.warmup_to_convergence_format:
                    return 'convergence'

        elif self.current_phase == 'convergence':
            if (self._ema_em >= self.convergence_to_transition_em
                    or step >= self.convergence_max_step):
                return 'transition'

        elif self.current_phase == 'transition':
            if (self._ema_em >= self.transition_to_refinement_em
                    or step >= self.transition_max_step):
                return 'refinement'

        return self.current_phase

    def get_metrics_summary(self) -> Dict:
        return {
            'phase': self.current_phase,
            'ema_em': self._ema_em,
            'ema_format': self._ema_format,
            'ema_f1': self._ema_f1,
            'step': self._step_count,
        }
