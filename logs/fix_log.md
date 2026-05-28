# V3 Training Fix Log

## Fix #1: normalize_answer AttributeError (2026-05-28)

**File:** `mem1/Mem1/train/grpo_v3/reward/rule_reward_v3.py`
**Error:** `AttributeError: 'numpy.ndarray' object has no attribute 'lower'`
**Root Cause:** `answer_targets` contains numpy arrays instead of plain strings. When `compute_per_turn_scores` iterates `answer_targets` and passes each `tgt` to `extract_key_phrases → normalize_answer`, the function calls `.lower()` on a numpy array.

**Fix Applied:**
```python
# BEFORE (line 17-25):
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

# AFTER:
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
```

**Revert:** Replace lines 17-30 of `rule_reward_v3.py` with the BEFORE version.

---

## Fix #2: CurriculumController.phase_name AttributeError (2026-05-28)

**File:** `mem1/Mem1/train/grpo_v3/main_ppo_v3.py` (line 248)
**Error:** `AttributeError: 'CurriculumController' object has no attribute 'phase_name'`
**Root Cause:** `main_ppo_v3.py` references `self.curriculum.phase_name` but the `CurriculumController` class (in `core/curriculum.py`) uses `current_phase` as the attribute name for the active phase string.

**Fix Applied:**
```python
# BEFORE (line 248):
'v3/curriculum_phase': float(list(PHASES.keys()).index(self.curriculum.phase_name)),

# AFTER:
'v3/curriculum_phase': float(list(PHASES.keys()).index(self.curriculum.current_phase)),
```

**Revert:** In `main_ppo_v3.py` line 248, change `.current_phase` back to `.phase_name`.

---
