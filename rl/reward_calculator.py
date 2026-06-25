from __future__ import annotations

from typing import Final

_TP_REWARD: Final[float] = 1.0
_FP_REWARD: Final[float] = -0.5
_FN_REWARD: Final[float] = -2.0
_CRITICAL_FN_REWARD: Final[float] = -4.0  # Highest penalty reflects severe impact of missed critical threats.


def compute_reward(verdict: str, severity: str) -> float:
    """Map analyst outcomes to asymmetric rewards for SOC-safe learning behavior.

    The values are intentionally asymmetric: true positives are rewarded modestly
    (+1.0) to reinforce good detections without encouraging aggressive threshold
    collapse; false positives are penalized lightly (-0.5) because analyst time
    is costly but not typically catastrophic; false negatives are penalized much
    more heavily (-2.0) because missed malicious activity creates direct exposure;
    and critical false negatives are penalized most severely (-4.0) because
    missing high-impact attacks can cause major operational, legal, and financial harm.
    """

    if verdict == "TP":
        return _TP_REWARD
    if verdict == "FP":
        return _FP_REWARD
    if verdict == "FN":
        if severity.lower() == "critical":
            return _CRITICAL_FN_REWARD  # Critical misses carry disproportionate SOC risk.
        return _FN_REWARD
    raise ValueError("verdict must be one of TP, FP, FN")
