from __future__ import annotations

import logging
from statistics import fmean, pstdev

logger = logging.getLogger(__name__)


class ADWINDetector:
    """Lightweight window-based drift detector approximating ADWIN-style behavior.

    This detector provides fast online drift signals without heavy state requirements.
    """

    def __init__(self, max_window: int = 50) -> None:
        """Initialize bounded sliding window used for mean-shift drift checks."""

        if max_window < 10:
            raise ValueError("max_window must be at least 10")
        self._window: list[float] = []
        self._max_window = max_window

    def add_element(self, value: float) -> bool:
        """Add one observation and return whether drift is currently detected."""

        self._window.append(float(value))
        if len(self._window) > self._max_window:
            self._window.pop(0)  # Keep only the newest observations inside fixed memory budget.
        if len(self._window) < 10:
            return False

        mid = len(self._window) // 2
        left = self._window[:mid]
        right = self._window[mid:]

        if len(left) < 5 or len(right) < 5:
            return False

        left_mean = fmean(left)
        right_mean = fmean(right)
        combined_std = pstdev(self._window)
        if combined_std == 0.0:
            return False

        drifted = abs(left_mean - right_mean) > combined_std
        if drifted:
            logger.warning("drift_detected method=adwin")
        return bool(drifted)

    def reset(self) -> None:
        """Reset detector state after retraining or controlled detector restart."""

        self._window.clear()
        logger.info("adwin_reset")
