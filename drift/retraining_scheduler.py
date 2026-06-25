from __future__ import annotations

import logging

from scipy.stats import ks_2samp

from .adwin_detector import ADWINDetector

logger = logging.getLogger(__name__)


def check_and_schedule_retrain(
    current: list[float],
    reference: list[float],
    detector: ADWINDetector,
) -> bool:
    """Combine batch and online drift signals to decide retraining necessity.

    Using both KS and ADWIN catches both distribution shifts and streaming mean
    changes that either method alone might miss.
    """

    if not current or not reference:
        raise ValueError("current and reference must not be empty")

    ks_result = ks_2samp(current, reference)
    adwin_triggered = False
    for value in current:
        if detector.add_element(value):
            adwin_triggered = True

    drift = ks_result.pvalue < 0.05 or adwin_triggered  # Trigger if either detector flags drift risk.
    if drift:
        logger.warning(
            "drift_signal ks_pvalue=%.6f adwin=%s",
            ks_result.pvalue,
            adwin_triggered,
        )
    return bool(drift)
