from __future__ import annotations

import logging
from math import ceil

logger = logging.getLogger(__name__)


def calibrate(val_scores: list[float], fpr_target: float = 0.01) -> float:
    """Select anomaly threshold from validation scores at target false-positive rate.

    Calibration converts raw score distributions into actionable alert cutoffs.
    """

    if not val_scores:
        raise ValueError("val_scores must not be empty")
    if not 0.0 < fpr_target < 1.0:
        raise ValueError("fpr_target must be between 0 and 1")

    scores = sorted(val_scores)
    rank = ceil((1.0 - fpr_target) * (len(scores) - 1))
    index = min(max(rank, 0), len(scores) - 1)
    threshold = float(scores[index])
    logger.info("threshold_calibrated samples=%d", len(scores))
    return threshold
