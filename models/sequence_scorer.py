from __future__ import annotations

import logging

from torch import Tensor

from .lstm_predictor import SessionLSTM

logger = logging.getLogger(__name__)


def score_sequence(seq: Tensor, model: SessionLSTM, baseline_ll: float, baseline_std: float) -> float:
    """Convert sequence log-likelihood into a z-score anomaly measure.

    Standardization against a reference baseline makes scores comparable over time.
    """

    if baseline_std <= 0.0:
        raise ValueError("baseline_std must be positive")
    seq_ll = model.sequence_log_likelihood(seq)
    z_score = (baseline_ll - seq_ll) / baseline_std
    logger.info("sequence_scored")
    return float(z_score)
