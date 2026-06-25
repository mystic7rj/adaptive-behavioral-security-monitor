from __future__ import annotations

import logging
from dataclasses import dataclass

from torch import Tensor

from models.autoencoder import BehaviorAutoencoder
from models.lstm_predictor import SessionLSTM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsembleScore:
    """Hold component and blended anomaly scores for downstream decision logic."""

    ae_score: float
    lstm_score: float
    combined_score: float


class EnsembleScorer:
    """Combine autoencoder and LSTM signals into one weighted anomaly score.

    Ensembling reduces single-model blind spots by fusing complementary signals.
    """

    def __init__(
        self,
        autoencoder: BehaviorAutoencoder,
        lstm: SessionLSTM,
        ae_weight: float = 0.5,
        lstm_weight: float = 0.5,
    ) -> None:
        """Initialize score combiner with validated non-negative component weights."""

        if ae_weight < 0 or lstm_weight < 0:
            raise ValueError("weights must be non-negative")
        if ae_weight + lstm_weight == 0:
            raise ValueError("weights must sum to a positive value")
        self.autoencoder = autoencoder
        self.lstm = lstm
        self.ae_weight = ae_weight
        self.lstm_weight = lstm_weight

    def score(self, feature_vec: Tensor, event_seq: Tensor) -> EnsembleScore:
        """Compute and return component scores plus normalized weighted blend."""

        ae_score = self.autoencoder.anomaly_score(feature_vec)
        lstm_score = self.lstm.sequence_log_likelihood(event_seq)
        combined = (ae_score * self.ae_weight + lstm_score * self.lstm_weight) / (
            self.ae_weight + self.lstm_weight
        )
        logger.info("ensemble_scored")
        return EnsembleScore(ae_score=ae_score, lstm_score=lstm_score, combined_score=combined)
