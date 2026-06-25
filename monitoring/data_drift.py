"""Feature-level drift detection helpers based on lightweight statistics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, stdev

from .metrics import feature_drift_score


@dataclass(frozen=True)
class DriftResult:
    """Result object for one feature drift evaluation."""

    feature_name: str
    drifted: bool
    deviation: float


def detect_feature_drift(current: list[float], reference: list[float], feature_name: str) -> DriftResult:
    """Detect mean shift drift using a two-standard-deviation reference threshold.

    We use 2 * stdev(reference) as the cutoff because, under approximately stable
    distributions, values beyond about two standard deviations represent an uncommon
    shift magnitude and are a practical early-warning threshold without over-triggering.
    """

    if not current:
        raise ValueError("current must not be empty")
    if not reference:
        raise ValueError("reference must not be empty")
    if len(reference) < 2:
        raise ValueError("reference must contain at least 2 values")

    current_mean = fmean(current)
    reference_mean = fmean(reference)
    reference_stdev = stdev(reference)
    deviation = abs(current_mean - reference_mean)
    threshold = 2.0 * reference_stdev
    drifted = deviation > threshold

    # Label tracks per-feature drift signal without recording any raw user samples.
    feature_drift_score.labels(feature_name=feature_name).set(deviation)
    return DriftResult(feature_name=feature_name, drifted=drifted, deviation=deviation)
