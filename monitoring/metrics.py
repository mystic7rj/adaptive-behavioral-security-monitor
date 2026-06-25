"""Prometheus metric primitives used by runtime monitoring checks."""

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "prometheus-client is required for monitoring metrics. Install project dependencies before importing monitoring.metrics."
    ) from exc

alert_rate_total = Counter(
    "alert_rate_total",
    "Counts every dispatched alert so sudden spikes reveal possible alert storms or detector instability.",
)
fp_rate_gauge = Gauge(
    "fp_rate_gauge",
    "Tracks current false-positive rate; sustained increases indicate precision degradation and analyst fatigue risk.",
)
reconstruction_error_histogram = Histogram(
    "reconstruction_error_histogram",
    "Measures reconstruction-error distribution to detect anomaly-score inflation or distributional shifts.",
    buckets=(0.1, 0.3, 0.5, 1.0, 2.0, 5.0),
)
pipeline_lag_seconds = Histogram(
    "pipeline_lag_seconds",
    "Captures ingest-to-alert latency so tail growth exposes operational bottlenecks before SLA misses escalate.",
)
feature_drift_score = Gauge(
    "feature_drift_score",
    "Stores per-feature drift deviation magnitude; rising values identify which feature distributions are diverging first.",
    labelnames=("feature_name",),
)


def get_metrics() -> bytes:
    """Expose all Prometheus metrics as bytes for scrape endpoints.

    Metric intent and degradation signals:
    - alert_rate_total: detects sudden alert volume spikes that can imply model instability.
    - fp_rate_gauge: surfaces precision regressions that increase false investigation load.
    - reconstruction_error_histogram: catches model-reconstruction behavior drift over time.
    - pipeline_lag_seconds: highlights latency growth threatening near-real-time response.
    - feature_drift_score: pinpoints which feature distributions are diverging from baseline.
    """

    metric_payload = generate_latest()
    return metric_payload
