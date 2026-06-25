from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MONITORING_MODEL_BASELINE_P95_DEFAULT = 0.5
MONITORING_SLA_THRESHOLD_SECONDS_DEFAULT = 60.0


class Settings(BaseSettings):
    """Centralize runtime configuration loaded from environment variables.

    A typed settings object provides a single validation boundary so operational
    misconfiguration is detected early and consistently.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
        case_sensitive=True,
    )

    KAFKA_BROKER: str = Field(..., description="Kafka bootstrap server(s)")
    KAFKA_TOPIC: str = Field(..., description="Kafka topic to consume")
    KAFKA_GROUP_ID: str = Field(..., description="Kafka consumer group id")
    DB_URL: str = Field(..., description="TimescaleDB connection string")
    SCRUBBER_HMAC_KEY: str = Field(..., description="HMAC key for PII scrubbing")
    REDIS_URL: str = Field(..., description="Redis connection URL")
    MODEL_REGISTRY_DIR: str = Field(..., description="Base directory for model registry")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    RL_FEEDBACK_LOG_PATH: str = Field(
        "storage/rl_feedback.jsonl",
        description="Append-only JSONL file used to persist analyst RL feedback",
    )
    RL_Q_TABLE_PATH: str = Field(
        "storage/rl_q_table.json",
        description="JSON file used to persist contextual bandit Q-values",
    )
    PEER_STATS_PATH: str = Field(
        "storage/peer_stats.json",
        description="Aggregated same-role peer statistics used for percentile context",
    )
    ALERT_DEAD_LETTER_PATH: str = Field(
        "storage/alert_dead_letter.jsonl",
        description="Dead-letter JSONL file for alerts that fail all dispatch retries",
    )
    JWT_SECRET_KEY: str = Field(..., description="Secret key used to validate JWT signatures")
    JWT_ISSUER: str = Field(..., description="Expected JWT issuer claim")
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins for API clients",
    )
    MONITORING_MODEL_BASELINE_P95: float = Field(
        MONITORING_MODEL_BASELINE_P95_DEFAULT,
        description="Baseline p95 reconstruction error used by model health checks",
    )
    MONITORING_SLA_THRESHOLD_SECONDS: float = Field(
        MONITORING_SLA_THRESHOLD_SECONDS_DEFAULT,
        description="Default ingest-to-alert latency SLA threshold in seconds",
    )


def load_settings() -> Settings:
    """Load and validate application settings from environment sources.

    This wrapper exists so callers do not depend directly on settings library internals.
    """

    return Settings()  # type: ignore[call-arg]


def configure_logging(level: str) -> None:
    """Configure global logging format/level for structured operational observability.

    Centralized logging setup keeps output consistent across all submodules.
    """

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
