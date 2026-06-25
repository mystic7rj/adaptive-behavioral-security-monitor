from __future__ import annotations

import json
import logging
from typing import Any, Callable, cast

from confluent_kafka import Consumer, KafkaError, Message

from config.settings import Settings, configure_logging, load_settings
from ingest.normalizer import normalize
from ingest.schema import NormalizedEvent

logger = logging.getLogger(__name__)


def _deserialize_message(message: Message) -> dict[str, object] | None:
    """Decode and parse Kafka message bytes into JSON dictionaries.

    This helper isolates message parsing so the consume loop can focus on control
    flow, retries, and offset management.
    """

    payload = message.value()
    if payload is None:
        logger.warning("empty_message topic=%s partition=%s offset=%s", message.topic(), message.partition(), message.offset())
        return None
    try:
        return cast(dict[str, object], json.loads(payload.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "malformed_json topic=%s partition=%s offset=%s error=%s",
            message.topic(),
            message.partition(),
            message.offset(),
            exc,
        )
        return None


def consume_loop(
    process_event: Callable[[NormalizedEvent], None],
    settings: Settings | None = None,
    poll_timeout: float = 1.0,
) -> None:
    """Continuously consume, normalize, and process events from Kafka.

    This long-running loop exists to provide at-least-once processing with
    explicit commits only after successful downstream handling.
    """

    resolved = settings or load_settings()
    configure_logging(resolved.LOG_LEVEL)

    consumer = Consumer(
        {
            "bootstrap.servers": resolved.KAFKA_BROKER,
            "group.id": resolved.KAFKA_GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([resolved.KAFKA_TOPIC])
    logger.info(
        "consumer_started broker=%s topic=%s group_id=%s",
        resolved.KAFKA_BROKER,
        resolved.KAFKA_TOPIC,
        resolved.KAFKA_GROUP_ID,
    )

    try:
        while True:
            message = consumer.poll(poll_timeout)
            if message is None:
                continue  # Poll timeouts are expected; keep the loop responsive.
            if message.error():
                error = message.error()
                if isinstance(error, KafkaError):
                    if error.code() != KafkaError._PARTITION_EOF:
                        logger.error("kafka_error error=%s", error)
                else:
                    logger.error("kafka_error error=%s", error)
                continue

            raw = _deserialize_message(message)
            if raw is None:
                continue

            event = normalize(raw)
            if event is None:
                logger.warning(
                    "event_dropped topic=%s partition=%s offset=%s",
                    message.topic(),
                    message.partition(),
                    message.offset(),
                )
                continue

            process_event(event)
            # Commit only after processing succeeds to avoid acknowledging dropped work.
            consumer.commit(message=message, asynchronous=False)
    except KeyboardInterrupt:
        logger.info("consumer_stopped")
    finally:
        consumer.close()
