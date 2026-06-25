from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

from config.settings import load_settings

from .alert_builder import Alert

logger = logging.getLogger(__name__)


def _write_dead_letter(alert: Alert, reason: str) -> None:
    """Persist undelivered alerts so they can be reprocessed instead of silently lost."""

    path = Path(load_settings().ALERT_DEAD_LETTER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the dead-letter destination exists.
    payload = {
        "alert_id": str(alert.alert_id),
        "severity": alert.severity,
        "mitre_tactic": alert.mitre_tactic,
        "recommended_actions": alert.recommended_actions,
        "explanation": asdict(alert.explanation),
        "peer_context": asdict(alert.peer_context),
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    with path.open("a", encoding="utf-8") as handle:  # Append-only DLQ preserves delivery history.
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _deliver(alert: Alert) -> None:
    """Deliver alert using severity-specific channels for operational triage priority."""

    if alert.severity == "critical":
        logger.critical("alert_dispatched severity=critical alert_id=%s", str(alert.alert_id))
        print(f"CRITICAL ALERT: {alert.alert_id}")  # Immediate console signal for urgent response.
        return
    if alert.severity in {"high", "medium"}:
        logger.warning("alert_dispatched severity=%s alert_id=%s", alert.severity, str(alert.alert_id))
        return
    logger.info("alert_dispatched severity=low alert_id=%s", str(alert.alert_id))


def dispatch(alert: Alert) -> None:
    """Dispatch alerts with bounded retries and dead-letter fallback for reliability.

    Dead-letter handling matters in security operations because dropped alerts can
    hide ongoing incidents. Persisting exhausted deliveries ensures investigation
    teams can recover and replay failed notifications.
    """

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            _deliver(alert)
            return
        except Exception as exc:
            # Catch-and-retry is intentional here to implement bounded transient-failure handling.
            logger.warning(
                "alert_dispatch_retry attempt=%d max_attempts=%d alert_id=%s",
                attempt,
                max_attempts,
                str(alert.alert_id),
            )
            if attempt == max_attempts:
                _write_dead_letter(alert, reason=str(exc))
                logger.error("alert_dispatched_to_dead_letter alert_id=%s", str(alert.alert_id))
                return
            sleep(0.1 * (2 ** (attempt - 1)))  # Exponential backoff reduces repeated contention.
