from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from config.settings import load_settings

logger = logging.getLogger(__name__)

AllowedVerdict = Literal["TP", "FP", "FN"]  # Constrain accepted verdict labels for type-safe downstream logic.
_ALLOWED_VERDICTS: set[str] = {"TP", "FP", "FN"}


@dataclass(frozen=True)
class FeedbackRecord:
    """Represent a single analyst feedback decision for RL updates and auditability."""

    alert_id: str
    verdict: Literal["TP", "FP", "FN"]
    reason: str
    analyst_id: str
    timestamp: str


def _mask_identifier(identifier: str) -> str:
    """Mask an identifier before logging so raw alert or analyst IDs are never emitted."""

    if len(identifier) <= 4:
        return "***"  # Short identifiers are fully masked to avoid easy reconstruction.
    return f"{identifier[:2]}***{identifier[-2:]}"  # Keep tiny context for traceability only.


def _validate_verdict(verdict: str) -> AllowedVerdict:
    """Validate verdict values to guarantee downstream reward mapping remains well-defined."""

    if verdict not in _ALLOWED_VERDICTS:
        raise ValueError("verdict must be one of TP, FP, FN")
    return cast(AllowedVerdict, verdict)  # Cast after membership check for strict type-checkers.


def _append_jsonl_record(record: FeedbackRecord, log_path: Path) -> None:
    """Append feedback as a JSONL line so history is immutable and never overwritten."""

    log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent exists for first write.
    with log_path.open("a", encoding="utf-8") as handle:  # Append mode preserves prior records.
        handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")


def collect_feedback(
    alert_id: str,
    verdict: str,
    reason: str,
    analyst_id: str,
) -> FeedbackRecord:
    """Validate and persist analyst verdicts as append-only records for RL learning.

    This function centralizes validation and persistence so reward updates and
    audit trails stay consistent across the system while protecting sensitive
    user context from logs.
    """

    _validate_verdict(verdict)
    validated_verdict: Literal["TP", "FP", "FN"] = cast(Literal["TP", "FP", "FN"], verdict)
    settings = load_settings()
    log_path = Path(settings.RL_FEEDBACK_LOG_PATH)
    record = FeedbackRecord(
        alert_id=alert_id,
        verdict=validated_verdict,
        reason=reason,
        analyst_id=analyst_id,
        timestamp=datetime.now(timezone.utc).isoformat(),  # UTC keeps ordering stable globally.
    )
    _append_jsonl_record(record, log_path)
    logger.info(
        "feedback_collected verdict=%s alert_id=%s analyst_id=%s",
        validated_verdict,
        _mask_identifier(alert_id),  # Mask identifiers to avoid raw security data in logs.
        _mask_identifier(analyst_id),
    )
    return record
