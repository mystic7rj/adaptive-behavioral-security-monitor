from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from enrich.peer_comparator import PeerContext
from explain.shap_explainer import Explanation

Severity = Literal["critical", "high", "medium", "low"]

_FEATURE_TO_MITRE_TACTIC: dict[str, str] = {
    "privileged_command_rate": "Privilege Escalation",
    "credential_stuffing_signals": "Credential Access",
    "beaconing_score": "Command and Control",
    "data_exfil_score": "Exfiltration",
    "impossible_travel_score": "Initial Access",
}


@dataclass(frozen=True)
class Alert:
    """Represent a security alert package ready for analyst-facing dispatch."""

    alert_id: UUID
    severity: Severity
    mitre_tactic: str
    recommended_actions: list[str]
    explanation: Explanation
    peer_context: PeerContext


def _severity_from_score(anomaly_score: float) -> Severity:
    """Map anomaly score to severity tiers so downstream routing remains deterministic."""

    if anomaly_score > 0.8:
        return "critical"
    if anomaly_score > 0.6:
        return "high"
    if anomaly_score > 0.4:
        return "medium"
    return "low"


def _mitre_tactic_from_explanation(explanation: Explanation) -> str:
    """Map the top contributing feature to a MITRE tactic for triage context."""

    if not explanation.top_features:
        return "Discovery"  # Conservative default keeps alert schema complete.
    top_feature = explanation.top_features[0].feature_name
    return _FEATURE_TO_MITRE_TACTIC.get(top_feature, "Discovery")


def _recommended_actions(severity: Severity) -> list[str]:
    """Provide severity-aligned analyst guidance to reduce triage latency."""

    if severity == "critical":
        return ["Isolate endpoint", "Disable account session", "Open incident bridge"]
    if severity == "high":
        return ["Review account activity", "Validate MFA events", "Escalate to Tier 2"]
    if severity == "medium":
        return ["Queue analyst review", "Correlate with recent alerts"]
    return ["Monitor trend", "Re-evaluate if repeated"]


def build_alert(
    anomaly_score: float,
    explanation: Explanation,
    peer_context: PeerContext,
    user_id: str,
) -> Alert:
    """Build immutable alert payloads with UUID IDs for distributed-safe uniqueness.

    UUIDs are used instead of sequential IDs because distributed security systems
    can generate alerts across many workers and regions. UUID generation avoids
    central counters, reduces collision risk, and prevents leaking alert volume
    patterns that sequential IDs can expose.
    """

    _ = user_id  # User ID is accepted for API parity but intentionally not embedded in alert payload.
    severity = _severity_from_score(anomaly_score)
    return Alert(
        alert_id=uuid4(),  # Random UUID avoids guessable, sequential identifiers.
        severity=severity,
        mitre_tactic=_mitre_tactic_from_explanation(explanation),
        recommended_actions=_recommended_actions(severity),
        explanation=explanation,
        peer_context=peer_context,
    )
