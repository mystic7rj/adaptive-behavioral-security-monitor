from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

if TYPE_CHECKING:
    from models.autoencoder import BehaviorAutoencoder

_FEATURE_NAMES: tuple[str, ...] = (
    "login_hour",
    "session_duration_s",
    "bytes_out",
    "unique_ip_count",
    "failure_ratio",
    "geo_velocity",
    "device_switch_rate",
    "new_ip_ratio",
    "offhours_access",
    "sensitive_data_access",
    "privilege_change_events",
    "download_volume",
    "upload_volume",
    "api_call_rate",
    "process_spawn_rate",
    "command_entropy",
    "dns_query_rate",
    "http_error_rate",
    "vpn_toggle_rate",
    "mfa_bypass_signals",
    "token_refresh_rate",
    "session_count",
    "remote_admin_tool_use",
    "impossible_travel_score",
    "endpoint_alert_rate",
    "malware_indicator_rate",
    "phishing_indicator_rate",
    "credential_stuffing_signals",
    "bruteforce_signals",
    "lateral_movement_signals",
    "beaconing_score",
    "data_exfil_score",
    "privileged_command_rate",
    "new_device_count",
    "resource_access_spike",
    "service_account_usage",
    "role_change_frequency",
    "access_denied_ratio",
    "suspicious_attachment_rate",
    "macro_execution_signals",
    "script_execution_rate",
    "registry_modification_rate",
    "kernel_event_rate",
    "memory_injection_signals",
    "persistence_indicator_rate",
    "cloud_admin_action_rate",
    "iam_policy_change_rate",
    "storage_bucket_access_spike",
    "db_query_anomaly_score",
    "container_escape_signals",
    "k8s_privilege_escalation_signals",
    "secrets_access_rate",
    "external_share_rate",
    "unusual_app_install_rate",
    "new_process_lineage_rate",
    "log_tamper_signals",
    "threat_intel_match_rate",
    "edr_suppression_signals",
    "unusual_timezone_access",
    "credential_reset_rate",
    "new_geo_location_rate",
    "admin_panel_access_rate",
    "sso_failure_ratio",
    "risky_oauth_grant_rate",
)


@dataclass(frozen=True)
class FeatureContribution:
    """Represent one high-impact feature contribution in privacy-preserving terms."""

    feature_name: str
    contribution: float
    description: str


@dataclass(frozen=True)
class Explanation:
    """Represent the top explainability factors that justify an anomaly decision."""

    top_features: list[FeatureContribution]


def _feature_name(index: int) -> str:
    """Map feature indices to stable names so analyst output is interpretable."""

    if 0 <= index < len(_FEATURE_NAMES):
        return _FEATURE_NAMES[index]
    return f"feature_{index}"  # Fallback keeps output stable for future feature expansion.


def _description_for(feature_name: str, contribution: float) -> str:
    """Create human-readable deviation summaries without exposing raw user values."""

    direction = "increased" if contribution >= 0.0 else "decreased"
    sigma = abs(contribution)
    return (
        f"{feature_name.replace('_', ' ').title()} deviated {direction} "
        f"{sigma:.1f}sigma from baseline"
    )  # Sigma-like phrasing conveys distance from baseline while protecting raw values.


def explain(feature_vec: list[float], model: BehaviorAutoencoder) -> Explanation:
    """Explain model output with SHAP to provide local, additive, analyst-auditable reasons.

    SHAP is used over simpler importance heuristics because it provides local feature
    attributions for each individual prediction with additive consistency, allowing
    SOC analysts to understand why *this* event was flagged rather than only global
    trends. This improves triage quality and post-incident auditability.
    """

    if len(feature_vec) == 0:
        raise ValueError("feature_vec must not be empty")
    if len(feature_vec) != 64:
        raise ValueError("feature_vec must contain exactly 64 features for the current model")

    shap = import_module("shap")  # SHAP is imported lazily to keep module import lightweight.
    tensor = torch.tensor(feature_vec, dtype=torch.float32).reshape(1, -1)

    def _wrapped_fn(arr: list[list[float]] | torch.Tensor) -> torch.Tensor:
        """Convert SHAP input into anomaly scores for contribution estimation."""

        if isinstance(arr, torch.Tensor):
            x = arr.float()
        else:
            x = torch.tensor(arr, dtype=torch.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)  # SHAP may provide a flat vector for single-sample explanation.
        baseline = torch.zeros((1, x.shape[1]), dtype=torch.float32)
        reconstructed, _ = model(x)
        # Return elementwise reconstruction residual to let SHAP attribute per-feature impact.
        result = (x - reconstructed) - baseline
        return cast(Tensor, result)

    explainer = shap.Explainer(_wrapped_fn, tensor)
    shap_result = explainer(tensor)
    values = shap_result.values
    if values is None:
        raise ValueError("SHAP did not return contribution values")

    if hasattr(values, "tolist"):
        values_list = values.tolist()
    else:
        values_list = values
    row = values_list[0]
    if not isinstance(row, list):
        raise TypeError("SHAP result row has unexpected shape")

    indexed = [(idx, float(val)) for idx, val in enumerate(row)]
    top = sorted(indexed, key=lambda item: abs(item[1]), reverse=True)[:3]
    contributions = [
        FeatureContribution(
            feature_name=_feature_name(idx),
            contribution=val,
            description=_description_for(_feature_name(idx), val),
        )
        for idx, val in top
    ]
    return Explanation(top_features=contributions)
