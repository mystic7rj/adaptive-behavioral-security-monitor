from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from config.settings import load_settings


@dataclass(frozen=True)
class PeerFeatureRank:
    """Represent one feature's percentile rank against same-role peer aggregates."""

    feature_name: str
    percentile_rank: float


@dataclass(frozen=True)
class PeerContext:
    """Capture percentile-based context so analysts can compare behavior against role norms."""

    user_id: str
    role: str
    feature_percentiles: list[PeerFeatureRank]


def _load_peer_stats(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load aggregated peer statistics from disk to avoid querying individual peer records."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("peer stats must be a JSON object")
    return data


def _to_percentile(value: float, mean: float, std: float) -> float:
    """Approximate percentile from z-score so outputs are normalized and easy to compare."""

    if std <= 0.0:
        return 50.0  # Degenerate distributions default to median percentile.
    z_score = (value - mean) / std
    # Logistic mapping gives a smooth percentile-like score without exposing raw peer samples.
    percentile = 100.0 / (1.0 + pow(2.718281828, -1.702 * z_score))
    result = max(0.0, min(100.0, percentile))
    return cast(float, result)


def compare_to_peers(user_id: str, feature_vec: dict[str, float], role: str) -> PeerContext:
    """Compare user features to same-role aggregate distributions via percentile rank.

    Percentile rank is used because it is intuitive for analysts ("top 5%" vs raw
    z-scores) and robust across mixed feature scales. Only aggregated role-level
    stats are used to preserve peer privacy and avoid exposing individual records.
    """

    stats_path = Path(load_settings().PEER_STATS_PATH)
    all_stats = _load_peer_stats(stats_path)
    role_stats = all_stats.get(role, {})
    ranks: list[PeerFeatureRank] = []
    for feature_name, value in feature_vec.items():
        feature_stats = role_stats.get(feature_name, {})
        mean = float(feature_stats.get("mean", value))
        std = float(feature_stats.get("std", 0.0))
        ranks.append(
            PeerFeatureRank(
                feature_name=feature_name,
                percentile_rank=_to_percentile(value, mean, std),
            )
        )
    return PeerContext(
        user_id=user_id,
        role=role,
        feature_percentiles=ranks,
    )
