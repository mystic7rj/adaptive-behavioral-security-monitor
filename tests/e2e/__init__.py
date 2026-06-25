"""End-to-end synthetic scenario runners for validating security-detection behavior."""

from .test_compromised_creds import run_scenario as run_compromised_creds_scenario
from .test_insider_threat import run_scenario as run_insider_threat_scenario

__all__ = ["run_compromised_creds_scenario", "run_insider_threat_scenario"]
