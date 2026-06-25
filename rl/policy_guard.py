from __future__ import annotations

import logging

from .threshold_policy import ContextualBanditPolicy, State

logger = logging.getLogger(__name__)


class PolicyViolationError(Exception):
    """Signal that a policy proposed an unsafe threshold outside permitted bounds."""


class GuardedPolicy:
    """Wrap policy actions with hard safety limits to prevent reward-hacking behavior.

    Hard limits exist because learned policies can drift toward pathological values
    if reward signals are sparse, delayed, or noisy. This includes reward hacking:
    the policy finds shortcuts that optimize numeric reward while violating the
    intended security objective (for example, using extremes that suppress useful alerts).
    """

    MIN_THRESHOLD: float = 0.3
    MAX_THRESHOLD: float = 0.95

    def __init__(self, policy: ContextualBanditPolicy) -> None:
        """Store the wrapped policy used to generate raw threshold suggestions."""

        self._policy = policy

    def get_threshold(self, state: State) -> float:
        """Return a safe threshold or raise if the underlying policy breaches limits."""

        raw_action = self._policy.select_threshold(state)
        # Reject outright when the policy proposes values outside the hard safety envelope.
        if raw_action < self.MIN_THRESHOLD or raw_action > self.MAX_THRESHOLD:
            logger.error(
                "policy_action_rejected reason=out_of_range min=%.2f max=%.2f action=%.3f",
                self.MIN_THRESHOLD,
                self.MAX_THRESHOLD,
                raw_action,
            )
            raise PolicyViolationError("policy produced out-of-range threshold")
        # Log a clip event path for future-proofing if clipping behavior is introduced later.
        if raw_action != min(max(raw_action, self.MIN_THRESHOLD), self.MAX_THRESHOLD):
            logger.warning(
                "policy_action_clipped min=%.2f max=%.2f action=%.3f",
                self.MIN_THRESHOLD,
                self.MAX_THRESHOLD,
                raw_action,
            )
        return raw_action
