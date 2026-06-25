from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from config.settings import load_settings

logger = logging.getLogger(__name__)

State = tuple[str, int]  # (user_role, hour_bucket) context tuple used by the bandit policy.


class ContextualBanditPolicy:
    """Choose alert thresholds per context with epsilon-greedy exploration.

    Epsilon-greedy means the policy mostly exploits its best-known action
    (highest Q-value) but occasionally explores a random action with probability
    epsilon. Using epsilon=0.05 balances stability and adaptation in SOC systems:
    95% of decisions follow learned behavior to reduce alert volatility, while 5%
    of decisions keep probing alternatives to handle drift in attacker behavior.
    """

    ACTIONS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    def __init__(self, epsilon: float = 0.05, alpha: float = 0.2) -> None:
        """Initialize policy parameters and load persisted Q-values if available."""

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._epsilon = epsilon
        self._alpha = alpha
        self._q_table_path = Path(load_settings().RL_Q_TABLE_PATH)
        self._q_table: dict[str, dict[str, float]] = {}
        self._load_q_table()

    @staticmethod
    def _state_key(state: State) -> str:
        """Encode state as a compact key so it can be used safely in JSON maps."""

        user_role, hour_bucket = state
        return f"{user_role}|{hour_bucket}"

    def _load_q_table(self) -> None:
        """Load Q-values from disk so learning survives process restarts."""

        if not self._q_table_path.exists():
            self._q_table = {}  # Empty table is a valid cold-start state.
            return
        with self._q_table_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise TypeError("Q-table file must contain a JSON object")
        parsed: dict[str, dict[str, float]] = {}
        for state_key, action_map in data.items():
            if not isinstance(state_key, str) or not isinstance(action_map, dict):
                raise TypeError("Q-table structure is invalid")
            parsed[state_key] = {}
            for action_key, value in action_map.items():
                parsed[state_key][action_key] = float(value)
        self._q_table = parsed

    def _save_q_table(self) -> None:
        """Persist Q-values after each update so runtime learning is durable."""

        self._q_table_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure destination exists.
        with self._q_table_path.open("w", encoding="utf-8") as handle:
            json.dump(self._q_table, handle, separators=(",", ":"))

    def _ensure_state(self, state_key: str) -> dict[str, float]:
        """Create default action values for unseen states to support safe exploration."""

        if state_key not in self._q_table:
            self._q_table[state_key] = {
                str(action): 0.0 for action in self.ACTIONS
            }  # Optimistic-neutral init keeps all actions initially comparable.
        return self._q_table[state_key]

    def select_threshold(self, state: State) -> float:
        """Select an action via epsilon-greedy to balance exploitation and exploration."""

        state_key = self._state_key(state)
        action_values = self._ensure_state(state_key)
        if random.random() < self._epsilon:
            selected = random.choice(self.ACTIONS)  # Random probe prevents local optima lock-in.
            logger.info("policy_action_selected mode=explore epsilon=%.3f", self._epsilon)
            return selected
        best_action = max(
            self.ACTIONS,
            key=lambda action: action_values[str(action)],
        )  # Deterministic tie-breaking follows ACTIONS order.
        logger.info("policy_action_selected mode=exploit epsilon=%.3f", self._epsilon)
        return best_action

    def update(self, state: State, action: float, reward: float) -> None:
        """Apply incremental Q-learning update for the observed context-action reward."""

        if action not in self.ACTIONS:
            raise ValueError("action must be one of the configured threshold actions")
        state_key = self._state_key(state)
        action_values = self._ensure_state(state_key)
        action_key = str(action)
        old_value = action_values[action_key]
        action_values[action_key] = old_value + self._alpha * (
            reward - old_value
        )  # One-step bandit update moves value toward observed reward.
        self._save_q_table()
        logger.info(
            "policy_q_updated role_length=%d hour_bucket=%d action=%.1f",
            len(state[0]),  # Log only role length to avoid leaking raw role labels.
            state[1],
            action,
        )
