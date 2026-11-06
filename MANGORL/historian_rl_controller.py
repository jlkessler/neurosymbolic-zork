"""Online TD(0) Historian critic with action-conditioned transition retrieval.

Only the small external linear value estimator changes; all language-model
and optional embedding-model parameters remain frozen.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .actor_experience import ActorExperienceMemory


def _clean_words(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    return [token.casefold() for token in re.findall(r"[A-Za-z0-9']+", text) if len(token) > 1]


class HistorianRLController:
    """Sparse linear TD value estimator and persistent transition memory."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: str | Path = "memory/historian_td.json",
        *,
        enabled: bool = False,
        gamma: float = 0.995,
        learning_rate: float = 0.025,
        l2: float = 0.0001,
        max_features_per_state: int = 128,
        state_encoder=None,
    ):
        self.path = Path(path).expanduser()
        self.state_encoder = state_encoder
        self.enabled = bool(enabled)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.l2 = max(0.0, float(l2))
        self.max_features_per_state = max(16, int(max_features_per_state))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1 inclusive")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.weights: dict[str, float] = {}
        self.state_visits = 0
        self.episodes_trained = 0
        self._episode_transitions: list[dict[str, Any]] = []
        self.td_memory: list[dict[str, Any]] = []
        if self.enabled:
            self.load()

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Critic memory must be an object.")
            if payload.get("schema_version") not in (1, self.SCHEMA_VERSION):
                raise ValueError("Unsupported critic memory schema.")
            weights = payload.get("weights", {})
            if not isinstance(weights, dict) or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in weights.values()
            ):
                raise ValueError("Invalid critic coefficients.")
            self.weights = {str(key): float(value) for key, value in weights.items()}
            self.state_visits = int(payload.get("state_visits", 0))
            self.episodes_trained = int(payload.get("episodes_trained", 0))
            self.td_memory = payload.get("td_memory", [])[-20000:]
            for row in self.td_memory:
                if (not isinstance(row, dict) or not isinstance(row.get("state"), dict)
                    or not isinstance(row.get("next_state"), dict) or not isinstance(row.get("action"), str)
                    or not isinstance(row.get("terminal"), bool)
                    or not isinstance(row.get("reward"), (int, float)) or not math.isfinite(row["reward"])):
                    raise ValueError("Invalid critic transition.")
            return True
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Cannot load critic memory {self.path}; original file preserved.") from exc

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "algorithm": "online_td0_with_action_conditioned_memory",
            "td_memory": self.td_memory,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "state_visits": self.state_visits,
            "episodes_trained": self.episodes_trained,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.path)
        return self.path

    @staticmethod
    def _tokens(state: dict[str, Any], key: str) -> list[str]:
        return _clean_words(state.get(key, []))

    def featurize(self, state: dict[str, Any]) -> dict[str, float]:
        """Build sparse, interpretable features with capped text vocabularies."""
        state = state if isinstance(state, dict) else {}
        features: dict[str, float] = {"bias": 1.0}

        def add(name: str, value: float = 1.0):
            if len(features) < self.max_features_per_state:
                features[name] = features.get(name, 0.0) + value

        location_id = str(state.get("location_id", "") or "").strip()
        location = str(state.get("location", "") or "").strip().casefold()
        if location_id:
            add(f"room_id:{location_id}")
        if location:
            add(f"room:{location}")

        goal_kind = str(state.get("goal_kind", "unknown") or "unknown").casefold()
        add(f"goal:{goal_kind}")
        route_status = str(state.get("home_route_status", "unknown") or "unknown").casefold()
        add(f"route:{route_status}")

        for key, prefix in (
            ("inventory_items", "inventory"),
            ("treasure_candidates", "treasure"),
            ("ordinary_items", "ordinary"),
            ("useful_items", "useful"),
            ("puzzle_clues", "clue"),
            ("observation", "obs"),
            ("goal_description", "goalword"),
        ):
            words = self._tokens(state, key)
            for token in words[: max(0, self.max_features_per_state - len(features))]:
                add(f"{prefix}:{token}")
        return features

    def value(self, state: dict[str, Any]) -> float:
        """Estimate V(s); returns zero while the critic is disabled."""
        if not self.enabled:
            return 0.0
        return sum(
            self.weights.get(feature, 0.0) * value
            for feature, value in self.featurize(state).items()
        )

    def estimate_advantage(
        self,
        state: dict[str, Any],
        reward: float,
        next_state: dict[str, Any],
        *,
        terminal: bool = False,
    ) -> float:
        """One-step TD residual, suitable as an optional Actor baseline."""
        if not self.enabled:
            return 0.0
        bootstrap = 0.0 if terminal else self.gamma * self.value(next_state)
        return float(reward) + bootstrap - self.value(state)

    def start_episode(self, episode_id: str | int | None = None) -> None:
        if self._episode_transitions:
            raise RuntimeError("Finish the previous critic episode before starting another.")
        self._episode_id = str(episode_id) if episode_id is not None else None

    def observe_transition(
        self,
        *,
        state: dict[str, Any],
        action: str,
        reward: float | int | None,
        next_state: dict[str, Any],
        terminal: bool = False,
    ) -> dict[str, Any]:
        """Update the external critic online; backbone LLM weights stay frozen."""
        if not self.enabled:
            return {"enabled": False}
        reward = float(reward or 0.0)
        if not math.isfinite(reward):
            raise ValueError("Critic reward must be finite.")
        target = reward + (0.0 if terminal else self.gamma * self.value(next_state))
        residual = target - self.value(state)
        transition = {
            "state": dict(state or {}),
            "action": str(action),
            "reward": reward,
            "next_state": dict(next_state or {}),
            "terminal": bool(terminal),
            "td_error": residual,
            "episode_id": getattr(self, "_episode_id", None),
        }
        self._episode_transitions.append(transition)
        self.td_memory.append(transition)
        self.td_memory = self.td_memory[-20000:]
        self._update_state(state, target)
        return {"enabled": True, "td_error": residual, "target": target, "terminal": bool(terminal)}

    def action_advantages(self, state, candidate_actions, *, top_k=40, min_similarity=0.65):
        """Retrieve action-conditioned TD targets using the current critic.

        These are small-data estimates under a changing policy, not an unbiased
        off-policy estimator. The state includes inventory, clues and progress.
        """
        result = {}
        if not self.enabled:
            return result
        matches = []
        rows_to_search = self.td_memory
        if self.state_encoder is not None:
            rows_to_search = sorted(rows_to_search, key=lambda row: ActorExperienceMemory.state_similarity(
                state, row["state"]), reverse=True)[:min(512, max(64, int(top_k) * 4))]
        for row in rows_to_search:
            similarity = ActorExperienceMemory.state_similarity(state, row["state"])
            if self.state_encoder is not None:
                similarity = 0.5 * similarity + 0.5 * self.state_encoder.similarity(
                    ActorExperienceMemory.normalize_state(state), ActorExperienceMemory.normalize_state(row["state"]))
            if similarity >= min_similarity:
                matches.append((similarity, row))
        matches.sort(key=lambda pair: pair[0], reverse=True)
        matches = matches[:top_k]
        for action in candidate_actions:
            rows = [(sim, row) for sim, row in matches if row["action"].casefold() == action.casefold()]
            if not rows:
                continue
            residuals = [(sim ** 2, self.estimate_advantage(
                row["state"], row["reward"], row["next_state"], terminal=row["terminal"]
            )) for sim, row in rows]
            value = sum(w * delta for w, delta in residuals) / sum(w for w, _ in residuals)
            result[action.casefold()] = {
                "action": action, "advantage": value * len(rows) / (len(rows) + 3),
                "raw_advantage": value, "sample_count": len(rows),
                "best_similarity": max(sim for sim, _ in rows), "source": "historian_td",
            }
        return result

    def _update_state(self, state: dict[str, Any], target: float) -> float:
        features = self.featurize(state)
        prediction = sum(self.weights.get(key, 0.0) * value for key, value in features.items())
        error = float(target) - prediction
        norm = sum(value * value for value in features.values()) or 1.0
        scale = self.learning_rate / norm
        for key, feature_value in features.items():
            old = self.weights.get(key, 0.0)
            self.weights[key] = old + scale * error * feature_value - self.learning_rate * self.l2 * old
        self.state_visits += 1
        return error

    def finish_episode(
        self,
        *,
        gamma: float | None = None,
        termination_reason: str = "unknown",
    ) -> dict[str, Any]:
        """Persist the online critic without training the same trajectory twice."""
        if not self.enabled:
            self._episode_transitions = []
            self._episode_id = None
            return {"enabled": False, "updates": 0, "reason": "critic_disabled"}
        discount = self.gamma if gamma is None else float(gamma)
        if not 0.0 <= discount <= 1.0:
            raise ValueError("gamma must be between 0 and 1 inclusive")
        errors = [transition["td_error"] for transition in self._episode_transitions]
        updates = len(errors)
        self.episodes_trained += 1 if updates else 0
        self.save()
        summary = {
            "enabled": True,
            "updates": updates,
            "mean_absolute_prediction_error": (
                sum(abs(item) for item in errors) / len(errors) if errors else 0.0
            ),
            "state_visits": self.state_visits,
            "episodes_trained": self.episodes_trained,
            "termination_reason": termination_reason,
            "path": str(self.path),
        }
        self._episode_transitions = []
        self._episode_id = None
        return summary
