"""Persistent experience memory for the Zork Actor.

The memory stores Actor-selected decisions, but computes their Monte-Carlo
returns over *all* environment transitions in an episode. This lets an Actor
decision receive credit for a score increase that occurs later, including
when the symbolic navigator executes intermediate commands.

The JSONL file is append-only: one completed Actor decision per line. It is
separate from the WorldMap and the Historian's prose/strategic memory.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _clean_strings(values: Any, limit: int = 40) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value).split()).strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        result.append(item[:240])
        seen.add(key)
        if len(result) >= limit:
            break
    return result


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9']+", str(value or ""))
        if len(token) > 1
    }


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ActorExperienceMemory:
    """Collect, persist, and retrieve Actor action/return experiences."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path = "memory/actor_experiences.jsonl",
        *,
        max_records: int = 20_000,
        environment_reward_weight: float = 0.0,
        no_progress_penalty: float = 0.0,
    ):
        self.path = Path(path).expanduser()
        self.max_records = max(100, int(max_records))
        self.environment_reward_weight = float(environment_reward_weight)
        self.no_progress_penalty = float(no_progress_penalty)
        self.experiences: list[dict[str, Any]] = []
        self.episode_id: str | None = None
        self._episode_steps: list[dict[str, Any]] = []
        self.last_finish_summary: dict[str, Any] = {}
        self.last_episode_records: list[dict[str, Any]] = []
        self.load()

    def load(self) -> int:
        """Load valid JSONL records, ignoring malformed/truncated final lines."""
        self.experiences = []
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A partially written final line should not make prior
                    # experience inaccessible.
                    continue
                if (
                    isinstance(record, dict)
                    and record.get("record_type") == "actor_experience"
                    and isinstance(record.get("state"), dict)
                    and record.get("action")
                ):
                    self.experiences.append(record)
        self.experiences = self.experiences[-self.max_records :]
        return len(self.experiences)

    def start_episode(self, episode_id: str | int) -> None:
        """Begin collecting a new game trajectory."""
        if self._episode_steps:
            raise RuntimeError(
                "Cannot start a new Actor experience episode before finishing the previous one."
            )
        self.episode_id = str(episode_id)
        self._episode_steps = []

    @staticmethod
    def normalize_state(state: dict[str, Any] | None) -> dict[str, Any]:
        state = state if isinstance(state, dict) else {}
        return {
            "location_id": str(state.get("location_id", "") or ""),
            "location": str(state.get("location", "") or "")[:160],
            "observation": " ".join(str(state.get("observation", "") or "").split())[:700],
            "inventory_items": _clean_strings(state.get("inventory_items", []), 50),
            "treasure_candidates": _clean_strings(state.get("treasure_candidates", []), 30),
            "ordinary_items": _clean_strings(state.get("ordinary_items", []), 30),
            "useful_items": _clean_strings(state.get("useful_items", []), 30),
            "puzzle_clues": _clean_strings(state.get("puzzle_clues", []), 20),
            "goal_kind": str(state.get("goal_kind", "unknown") or "unknown"),
            "goal_description": " ".join(
                str(state.get("goal_description", "") or "").split()
            )[:300],
            "home_route_status": str(state.get("home_route_status", "unknown") or "unknown"),
            "score": _number(state.get("score"), default=None),
        }

    def record_step(
        self,
        *,
        state: dict[str, Any],
        action: str,
        next_state: dict[str, Any],
        score_delta: float | int | None = None,
        environment_reward: float | int | None = None,
        action_source: str = "unknown",
        actor_decision: bool = False,
        candidate_actions: list[str] | None = None,
        selection_info: dict[str, Any] | None = None,
        action_logprob: float | None = None,
        no_progress: bool = False,
    ) -> dict[str, Any]:
        """Append one environment transition to the active episode trajectory.

        All transitions are retained temporarily for return calculation. Only
        Actor decisions are persisted as policy experiences when the episode
        is finalized.
        """
        if self.episode_id is None:
            raise RuntimeError("Call start_episode() before record_step().")
        if not str(action).strip():
            raise ValueError("An experience transition requires a non-empty action.")

        record = {
            "step": len(self._episode_steps),
            "state": self.normalize_state(state),
            "action": " ".join(str(action).split())[:240],
            "next_state": self.normalize_state(next_state),
            "score_delta": _number(score_delta),
            "environment_reward": _number(environment_reward),
            "action_source": str(action_source or "unknown"),
            "actor_decision": bool(actor_decision),
            "candidate_actions": _clean_strings(candidate_actions or [], 12),
            "selection_info": selection_info if isinstance(selection_info, dict) else {},
            "action_logprob": (
                _number(action_logprob) if action_logprob is not None else None
            ),
            "no_progress": bool(no_progress),
        }
        self._episode_steps.append(record)
        return record

    def _step_reward(self, step: dict[str, Any]) -> float:
        reward = _number(step.get("score_delta"))
        reward += self.environment_reward_weight * _number(
            step.get("environment_reward")
        )
        if step.get("no_progress"):
            reward += self.no_progress_penalty
        return reward

    def finish_episode(
        self,
        *,
        gamma: float = 0.995,
        termination_reason: str = "unknown",
        final_score: float | int | None = None,
    ) -> dict[str, Any]:
        """Compute discounted returns and append Actor decisions to JSONL."""
        if self.episode_id is None:
            return {"episode_id": None, "steps": 0, "actor_experiences": 0}
        gamma = float(gamma)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1 inclusive.")

        running_return = 0.0
        completed = list(self._episode_steps)
        # Monte-Carlo returns include rewards on subsequent symbolic-navigation
        # steps; those steps are not themselves stored as Actor policy samples.
        for step in reversed(completed):
            running_return = self._step_reward(step) + gamma * running_return
            step["discounted_return"] = running_return
            step["training_reward"] = self._step_reward(step)

        actor_records: list[dict[str, Any]] = []
        for step in completed:
            if not step.get("actor_decision"):
                continue
            record = {
                "record_type": "actor_experience",
                "schema_version": self.SCHEMA_VERSION,
                "episode_id": self.episode_id,
                "step": step["step"],
                "state": step["state"],
                "action": step["action"],
                "next_state": step["next_state"],
                "candidate_actions": step["candidate_actions"],
                "action_source": step["action_source"],
                "score_delta": step["score_delta"],
                "environment_reward": step["environment_reward"],
                "training_reward": step["training_reward"],
                "discounted_return": step["discounted_return"],
                "selection_info": step["selection_info"],
                "action_logprob": step["action_logprob"],
            }
            actor_records.append(record)

        self.last_episode_records = actor_records
        if actor_records:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                for record in actor_records:
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                file.flush()

        self.experiences.extend(actor_records)
        self.experiences = self.experiences[-self.max_records :]
        summary = {
            "episode_id": self.episode_id,
            "steps": len(completed),
            "actor_experiences": len(actor_records),
            "score_reward_total": sum(_number(x.get("score_delta")) for x in completed),
            "discounted_return_first_step": (
                completed[0].get("discounted_return") if completed else None
            ),
            "termination_reason": termination_reason,
            "final_score": _number(final_score, default=None),
            "memory_path": str(self.path),
        }
        self.last_finish_summary = summary
        self.episode_id = None
        self._episode_steps = []
        return summary

    @staticmethod
    def state_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        """A transparent weighted similarity for retrieval, not an embedding."""
        a = ActorExperienceMemory.normalize_state(left)
        b = ActorExperienceMemory.normalize_state(right)
        weights = {
            "location": 0.35,
            "goal": 0.15,
            "inventory": 0.10,
            "clues": 0.10,
            "observation": 0.15,
            "route": 0.10,
            "score": 0.05,
        }
        values: dict[str, float] = {}
        if a["location_id"] and b["location_id"]:
            values["location"] = float(a["location_id"] == b["location_id"])
        else:
            values["location"] = float(
                bool(a["location"] and b["location"])
                and a["location"].casefold() == b["location"].casefold()
            )
        values["goal"] = float(a["goal_kind"] == b["goal_kind"])
        values["inventory"] = _jaccard(
            (x.casefold() for x in a["inventory_items"]),
            (x.casefold() for x in b["inventory_items"]),
        )
        values["clues"] = _jaccard(
            (x.casefold() for x in a["puzzle_clues"]),
            (x.casefold() for x in b["puzzle_clues"]),
        )
        values["observation"] = _jaccard(
            _tokens(a["observation"]), _tokens(b["observation"])
        )
        values["route"] = float(a["home_route_status"] == b["home_route_status"])
        if a["score"] is None or b["score"] is None:
            values["score"] = 0.5
        else:
            values["score"] = max(0.0, 1.0 - abs(a["score"] - b["score"]) / 100.0)

        # If a feature is missing from both states (e.g. no inventory reported),
        # its neutral similarity should not dominate the location/goal match.
        active_weights = dict(weights)
        if not a["inventory_items"] and not b["inventory_items"]:
            active_weights.pop("inventory")
        if not a["puzzle_clues"] and not b["puzzle_clues"]:
            active_weights.pop("clues")
        if not a["observation"] and not b["observation"]:
            active_weights.pop("observation")
        if a["score"] is None or b["score"] is None:
            active_weights.pop("score")
        total = sum(active_weights.values()) or 1.0
        return sum(values[key] * weight for key, weight in active_weights.items()) / total

    def retrieve_similar(
        self,
        state: dict[str, Any],
        *,
        top_k: int = 12,
        min_similarity: float = 0.20,
        exclude_episode_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most similar historical Actor decisions."""
        matches = []
        for record in self.experiences:
            if exclude_episode_id and record.get("episode_id") == exclude_episode_id:
                continue
            similarity = self.state_similarity(state, record.get("state", {}))
            if similarity < min_similarity:
                continue
            item = dict(record)
            item["similarity"] = similarity
            matches.append(item)
        matches.sort(key=lambda item: (item["similarity"], item.get("step", -1)), reverse=True)
        return matches[: max(1, int(top_k))]

    def estimate_action_values(
        self,
        state: dict[str, Any],
        candidate_actions: Iterable[str],
        *,
        top_k: int = 40,
        min_similarity: float = 0.20,
        exclude_episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Estimate candidate Q values from similar state/action returns.

        Returns a dictionary with a state baseline and one estimate per
        candidate. Estimates are empirical Monte-Carlo means, not calibrated
        neural Q-values; ``sample_count`` makes their support explicit.
        """
        similar = self.retrieve_similar(
            state,
            top_k=top_k,
            min_similarity=min_similarity,
            exclude_episode_id=exclude_episode_id,
        )
        if not similar:
            return {"state_value": None, "samples": 0, "actions": {}}

        baseline_weights = [max(1e-6, item["similarity"] ** 2) for item in similar]
        denominator = sum(baseline_weights)
        baseline = sum(
            weight * _number(item.get("discounted_return"))
            for weight, item in zip(baseline_weights, similar)
        ) / denominator

        candidate_list = list(candidate_actions)
        estimates: dict[str, Any] = {}
        for candidate in candidate_list:
            action_key = " ".join(str(candidate).split()).casefold()
            if not action_key or action_key in estimates:
                continue
            rows = [
                item for item in similar
                if " ".join(str(item.get("action", "")).split()).casefold() == action_key
            ]
            weights = [max(1e-6, item["similarity"] ** 2) for item in rows]
            weight_sum = sum(weights)
            count = len(rows)
            if count:
                q_value = sum(
                    weight * _number(item.get("discounted_return"))
                    for weight, item in zip(weights, rows)
                ) / weight_sum
                raw_advantage = q_value - baseline
                confidence = count / (count + 3.0)
            else:
                q_value = None
                raw_advantage = 0.0
                confidence = 0.0
            estimates[action_key] = {
                "action": str(candidate),
                "q_value": q_value,
                "state_value_baseline": baseline,
                "advantage": raw_advantage * confidence,
                "raw_advantage": raw_advantage,
                "sample_count": count,
                "confidence": confidence,
                "best_similarity": max((row["similarity"] for row in rows), default=0.0),
            }
        return {"state_value": baseline, "samples": len(similar), "actions": estimates}

    def __len__(self) -> int:
        return len(self.experiences)
