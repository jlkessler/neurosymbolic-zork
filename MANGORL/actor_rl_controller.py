"""Experience-based Actor policy adaptation for Zork.

The first implementation is intentionally model-agnostic. For remote
OpenAI-compatible inference, it combines the Actor's ordered candidate list
with empirical action advantages retrieved from ActorExperienceMemory. This
is a *candidate-rank adjustment*, not access to the LLM's native token logits.

The controller also exposes ``adjust_logits`` for a future local inference
backend that can supply genuine candidate logits/log-probabilities.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


class ActorRLController:
    def __init__(
        self,
        experience_memory,
        *,
        enabled: bool = True,
        beta: float = 0.35,
        exploration_coefficient: float = 0.08,
        rank_penalty: float = 0.30,
        top_k: int = 40,
        min_similarity: float = 0.20,
        max_abs_advantage: float = 2.0,
        value_controller=None,
    ):
        self.experience_memory = experience_memory
        self.enabled = bool(enabled)
        self.beta = max(0.0, float(beta))
        self.exploration_coefficient = max(0.0, float(exploration_coefficient))
        self.rank_penalty = max(0.0, float(rank_penalty))
        self.top_k = max(1, int(top_k))
        self.min_similarity = max(0.0, min(1.0, float(min_similarity)))
        self.max_abs_advantage = max(0.1, float(max_abs_advantage))
        self.value_controller = value_controller

    @staticmethod
    def _deduplicate(actions: Iterable[str]) -> list[str]:
        result, seen = [], set()
        for action in actions:
            text = " ".join(str(action).split()).strip()
            key = text.casefold()
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result

    def estimate(self, state: dict[str, Any], candidate_actions: Iterable[str]) -> dict[str, Any]:
        estimates = self.experience_memory.estimate_action_values(
            state,
            candidate_actions,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )
        # Optional Historian critic baseline. The critic remains entirely
        # inactive unless explicitly enabled; when enabled, it estimates V(s)
        # while ActorExperienceMemory supplies empirical action-return Q(s,a).
        if self.value_controller is not None and getattr(self.value_controller, "enabled", False):
            critic_value = float(self.value_controller.value(state))
            estimates["critic_state_value"] = critic_value
            estimates["state_value"] = critic_value
            for item in estimates.get("actions", {}).values():
                q_value = item.get("q_value")
                if q_value is None:
                    item["raw_advantage"] = 0.0
                    item["advantage"] = 0.0
                    continue
                raw_advantage = float(q_value) - critic_value
                confidence = float(item.get("confidence", 0.0))
                item["raw_advantage"] = raw_advantage
                item["advantage"] = raw_advantage * confidence
        return estimates

    def build_guidance(
        self,
        state: dict[str, Any],
        valid_actions: Iterable[str],
        *,
        max_lines: int = 6,
    ) -> str:
        """Create a concise, evidence-qualified hint for the Actor prompt."""
        if not self.enabled:
            return "Actor reinforcement learning is disabled."
        actions = self._deduplicate(valid_actions)
        estimates = self.estimate(state, actions)
        known = [
            data for data in estimates.get("actions", {}).values()
            if data.get("sample_count", 0) > 0
        ]
        if not known:
            return (
                "No sufficiently similar Actor experiences are available yet. "
                "Treat every valid action as uncertain; do not infer that an "
                "untried action is bad. Explore using current observations."
            )
        known.sort(
            key=lambda item: (
                item.get("advantage", 0.0),
                item.get("sample_count", 0),
            ),
            reverse=True,
        )
        lines = [
            "EXPERIENCE-BASED ACTION EVIDENCE (soft guidance, not a rule):",
            "These estimates come from similar past states and discounted returns; "
            "sample counts are small-data evidence, not certainty.",
        ]
        for item in known[:max_lines]:
            q = item.get("q_value")
            q_text = "unknown" if q is None else f"{q:.3f}"
            lines.append(
                f"- {item['action']}: estimated return={q_text}, "
                f"advantage={item.get('advantage', 0.0):+.3f}, "
                f"samples={item.get('sample_count', 0)}, "
                f"similarity={item.get('best_similarity', 0.0):.2f}."
            )
        lines.append(
            "Do not repeat an action solely because it scored well elsewhere; "
            "check whether the current inventory, puzzle state, and observation match."
        )
        return "\n".join(lines)

    def _normalized_advantages(self, estimates: dict[str, Any]) -> dict[str, float]:
        entries = estimates.get("actions", {})
        known_values = [
            float(item.get("advantage", 0.0))
            for item in entries.values()
            if item.get("sample_count", 0) > 0
        ]
        scale = max(1.0, statistics.pstdev(known_values) if len(known_values) > 1 else 1.0)
        normalized = {}
        for key, item in entries.items():
            advantage = float(item.get("advantage", 0.0)) / scale
            normalized[key] = max(
                -self.max_abs_advantage,
                min(self.max_abs_advantage, advantage),
            )
        return normalized

    def rerank_candidates(
        self,
        state: dict[str, Any],
        candidate_actions: Iterable[str],
        *,
        base_action: str | None = None,
    ) -> dict[str, Any]:
        """Combine the Actor's preference order with empirical advantages.

        Candidate order acts as a weak base-policy prior. It is not presented
        as a native LLM logit. If no historical support exists, order is kept.
        """
        candidates = self._deduplicate(candidate_actions)
        if base_action:
            base_action = " ".join(str(base_action).split()).strip()
            if base_action and base_action.casefold() not in {x.casefold() for x in candidates}:
                candidates.insert(0, base_action)
            elif base_action:
                candidates = [base_action] + [
                    item for item in candidates if item.casefold() != base_action.casefold()
                ]
        if not candidates:
            return {
                "action": base_action,
                "candidates": [],
                "scores": [],
                "method": "disabled_or_no_candidates",
            }
        if not self.enabled:
            return {
                "action": candidates[0],
                "candidates": candidates,
                "scores": [],
                "method": "disabled",
            }

        estimates = self.estimate(state, candidates)
        normalized_advantages = self._normalized_advantages(estimates)
        rows = []
        for rank, action in enumerate(candidates):
            key = action.casefold()
            evidence = estimates.get("actions", {}).get(key, {})
            count = int(evidence.get("sample_count", 0))
            base_score = -self.rank_penalty * rank
            advantage_adjustment = self.beta * normalized_advantages.get(key, 0.0)
            exploration_bonus = self.exploration_coefficient / math.sqrt(count + 1.0)
            adjusted_score = base_score + advantage_adjustment + exploration_bonus
            rows.append({
                "action": action,
                "rank": rank,
                "base_rank_score": base_score,
                "estimated_advantage": float(evidence.get("advantage", 0.0)),
                "sample_count": count,
                "exploration_bonus": exploration_bonus,
                "adjusted_score": adjusted_score,
            })
        rows.sort(key=lambda row: (row["adjusted_score"], -row["rank"]), reverse=True)
        return {
            "action": rows[0]["action"],
            "candidates": [row["action"] for row in rows],
            "scores": rows,
            "state_value_baseline": estimates.get("state_value"),
            "retrieved_samples": estimates.get("samples", 0),
            "method": "candidate_rank_plus_experience_advantage",
        }

    def adjust_logits(
        self,
        state: dict[str, Any],
        candidate_actions: Iterable[str],
        base_logits: Iterable[float],
    ) -> dict[str, Any]:
        """Apply advantage corrections to genuine candidate logits.

        A local Transformers/vLLM adapter can call this method when it has
        actual logits. This method deliberately does not fabricate logits from
        remote API response text.
        """
        actions = self._deduplicate(candidate_actions)
        logits = [float(value) for value in base_logits]
        if len(actions) != len(logits):
            raise ValueError("candidate_actions and base_logits must have equal lengths")
        estimates = self.estimate(state, actions) if self.enabled else {"actions": {}}
        normalized = self._normalized_advantages(estimates) if self.enabled else {}
        adjusted = [
            logit + self.beta * normalized.get(action.casefold(), 0.0)
            for action, logit in zip(actions, logits)
        ]
        return {
            "actions": actions,
            "base_logits": logits,
            "adjusted_logits": adjusted,
            "advantages": normalized,
            "method": "native_candidate_logit_adjustment" if self.enabled else "disabled",
        }

    def select_action(
        self,
        state: dict[str, Any],
        candidate_actions: Iterable[str],
        *,
        base_action: str | None = None,
    ) -> dict[str, Any]:
        return self.rerank_candidates(state, candidate_actions, base_action=base_action)
