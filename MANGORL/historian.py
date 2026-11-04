"""Persistent Historian for score-seeking Zork episodes.

The Historian receives only observations, Jericho score deltas, the Actor's
reported inventory/valuables/clues, and the symbolic map. It contains no
walkthrough or guide-derived point table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class Historian:
    def __init__(
        self,
        memory_path="memory/historian.json",
        target_score=76,
        repeat_threshold=2,
        lessons_per_category=3,
    ):
        self.memory_path = Path(memory_path)
        self.target_score = int(target_score)
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.lessons_per_category = int(lessons_per_category)
        if not 1 <= self.lessons_per_category <= 10:
            raise ValueError("lessons_per_category must be between 1 and 10")
        self.memory = self._empty_memory()
        self.episode = None
        self._recent_attempts = []
        self._completed_score_tasks = set()
        self._attempted_score_tasks = set()
        self.agent_state = self._empty_agent_state()
        # These lessons intentionally live only for one multi-episode trial.
        # The persistent Historian memory remains useful across runs, but a
        # trial's review file should not silently inherit coaching from an
        # unrelated experiment.
        self.trial_lessons = {"positives": [], "negatives": []}
        self.load()

    @staticmethod
    def _empty_agent_state():
        return {
            "inventory_items": [],
            "carried_treasures": [],
            "puzzle_clues": [],
            "state_note": "",
        }

    @staticmethod
    def _empty_memory():
        return {
            "schema_version": 2,
            "episodes": [],
            "score_profiles": {},
            "score_events": [],
            "blocked_actions": {},
            "repeat_failures": {},
            "puzzle_clues": [],
            "lessons": [],
            "agent_reports": [],
        }

    @staticmethod
    def normalize_action(action):
        action = str(action).strip().lower()
        if action.startswith("go "):
            action = action[3:].strip()
        aliases = {
            "n": "north", "s": "south", "e": "east", "w": "west",
            "ne": "northeast", "nw": "northwest",
            "se": "southeast", "sw": "southwest",
            "u": "up", "d": "down",
        }
        return aliases.get(action, action)

    @staticmethod
    def _is_direction(action):
        return Historian.normalize_action(action) in {
            "north", "south", "east", "west", "northeast", "northwest",
            "southeast", "southwest", "up", "down",
        }

    @staticmethod
    def _as_number(value):
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_strings(values, limit=30):
        if not isinstance(values, (list, tuple, set)):
            return []
        result = []
        seen = set()
        for value in values:
            item = " ".join(str(value).split()).strip()
            if not item:
                continue
            key = item.casefold()
            if key not in seen:
                result.append(item[:180])
                seen.add(key)
            if len(result) >= limit:
                break
        return result

    def load(self):
        if not self.memory_path.exists():
            self.memory = self._empty_memory()
            return
        with self.memory_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        defaults = self._empty_memory()
        defaults.update(loaded)
        self.memory = defaults

    def save(self):
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self.memory, file, indent=2)
        temporary.replace(self.memory_path)

    def start_episode(self, start_node, start_location, initial_score=None):
        self.episode = {
            "start_node": str(start_node),
            "start_location": start_location,
            "start_score": self._as_number(initial_score),
            "end_score": None,
            "score_delta": None,
            "steps": 0,
            "visited_nodes": [str(start_node)],
            "new_locations": [],
            "score_events": [],
            "repeated_actions": [],
            "deviations": [],
            "goal_history": [],
            "action_log": [],
        }
        self._recent_attempts = []
        self._completed_score_tasks = set()
        self._attempted_score_tasks = set()
        # Inventory resets with a fresh Jericho episode. Cross-episode lessons
        # and puzzle clues remain in persistent memory.
        self.agent_state = self._empty_agent_state()

    def record_agent_report(self, report):
        """Store Actor's compact state report; no external guide is consulted."""
        if not isinstance(report, dict):
            return
        self.agent_state["inventory_items"] = self._clean_strings(
            report.get("inventory_items", []), limit=50
        )
        self.agent_state["carried_treasures"] = self._clean_strings(
            report.get("carried_treasures", []), limit=30
        )
        self.agent_state["puzzle_clues"] = self._clean_strings(
            report.get("puzzle_clues", []), limit=12
        )
        self.agent_state["state_note"] = " ".join(
            str(report.get("state_note", "")).split()
        )[:500]

        for clue in self.agent_state["puzzle_clues"]:
            record = {
                "clue": clue,
                "location": str(report.get("location", ""))[:120],
                "episode": len(self.memory["episodes"]) + 1,
            }
            if not any(
                item.get("clue", "").casefold() == clue.casefold()
                for item in self.memory["puzzle_clues"]
            ):
                self.memory["puzzle_clues"].append(record)
        self.memory["puzzle_clues"] = self.memory["puzzle_clues"][-40:]

        self.memory["agent_reports"].append({
            "location": str(report.get("location", ""))[:120],
            "inventory_items": self.agent_state["inventory_items"],
            "carried_treasures": self.agent_state["carried_treasures"],
            "state_note": self.agent_state["state_note"],
            "episode": len(self.memory["episodes"]) + 1,
        })
        self.memory["agent_reports"] = self.memory["agent_reports"][-50:]

    def observe_step(
        self,
        *,
        before_node,
        before_location,
        action,
        after_node,
        after_location,
        score_before=None,
        score_after=None,
        reward=None,
        observation_before=None,
        observation=None,
        navigation_result=None,
        goal=None,
        action_source=None,
        actor_report=None,
    ):
        if self.episode is None:
            raise RuntimeError("Call start_episode() before observe_step().")

        before_key, after_key = str(before_node), str(after_node)
        normalized = self.normalize_action(action)
        before_score, after_score = self._as_number(score_before), self._as_number(score_after)
        score_delta = (
            after_score - before_score
            if before_score is not None and after_score is not None
            else None
        )
        moved = before_key != after_key
        before_text = " ".join(str(observation_before or "").split())[:300]
        after_text = " ".join(str(observation or "").split())[:300]
        state_changed = bool(before_text and after_text and before_text != after_text)

        profile_key = json.dumps([before_key, normalized], separators=(",", ":"))
        previous_attempt = self.memory["repeat_failures"].get(profile_key, {})
        same_result = bool(
            after_text and previous_attempt.get("last_observation") == after_text
        )
        no_progress = (
            not moved
            and (score_delta is None or score_delta <= 0)
            and not state_changed
            and same_result
        )

        existing = self.memory["score_profiles"].get(profile_key)
        if existing is not None:
            existing["attempts"] = existing.get(
                "attempts", existing.get("samples", 1)
            ) + 1
            if score_delta is not None:
                existing["total_gain"] += score_delta
            if score_delta is not None and score_delta > 0:
                existing["samples"] = existing.get("samples", 0) + 1
            else:
                self._attempted_score_tasks.add(profile_key)

        failure = self.memory["repeat_failures"].setdefault(
            profile_key,
            {
                "node": before_key,
                "location": before_location,
                "action": normalized,
                "no_progress_count": 0,
                "last_episode": len(self.memory["episodes"]) + 1,
            },
        )
        if no_progress:
            failure["no_progress_count"] += 1
            failure["last_episode"] = len(self.memory["episodes"]) + 1
        elif moved or state_changed or (score_delta is not None and score_delta > 0):
            failure["no_progress_count"] = max(
                0, failure["no_progress_count"] - 1
            )
        if after_text:
            failure["last_observation"] = after_text

        if self._is_direction(action) and not moved:
            blocked = self.memory["blocked_actions"].setdefault(before_key, {})
            blocked[normalized] = blocked.get(normalized, 0) + 1

        self._recent_attempts.append({
            "node": before_key, "action": normalized, "no_progress": no_progress
        })
        self._recent_attempts = self._recent_attempts[-12:]
        repeated = [
            item for item in self._recent_attempts
            if item["node"] == before_key
            and item["action"] == normalized
            and item["no_progress"]
        ]
        if no_progress and len(repeated) >= self.repeat_threshold:
            self.episode["repeated_actions"].append({
                "location": before_location,
                "node": before_key,
                "action": normalized,
                "count_in_recent_window": len(repeated),
            })

        if score_delta is not None and score_delta > 0:
            event = {
                "node": before_key,
                "location": before_location,
                "action": normalized,
                "score_delta": score_delta,
                "observation": str(observation or "")[:500],
                "episode": len(self.memory["episodes"]) + 1,
            }
            self.episode["score_events"].append(event)
            self.memory["score_events"].append(event)
            self.memory["score_events"] = self.memory["score_events"][-250:]

            if existing is None:
                self.memory["score_profiles"][profile_key] = {
                    "node": before_key,
                    "location": before_location,
                    "action": normalized,
                    "total_gain": score_delta,
                    "attempts": 1,
                    "samples": 1,
                    "best_gain": score_delta,
                }
            else:
                existing["best_gain"] = max(
                    existing.get("best_gain", 0.0), score_delta
                )
            self._completed_score_tasks.add(profile_key)
            self._attempted_score_tasks.add(profile_key)

        if moved and after_key not in self.episode["visited_nodes"]:
            self.episode["visited_nodes"].append(after_key)
            self.episode["new_locations"].append(after_location)

        if navigation_result and navigation_result.get("deviation"):
            self.episode["deviations"].append({
                "before_location": before_location,
                "action": str(action),
                "after_location": after_location,
                "navigation_status": navigation_result.get("plan_status"),
            })

        record = {
            "before_node": before_key,
            "before_location": before_location,
            "action": str(action),
            "after_node": after_key,
            "after_location": after_location,
            "moved": moved,
            "score_delta": score_delta,
            "reward": self._as_number(reward),
            "state_changed": state_changed,
            "observation_before": before_text,
            "observation": after_text,
            "navigation_status": navigation_result.get("plan_status")
            if navigation_result else None,
            "action_source": str(action_source or "unknown"),
            "goal_kind": str((goal or {}).get("kind", "unknown")),
            "goal_description": str((goal or {}).get("description", ""))[:300],
            "actor_state_note": str((actor_report or {}).get("state_note", ""))[:300],
        }
        self.episode["steps"] += 1
        # Keep the full episode trajectory for the post-episode review. The
        # console/chat logs remain the authoritative detailed transcripts.
        self.episode.setdefault("action_log", []).append(record)
        self.episode.setdefault("recent_steps", []).append(record)
        self.episode["recent_steps"] = self.episode["recent_steps"][-40:]
        return record

    def finish_episode(self, final_score=None, moves=None, termination_reason="step_limit"):
        if self.episode is None:
            return None
        end_score = self._as_number(final_score)
        start_score = self.episode.get("start_score")
        self.episode["end_score"] = end_score
        if start_score is not None and end_score is not None:
            self.episode["score_delta"] = end_score - start_score
        self.episode["moves"] = moves
        self.episode["termination_reason"] = termination_reason
        self.episode["target_score"] = self.target_score
        self.episode["target_achieved"] = (
            end_score is not None and end_score >= self.target_score
        )
        self.memory["episodes"].append(self.episode)
        self.memory["episodes"] = self.memory["episodes"][-50:]
        finished = self.episode
        self.episode = None
        self.save()
        return finished

    @staticmethod
    def _parse_review_json(answer):
        answer = (answer or "").strip()
        answer = re.sub(r"^```(?:json)?\s*", "", answer, flags=re.I)
        answer = re.sub(r"\s*```$", "", answer)
        try:
            value = json.loads(answer)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", answer, flags=re.S)
            if match:
                try:
                    value = json.loads(match.group(0))
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _fit_lesson_count(values, fallback, count):
        """Return exactly ``count`` distinct, non-empty lessons.

        The deterministic fallback pool is deliberately longer than the old
        three-item list so API failures still produce the configured number of
        useful prompt lessons.
        """
        count = int(count)
        cleaned = Historian._clean_strings(values, limit=max(count, 20))
        result = cleaned[:count]
        seen = {entry.casefold() for entry in result}
        for item in fallback:
            if len(result) >= count:
                break
            normalized = " ".join(str(item).split()).strip()
            if normalized and normalized.casefold() not in seen:
                result.append(normalized[:180])
                seen.add(normalized.casefold())
        return result[:count]

    @staticmethod
    def _exactly_three(values, fallback):
        """Backward-compatible helper retained for existing callers/tests."""
        return Historian._fit_lesson_count(values, fallback, 3)

    def _fallback_episode_review(self, episode):
        """Produce evidence-based coaching if the review model is unavailable."""
        score_delta = episode.get("score_delta")
        start_score = episode.get("start_score")
        end_score = episode.get("end_score")
        steps = episode.get("action_log", episode.get("recent_steps", []))
        score_events = episode.get("score_events", [])
        repeated = episode.get("repeated_actions", [])
        new_locations = episode.get("new_locations", [])
        reason = episode.get("termination_reason", "unknown")

        score_text = (
            f"The score changed from {start_score:g} to {end_score:g} "
            f"({score_delta:+g} points)"
            if isinstance(start_score, (int, float))
            and isinstance(end_score, (int, float))
            and isinstance(score_delta, (int, float))
            else "A complete score comparison was unavailable"
        )
        score_actions = [
            f"'{event.get('action')}' at {event.get('location')} (+{event.get('score_delta')} points)"
            for event in score_events[:3]
        ]
        first = (
            f"Episode review: {score_text} across {episode.get('steps', len(steps))} "
            f"recorded commands, ending because of '{reason}'. "
        )
        if score_actions:
            first += "Observed score-positive behavior included " + "; ".join(score_actions) + "."
        else:
            first += "No positive score event was recorded in the episode data."

        second = (
            f"The run visited {len(episode.get('visited_nodes', []))} distinct mapped "
            f"locations and recorded {len(new_locations)} newly encountered locations, "
            f"with {len(repeated)} repeated-action warning(s). "
        )
        if repeated:
            examples = [
                f"'{item.get('action')}' at {item.get('location')}"
                for item in repeated[:3]
            ]
            second += "The clearest behavior to reconsider was " + ", ".join(examples) + "."
        else:
            second += "The stored episode summary did not flag repeated no-progress actions."

        positives = []
        if score_actions:
            positives.append("Prioritize interactions that produced an observed score increase; revisit them only when the context supports it.")
        if new_locations:
            positives.append("Continue exploring newly discovered locations and recording their transitions in the persistent map.")
        if episode.get("score_events"):
            positives.append("Use Jericho score deltas as feedback rather than relying only on whether an action sounds promising.")
        if episode.get("deviations"):
            positives.append("Let the symbolic navigator validate each movement transition before continuing along a mapped route.")
        positive_fallback = [
            "Use score changes and environment responses as concrete feedback for choosing subsequent actions.",
            "Inspect unfamiliar rooms and objects before committing to a long sequence of movement commands.",
            "Record successful transitions and useful environmental clues so later decisions can reuse them.",
            "Verify that a proposed route works one transition at a time before relying on it.",
            "Compare the current observation with earlier responses before repeating an interaction.",
            "Reassess the active goal when new score events or puzzle clues change the situation.",
            "Preserve potentially useful inventory items until their purpose is better understood.",
            "Use the available action list to keep commands grounded in actions the game accepts.",
            "Prefer focused exploration of a room's objects and exits over unstructured movement.",
            "Carry forward only puzzle conclusions that are supported by explicit environment feedback.",
        ]

        negatives = []
        if repeated:
            negatives.append("Do not repeat commands that produce the same no-progress response unless new evidence changes the situation.")
        if episode.get("deviations"):
            negatives.append("Do not assume a historical map edge is open; verify each transition and abandon a route after a failed step.")
        if reason == "step_limit":
            negatives.append("Avoid spending too many commands on an unproductive line of investigation; change approach when observations stop changing.")
        negative_fallback = [
            "Avoid repeating an unsuccessful command without a new clue, item, or change in circumstances.",
            "Do not treat remembered routes, inventory guesses, or puzzle hypotheses as confirmed facts.",
            "Avoid exploring without a concrete short-term objective when a score opportunity or unresolved obstacle is already evident.",
            "Do not continue a route after the environment has shown that its next transition is blocked.",
            "Do not assume that a movement command returns to the room from which the player came.",
            "Avoid abandoning a potentially useful object without first examining its description or behavior.",
            "Do not confuse an Actor-reported inventory guess with an environment-verified inventory state.",
            "Avoid pursuing the same puzzle hypothesis after repeated contradictory feedback.",
            "Do not spend many consecutive steps on an action that produces no new information.",
            "Avoid treating another episode's successful puzzle state as proof that the current episode has the same state.",
        ]

        positives = self._fit_lesson_count(
            positives, positive_fallback, self.lessons_per_category
        )
        negatives = self._fit_lesson_count(
            negatives, negative_fallback, self.lessons_per_category
        )
        return {
            "report": first + "\n\n" + second,
            "positives": positives,
            "negatives": negatives,
            "review_source": "deterministic_fallback",
        }

    @staticmethod
    def _review_payload(episode):
        """Build a compact, bounded review context from the recorded trajectory."""
        action_log = episode.get("action_log", episode.get("recent_steps", []))
        # Preserve score events and the end of the trajectory while avoiding
        # an excessively large second LLM request for long episodes.
        selected = action_log[-160:]
        compact_steps = []
        for step in selected:
            compact = {
                key: step.get(key)
                for key in (
                    "before_location", "action", "after_location", "moved",
                    "score_delta", "navigation_status", "action_source",
                    "goal_kind", "goal_description", "actor_state_note",
                )
                if step.get(key) not in (None, "")
            }
            if step.get("score_delta") not in (None, 0):
                compact["observation"] = step.get("observation", "")[:220]
            elif not step.get("moved"):
                compact["observation"] = step.get("observation", "")[:120]
            compact_steps.append(compact)

        return {
            "start_location": episode.get("start_location"),
            "start_score": episode.get("start_score"),
            "end_score": episode.get("end_score"),
            "score_delta": episode.get("score_delta"),
            "target_score": episode.get("target_score"),
            "target_achieved": episode.get("target_achieved"),
            "steps": episode.get("steps", len(action_log)),
            "moves": episode.get("moves"),
            "termination_reason": episode.get("termination_reason"),
            "visited_locations": episode.get("new_locations", [])[-80:],
            "score_events": episode.get("score_events", [])[-30:],
            "repeated_actions": episode.get("repeated_actions", [])[-30:],
            "navigation_deviations": episode.get("deviations", [])[-30:],
            "trajectory_is_truncated": len(action_log) > len(selected),
            "trajectory": compact_steps,
        }

    def review_episode(
        self,
        episode,
        *,
        client,
        model,
        temperature=0.2,
        max_tokens=900,
        event_logger=None,
        extra_body=None,
        trial_episode_number=None,
    ):
        """Review a completed episode and refresh this trial's configurable coaching lessons.

        The model sees only runtime episode evidence and prior lessons. It does
        not receive the walkthrough, guide, or a treasure point-value table.
        A deterministic fallback keeps the next episode usable if the review
        endpoint is unavailable or returns malformed output.
        """
        episode_number = (
            int(trial_episode_number)
            if trial_episode_number is not None
            else len(self.memory.get("episodes", []))
        )
        previous = {
            "positives": list(self.trial_lessons.get("positives", [])),
            "negatives": list(self.trial_lessons.get("negatives", [])),
        }
        review_context = self._review_payload(episode)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Historian for a neuro-symbolic agent playing Zork I. "
                    "Review the completed episode using only the supplied runtime evidence. "
                    "Be candid, specific, and fair. Distinguish "
                    "confirmed outcomes from hypotheses. Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write a concise 1-2 paragraph review of how the Actor performed. "
                    "Discuss its strongest decisions, missed opportunities, ineffective "
                    "behavior, and actionable advice for its next episode. Assess the "
                    "Actor separately from symbolic-navigation actions where the evidence "
                    "allows. Do not claim an action caused a score increase unless the "
                    "recorded score event supports that conclusion. Add a newline after each sentence for readability. \n\n"
                    f"Then update the cross-episode lessons. Return exactly {self.lessons_per_category} "
                    "concise, actionable positives to repeat and exactly "
                    f"{self.lessons_per_category} negatives to avoid. Treat previous lessons "
                    "as candidates: retain useful ones, revise or replace them when this "
                    "episode provides better evidence. The lessons will be included in "
                    "the next episode's Actor prompt. Avoid generic platitudes and do not "
                    "disclose puzzle solutions.\n\n"
                    "Required JSON fields: report (a string containing one or two "
                    "paragraphs), positives (an array of strings), and negatives "
                    "(an array of strings). Each array must contain exactly "
                    f"{self.lessons_per_category} distinct items. Return valid JSON only.\n\n"
                    f"PREVIOUS LESSONS:\n{json.dumps(previous, ensure_ascii=False)}\n\n"
                    f"COMPLETED EPISODE EVIDENCE:\n"
                    f"{json.dumps(review_context, ensure_ascii=False, default=str)}"
                ),
            },
        ]
        request = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if extra_body is not None:
            request["extra_body"] = extra_body

        if event_logger:
            event_logger("historian_review_request", request)

        result = None
        raw_response = ""
        try:
            if client is None:
                raise RuntimeError("No review client is configured.")
            response = client.chat.completions.create(**request)
            raw_response = response.choices[0].message.content or ""
            if event_logger:
                event_logger("historian_review_response", {
                    "model": model,
                    "content": raw_response,
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                })
            parsed = self._parse_review_json(raw_response)
            report = "\n\n".join(
                part.strip()
                for part in str(parsed.get("report", "")).split("\n\n")
                if part.strip()
            )
            requested_count = self.lessons_per_category
            positives = self._clean_strings(
                parsed.get("positives", []), limit=max(20, requested_count)
            )
            negatives = self._clean_strings(
                parsed.get("negatives", []), limit=max(20, requested_count)
            )
            if not report or len(positives) < requested_count or len(negatives) < requested_count:
                raise ValueError(
                    "Historian response did not contain a report and "
                    f"{requested_count} lessons in each category."
                )
            result = {
                "report": report,
                "positives": positives[:requested_count],
                "negatives": negatives[:requested_count],
                "review_source": "llm",
            }
        except Exception as exc:
            if event_logger:
                event_logger("historian_review_error", {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_response": raw_response,
                })
            result = self._fallback_episode_review(episode)
            result["review_error"] = f"{type(exc).__name__}: {exc}"

        # Normalize both LLM and deterministic-fallback results to the exact
        # requested count, so prompt size and per-episode artifacts are stable.
        self.trial_lessons = {
            "positives": self._fit_lesson_count(
                result.get("positives", []),
                [
                    "Use score changes and environment responses as concrete feedback.",
                    "Inspect unfamiliar rooms and potentially useful objects.",
                    "Record successful transitions and environmental clues.",
                    "Verify routes one transition at a time.",
                    "Reassess goals when new evidence appears.",
                    "Preserve potentially useful inventory items.",
                    "Use valid actions and explicit observations to ground decisions.",
                    "Compare outcomes before repeating an interaction.",
                    "Focus exploration on relevant objects and exits.",
                    "Retain only puzzle conclusions supported by feedback.",
                ],
                self.lessons_per_category,
            ),
            "negatives": self._fit_lesson_count(
                result.get("negatives", []),
                [
                    "Avoid repeating unsuccessful commands without new evidence.",
                    "Do not assume historical routes are currently open.",
                    "Change approach when an investigation repeatedly makes no progress.",
                    "Do not assume reverse movement returns to the previous room.",
                    "Do not treat guessed inventory as verified inventory.",
                    "Do not pursue hypotheses contradicted by environment feedback.",
                    "Avoid long sequences that produce no new information.",
                    "Do not assume puzzle state carries across episodes.",
                    "Do not abandon useful objects without inspection.",
                    "Do not treat a blocked route as temporarily open without evidence.",
                ],
                self.lessons_per_category,
            ),
        }
        result["episode_number"] = episode_number
        result["lessons_per_category"] = self.lessons_per_category
        result["lessons"] = {
            "positives": list(self.trial_lessons["positives"]),
            "negatives": list(self.trial_lessons["negatives"]),
        }
        episode["historian_review"] = result["report"]
        episode["lessons_after_review"] = result["lessons"]
        episode["review_source"] = result["review_source"]
        self.save()
        if event_logger:
            event_logger("historian_review_result", result)
        return result

    def format_trial_lessons(self):
        positives = self.trial_lessons.get("positives", [])
        negatives = self.trial_lessons.get("negatives", [])
        if not positives and not negatives:
            return "No completed-episode lessons are available yet."
        lines = ["LESSONS LEARNED FROM PREVIOUS EPISODES:", "Repeat these positive behaviors:"]
        lines.extend(f"{index}. {item}" for index, item in enumerate(positives, 1))
        lines.append("Avoid these negative behaviors:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(negatives, 1))
        lines.append("Apply these lessons when relevant; current game feedback takes precedence.")
        return "\n".join(lines)

    def get_avoid_actions(self, current_node):
        key = str(current_node)
        blocked = self.memory["blocked_actions"].get(key, {})
        repeated = []
        for record in self.memory["repeat_failures"].values():
            if record.get("node") == key and record.get(
                "no_progress_count", 0
            ) >= self.repeat_threshold:
                repeated.append(record.get("action"))
        return sorted(set(blocked) | set(repeated))

    def build_guidance(self, current_node, current_location, current_score=None):
        score = self._as_number(current_score)
        remaining = (
            max(0, self.target_score - score)
            if score is not None else self.target_score
        )
        lines = [
            f"Primary objective: exceed 75 points (target {self.target_score}; current {score if score is not None else 'unknown'}; at least {remaining} additional point(s) needed).",
            "Do not optimize for fewest moves. A long detour can be worthwhile if it helps discover score-producing opportunities.",
            "Investigate objects and interactions that may be valuable. When carrying likely valuables, prioritize bringing them to the Living Room and securing them in the trophy case.",
            "If the route home is unknown or blocked, investigate clues and alternate return paths. Consider what the environment implies about barriers, guardians, keys, unusual locations, and special actions; infer solutions from feedback rather than repeating guesses.",
            "Treat score changes as reliable feedback. Remember actions that increased score and avoid repeating identical no-progress actions.",
        ]

        avoid = self.get_avoid_actions(current_node)
        if avoid:
            lines.append(
                f"At {current_location}, these actions repeatedly failed to make progress: {', '.join(avoid)}. Do not retry without new evidence."
            )

        positive = sorted(
            self.memory["score_profiles"].values(),
            key=lambda item: (
                item.get("total_gain", 0.0),
                item.get("best_gain", 0.0),
            ),
            reverse=True,
        )[:5]
        if positive:
            lessons = []
            for item in positive:
                if item.get("total_gain", 0) > 0:
                    lessons.append(
                        f"Observed score increase after '{item['action']}' at {item['location']}."
                    )
            if lessons:
                lines.append("Learned from actual score feedback: " + " ".join(lessons))

        clues = self.memory.get("puzzle_clues", [])[-5:]
        if clues:
            lines.append(
                "Previously reported environmental clues: "
                + " | ".join(f"{c.get('location', 'unknown')}: {c.get('clue', '')}" for c in clues)
            )

        trial_lessons = self.format_trial_lessons()
        if trial_lessons != "No completed-episode lessons are available yet.":
            lines.append(trial_lessons)

        return "\n".join(lines)

    def choose_goal(
        self,
        *,
        current_node,
        current_location,
        valid_actions,
        known_routes,
        known_outgoing_actions=(),
        current_score=None,
        home_node=None,
        home_label="Living Room",
        home_route_status="unknown",
    ):
        """Choose a strategy from observed score/inventory/map feedback.

        No guide-derived score table or walkthrough is loaded. Goal order:
        target reached -> recover carried valuables -> repeat observed positive
        scoring action -> explore unknown transitions -> investigate puzzles.
        """
        score = self._as_number(current_score)
        if score is not None and score >= self.target_score:
            return {
                "kind": "target_reached",
                "description": f"Target score achieved: {score} >= {self.target_score}.",
                "target_node": None,
                "target_label": None,
                "expected_score_gain": 0,
                "source": "jericho_score",
            }

        carried = self.agent_state.get("carried_treasures", [])
        home_key = str(home_node) if home_node is not None else None
        current_key = str(current_node)

        # If treasure is believed to be carried, secure it before further roaming.
        if carried:
            if home_key is None:
                return {
                    "kind": "solve_return_route",
                    "description": (
                        f"Find a way back to {home_label} while preserving the carried valuables. "
                        "Investigate environmental clues and alternate routes; keep the return objective in mind."
                    ),
                    "target_node": None,
                    "target_label": home_label,
                    "carried_treasures": carried,
                    "route_available": False,
                    "source": "actor_inventory_report",
                }

            if current_key != home_key:
                # A persistent map path is a candidate. The navigator probes
                # its transitions individually and abandons it on failure.
                if home_route_status in {"confirmed", "historical_unconfirmed"}:
                    route_instruction = (
                        "Use the route confirmed in this episode."
                        if home_route_status == "confirmed"
                        else (
                            "The persistent map has a previously observed candidate route, "
                            "but it is not fully confirmed this episode. Follow it cautiously "
                            "one transition at a time; stop and investigate if a passage is blocked."
                        )
                    )
                    return {
                        "kind": "navigate_to_home",
                        "description": (
                            f"Return to {home_label} with carried valuables. {route_instruction} "
                            "Then deposit the carried valuables in the trophy case."
                        ),
                        "target_node": home_key,
                        "target_label": home_label,
                        "carried_treasures": carried,
                        "route_available": True,
                        "route_status": home_route_status,
                        "source": "actor_inventory_report",
                    }

                if home_route_status == "blocked":
                    return_description = (
                        f"Find another way back to {home_label} while preserving the carried valuables. "
                        "All currently known candidate paths have failed during this episode. Treat those "
                        "passages as blocked for now; investigate environmental clues, prerequisites, and alternate routes."
                    )
                else:
                    return_description = (
                        f"Find a way back to {home_label} while preserving the carried valuables. "
                        "No usable route is currently known; investigate environmental clues and alternate "
                        "routes, and keep the return objective in mind."
                    )

                return {
                    "kind": "solve_return_route",
                    "description": return_description,
                    "target_node": home_key,
                    "target_label": home_label,
                    "carried_treasures": carried,
                    "route_available": False,
                    "route_status": home_route_status,
                    "source": "actor_inventory_report",
                }

            if home_key is not None and current_key == home_key:
                return {
                    "kind": "deposit_treasure",
                    "description": (
                        f"Deposit carried valuables ({', '.join(carried)}) in the trophy case and verify the result."
                    ),
                    "target_node": home_key,
                    "target_label": home_label,
                    "carried_treasures": carried,
                    "source": "actor_inventory_report",
                }

        # Retry only observed score-positive tasks, not speculative values.
        valid_by_normalized = {
            self.normalize_action(action): action for action in valid_actions
        }
        route_by_destination = {
            str(route.get("destination")): route for route in (known_routes or [])
        }
        candidates = []
        for profile_key, pattern in self.memory["score_profiles"].items():
            attempts = max(
                1, int(pattern.get("attempts", pattern.get("samples", 0)))
            )
            average_gain = pattern.get("total_gain", 0.0) / attempts
            if average_gain <= 0:
                continue
            if profile_key in self._completed_score_tasks or profile_key in self._attempted_score_tasks:
                continue
            target_node = str(pattern["node"])
            action = pattern["action"]
            if target_node == current_key:
                if action not in valid_by_normalized:
                    continue
                kind = "score_action"
                target_route = None
            else:
                target_route = route_by_destination.get(target_node)
                if target_route is None:
                    continue
                kind = "navigate_score_goal"

            candidates.append({
                "kind": kind,
                "description": (
                    f"Revisit an action that previously increased score: '{action}' at {pattern['location']}."
                ),
                "target_node": target_node,
                "target_label": pattern["location"],
                "suggested_action": valid_by_normalized.get(action) if kind == "score_action" else None,
                "expected_score_gain": average_gain,
                "evidence_samples": pattern.get("samples", 0),
                "observed_attempts": attempts,
                "route": target_route,
                "source": "observed_score_delta",
            })

        if candidates:
            # Score gain dominates; travel distance is not a utility penalty.
            candidates.sort(
                key=lambda item: (
                    -item["expected_score_gain"],
                    -item["evidence_samples"],
                    item["target_label"],
                )
            )
            return candidates[0]

        known = {self.normalize_action(a) for a in known_outgoing_actions}
        blocked_at_node = self.memory["blocked_actions"].get(current_key, {})
        blocked = {
            action for action, count in blocked_at_node.items()
            if count >= self.repeat_threshold
        }
        frontier = [
            action for action in valid_actions
            if self._is_direction(action)
            and self.normalize_action(action) not in known
            and self.normalize_action(action) not in blocked
        ]
        if frontier:
            return {
                "kind": "explore_frontier",
                "description": (
                    "Explore an unmapped exit to discover new rooms and possible scoring opportunities. "
                    "Do not prioritize speed over useful discovery."
                ),
                "target_node": None,
                "target_label": None,
                "preferred_actions": frontier,
                "source": "map_frontier",
            }

        return {
            "kind": "investigate",
            "description": (
                "Investigate the current environment for score-producing actions, treasure, puzzle clues, "
                "or a way to unlock new routes. Examine objects and try contextually plausible interactions."
            ),
            "target_node": None,
            "target_label": None,
            "preferred_actions": [],
            "source": "fallback_reasoning",
        }
