"""Persistent Historian for score-seeking Zork episodes.

The Historian receives only observations, Jericho score deltas, the Actor's
reported inventory/valuables/clues, and the symbolic map. It contains no
walkthrough or guide-derived point table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Historian:
    def __init__(
        self,
        memory_path="memory/historian.json",
        target_score=76,
        repeat_threshold=2,
    ):
        self.memory_path = Path(memory_path)
        self.target_score = int(target_score)
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.memory = self._empty_memory()
        self.episode = None
        self._recent_attempts = []
        self._completed_score_tasks = set()
        self._attempted_score_tasks = set()
        self.agent_state = self._empty_agent_state()
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
        }
        self.episode["steps"] += 1
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
                route_map = {
                    str(route.get("destination")): route
                    for route in (known_routes or [])
                }
                home_route = route_map.get(home_key)
                if home_route is not None:
                    return {
                        "kind": "navigate_to_home",
                        "description": (
                            f"Return to {home_label} with carried valuables, then deposit them in the trophy case."
                        ),
                        "target_node": home_key,
                        "target_label": home_label,
                        "carried_treasures": carried,
                        "route_available": True,
                        "source": "actor_inventory_report",
                    }
                return {
                    "kind": "solve_return_route",
                    "description": (
                        f"Find a way back to {home_label} while preserving the carried valuables. "
                        "Investigate environmental clues and alternate routes; do not keep wandering away from the return objective."
                    ),
                    "target_node": home_key,
                    "target_label": home_label,
                    "carried_treasures": carried,
                    "route_available": False,
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
