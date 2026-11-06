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
            "treasure_candidates": [],
            "ordinary_items": [],
            "useful_items": [],
            "uncertain_items": [],
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
            "item_value_evidence": {},
            "productive_locations": {},
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
            "underground_reached": False,
            "first_underground_location": None,
            "home_return_verified": False,
            "home_return_step": None,
        }
        self._recent_attempts = []
        self._completed_score_tasks = set()
        self._attempted_score_tasks = set()
        # Inventory resets with a fresh Jericho episode. Cross-episode lessons
        # and puzzle clues remain in persistent memory.
        self.agent_state = self._empty_agent_state()

    @staticmethod
    def _normalize_item_name(item):
        value = " ".join(str(item or "").casefold().split())
        value = re.sub(r"^(?:the|a|an)\s+", "", value)
        return re.sub(r"[^a-z0-9 ]+", "", value).strip()

    @classmethod
    def _items_match(cls, left, right):
        left_key = cls._normalize_item_name(left)
        right_key = cls._normalize_item_name(right)
        if not left_key or not right_key:
            return False
        if left_key == right_key:
            return True
        left_tokens, right_tokens = set(left_key.split()), set(right_key.split())
        shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
        return len(shorter) >= 1 and shorter.issubset(longer)

    @classmethod
    def _is_obvious_non_treasure(cls, item):
        """Conservatively exclude common information/tool/utility objects.

        This is not a treasure whitelist. Positive in-game score evidence always
        overrides this heuristic, and the Actor may still retain these items as
        useful tools or puzzle objects.
        """
        normalized = cls._normalize_item_name(item)
        ordinary_exact = {
            "leaflet", "bottle of water", "water", "brass lantern", "lantern",
            "sword", "rope", "nasty looking knife", "broken lantern",
        }
        return normalized in ordinary_exact

    @classmethod
    def _item_from_rewarded_action(cls, action):
        """Extract an item name when a score-positive command names one."""
        text = " ".join(str(action or "").strip().split())
        patterns = (
            r"^(?:take|pick up|get|grab|collect|remove|lift)\s+(?:the\s+)?(.+?)$",
            r"^(?:put|place|insert|drop)\s+(?:the\s+)?(.+?)\s+(?:in|into)\s+(?:the\s+)?(?:trophy case|case)$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.I)
            if match:
                item = match.group(1).strip(" .,!?")
                if item.casefold() not in {"all", "everything", "it", "them"}:
                    return item
        return None

    def _record_item_value_evidence(self, item, *, action, score_delta, location):
        normalized = self._normalize_item_name(item)
        if not normalized:
            return
        evidence = self.memory.setdefault("item_value_evidence", {}).setdefault(
            normalized,
            {
                "item": item,
                "score_positive_events": 0,
                "total_score_gain": 0.0,
                "locations": [],
                "actions": [],
                "confirmed_treasure": False,
            },
        )
        evidence["score_positive_events"] = int(evidence.get("score_positive_events", 0)) + 1
        evidence["total_score_gain"] = float(evidence.get("total_score_gain", 0.0)) + float(score_delta)
        evidence["confirmed_treasure"] = True
        if location and location not in evidence.setdefault("locations", []):
            evidence["locations"].append(str(location)[:120])
            evidence["locations"] = evidence["locations"][-10:]
        if action and action not in evidence.setdefault("actions", []):
            evidence["actions"].append(str(action)[:120])
            evidence["actions"] = evidence["actions"][-10:]

    def record_agent_report(self, report):
        """Store Actor inventory classifications and clues.

        The Actor supplies tentative semantic classifications. Actual positive
        Jericho score events are stronger evidence and can confirm an item's
        score relevance across episodes.
        """
        if not isinstance(report, dict):
            return
        inventory = self._clean_strings(report.get("inventory_items", []), limit=50)
        reported_treasures = self._clean_strings(
            report.get("treasure_candidates") or report.get("carried_treasures", []),
            limit=30,
        )
        ordinary = self._clean_strings(report.get("ordinary_items", []), limit=40)
        useful = self._clean_strings(report.get("useful_items", []), limit=40)
        uncertain = self._clean_strings(report.get("uncertain_items", []), limit=40)
        for item in inventory:
            if self._is_obvious_non_treasure(item) and not any(
                self._items_match(item, existing) for existing in ordinary
            ):
                ordinary.append(item)

        ordinary_keys = {self._normalize_item_name(item) for item in ordinary}
        evidence = self.memory.setdefault("item_value_evidence", {})
        confirmed_carried = []
        if inventory:
            for inventory_item in inventory:
                if any(
                    value.get("confirmed_treasure") and self._items_match(key, inventory_item)
                    for key, value in evidence.items()
                ):
                    confirmed_carried.append(inventory_item)

        ordinary = [
            item for item in ordinary
            if not any(
                value.get("confirmed_treasure") and self._items_match(key, item)
                for key, value in evidence.items()
            )
        ]
        ordinary_keys = {self._normalize_item_name(item) for item in ordinary}
        treasure_candidates = []
        for item in reported_treasures + confirmed_carried:
            key = self._normalize_item_name(item)
            has_reward_evidence = any(
                value.get("confirmed_treasure") and self._items_match(evidence_key, item)
                for evidence_key, value in evidence.items()
            )
            if not has_reward_evidence and (
                key in ordinary_keys or self._is_obvious_non_treasure(item)
            ):
                if item not in ordinary:
                    ordinary.append(item)
                continue
            if inventory and not any(self._items_match(item, present) for present in inventory):
                continue
            if item.casefold() not in {existing.casefold() for existing in treasure_candidates}:
                treasure_candidates.append(item)

        # Keep backward compatibility: carried_treasures now means the subset
        # of current inventory worth considering for trophy-case return.
        self.agent_state["inventory_items"] = inventory
        self.agent_state["treasure_candidates"] = treasure_candidates
        self.agent_state["carried_treasures"] = list(treasure_candidates)
        self.agent_state["ordinary_items"] = ordinary
        self.agent_state["useful_items"] = useful
        self.agent_state["uncertain_items"] = uncertain
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
            "inventory_items": inventory,
            "carried_treasures": self.agent_state["carried_treasures"],
            "ordinary_items": ordinary,
            "useful_items": useful,
            "uncertain_items": uncertain,
            "state_note": self.agent_state["state_note"],
            "episode": len(self.memory["episodes"]) + 1,
        })
        self.memory["agent_reports"] = self.memory["agent_reports"][-50:]

    def _update_inventory_from_action_result(self, action, observation, score_delta):
        """Reconcile one explicit take/deposit action after Jericho responds.

        Actor reports are produced before the selected command executes. This
        small deterministic update avoids keeping the inventory one step stale
        when the Actor takes or deposits an item.
        """
        text = " ".join(str(action or "").split())
        observation_text = " ".join(str(observation or "").casefold().split())
        take_match = re.match(
            r"^(?:take|pick up|get|grab|collect|lift)\s+(?:the\s+)?(.+?)$",
            text,
            flags=re.I,
        )
        deposit_match = re.match(
            r"^(?:put|place|insert)\s+(?:the\s+)?(.+?)\s+(?:in|into)\s+(?:the\s+)?(?:trophy case|case)$",
            text,
            flags=re.I,
        )
        if take_match:
            item = take_match.group(1).strip(" .,!? ")
            if item.casefold() in {"all", "everything", "it", "them"}:
                return
            success = bool(
                (score_delta is not None and score_delta > 0)
                or re.search(
                    r"\b(taken|you take|you pick up|you now have|you are now carrying)\b",
                    observation_text,
                )
            )
            if not success:
                return
            inventory = self.agent_state.setdefault("inventory_items", [])
            if not any(self._items_match(item, existing) for existing in inventory):
                inventory.append(item)
            evidence = self.memory.setdefault("item_value_evidence", {})
            has_reward_evidence = any(
                value.get("confirmed_treasure") and self._items_match(key, item)
                for key, value in evidence.items()
            )
            if has_reward_evidence or (score_delta is not None and score_delta > 0):
                treasure_items = self.agent_state.setdefault("treasure_candidates", [])
                if not any(self._items_match(item, existing) for existing in treasure_items):
                    treasure_items.append(item)
                self.agent_state["carried_treasures"] = list(treasure_items)
                self.agent_state["ordinary_items"] = [
                    old for old in self.agent_state.get("ordinary_items", [])
                    if not self._items_match(old, item)
                ]
                self.agent_state["uncertain_items"] = [
                    old for old in self.agent_state.get("uncertain_items", [])
                    if not self._items_match(old, item)
                ]
            elif self._is_obvious_non_treasure(item):
                ordinary = self.agent_state.setdefault("ordinary_items", [])
                if not any(self._items_match(item, existing) for existing in ordinary):
                    ordinary.append(item)
            else:
                uncertain = self.agent_state.setdefault("uncertain_items", [])
                if not any(self._items_match(item, existing) for existing in uncertain):
                    uncertain.append(item)
            return

        if deposit_match:
            item = deposit_match.group(1).strip(" .,!? ")
            if item.casefold() in {"all", "everything", "it", "them"}:
                return
            # A trophy-case command with an explicit item is treated as
            # successful only when the response indicates acceptance, or a
            # score increase confirms the deposit.
            success = bool(
                (score_delta is not None and score_delta > 0)
                or re.search(
                    r"\b(you put|you place|you deposit|placed in|is now in the trophy case|case now contains|you secure)\b",
                    observation_text,
                )
            )
            if not success:
                return
            for key in ("inventory_items", "treasure_candidates", "carried_treasures", "ordinary_items", "useful_items", "uncertain_items"):
                self.agent_state[key] = [
                    old for old in self.agent_state.get(key, [])
                    if not self._items_match(item, old)
                ]

    @staticmethod
    def _is_underground_location(location, observation=""):
        """Identify known subterranean room labels without treating surface clues as entry.

        The room label is authoritative. Observation text is only used for
        explicit statements that the player is already underground; merely
        seeing a grating or trap door on the surface does not trigger the goal.
        """
        label = " ".join(str(location or "").casefold().split())
        label_markers = (
            "cellar", "underground", "maze", "troll room", "cyclops",
            "dungeon", "cave", "temple", "dam base", "reservoir",
            "mine", "shaft", "torch room", "gas room", "loud room",
            "round room", "deep canyon", "volcano", "treasure room",
            "east-west passage", "north-south passage",
        )
        if any(marker in label for marker in label_markers):
            return True

        description = " ".join(str(observation or "").casefold().split())
        explicit_entry_phrases = (
            "you are underground", "you are in the cellar",
            "you are in a maze", "you have entered the underground",
            "you are deep underground", "you are in the cave",
        )
        return any(phrase in description for phrase in explicit_entry_phrases)

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
        home_node=None,
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

        if self._is_underground_location(after_location, after_text):
            if not self.episode.get("underground_reached"):
                self.episode["first_underground_location"] = str(after_location)
                self.episode["underground_entry_step"] = self.episode.get("steps", 0) + 1
            self.episode["underground_reached"] = True
        if (
            self.episode.get("underground_reached")
            and home_node is not None
            and after_key == str(home_node)
        ):
            self.episode["home_return_verified"] = True
            if self.episode.get("home_return_step") is None:
                self.episode["home_return_step"] = self.episode.get("steps", 0) + 1

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
            rewarded_item = self._item_from_rewarded_action(normalized)
            if rewarded_item:
                event["item"] = rewarded_item
                self._record_item_value_evidence(
                    rewarded_item,
                    action=normalized,
                    score_delta=score_delta,
                    location=before_location,
                )
            self.episode["score_events"].append(event)
            self.memory["score_events"].append(event)
            self.memory["score_events"] = self.memory["score_events"][-250:]

            productive = self.memory.setdefault("productive_locations", {}).setdefault(
                before_key,
                {
                    "node": before_key,
                    "location": before_location,
                    "score_events": 0,
                    "total_score_gain": 0.0,
                    "best_score_gain": 0.0,
                    "actions": {},
                },
            )
            productive["score_events"] = int(productive.get("score_events", 0)) + 1
            productive["total_score_gain"] = float(productive.get("total_score_gain", 0.0)) + float(score_delta)
            productive["best_score_gain"] = max(
                float(productive.get("best_score_gain", 0.0)), float(score_delta)
            )
            action_profile = productive.setdefault("actions", {}).setdefault(
                normalized, {"count": 0, "total_gain": 0.0, "best_gain": 0.0}
            )
            action_profile["count"] = int(action_profile.get("count", 0)) + 1
            action_profile["total_gain"] = float(action_profile.get("total_gain", 0.0)) + float(score_delta)
            action_profile["best_gain"] = max(
                float(action_profile.get("best_gain", 0.0)), float(score_delta)
            )

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

        self._update_inventory_from_action_result(normalized, observation, score_delta)

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

        if goal:
            goal_record = {
                "step": self.episode["steps"] + 1,
                "kind": str(goal.get("kind", "unknown")),
                "description": str(goal.get("description", ""))[:600],
                "sequence": goal.get("sequence", []),
                "source": goal.get("source", "unknown"),
            }
            history = self.episode.setdefault("goal_history", [])
            if not history or (
                history[-1].get("kind") != goal_record["kind"]
                or history[-1].get("description") != goal_record["description"]
            ):
                history.append(goal_record)

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
            "goal_description": str((goal or {}).get("description", ""))[:600],
            "goal_sequence": (goal or {}).get("sequence", []),
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
                    "goal_kind", "goal_description", "goal_sequence", "actor_state_note",
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
            "strategic_goal_history": episode.get("goal_history", [])[-30:],
            "underground_return_milestones": {
                "underground_reached": episode.get("underground_reached", False),
                "first_underground_location": episode.get("first_underground_location"),
                "home_return_verified": episode.get("home_return_verified", False),
                "home_return_step": episode.get("home_return_step"),
            },
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
        """Ask the configured LLM to review an episode and update trial lessons.

        There is deliberately no deterministic review or lesson fallback. If
        the model cannot be reached, returns malformed JSON, or omits required
        lessons, the failure is logged and raised. Existing trial lessons are
        left untouched so an unsuccessful review cannot silently invent
        coaching or replace the previous successful review.
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
                    "Do not use or invent walkthrough knowledge, puzzle solutions, treasure "
                    "lists, or point values. Be candid, specific, and fair. Distinguish "
                    "confirmed outcomes from hypotheses. Return valid JSON only.\n" +
                    getattr(self, "evolved_strategy", "")
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
                    "recorded score event supports that conclusion.\n\n"
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

        raw_response = ""
        try:
            if client is None:
                raise RuntimeError(
                    "No Historian review client is configured; the LLM review was not called."
                )

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
            if not report:
                raise ValueError(
                    "Historian model response did not contain a non-empty 'report' field."
                )
            if len(positives) != requested_count:
                raise ValueError(
                    "Historian model response must contain exactly "
                    f"{requested_count} distinct positive lessons; received {len(positives)}."
                )
            if len(negatives) != requested_count:
                raise ValueError(
                    "Historian model response must contain exactly "
                    f"{requested_count} distinct negative lessons; received {len(negatives)}."
                )

            result = {
                "report": report,
                "positives": positives,
                "negatives": negatives,
                "review_source": "llm",
            }
        except Exception as exc:
            error_payload = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_response": raw_response,
                "fallback_used": False,
            }
            episode["historian_review_error"] = error_payload
            self.save()
            if event_logger:
                event_logger("historian_review_error", error_payload)
            raise

        # Update cross-episode coaching only after a complete, valid model
        # response. Failed calls never substitute deterministic lessons.
        self.trial_lessons = {
            "positives": list(result["positives"]),
            "negatives": list(result["negatives"]),
        }
        result["episode_number"] = episode_number
        result["lessons_per_category"] = self.lessons_per_category
        result["lessons"] = {
            "positives": list(self.trial_lessons["positives"]),
            "negatives": list(self.trial_lessons["negatives"]),
        }
        episode.pop("historian_review_error", None)
        episode["historian_review"] = result["report"]
        episode["lessons_after_review"] = result["lessons"]
        episode["review_source"] = "llm"
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
        """Build a strategic prompt from reward, inventory, route, and lesson evidence."""
        score = self._as_number(current_score)
        remaining = (
            max(0, self.target_score - score)
            if score is not None else self.target_score
        )
        lines = [
            f"Primary objective: exceed 75 points and reach the configured target of {self.target_score} (current score {score if score is not None else 'unknown'}; approximately {remaining} additional point(s) needed).",
            "Do not optimize for fewest moves. Pursue a sequence of meaningful objectives: establish a reliable way home, exploit proven score opportunities, secure valuables, and continue exploring productive areas.",
            "Treat previous score increases as empirical evidence. Prefer repeating a previously successful interaction when you are at the same room with comparable inventory and puzzle state; use the persistent map to cautiously revisit rooms where those interactions worked.",
            "A historical route is a candidate, not a guarantee. Follow it one transition at a time and switch back to puzzle reasoning if a passage is blocked or unavailable.",
        ]

        inventory = self.agent_state.get("inventory_items", [])
        treasures = self.agent_state.get("carried_treasures", [])
        ordinary = self.agent_state.get("ordinary_items", [])
        useful = self.agent_state.get("useful_items", [])
        if inventory:
            lines.append("Current reported inventory: " + ", ".join(inventory) + ".")
        if treasures:
            lines.append(
                "Prioritize returning these treasure candidates to the Living Room trophy case: "
                + ", ".join(treasures)
                + ". These are candidates unless reinforced by recorded score evidence."
            )
        if ordinary:
            lines.append(
                "Do not treat these reported ordinary items as trophy-case valuables merely because they are carried: "
                + ", ".join(ordinary)
                + ". They may still have informational or puzzle utility."
            )
        if useful:
            lines.append(
                "Preserve these reported tools/puzzle items when useful: "
                + ", ".join(useful)
                + "."
            )

        avoid = self.get_avoid_actions(current_node)
        if avoid:
            lines.append(
                f"At {current_location}, these actions repeatedly failed to make progress: {', '.join(avoid)}. Do not retry without new evidence."
            )

        positive = sorted(
            self.memory.get("score_profiles", {}).values(),
            key=lambda item: (
                float(item.get("total_gain", 0.0)) /
                max(1, int(item.get("attempts", item.get("samples", 1)))),
                item.get("best_gain", 0.0),
                item.get("samples", 0),
            ),
            reverse=True,
        )[:7]
        successful = []
        for item in positive:
            if item.get("total_gain", 0) > 0:
                attempts = max(1, int(item.get("attempts", item.get("samples", 1))))
                average = float(item.get("total_gain", 0.0)) / attempts
                successful.append(
                    f"{item.get('location', item.get('node'))}: '{item.get('action')}' "
                    f"(observed total +{float(item.get('total_gain', 0.0)):g} over {attempts} attempt(s); "
                    f"best single gain +{float(item.get('best_gain', 0.0)):g}; average +{average:.2f}/attempt)"
                )
        if successful:
            lines.append(
                "Previously fruitful interactions to exploit when context matches: "
                + "; ".join(successful)
                + "."
            )

        confirmed_items = [
            value.get("item", key)
            for key, value in self.memory.get("item_value_evidence", {}).items()
            if value.get("confirmed_treasure")
        ]
        if confirmed_items:
            lines.append(
                "Items associated with an actual score-positive pickup/deposit event in prior play: "
                + ", ".join(confirmed_items[:15])
                + ". Return them when currently carried."
            )

        episode = self.episode or {}
        if episode.get("underground_reached") and not episode.get("home_return_verified"):
            lines.append(
                "EARLY UNDERGROUND PRIORITY: establish and test at least one route back to the Living Room before committing to a long underground treasure expedition. Once that return is verified, continue exploring and use it as a recovery route."
            )
        elif episode.get("home_return_verified"):
            lines.append(
                "A return from the underground to the Living Room has been verified during this episode. Preserve that knowledge, but still treat any untested alternate passages as uncertain."
            )

        clues = self.memory.get("puzzle_clues", [])[-7:]
        if clues:
            lines.append(
                "Previously reported environmental clues: "
                + " | ".join(
                    f"{c.get('location', 'unknown')}: {c.get('clue', '')}"
                    for c in clues
                )
            )

        trial_lessons = self.format_trial_lessons()
        if trial_lessons != "No completed-episode lessons are available yet.":
            lines.append(trial_lessons)
        return "\n".join(lines)

    @staticmethod
    def _goal_sequence(*steps):
        """Normalize a human-readable ordered strategic plan."""
        result = []
        for index, step in enumerate(steps, 1):
            if step:
                result.append({"order": index, **step})
        return result

    def _productive_goal_candidates(
        self, *, current_key, valid_by_normalized, route_by_destination
    ):
        candidates = []
        for profile_key, pattern in self.memory.get("score_profiles", {}).items():
            attempts = max(1, int(pattern.get("attempts", pattern.get("samples", 1))))
            total_gain = float(pattern.get("total_gain", 0.0))
            average_gain = total_gain / attempts
            if average_gain <= 0:
                continue
            if profile_key in self._completed_score_tasks or profile_key in self._attempted_score_tasks:
                continue

            target_node = str(pattern.get("node", ""))
            action = str(pattern.get("action", ""))
            if target_node == current_key:
                suggested_action = valid_by_normalized.get(self.normalize_action(action))
                if suggested_action is None:
                    continue
                kind = "score_action"
                target_route = None
            else:
                target_route = route_by_destination.get(target_node)
                if target_route is None:
                    continue
                suggested_action = None
                kind = "navigate_score_goal"

            candidates.append({
                "kind": kind,
                "description": (
                    f"Exploit a previously score-positive opportunity at {pattern.get('location', target_node)}: "
                    f"the interaction '{action}' produced an observed total gain of +{total_gain:g} "
                    f"({average_gain:.2f} points per recorded attempt). Revisit it only if the current "
                    "observation and prerequisites remain compatible; verify the new score response."
                ),
                "target_node": target_node,
                "target_label": pattern.get("location", target_node),
                "suggested_action": suggested_action,
                "productive_action": action,
                "expected_score_gain": average_gain,
                "total_observed_gain": total_gain,
                "evidence_samples": int(pattern.get("samples", 0)),
                "observed_attempts": attempts,
                "route": target_route,
                "route_status": "historical_candidate" if target_route else "current_location",
                "preferred_actions": [suggested_action] if suggested_action else [],
                "sequence": self._goal_sequence(
                    {
                        "goal": "Reach the previously productive location",
                        "details": (
                            "Use the persistent directed map route as a historical candidate. "
                            "Probe it one transition at a time; if an edge is blocked, stop and let "
                            "the Actor investigate instead of repeating it."
                        ) if target_route else "You are already at the room where the interaction previously produced score."
                    },
                    {
                        "goal": f"Reassess and try '{action}'",
                        "details": "Check the current observation, inventory, and puzzle prerequisites before repeating the known productive interaction."
                    },
                    {
                        "goal": "Verify progress and secure valuables",
                        "details": "Compare the score before and after the action. Classify inventory items, return confirmed or likely treasures to the Living Room, and deposit them in the trophy case."
                    },
                    {
                        "goal": "Continue to the next productive opportunity",
                        "details": "Use other previously fruitful locations/actions and then explore unmapped frontiers."
                    },
                ),
                "source": "observed_score_delta_and_persistent_world_map",
            })
        candidates.sort(
            key=lambda item: (
                -item["expected_score_gain"],
                -item["evidence_samples"],
                item["target_label"],
            )
        )
        return candidates

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
        """Choose an ambitious, sequential strategy from accumulated evidence.

        Priority: target reached -> return/deposit carried valuables -> establish
        an early underground return route -> revisit reward-positive actions and
        locations using the persistent map -> explore new frontiers/puzzles.
        """
        score = self._as_number(current_score)
        if score is not None and score >= self.target_score:
            return {
                "kind": "target_reached",
                "description": f"Target score achieved: {score} >= {self.target_score}.",
                "target_node": None,
                "target_label": None,
                "expected_score_gain": 0,
                "sequence": [],
                "source": "jericho_score",
            }

        current_key = str(current_node)
        home_key = str(home_node) if home_node is not None else None
        episode = self.episode or {}
        if self._is_underground_location(current_location):
            episode["underground_reached"] = True
            episode.setdefault("first_underground_location", str(current_location))
        if (
            episode.get("underground_reached")
            and home_key is not None
            and current_key == home_key
        ):
            episode["home_return_verified"] = True

        carried = list(self.agent_state.get("carried_treasures", []))
        inventory = list(self.agent_state.get("inventory_items", []))
        ordinary = list(self.agent_state.get("ordinary_items", []))
        useful = list(self.agent_state.get("useful_items", []))
        treasure_priority = bool(carried)

        # Goal sequence A: recover and bank score-relevant valuables before
        # pursuing optional exploration. Ordinary tools/information objects are
        # explicitly not included in the return/deposit requirement.
        if treasure_priority:
            follow_on = self._goal_sequence(
                {
                    "goal": f"Return to {home_label}",
                    "details": "Use a confirmed route if available; otherwise cautiously test the best historical path one transition at a time, or investigate its prerequisites if blocked."
                },
                {
                    "goal": "Deposit the treasure candidates",
                    "details": "Put only the identified treasure candidates in the trophy case. Keep ordinary and useful puzzle items unless there is a separate reason to dispose of them. Verify the score change and inventory afterward."
                },
                {
                    "goal": "Resume the highest-value exploration",
                    "details": "Return to productive underground areas using previously successful paths, repeat score-positive interactions when contextually valid, and continue searching for additional score opportunities."
                },
            )
            if home_key is None:
                return {
                    "kind": "solve_return_route",
                    "description": (
                        f"Return the carried treasure candidates ({', '.join(carried)}) to {home_label}. "
                        "The home room is not yet identified in the current map. Preserve the items, inspect route clues, "
                        "and discover a way back before attempting further optional exploration."
                    ),
                    "target_node": None,
                    "target_label": home_label,
                    "carried_treasures": carried,
                    "ordinary_items": ordinary,
                    "useful_items": useful,
                    "sequence": follow_on,
                    "route_available": False,
                    "source": "actor_inventory_and_reward_evidence",
                }
            if current_key == home_key:
                return {
                    "kind": "deposit_treasure",
                    "description": (
                        f"You are in {home_label} carrying treasure candidates: {', '.join(carried)}. "
                        "Deposit those candidates in the trophy case, but do not mistake ordinary items "
                        f"({', '.join(ordinary) if ordinary else 'none currently reported'}) or useful tools "
                        f"({', '.join(useful) if useful else 'none currently reported'}) for treasure. "
                        "Then inspect inventory and score feedback to verify what was secured."
                    ),
                    "target_node": home_key,
                    "target_label": home_label,
                    "carried_treasures": carried,
                    "ordinary_items": ordinary,
                    "useful_items": useful,
                    "sequence": follow_on,
                    "source": "actor_inventory_and_reward_evidence",
                }
            if home_route_status in {"confirmed", "historical_unconfirmed"}:
                instruction = (
                    "The route has been traversed successfully this episode."
                    if home_route_status == "confirmed"
                    else "The map contains a historical candidate route, not a guarantee; probe one transition at a time and stop immediately if a passage is blocked."
                )
                return {
                    "kind": "navigate_to_home",
                    "description": (
                        f"Return to {home_label} with treasure candidates {', '.join(carried)}. {instruction} "
                        "Do not divert to ordinary items or unrelated exploration until these valuables have been banked."
                    ),
                    "target_node": home_key,
                    "target_label": home_label,
                    "carried_treasures": carried,
                    "ordinary_items": ordinary,
                    "useful_items": useful,
                    "route_available": True,
                    "route_status": home_route_status,
                    "sequence": follow_on,
                    "source": "actor_inventory_and_reward_evidence",
                }
            return {
                "kind": "solve_return_route",
                "description": (
                    f"Find a way back to {home_label} while preserving treasure candidates {', '.join(carried)}. "
                    f"Current route status is {home_route_status}; investigate environmental clues, puzzle prerequisites, "
                    "and alternate routes/paths. Do not repeatedly retry a blocked transition. Keep ordinary items and tools "
                    "separate from the valuables that need to be deposited."
                ),
                "target_node": home_key,
                "target_label": home_label,
                "carried_treasures": carried,
                "ordinary_items": ordinary,
                "useful_items": useful,
                "route_available": False,
                "route_status": home_route_status,
                "sequence": follow_on,
                "source": "actor_inventory_and_reward_evidence",
            }

        # Goal sequence B: after first entering the underground, make a return
        # trip a deliberate early milestone even when no treasure is carried.
        if episode.get("underground_reached") and not episode.get("home_return_verified"):
            sequence = self._goal_sequence(
                {
                    "goal": f"Establish a return route to {home_label}",
                    "details": "Do this early, before committing to a long treasure expedition. Use a historical route as a cautious candidate, not as proof that the door or puzzle prerequisite is currently solved."
                },
                {
                    "goal": "Verify the route by reaching {home_label}".format(home_label=home_label),
                    "details": "A route is established only after the environment actually returns the player to the Living Room. Record any obstacles and newly discovered transitions."
                },
                {
                    "goal": "Resume underground exploration",
                    "details": "Once the return trip is verified, explore productive areas, collect likely treasures, and use the tested route to bank valuables."
                },
            )
            if home_key is not None and current_key != home_key and home_route_status in {"confirmed", "historical_unconfirmed"}:
                return {
                    "kind": "establish_return_route",
                    "description": (
                        f"Before deep exploration, make an early return trip to {home_label}. A candidate path is present "
                        "in the learned map. Follow it one action/transition at a time; if it fails, stop and investigate "
                        "the prerequisite or alternate route rather than repeating it. Reaching the Living Room will "
                        "verify a practical return path for this episode."
                    ),
                    "target_node": home_key,
                    "target_label": home_label,
                    "route_available": True,
                    "route_status": home_route_status,
                    "sequence": sequence,
                    "source": "early_underground_safety_milestone",
                }
            return {
                "kind": "solve_return_route",
                "description": (
                    f"Make finding a way back to {home_label} your immediate strategic objective, even though you are "
                    "not currently carrying identified treasure. Investigate exits, room descriptions, keys, mechanisms, "
                    "guardians, and special interactions. Use previously learned puzzle clues and avoid repeating blocked "
                    "actions. Establish the return route before beginning a lengthy treasure-hunting sequence."
                ),
                "target_node": home_key,
                "target_label": home_label,
                "route_available": False,
                "route_status": home_route_status,
                "sequence": sequence,
                "source": "early_underground_safety_milestone",
            }

        # Goal sequence C: exploit earlier score-positive actions and navigate
        # through historical map routes to productive rooms.
        valid_by_normalized = {
            self.normalize_action(action): action for action in valid_actions
        }
        route_by_destination = {
            str(route.get("destination")): route for route in (known_routes or [])
        }
        candidates = self._productive_goal_candidates(
            current_key=current_key,
            valid_by_normalized=valid_by_normalized,
            route_by_destination=route_by_destination,
        )
        if candidates:
            chosen = candidates[0]
            chosen["next_productive_opportunities"] = [
                {
                    "location": candidate["target_label"],
                    "action": candidate.get("productive_action"),
                    "expected_score_gain": candidate["expected_score_gain"],
                }
                for candidate in candidates[1:4]
            ]
            return chosen

        # Goal sequence D: explore systematically when no known score-positive
        # task is available from the current map position.
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
                    "Explore an unmapped exit, but do so deliberately: inspect the current room and available objects, "
                    "record the resulting transition, and look for score-producing interactions. If you enter the "
                    "underground, make establishing a return route your next milestone before extensive exploration."
                ),
                "target_node": None,
                "target_label": None,
                "preferred_actions": frontier,
                "sequence": self._goal_sequence(
                    {"goal": "Explore an unmapped exit", "details": "Choose a valid frontier action and inspect the destination."},
                    {"goal": "Investigate the new room", "details": "Examine notable objects and available interactions; compare score before and after each action."},
                    {"goal": "Preserve recoverability", "details": "If the room is underground, establish a route to the Living Room before pursuing a lengthy treasure sequence."},
                    {"goal": "Pursue productive discoveries", "details": "Record newly successful actions and routes so they can be exploited in later episodes."},
                ),
                "source": "map_frontier_and_reward_history",
            }

        return {
            "kind": "investigate",
            "description": (
                "Investigate the current environment for score-producing actions, treasure candidates, puzzle clues, "
                "or a way to unlock new routes. Reuse prior successful interactions only when their preconditions "
                "match the current state. Inspect objects, reason from textual clues, and change approach when feedback "
                "shows that an action is ineffective."
            ),
            "target_node": None,
            "target_label": None,
            "preferred_actions": [],
            "sequence": self._goal_sequence(
                {"goal": "Resolve the most informative local uncertainty", "details": "Inspect the environment or test a contextually plausible interaction."},
                {"goal": "Record reward and transition evidence", "details": "Use score changes and room transitions to update what is known."},
                {"goal": "Choose the next productive target", "details": "Prefer previously successful opportunities when state conditions match; otherwise explore a new frontier."},
            ),
            "source": "fallback_reasoning_and_reward_history",
        }
