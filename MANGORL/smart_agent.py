"""Smart LLM Actor for Zork I.

The Actor is not given a walkthrough, treasure-point table, or puzzle solutions.
It reasons from observations, score feedback, Historian goals, and known routes.
"""
from __future__ import annotations

# Load project-local .env before reading configuration defaults.
import config  # noqa: F401

import json
import os
import random
import re

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class SmartAgent:
    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        extra_body=None,
        history_length=10,
        temperature=0.25,
        max_tokens=300,
        event_logger=None,
        client=None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "gemma4:31b")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "http://cs-steverw.umflint.edu:8080/api"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.extra_body = extra_body
        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                if self.base_url.rstrip("/").endswith("/v1"):
                    self.api_key = "ollama"
                else:
                    raise RuntimeError(
                        "Set OPENAI_API_KEY to your Open WebUI/API key before starting."
                    )
            if OpenAI is None:
                raise RuntimeError("Install the OpenAI Python package: pip install openai")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.history_length = int(history_length)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.event_logger = event_logger
        self.history = []

    def reset_history(self):
        """Clear short-term Actor context at the start of a fresh episode."""
        self.history.clear()

    def _log_event(self, event, payload):
        if self.event_logger is None:
            return
        try:
            self.event_logger(event, payload)
        except Exception as exc:
            # Logging must not prevent the agent from interacting with Zork.
            print(f"Could not write chat trace ({event}): {exc}")

    def add_history(self, observation, action, location, score_delta=None):
        self.history.append({
            "observation": str(observation)[:500],
            "action": str(action),
            "location": str(location),
            "score_delta": score_delta,
        })
        self.history = self.history[-self.history_length:]

    def format_history(self):
        if not self.history:
            return "No previous actions."
        return "\n\n".join(
            f"Location: {row['location']}\n"
            f"Observation: {row['observation']}\n"
            f"Action: {row['action']}\n"
            f"Score change: {row['score_delta']}"
            for row in self.history
        )

    @staticmethod
    def _parse_json_response(answer):
        answer = (answer or "").strip()
        answer = re.sub(r"^```(?:json)?\s*", "", answer, flags=re.I)
        answer = re.sub(r"\s*```$", "", answer)
        try:
            value = json.loads(answer)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            # Recover a JSON object embedded in short model preamble.
            match = re.search(r"\{.*\}", answer, flags=re.S)
            if match:
                try:
                    value = json.loads(match.group(0))
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _clean_list(value, limit=30):
        if not isinstance(value, list):
            return []
        result, seen = [], set()
        for item in value:
            text = " ".join(str(item).split()).strip()
            if not text:
                continue
            key = text.casefold()
            if key not in seen:
                result.append(text[:180])
                seen.add(key)
            if len(result) >= limit:
                break
        return result

    def choose_decision(
        self,
        *,
        observation,
        current_location,
        valid_actions,
        current_score=None,
        target_score=76,
        known_locations=None,
        known_routes=None,
        goal=None,
        historian_guidance="",
        avoid_actions=None,
        preferred_actions=None,
        navigation_context="No active symbolic navigation plan.",
        rl_guidance="",
    ):
        """Return action plus Actor's compact inventory/clue report."""
        if not valid_actions:
            raise ValueError("The Actor cannot choose from an empty action list.")

        known_locations = known_locations or []
        known_routes = known_routes or []
        avoid_actions = avoid_actions or []
        preferred_actions = preferred_actions or []
        goal = goal or {"kind": "explore", "description": "Explore the world."}

        route_lines = []
        for route in known_routes[:15]:
            route_lines.append(
                f"- {route.get('destination_label', route.get('destination'))} "
                f"({route.get('total_cost', '?')} movement commands): "
                f"{' -> '.join(route.get('actions', []))}"
            )
        navigation_text = "\n".join(route_lines) or "No known routes are available."

        system_prompt = f"""You are the reasoning Actor in a neuro-symbolic agent playing Zork I.

The objective is to achieve a score greater than 75 (at least {target_score}).
You do not have a walkthrough, official treasure list, or point-value table.
Discover how the game works from its observations and score feedback.

You choose one game command at a time. A symbolic navigation controller may
execute known routes. The Historian maintains strategic goals and lessons.
The Historian may also provide a configurable set of positive behaviors learned
from prior episodes and negative behaviors to avoid. Apply those lessons when
they fit the current state, but let fresh environment feedback take precedence.
The list length may vary by experiment; evaluate each lesson on its relevance
rather than assuming every item applies to every situation.

STRATEGY:
- Explore thoroughly and pursue likely score-producing opportunities. Do not
  optimize for speed; episodes may last hundreds of commands.
- Pay attention to unique objects, descriptions, treasure-like items, and
  interactions that may produce score. Learn from actual score increases.
- When you acquire an item that appears valuable, report it as a carried
  treasure candidate. When carrying such items, prioritize returning to the
  Living Room and putting them in the trophy case, then verify the result.
- Treat persistent map routes as historical hypotheses, not proof that a
  passage is currently open. The symbolic navigator may cautiously probe a
  historical candidate one transition at a time; each step must succeed before
  it advances. If a step fails or is unavailable, the plan is abandoned.
- When a candidate return route is blocked, or no route is known, make solving
  the return route a strategic objective. Use textual clues to infer barriers,
  prerequisites, and alternate paths rather than repeating the failed command.
- For puzzles, inspect the situation and reason from textual clues. Consider
  relevant objects, keys, dialogue, special actions, environmental mechanisms,
  and alternate routes. Try plausible actions, observe the response, and adapt.
- Do not assume an obstacle is solved until the environment confirms it.
- Avoid repeating an action when the location, score, and observation do not
  meaningfully change. Repeated-failure warnings are evidence, not absolute
  bans if new clues appear.
- Preserve useful tools and items. Consider inventory capacity and whether an
  item may be needed later.
- Return exactly one valid command from AVAILABLE ACTIONS.

OUTPUT:
Return one JSON object only, with this schema:
{{
  "action": "<your single best command from AVAILABLE ACTIONS>",
  "candidate_actions": ["best candidate first", "other plausible candidates"],
  "inventory_items": ["items you currently believe are carried"],
  "carried_treasures": ["items currently carried that may be valuable/score-relevant"],
  "puzzle_clues": ["new concise environmental clues learned this turn"],
  "state_note": "brief note about changed state, obstacle, or next reasoning step"
}}
candidate_actions should contain up to five distinct commands from AVAILABLE
ACTIONS, ordered from your most preferred to least preferred. Put action first.
If only one command is reasonable, return a one-item list. Do not invent
commands. Do not include markdown fences or prose outside the JSON.
"""

        goal_text = (
            f"Type: {goal.get('kind')}\n"
            f"Objective: {goal.get('description')}\n"
            f"Target: {goal.get('target_label')}\n"
            f"Carried treasure candidates: {goal.get('carried_treasures', [])}\n"
            f"Home route status: {goal.get('route_status', 'not supplied')}\n"
            f"Suggested action: {goal.get('suggested_action')}\n"
            f"Preferred actions: {goal.get('preferred_actions', [])}"
        )
        actions_text = "\n".join(f"- {action}" for action in valid_actions)
        user_prompt = f"""CURRENT LOCATION:
{current_location}

CURRENT SCORE:
{current_score}

CURRENT OBSERVATION:
{observation}

HISTORIAN GOAL:
{goal_text}

HISTORIAN GUIDANCE (includes lessons from completed episodes when available):
{historian_guidance or 'No persistent lessons yet.'}

ACTOR REINFORCEMENT-LEARNING GUIDANCE:
{rl_guidance or 'No Actor experience-based guidance is currently available.'}

NAVIGATION CONTROLLER:
{navigation_context}

KNOWN LOCATIONS:
{', '.join(known_locations[:100]) if known_locations else 'None yet.'}

KNOWN ROUTES (persistent historical map; these are candidate paths, not
proof that every passage is currently open):
{navigation_text}

ACTIONS TO AVOID REPEATING WITHOUT NEW EVIDENCE:
{', '.join(avoid_actions) if avoid_actions else 'None recorded.'}

RECENT ACTION HISTORY:
{self.format_history()}

AVAILABLE ACTIONS:
{actions_text}

Choose the single best next command. Report inventory, likely carried valuables,
and any new puzzle clue in the required JSON object.
"""

        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body

        self._log_event("request", {
            "model": request["model"],
            "temperature": request["temperature"],
            "max_tokens": request["max_tokens"],
            "messages": request["messages"],
        })

        try:
            response = self.client.chat.completions.create(**request)
            answer = response.choices[0].message.content or ""
            self._log_event("response", {
                "model": self.model,
                "content": answer,
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
            })
            parsed = self._parse_json_response(answer)
            action = str(parsed.get("action", "")).strip().strip("`\"' ")
            if action not in valid_actions:
                # Permit exact case-insensitive action match.
                action = next(
                    (candidate for candidate in valid_actions
                     if candidate.casefold() == action.casefold()),
                    "",
                )
            if not action:
                raise ValueError(f"Actor returned invalid action: {answer[:500]!r}")

            candidate_actions = []
            for candidate in self._clean_list(parsed.get("candidate_actions", []), 12):
                matched = next(
                    (item for item in valid_actions if item.casefold() == candidate.casefold()),
                    None,
                )
                if matched and matched not in candidate_actions:
                    candidate_actions.append(matched)
            # Keep the Actor's selected action as the highest-ranked base-policy
            # candidate; older responses that only return "action" remain valid.
            candidate_actions = [action] + [
                item for item in candidate_actions if item != action
            ]
            candidate_actions = candidate_actions[:5]

            report = {
                "location": current_location,
                "inventory_items": self._clean_list(parsed.get("inventory_items"), 50),
                "carried_treasures": self._clean_list(parsed.get("carried_treasures"), 30),
                "puzzle_clues": self._clean_list(parsed.get("puzzle_clues"), 12),
                "state_note": str(parsed.get("state_note", ""))[:500],
            }
            return {
                "action": action,
                "candidate_actions": candidate_actions,
                "report": report,
                "raw_response": answer,
            }
        except Exception as exc:
            self._log_event("error", {
                "model": self.model,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"LLM request failed or returned invalid JSON: {exc}")

        # Deterministic fallback; no model response means do not invent reports.
        fallback = next((item for item in preferred_actions if item in valid_actions), None)
        if fallback is None:
            fallback = random.choice(valid_actions)
        self._log_event("fallback", {
            "selected_action": fallback,
            "reason": "The model request failed or did not return a valid action.",
        })
        return {
            "action": fallback,
            "candidate_actions": [fallback],
            "report": {
                "location": current_location,
                "inventory_items": [],
                "carried_treasures": [],
                "puzzle_clues": [],
                "state_note": "Actor fallback selected after request/parse failure.",
            },
            "raw_response": "",
        }

    def choose_action(self, **kwargs):
        """Compatibility wrapper for callers expecting only an action string."""
        return self.choose_decision(**kwargs)["action"]
