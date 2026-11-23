"""Non-parametric returns for revisited, observable Zork states."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .action_guards import inventory_from_text, snapshot_sections
from .checkpoint import atomic_json


def _text(value):
    return " ".join(str(value or "").casefold().split())


def _digest(value):
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _observed_lines(text):
    """Ignore listing order within a container, preserving parentage and status."""
    blocks = []
    parents = []
    for line in text.expandtabs().splitlines():
        label = _text(line)
        if not label:
            continue
        depth = len(line) - len(line.lstrip())
        if depth == 0 or not blocks:
            blocks.append((label, []))
            parents.clear()
        else:
            while parents and parents[-1][0] >= depth:
                parents.pop()
            parents.append((depth, label))
            blocks[-1][1].append(tuple(name for _, name in parents))
    return [(title, sorted(items)) for title, items in blocks]


def outcome_state_key(location_id, room, inventory, score, peak_score):
    facts = [location_id, _observed_lines(room), _observed_lines(inventory), score, peak_score]
    if (not str(location_id or "").strip() or not facts[1] or inventory_from_text(inventory) is None
            or score is None or peak_score is None):
        return None
    return _digest(facts)


class OutcomeMemory:
    """Publish one complete, finite-horizon trajectory at an episode boundary.

    Index physical observations independently of the Historian's changing goals.
    Keep the predecessor feedback and remaining horizon for ordering matches.
    """

    def __init__(self, directory, *, gamma, horizon, target, rom, map_navigation):
        self.gamma = gamma
        self.horizon = horizon
        self.protocol = dict(version=1, gamma=gamma, horizon=horizon, target=target,
                             rom=rom, map_navigation=map_navigation)
        self.directory = Path(directory) / _digest(self.protocol)
        self.by_state = defaultdict(list)
        self.episodes = set()
        self.visits = Counter()
        self.losses = Counter()
        self.cooldowns = Counter()
        self.active_episode = None
        for path in sorted(self.directory.glob("*.json")):
            episode = json.loads(path.read_text(encoding="utf-8"))
            if episode["protocol"] != self.protocol:
                raise ValueError(f"Incompatible outcome memory in {path}")
            self._index(episode)

    @staticmethod
    def snapshot(state):
        observation = str(state.get("observation") or "")
        room, inventory = snapshot_sections(observation)
        # Keep light status and container nesting from Jericho's inventory.
        # Empty or unavailable snapshots must not masquerade as an empty pack.
        location = state.get("location_id")
        score, peak = state.get("score"), state.get("peak_score")
        key = outcome_state_key(location, room, inventory, score, peak)
        feedback = observation.partition("CURRENT ROOM (look):\n")[0]
        return {"key": key, "room": room.strip(), "inventory": inventory.strip(),
                "location_id": location, "score": score, "peak_score": peak,
                "feedback": _digest(_text(feedback)),
                "remaining": state.get("steps_remaining", 0)}

    def start_episode(self, episode_id):
        self.active_episode = episode_id
        self.visits.clear()
        self.losses.clear()
        self.cooldowns.clear()

    def record_visit(self, snapshot, action, score_delta):
        if snapshot["key"]:
            pair = (snapshot["key"], _text(action))
            self.visits[pair] += 1
            if score_delta < 0:
                self.losses[pair] += 1
                if self.losses[pair] >= 2:
                    self.cooldowns[pair] = 3

    def apply_cooldown(self, state, actions):
        """Temporarily withhold repeated withdrawals upon an observed return.

        A loss alone cannot trigger suppression elsewhere. Revisiting this key
        requires restoring the score, peak, room contents and carried inventory.
        """
        key = self.snapshot(state)["key"]
        held = [action for action in actions if self.cooldowns[key, _text(action)] > 0]
        if len(held) == len(actions):
            return actions, []
        for action in held:
            pair = (key, _text(action))
            self.cooldowns[pair] -= 1
            if self.cooldowns[pair] == 0:
                del self.cooldowns[pair]
        return [action for action in actions if action not in held], held

    def repetitions(self, state, actions):
        key = self.snapshot(state)["key"]
        return [self.visits[key, _text(action)] for action in actions]

    def _index(self, episode):
        episode_id = episode["episode_id"]
        if episode_id in self.episodes:
            return
        for row in episode["steps"]:
            if row["key"]:
                self.by_state[row["key"]].append(dict(row, episode_id=episode_id))
        self.episodes.add(episode_id)

    def publish(self, episode_id, steps, reason, final_score):
        # At the configured time limit, the remaining benchmark return is zero.
        # Failures and provider-recovery trajectories supply no training targets.
        self.active_episode = None
        if (reason not in {"step_limit", "target_score_reached", "environment_done"}
                or not steps or any(s["action_source"] == "recovery_default" or not s["score_delta_known"]
                                    for s in steps)
                or (reason == "step_limit" and len(steps) != self.horizon)):
            return 0
        if episode_id in self.episodes:
            return 0
        states, rows = {}, []
        return_to_go = 0.0
        for step in reversed(steps):
            snapshot = step["outcome_snapshot"]
            key = snapshot["key"]
            if key and key not in states:
                states[key] = {k: snapshot[k] for k in
                               ("room", "inventory", "location_id", "score", "peak_score")}
            return_to_go = step["score_delta"] + self.gamma * return_to_go
            rows.append({"step": step["step"] + 1, "key": key, "action": step["action"],
                         "source": step["action_source"], "remaining": snapshot["remaining"],
                         "feedback": snapshot["feedback"], "score_delta": step["score_delta"],
                         "next_location": step["next_state"]["location_id"],
                         "next_score": step["next_state"]["score"], "return": return_to_go})
        episode = {"episode_id": episode_id, "protocol": self.protocol, "reason": reason,
                   "final_score": final_score, "states": states, "steps": rows[::-1]}
        atomic_json(self.directory / (_digest(episode_id) + ".json"), episode)
        self._index(episode)
        return len(rows)

    def estimate_action_values(self, state, actions, *, top_k, min_similarity):
        current = self.snapshot(state)
        best = {}
        for row in self.by_state.get(current["key"], ()):
            if row["episode_id"] == self.active_episode:
                continue
            budget_distance = abs(row["remaining"] - current["remaining"]) / max(1, self.horizon)
            similarity = (0.8 + 0.1 * (current["feedback"] == row["feedback"])
                          + 0.1 * max(0.0, 1.0 - budget_distance))
            if similarity < min_similarity:
                continue
            # Count at most one observation per episode and action. A long
            # oscillation should not drown out a different observed outcome.
            pair = (_text(row["action"]), row["episode_id"])
            if pair not in best or similarity > best[pair][0]:
                best[pair] = (similarity, row)
        neighbors = sorted(best.values(), key=lambda match: (
            -match[0], match[1]["episode_id"], match[1]["step"]))[:top_k]
        if not neighbors:
            return {"state_value": None, "samples": 0, "actions": {}}
        baseline = sum(row["return"] for _, row in neighbors) / len(neighbors)
        estimates = {}
        for action in actions:
            key = _text(action)
            matches = [(sim, row) for sim, row in neighbors if _text(row["action"]) == key]
            if not matches:
                continue
            similarity, example = matches[0]
            count = len(matches)
            q = sum(row["return"] for _, row in matches) / count
            confidence = count / (count + 3.0)
            estimates[key] = {"action": action, "q_value": q, "raw_advantage": q - baseline,
                              "advantage": confidence * (q - baseline), "confidence": confidence,
                              "sample_count": count, "best_similarity": similarity,
                              "source": "concrete_outcomes", "example": {
                                  k: example[k] for k in ("episode_id", "step", "score_delta",
                                                          "next_location", "next_score", "return", "source")}}
        return {"state_value": baseline, "samples": len(neighbors), "actions": estimates}
