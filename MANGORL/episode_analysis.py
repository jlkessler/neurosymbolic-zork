"""Offline measurements shared by trial summaries and traversal figures."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from .room_identity import RoomIndex, observed_room_title, room_id_key
import json
import re
from pathlib import Path
from statistics import mean, median


ITEMS = {
    "painting": r"(?:beautiful |valuable |large )*painting|art",
    "egg": r"(?:broken |jewel[- ]encrusted |jeweled |golden |clockwork )*egg",
    "coins": r"(?:(?:leather |old )*bag of (?:gold |silver |old )*)?coins",
    "bar": r"(?:platinum )?bar(?: of platinum)?",
    "canary": r"(?:broken |golden |clockwork )*canary",
    "chalice": r"(?:silver )?chalice",
}
ITEM_LABELS = {"painting": "Painting", "egg": "Egg", "coins": "Bag of coins", "bar": "Platinum bar",
               "canary": "Canary", "chalice": "Chalice"}
OBJECTIVES = {f"{item}_{verb}": f"{label} {verb}" for item, label in ITEM_LABELS.items()
              for verb in ("taken", "deposited")}
ROOM_MILESTONES = {"cellar": "cellar_reached", "treasure room": "treasure_room_reached",
                   "maze": "maze_reached", "kitchen": "kitchen_reached", "cyclops room": "cyclops_room_reached"}
OBJECTIVES.update({key: room.title() + " reached" for room, key in ROOM_MILESTONES.items()})
OBJECTIVES.update(troll_killed="Troll killed", return_route_found="Return route to surface demonstrated",
                  returned_home="Living Room reached after underground exploration",
                  cyclops_fled="Cyclops fled")
UNDERGROUND = {"cellar", "maze", "troll room", "treasure room", "cyclops room", "gallery", "studio",
               "torch room", "round room", "east of chasm", "east chasm", "reservoir south", "dam base",
               "altar", "temple", "up a chimney", "north-south passage", "east-west passage", "loud room",
               "strange passage", "grating room", "below the grating"}
SURFACE = {"kitchen", "living room", "attic", "forest", "forest path", "up a tree", "clearing",
           "grating clearing", "forest clearing", "behind house", "west house", "north house",
           "south house", "west of house", "north of house", "south of house"}
DEATH = re.compile(r"\byou have died\b|\byou are dead\b|\byou (?:have been|are) slain\b", re.I)
TROLL_DEATH = re.compile(
    r"\btroll (?:breathes his last|dies\b|is (?:dead|killed|slain)\b|lies dead\b)"
    r"|\byou (?:have )?(?:killed|slain) the troll\b", re.I)


def item_id(text):
    name = re.sub(r"^(?:a|an|the|some)\s+", "", text.strip().casefold())
    return next((item for item, pattern in ITEMS.items() if re.fullmatch(pattern, name)), None)


def _room(label):
    name = re.sub(r"\s*\(room #?[^)]+\)$", "", label or "").strip().casefold()
    return {"living": "living room", "troll": "troll room", "loud": "loud room",
            "treasure": "treasure room"}.get(name, name)


def objectives(steps):
    """First verified command number for each achieved milestone."""
    first = {}
    excursion = False
    last_step = None
    for row in steps:
        n = row["step"]
        before, after = _room(row["before_label"]), _room(row["after_label"])
        action = " ".join(row["action"].casefold().split())
        result = row["result"].strip()
        if last_step is None or n != last_step + 1:
            excursion = False
        last_step = n
        excursion = (excursion or before in UNDERGROUND or after in UNDERGROUND) and not row["relocation"]
        room_objective = ROOM_MILESTONES.get(after)
        if room_objective and not row["relocation"]:
            first.setdefault(room_objective, n)
        if excursion and after in SURFACE and before != after:
            first.setdefault("return_route_found", n)
            if after == "living room":
                first.setdefault("returned_home", n)
                excursion = False
        if TROLL_DEATH.search(result):
            first.setdefault("troll_killed", n)
        if re.search(r"\bcyclops\b[^.!?]*\bflees the room\b", result, re.I):
            first.setdefault("cyclops_fled", n)
        take = re.fullmatch(r"(?:take|get|pick up) (.+?)(?: (?:from|out of|off) .+)?", action)
        deposit = re.fullmatch(r"(?:put|place|insert|drop) (.+?) (?:in|into|inside) (?:the )?(?:trophy )?case", action)
        if take:
            subject = take[1]
            gained = {item_id(name) for name in row.get("inventory_added", [])}
            item = item_id(subject)
            for found in gained - {None}:
                first.setdefault(f"{found}_taken", n)
            if item and (item in gained or re.match(r"^Taken\b", result, re.I)):
                first.setdefault(f"{item}_taken", n)
            for found in _itemized_results(result, "taken"):
                first.setdefault(f"{found}_taken", n)
        if deposit and before == "living room":
            item = item_id(deposit[1])
            removed = {item_id(name) for name in row.get("inventory_removed", [])}
            if deposit[1] in {"bag", "the bag"} and "coins" in removed:
                item = "coins"
            if deposit[1] == "it" and len(removed) == 1 and not row["relocation"]:
                item = next(iter(removed))
            if item and re.match(r"^(?:Done\.|Placed\.|You put .+ (?:in|into) (?:the )?(?:trophy )?case\b)", result, re.I):
                first.setdefault(f"{item}_deposited", n)
            for found in _itemized_results(result, "done|placed"):
                first.setdefault(f"{found}_deposited", n)
    return first


def _itemized_results(result, verbs):
    for match in re.finditer(rf"(?im)^[ \t]*([^:\n]+):[ \t]*(?:{verbs})\b", result):
        found = item_id(match[1])
        if found:
            yield found


def objective_summary(episodes):
    """Denominator is evaluated episodes with complete trace coverage."""
    covered = {ep: values for ep, values in episodes.items() if values is not None}
    stats = {}
    for key, label in OBJECTIVES.items():
        hits = {str(ep): values[key] for ep, values in covered.items() if key in values}
        stats[key] = {"label": label, "episodes_met": len(hits),
                      "rate": len(hits) / len(covered) if covered else None,
                      "median_first_step": median(hits.values()) if hits else None,
                      "first_step_by_episode": hits}
    return {"evaluated_episodes": len(episodes), "episodes_with_evidence": len(covered),
            "episodes_without_evidence": sorted(episodes.keys() - covered.keys()),
            "distinct_by_episode": {str(ep): len(values) for ep, values in covered.items()},
            "mean_distinct_objectives": mean(map(len, covered.values())) if covered else None,
            "milestones": stats}


def transition(number, before, after, before_label, after_label, action, source, result,
               score, delta, inventory_before=None, inventory_after=None):
    row = {"step": number, "before": before, "after": after, "before_label": before_label,
           "after_label": after_label, "action": action, "source": source, "score": score,
           "delta": delta, "result": result, "relocation": bool(DEATH.search(result))}
    if inventory_before is not None and inventory_after is not None:
        row["inventory_added"] = sorted(set(inventory_after) - set(inventory_before))
        row["inventory_removed"] = sorted(set(inventory_before) - set(inventory_after))
    return row


def _console_trace(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"^========== STEP (\d+)/\d+ ==========\s*$", text, flags=re.M)
    start = re.search(r"^Starting location:[ \t]*([^\n]+)", parts[0], re.M)
    rows = []
    for number, block in zip(parts[1::2], parts[2::2]):
        def field(name):
            match = re.search(rf"^{name}:[ \t]*([^\n]*)", block, re.M)
            return match[1].strip() if match else None
        before, after, command = field("Location"), field("New location"), field("Action")
        if before is None or after is None or not command:
            continue
        result = re.search(r"^Observation: ?(.*?)\nNew location:", block, re.M | re.S)
        score = re.match(r"(-?[\d.]+) \(delta: (-?[\d.]+|None)\)", field("Score") or "")
        rows.append(transition(int(number), before, after, before, after, command,
                               field("Action source") or "unknown", result[1].strip() if result else "",
                               float(score[1]) if score else None,
                               float(score[2]) if score and score[2] != "None" else None))
    # A finished console transcript gives archived runs an independent endpoint.
    # A trailing unfinished command will fail the consecutive-step coverage check.
    finished = re.search(r"^Episode finished: .*reason=\w+", text, re.M)
    end_step = (int(parts[-2]) if len(parts) > 1 else 0) if finished else None
    return start[1].strip() if start else None, rows, end_step


@lru_cache(maxsize=2)
def _archive_index(map_path, map_version, critic_path, critic_version):
    # Cache only small indexes, never entire experience banks or chat transcripts.
    data = json.loads(map_path.read_text(encoding="utf-8"))
    rooms = RoomIndex(data["nodes"])
    transitions = defaultdict(dict)
    for row in data.get("graph", {}).get("action_experiences", []):
        if row.get("trajectory_id"):
            transitions[row["trajectory_id"]][row.get("step")] = (
                row.get("action"), row.get("source"), row.get("destination"))
    if critic_version is not None:
        critic = json.loads(critic_path.read_text(encoding="utf-8"))
        for row in critic.get("td_memory", []):
            if row.get("episode_id") not in transitions:
                continue
            for state in (row.get("state") or {}, row.get("next_state") or {}):
                key = rooms.machine(state.get("location_id"))
                if key:
                    label = rooms.labels[key]
                    short = "" if label == f"Room #{key[5:]}" else label
                    title = observed_room_title(short, state.get("observation"))
                    if title:
                        rooms.labels[key] = title
    return rooms, transitions


def _restore_room_ids(rows, experience_path):
    """Recover an archived graph key only from the same episode, step and command."""
    by_number = {r["step"]: r for r in rows}
    known = defaultdict(set)
    samples = {}
    episodes = set()
    if experience_path.exists():
        with experience_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        break
                    raise
                n = item.get("step", -2) + 1
                if n in by_number and by_number[n]["action"] == item.get("action"):
                    samples[n] = (item.get("state") or {}, item.get("next_state") or {})
                    episodes.add(item.get("episode_id"))
    rooms, archived = None, {}
    if len(episodes) == 1:
        episode_id = episodes.pop()
        directory = experience_path.parent
        for parent in (directory / "memory", directory, directory.parent):
            path = next((parent / name for name in ("world_model.json", "world_map.json")
                         if (parent / name).is_file()), None)
            if path:
                critic = parent / "historian_td.json"
                index, trajectories = _archive_index(path, path.stat().st_mtime_ns, critic,
                    critic.stat().st_mtime_ns if critic.is_file() else None)
                if episode_id in trajectories:
                    rooms, archived = index, trajectories[episode_id]
                    break

    for row in rows:
        n = row["step"]
        record = archived.get(n)
        for j, side in enumerate(("before", "after")):
            label = row[side + "_label"]
            raw = samples.get(n, ({}, {}))[j].get("location_id")
            if record and record[0] == row["action"]:
                raw = record[j + 1]
            embedded = re.search(r"\(room #(\d+)\)$", label)
            key = (rooms.machine(raw) or rooms.by_name(label)) if rooms else None
            key = key or room_id_key(raw) or (room_id_key(embedded[1]) if embedded else None)
            row[side] = None
            if key:
                row[side] = "id:" + key[5:]
                if rooms and key in rooms.labels and (not label or not embedded):
                    row[side + "_label"] = rooms.labels[key]
                elif not label:
                    row[side + "_label"] = f"Room #{key[5:]}"
            elif str(raw or "").strip():
                row[side] = "key:" + str(raw)
            if row[side] and label:
                known[label].add(row[side])

    for prev, curr in zip(rows, rows[1:]):
        if prev["step"] + 1 == curr["step"] and (not prev["after"] or not curr["before"]):
            if prev["after_label"] == curr["before_label"] or not (prev["after_label"] and curr["before_label"]):
                key = prev["after"] or curr["before"]
                prev["after"] = curr["before"] = key
                title = prev["after_label"] or curr["before_label"]
                prev["after_label"] = curr["before_label"] = title
    for row in rows:
        for side in ("before", "after"):
            label = row[side + "_label"]
            candidates = known[label]
            ambiguous = rooms and len(rooms.names.get(label.strip().casefold(), ())) > 1
            if row[side] is None and label and not ambiguous and _room(label) != "maze" and len(candidates) <= 1:
                row[side] = next(iter(candidates), "label:" + _room(label))


def read_episode(directory, episode):
    directory = Path(directory)
    stem = f"episode_{episode:04d}"
    journal = directory / (stem + "_traversal.jsonl")
    if journal.exists() and journal.stat().st_size:
        with journal.open(encoding="utf-8") as file:
            records = []
            for line in file:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if line.endswith("\n"):
                        raise
                    break  # interrupted append
        if records:
            return {"episode": episode, "identity": "Jericho room IDs", **records[0], "steps": records[1:]}
    console = directory / (stem + ".txt")
    if not console.exists():
        return None
    start_label, rows, end_step = _console_trace(console)
    _restore_room_ids(rows, directory / (stem + "_experience.jsonl"))
    identity = ("Recorded room IDs" if rows and all((r[s] or "").startswith("id:") for r in rows for s in ("before", "after"))
                else "Historical room evidence; unidentified endpoints omitted from graph")
    if rows and rows[0]["step"] == 1:
        start, start_label = rows[0]["before"], rows[0]["before_label"]
    else:
        start = "label:" + _room(start_label) if start_label else None
    return {"episode": episode, "identity": identity, "start": start, "start_label": start_label,
            "console_end_step": end_step, "steps": rows}


def measured_objectives(directory, metric):
    if metric.get("objectives_met") is not None:
        return metric["objectives_met"]
    trace = read_episode(directory, metric["episode"])
    if has_complete_steps(trace, metric.get("steps")):
        return objectives(trace["steps"])
    return None


def has_complete_steps(trace, expected):
    return (trace is not None and expected is not None and len(trace["steps"]) == expected
            and all(row["step"] == n for n, row in enumerate(trace["steps"], 1)))
