"""Conservative action checks grounded in the current room and inventory text.

These checks use observations, not hidden room knowledge or simulated outcomes.
They apply to both the Actor and symbolic navigation, before policy scoring.
"""
from __future__ import annotations

import re


def snapshot_sections(context):
    text = str(context or "")
    if "CURRENT ROOM (look):\n" not in text:
        return text, ""
    room = text.split("CURRENT ROOM (look):\n", 1)[1]
    room, _, inventory = room.partition("\n\nCURRENT INVENTORY:\n")
    return room, inventory


def inventory_from_text(text):
    """Return observed items, or None when the inventory format is unknown."""
    if re.search(r"empty[- ]handed|carrying nothing", text, re.I):
        return []
    match = re.search(r"you are carrying:\s*\n(.*)", text, re.I | re.S)
    if not match:
        return None
    items = []
    for line in match[1].splitlines():
        item = re.sub(r"\([^)]*\)", "", line).strip().rstrip(".")
        item = re.sub(r"^(?:a|an|the|some)\s+", "", item, flags=re.I)
        if item and not item.casefold().endswith("contains:"):
            items.append(item)
    return items


def ground_inventory(historian, items):
    """Keep model classifications only for objects present in the snapshot."""
    if items is None:
        return
    historian.agent_state["inventory_items"] = list(items)
    for field in ("treasure_candidates", "carried_treasures", "ordinary_items",
                  "useful_items", "uncertain_items"):
        historian.agent_state[field] = [
            item for item in historian.agent_state.get(field, [])
            if any(historian._items_match(item, present) for present in items)
        ]


def guard_actions(actions, context):
    """Return permitted commands and an auditable reason for every exclusion.

    Preserve the sole carried, active light. In observed darkness, prioritize
    a legal command restoring a carried light if one is available. No light
    requirement is invented for an unexplored room.
    """
    room, inventory = snapshot_sections(context)
    light_words = r"\b(?:lantern|lamp|torch|candle|candles)\b"
    lights, active = set(), set()
    for line in inventory.splitlines():
        names = re.findall(light_words, line.casefold())
        lights.update(names)
        if re.search(r"\((?:providing light|lit|lighted|on)\)", line, re.I):
            active.update(names)
    dark = bool(re.search(r"pitch (?:black|dark)|too dark to see|in (?:complete|total) darkness", room, re.I))
    restore = [action for action in actions if
               re.match(r"^(?:turn on|switch on|light|ignite)\s", action, re.I)
               and any(re.search(r"\b" + re.escape(light) + r"\b", action, re.I)
                       for light in lights)]
    rejected = {}
    for action in actions:
        command = " ".join(action.casefold().split())
        if dark and restore and action not in restore:
            rejected[action] = "Restore a carried light before another action in observed darkness."
            continue
        if len(active) != 1:
            continue
        light = next(iter(active))
        mentions_light = bool(re.search(r"\b" + re.escape(light) + r"\b", command))
        extinguish = re.match(r"^(?:turn off|switch off|put out|extinguish|blow out|douse)\b", command)
        discard = re.match(r"^(?:drop|put down|give|throw|break|destroy|put)\s", command)
        if ((extinguish and (mentions_light or re.search(r"\ball\b", command))) or
                (discard and not extinguish and
                 (mentions_light or re.search(r"\ball\b", command)))):
            rejected[action] = "Preserve the only carried light currently providing illumination."
    return [action for action in actions if action not in rejected], rejected
