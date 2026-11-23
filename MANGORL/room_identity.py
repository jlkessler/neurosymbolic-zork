"""Separate ROM-local object numbers, historical graph keys, and readable titles."""
from collections import defaultdict
import re

from .action_guards import snapshot_sections


def room_id_key(value):
    number = str(value).removeprefix("room:")
    return f"room:{int(number)}" if number.isascii() and number.isdecimal() else None


def observed_room_title(short_name, state_context):
    text = str(state_context or "").replace("\r\n", "\n")
    if "CURRENT ROOM (look):\n" not in text:
        return None
    room, _ = snapshot_sections(text)
    title = next((line.strip() for line in room.splitlines() if line.strip()), "")
    short = str(short_name or "").strip().casefold()
    if (not title or len(title) > 90 or len(title.split()) > 12
            or not title[0].isupper() or not re.fullmatch(r"[\w '\u2019()/,-]+", title)
            or title.casefold() in {"darkness", "dark", "it is pitch black", "you are carrying"}
            or (short and title.casefold() != short and not title.casefold().startswith(short + " "))):
        return None
    return title


class RoomIndex:
    """Exact historical machine keys may be blank. Display-only blanks never resolve."""

    def __init__(self, nodes):
        self.keys = {}
        self.names = defaultdict(set)
        self.labels = {}
        for node in nodes:
            old = node["id"]
            key = room_id_key(node.get("room_id")) or (
                room_id_key(old) if str(old).startswith("room:") else None)
            if key is None:
                continue
            for machine_key in [old, key, *node.get("legacy_keys", [])]:
                if machine_key in self.keys and self.keys[machine_key] != key:
                    raise ValueError(f"Conflicting room IDs for historical graph key {machine_key!r}")
                self.keys[machine_key] = key
            title = node.get("observed_title") or str(node.get("label") or "").strip()
            self.labels[key] = title or f"Room #{key[5:]}"
            for alias in [title, node.get("label"), *node.get("aliases", [])]:
                if alias and alias.strip():
                    self.names[alias.strip().casefold()].add(key)

    def machine(self, value):
        if value is None:
            return None
        return self.keys.get(value) or self.by_name(value)

    def by_name(self, label):
        name = str(label or "").strip().casefold()
        match = re.search(r"\(room #(\d+)\)$", name)
        if match:
            return self.keys.get(room_id_key(match[1]))
        matches = self.names.get(name, ())
        return next(iter(matches)) if len(matches) == 1 else None
