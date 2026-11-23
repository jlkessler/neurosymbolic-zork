"""Copy a saved Zork memory bundle into a new directory with ROM-local room IDs."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from MANGORL.outcome_memory import outcome_state_key
from MANGORL.world_map import WorldMap


def write_json(path, data):
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)


def remap_field(row, field, index, label=None):
    if field not in row or row[field] is None:
        return
    original = row[field]
    key = index.machine(original)
    if key is None:
        row["legacy_" + field] = original
    row[field] = key
    if key and label:
        row[label] = index.labels[key]


def remap_historian(memory, index):
    def records(value):
        if isinstance(value, list):
            for row in value:
                records(row)
        elif isinstance(value, dict):
            for field, label in (("node", "location"), ("start_node", "start_location"),
                                 ("before_node", "before_location"), ("after_node", "after_location"),
                                 ("target_node", "target_label"), ("destination", None)):
                remap_field(value, field, index, label)
            if "visited_nodes" in value:
                value["visited_nodes"] = [index.machine(n) for n in value["visited_nodes"]]
            for child in value.values():
                if isinstance(child, (dict, list)):
                    records(child)

    records(memory)
    for table in ("score_profiles", "repeat_failures", "blocked_actions", "productive_locations"):
        updated = {}
        for old, row in memory.get(table, {}).items():
            if table in {"score_profiles", "repeat_failures"}:
                key = json.dumps([row.get("node"), row["action"]], separators=(",", ":"))
                resolved = row.get("node") is not None
            else:
                key = index.machine(old)
                resolved = key is not None
            if not resolved:
                memory.setdefault("unresolved_room_evidence", {}).setdefault(table, {})[old] = row
            elif key in updated:
                raise ValueError(f"Conflicting {table} entries for {key!r}; inspect the originals.")
            else:
                updated[key] = row
        memory[table] = updated
    memory["room_identity_version"] = 2


def migrate_outcome(episode, index):
    states, keys = {}, {}
    for old, state in episode["states"].items():
        remap_field(state, "location_id", index)
        key = outcome_state_key(state["location_id"], state["room"], state["inventory"],
                                state["score"], state["peak_score"])
        keys[old] = key
        if key:
            states[key] = state
        else:
            episode.setdefault("unresolved_states", {})[old] = state
    episode["states"] = states
    for row in episode["steps"]:
        if row["key"] is not None:
            old = row["key"]
            row["key"] = keys[old]
            if row["key"] is None:
                row["legacy_key"] = old
        remap_field(row, "next_location", index)


def convert(source, output, rom):
    maps = [source / name for name in ("world_model.json", "world_map.json") if (source / name).is_file()]
    if len(maps) != 1:
        raise ValueError("Select a memory bundle containing exactly one world_model.json or world_map.json.")
    model = WorldMap(maps[0])
    model.load()
    hashes = set()
    for metadata in source.glob("*/trial_metadata.txt"):
        text = metadata.read_text(encoding="utf-8")
        if "{" in text:
            settings, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
            if Path(settings.get("map_path", "")).name == maps[0].name and settings.get("rom_sha256"):
                hashes.add(settings["rom_sha256"])
    if model.graph.graph.get("rom_sha256"):
        hashes.add(model.graph.graph["rom_sha256"])
    if len(hashes) > 1:
        raise ValueError("The archive contains conflicting ROM signatures for this map.")
    if hashes:
        model.graph.graph["rom_sha256"] = hashes.pop()
    if rom:
        if not rom.is_file():
            raise FileNotFoundError(rom)
        model.bind_rom(rom)
    index = model.legacy_rooms
    critic_file = source / "historian_td.json"
    critic = json.loads(critic_file.read_text(encoding="utf-8")) if critic_file.exists() else None
    if critic:
        for row in critic.get("td_memory", []):
            for field in ("state", "next_state"):
                state = row.get(field) or {}
                model.observe_room_title(index.machine(state.get("location_id")), state.get("observation"))
    for key in index.labels:
        index.labels[key] = model.display_location(key)
    for row in model.graph.graph["action_experiences"]:
        for field in ("source", "destination"):
            remap_field(row, field, index, field + "_label")
    model.save_path = output / maps[0].name
    model.save()

    def remap_states(row):
        for field in ("state", "next_state"):
            if row.get(field):
                remap_field(row[field], "location_id", index, "location")

    if critic is not None:
        for row in critic.get("td_memory", []):
            remap_states(row)
        weights = {}
        for feature, weight in critic.get("weights", {}).items():
            if feature.startswith("room_id:"):
                key = index.machine(feature[len("room_id:"):])
                if not key:
                    critic.setdefault("unresolved_weights", {})[feature] = weight
                    continue
                feature = "room_id:" + key
            if feature in weights:
                raise ValueError(f"Conflicting critic weights for {feature!r}")
            weights[feature] = weight
        critic["weights"] = weights
        write_json(output / critic_file.name, critic)
    historian = source / "historian.json"
    if historian.exists():
        data = json.loads(historian.read_text(encoding="utf-8"))
        remap_historian(data, index)
        write_json(output / historian.name, data)
    actor = source / "actor_experiences.jsonl"
    if actor.exists():
        with actor.open(encoding="utf-8") as src, (output / actor.name).open("w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    row = json.loads(line)
                    remap_states(row)
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    for path in source.glob("actor_experiences.jsonl.outcomes/*/*.json"):
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        episode = json.loads(path.read_text(encoding="utf-8"))
        protocol = episode["protocol"]
        if protocol.get("map_navigation") and not model.graph.graph.get("rom_sha256"):
            raise ValueError("Supply --rom to associate the saved outcome bank with its story file.")
        if protocol.get("map_navigation") and protocol.get("rom") == model.graph.graph.get("rom_sha256"):
            migrate_outcome(episode, index)
            write_json(target, episode)
        else:
            shutil.copyfile(path, target)
    if (source / "configurations.json").exists():
        shutil.copyfile(source / "configurations.json", output / "configurations.json")


def migrate(source, destination, rom=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source in destination.parents:
        raise ValueError("Choose a new destination outside the source bundle.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "memory"
        stage.mkdir()
        convert(source, stage, Path(rom) if rom else None)
        stage.rename(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rom", type=Path, help="Matching story file, to bind room IDs to its SHA-256")
    args = parser.parse_args()
    try:
        print(f"Migrated memories to {migrate(args.source, args.destination, args.rom)}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
