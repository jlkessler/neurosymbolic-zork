import hashlib
import json

import pytest

from MANGORL.episode_analysis import objectives, read_episode
from MANGORL.historian import Historian
from MANGORL.outcome_memory import OutcomeMemory
from MANGORL.world_map import WorldMap
from migrate_room_ids import migrate


def snapshot(title, facts):
    return f"CURRENT ROOM (look):\n{title}\n{facts}\n\nCURRENT INVENTORY:\nYou are empty-handed."


def test_names_and_legality_do_not_transfer_movement_reward(tmp_path):
    model = WorldMap(tmp_path / "map.json")
    cyclops = model.start_episode("", 185)
    assert model.display_location(cyclops) == "Room #185"
    assert not model.observe_room_title(cyclops, "The cyclops flees!\nTreasure Room")
    assert not model.observe_room_title(cyclops, snapshot("Darkness", "It is pitch black."))
    solved = snapshot("Cyclops Room", "The east wall has a large opening.")
    guarded = snapshot("Cyclops Room", "The cyclops blocks the staircase.")
    treasure = model.add_standard_location("Treasure ", 190)
    cellar = model.add_standard_location("Cellar", 12)
    model.add_transition(cyclops, treasure, "U")
    model.add_transition(cellar, cyclops, "up")
    model.bind_rom("z-machine-games-master/jericho-game-suite/zork1.z5")
    for action, before, after, dest, delta in (
        ("Odysseus", guarded, solved, cyclops, 0),
        ("up", solved, snapshot("Treasure Room", "A chamber."), treasure, 25),
    ):
        model.record_experience(source=cyclops, destination=dest, action=action,
                                before_context=before, after_context=after, score_before=55, score_after=55+delta)
    assert model.display_location(cyclops) == "Cyclops Room"
    assert model.display_location(treasure) == "Treasure Room"
    menu = ["odysse", "odyssex", "up"]

    def recall(room, state, blocked=()):
        return {r["action"]: r["score_delta"] for r in model.retrieve_action_experiences(
            current_node=room, state_context=state, valid_actions=menu, blocked_edges=blocked)}

    assert recall(cyclops, guarded) == {"odysse": 0}
    assert recall(cyclops, solved)["up"] == 25
    assert "up" not in recall(cyclops, solved, {(cyclops, treasure, 0)})
    assert recall(cellar, solved) == {}
    assert not model.get_reachable_routes(cellar, valid_actions=["look", "up"],
                                          blocked_edges={(cellar, cyclops, 0)})
    hist = Historian(tmp_path / "hist.json")
    hist.start_episode(cellar, "Cellar", 55)
    hist.memory["score_profiles"] = {"up": dict(node=cyclops, location="Cyclops Room", action="up",
                                                attempts=1, samples=1, total_gain=25)}
    goal = hist.choose_goal(current_node=cellar, current_location="Cellar", valid_actions=menu,
                            known_routes=[], supported_score_moves=set())
    assert goal["kind"] not in {"navigate_score_goal", "score_action"}
    assert "+25" not in hist.build_guidance(cellar, "Cellar", score_goal=goal)

    profile = hist.memory["score_profiles"].pop("up")
    hist.memory["score_profiles"]['["room:185","up"]'] = profile
    hist.cycle_recovery(cyclops, menu, state_context=guarded)
    hist.observe_step(before_node=cyclops, before_location="Cyclops Room", action="up",
                      after_node=cyclops, after_location="Cyclops Room", score_before=55, score_after=55)
    args = dict(current_node=cyclops, current_location="Cyclops Room", valid_actions=menu,
                known_routes=[], supported_score_moves={"up"})
    assert hist.choose_goal(**args)["kind"] != "score_action"
    hist.cycle_recovery(cyclops, menu, state_context=solved)
    assert hist.choose_goal(**args)["suggested_action"] == "up"
    hist.observe_step(before_node=cyclops, before_location="Cyclops Room", action="up",
                      after_node=treasure, after_location="Treasure Room", score_before=55, score_after=80)
    hist.cycle_recovery(cyclops, menu, state_context=guarded)
    assert hist.choose_goal(**args)["kind"] != "score_action"


def test_archive_keeps_unknown_commands_without_inventing_movement(tmp_path):
    (tmp_path / "episode_0001.txt").write_text(
        "========== STEP 1/2 ==========\nLocation: \nAction: odysse\nObservation: "
        "The cyclops, hearing the name, flees the room.\nNew location: \nScore: 55 (delta: 0)\n"
        "========== STEP 2/2 ==========\nLocation: \nAction: up\n", encoding="utf-8")
    rows = read_episode(tmp_path, 1)["steps"]
    assert len(rows) == 1 and rows[0]["before"] is rows[0]["after"] is None
    assert objectives(rows) == {"cyclops_fled": 1}


def test_migration_preserves_records_reviews_and_outcome_lookup(tmp_path):
    original = tmp_path / "old"
    original.mkdir()
    rom = tmp_path / "game.z5"
    rom.write_bytes(b"test story")
    rom_hash = hashlib.sha256(rom.read_bytes()).hexdigest()
    nodes = [{"id": "", "label": "", "room_id": 185, "node_type": "location"},
             {"id": "Treasure ", "label": "Treasure ", "room_id": 190, "node_type": "location"}]
    graph = dict(directed=True, multigraph=True, nodes=nodes, edges=[
        dict(source="", target="Treasure ", key=0, actions=["up"], cost=1)],
        graph=dict(action_experiences=[dict(source="", destination="Treasure ", action="up", score_delta=25)],
                   action_experience_schema=1))
    state = dict(location_id="", location="", observation=snapshot("Cyclops Room", "An opening."),
                 score=55, peak_score=55)
    after = dict(state, location_id="Treasure ", observation=snapshot("Treasure Room", "A chamber."))
    row = dict(state=state, next_state=after, action="up")
    profile = dict(node="", location="", action="up", total_gain=25)
    files = {"world_model.json": graph, "historian.json": dict(
        score_profiles={"old": profile}, episodes=[dict(historian_review="Verbatim episode review.")]),
        "historian_td.json": dict(td_memory=[row], weights={"room_id:Treasure": 0.4})}
    for name, data in files.items():
        (original / name).write_text(json.dumps(data))
    (original / "actor_experiences.jsonl").write_text(json.dumps(row) + "\n")
    bank = OutcomeMemory(original / "actor_experiences.jsonl.outcomes", gamma=1, horizon=1, target=80,
                         rom=rom_hash, map_navigation=True)
    bank.directory.mkdir(parents=True)
    (bank.directory / "episode.json").write_text(json.dumps(dict(protocol=bank.protocol, episode_id="ep",
        states={"old": dict(room="Cyclops Room\nAn opening.", inventory="You are empty-handed.",
                            location_id="", score=55, peak_score=55)},
        steps=[dict(key="old", action="up", next_location="Treasure ", score_delta=25)])))
    baseline = {p: p.read_bytes() for p in original.rglob("*") if p.is_file()}
    out = migrate(original, tmp_path / "new", rom)
    assert all(p.read_bytes() == contents for p, contents in baseline.items())
    h = json.loads((out / "historian.json").read_text())
    assert h["score_profiles"]['["room:185","up"]']["location"] == "Cyclops Room"
    assert h["episodes"][0]["historian_review"] == "Verbatim episode review."
    critic = json.loads((out / "historian_td.json").read_text())
    assert critic["weights"] == {"room_id:room:190": 0.4}
    assert critic["td_memory"][0]["state"]["location_id"] == "room:185"
    actor = json.loads((out / "actor_experiences.jsonl").read_text())
    memory = OutcomeMemory(out / "actor_experiences.jsonl.outcomes", gamma=1, horizon=1, target=80,
                           rom=rom_hash, map_navigation=True)
    key = memory.snapshot(actor["state"])["key"]
    assert memory.by_state[key][0]["next_location"] == "room:190"
    with pytest.raises(ValueError):
        migrate(original, out, rom)
