import json
from types import SimpleNamespace

import pytest

from MANGORL.walkthrough import run_walkthrough
from MANGORL.cognitive_map import CognitiveMap


class FakeEnv:
    def __init__(self, path, seed):
        self.index = 0
        self.closed = False

    def reset(self):
        return "Start", {"score": 0}

    def get_walkthrough(self):
        return ["take lamp", "north", "east"]

    def get_player_location(self):
        return SimpleNamespace(name="Maze", num=1 if self.index < 2 else self.index)

    def step(self, action):
        self.index += 1
        return "Observed", 1, False, {"score": self.index}

    def close(self):
        self.closed = True


def test_replay_preserves_interactions_room_ids_and_existing_map(tmp_path):
    path = tmp_path / "world_model.json"
    original = CognitiveMap(path)
    original.add_standard_location("Existing", 99)
    original.save()
    env = FakeEnv("rom", 0)
    run_walkthrough("rom", path, env_factory=lambda *a, **k: env)
    loaded = CognitiveMap(path)
    loaded.load()
    assert loaded.resolve_location("Existing") == "room:99"
    assert len(loaded.graph.edges) == 2
    replay = loaded.graph.graph["walkthrough"]
    assert [s["action"] for s in replay["steps"]] == env.get_walkthrough()
    assert replay["steps"][0]["source"] == replay["steps"][0]["destination"]
    assert replay["completed"]
    experiences = loaded.graph.graph["action_experiences"]
    assert [row["action"] for row in experiences] == env.get_walkthrough()
    assert all("before_features" in row and "score_delta" in row for row in experiences)
    assert env.closed


def test_world_model_retrieval_treats_reference_and_actor_observations_equally(tmp_path):
    model = CognitiveMap(tmp_path / "map.json")
    node = model.add_standard_location("Living Room", 1)
    context = ("LAST ACTION RESULT:\nDone.\n\nCURRENT ROOM (look):\n"
               "Living Room\nA closed trophy case is here.\n\nCURRENT INVENTORY:\n"
               "You are carrying:\nA brass lantern")
    for action, provenance, delta in (
        ("open trophy case", "reference_replay", 0),
        ("put egg in case", "smart_actor", 5),
    ):
        model.record_experience(
            source=node, action=action, destination=node,
            before_context=context, after_context="Done.",
            score_before=0, score_after=delta, trajectory_id=provenance,
            step=1, provenance=provenance,
        )

    rows = model.retrieve_action_experiences(
        current_node=node, state_context=context,
        valid_actions=["look", "open trophy case", "put egg in case"],
    )
    assert [row["action"] for row in rows] == ["put egg in case", "open trophy case"]
    assert {row["provenance"] for row in rows} == {"reference_replay", "smart_actor"}
    guidance = model.format_experience_guidance(rows)
    assert "put egg in case" in guidance and "open trophy case" in guidance
    assert "reference" not in guidance and "smart_actor" not in guidance


def test_terminal_relocation_is_not_a_route(tmp_path):
    env = FakeEnv("rom", 0)
    env.step = lambda action: ("Dead", 0, True, {"score": 0})
    env.get_player_location = lambda: SimpleNamespace(name="Room", num=env.index)
    path = tmp_path / "map.json"
    run_walkthrough("rom", path, env_factory=lambda *a, **k: env)
    data = json.loads(path.read_text())
    assert not data["edges"]
    assert not data["graph"]["walkthrough"]["completed"]
    assert data["graph"]["walkthrough"]["termination_reason"] == "game_over"
    assert env.closed


def test_missing_walkthrough_leaves_file_untouched(tmp_path):
    env = FakeEnv("rom", 0)
    env.get_walkthrough = lambda: []
    path = tmp_path / "map.json"
    path.write_text("existing")
    with pytest.raises(ValueError, match="no Jericho"):
        run_walkthrough("rom", path, env_factory=lambda *a, **k: env)
    assert path.read_text() == "existing"
    assert env.closed
