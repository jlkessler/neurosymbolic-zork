import json
import sys
from types import SimpleNamespace as NS

import pytest

import main
from MANGORL.action_guards import guard_actions, ground_inventory, inventory_from_text
from MANGORL.actor_experience import ActorExperienceMemory
from MANGORL.historian import Historian
from MANGORL.policy_logits import IncompleteLabelScores, LogitsUnavailable
from MANGORL.run_recovery import recoverable_episode_error


def context(room="Cellar\nA dark cellar.", inventory="A brass lantern (providing light)\nA manual"):
    return ("LAST ACTION RESULT:\nIt is pitch black.\n\nCURRENT ROOM (look):\n" + room +
            "\n\nCURRENT INVENTORY:\nYou are carrying:\n" + inventory + "\n")


def test_light_preservation_applies_to_disposal_and_all_without_masking_other_items():
    actions = ["put out lantern", "put down lantern", "give all to troll", "throw lantern at manual",
               "put down manual", "north", "look"]
    permitted, rejected = guard_actions(actions, context())
    assert permitted == ["put down manual", "north", "look"]
    assert len(rejected) == 4


def test_observed_darkness_prioritizes_restoring_light_but_stale_darkness_does_not():
    actions = ["put down all", "north", "turn on lantern", "look"]
    permitted, _ = guard_actions(actions, context("It is pitch black.", "A brass lantern\nA manual"))
    assert permitted == ["turn on lantern"]
    permitted, _ = guard_actions(actions, context("Living Room\nA bright room.", "A brass lantern"))
    assert permitted == actions  # LAST ACTION RESULT is not the current room.


def test_snapshot_inventory_clears_stale_treasures_including_empty_inventory(tmp_path):
    historian = Historian(tmp_path / "h.json")
    historian.agent_state.update(inventory_items=["egg", "lantern"], carried_treasures=["egg"],
                                 treasure_candidates=["egg"], useful_items=["lantern"])
    assert inventory_from_text("You are empty-handed.\n") == []
    assert inventory_from_text("unrecognized output") is None
    items = inventory_from_text("You are carrying:\n  A brass lantern (providing light)\n  A manual\n")
    assert items == ["brass lantern", "manual"]
    ground_inventory(historian, items)
    assert historian.agent_state["carried_treasures"] == []
    assert historian.agent_state["useful_items"] == ["lantern"]
    ground_inventory(historian, [])
    assert historian.agent_state["useful_items"] == []


@pytest.mark.parametrize("bootstrap,expected,eligible", [(10, 11, True), (None, 2, False)])
def test_interrupted_returns_are_bootstrapped_or_excluded(tmp_path, bootstrap, expected, eligible):
    memory = ActorExperienceMemory(tmp_path / "actor.jsonl")
    memory.start_episode("interrupted")
    memory.record_step(state={}, action="take art", next_state={}, score_delta=2, actor_decision=True)
    memory.finish_episode(gamma=.9, termination_reason="episode_exception", bootstrap_value=bootstrap)
    record = memory.last_episode_records[0]
    assert record["discounted_return"] == pytest.approx(expected)
    assert record["truncated"] and record["episode_truncated"]
    assert record["training_eligible"] is eligible
    assert len(memory.retrieve_similar({}, min_similarity=0)) == int(eligible)
    assert len(ActorExperienceMemory(memory.path)) == int(eligible)


def test_legacy_interrupted_returns_are_not_retrieved(tmp_path):
    path = tmp_path / "actor.jsonl"
    record = {"record_type": "actor_experience", "state": {}, "action": "north",
              "discounted_return": 0, "termination_reason": "episode_exception"}
    path.write_text(json.dumps(record) + "\n")
    assert len(ActorExperienceMemory(path)) == 0
    assert json.loads(path.read_text()) == record  # Preserve original research data.


def test_historical_score_action_is_reassessed_by_actor_and_room_alias_is_learned(tmp_path, monkeypatch):
    class Game:
        def __init__(self, *args, **kwargs):
            self.moves, self.lantern = 0, False

        def reset(self):
            return "Living Room\nA room with a trapdoor.", {"score": 0, "moves": 0}

        def get_player_location(self):
            return NS(name="Living ", num=193)

        def get_state(self):
            return self.moves, self.lantern

        def set_state(self, state):
            self.moves, self.lantern = state

        def get_valid_actions(self, **kwargs):
            return ["down", "look"] if self.lantern else ["down", "take lantern"]

        def step(self, action):
            self.moves += 1
            if action == "look":
                return "Living Room\nA room with a trapdoor.", 0, False, {}
            if action == "inventory":
                text = "You are carrying:\nA brass lantern (providing light)\n" if self.lantern else "You are empty-handed."
                return text, 0, False, {}
            if action == "take lantern":
                self.lantern = True
                return "Taken.", 0, False, {"score": 0, "moves": self.moves}
            assert self.lantern, "Historical score replay entered without prerequisites."
            return "A passage.", 25, True, {"score": 25, "moves": self.moves}

        def close(self):
            pass

    class Agent:
        calls = []

        def reset_history(self):
            pass

        def apply_configuration(self, config):
            pass

        def add_history(self, **kwargs):
            pass

        def choose_decision(self, **kwargs):
            self.calls.append(kwargs)
            action = "take lantern" if "take lantern" in kwargs["valid_actions"] else "down"
            return {"action": action, "candidate_actions": [action], "base_logits": [0], "report": {}}

    hpath = tmp_path / "historian.json"
    historian = Historian(hpath)
    historian.memory["score_profiles"][json.dumps(["Living ", "down"], separators=(",", ":"))] = {
        "node": "Living ", "location": "Living ", "action": "down",
        "total_gain": 25, "attempts": 1, "samples": 1, "best_gain": 25,
    }
    historian.save()
    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=Game))
    monkeypatch.setattr(main, "create_agent", lambda *args, **kwargs: Agent())
    run = main.main(["--episodes", "1", "--max-steps", "2", "--target-score", "350", "--no-evolve", "--no-historian-review",
                     "--historian-path", str(hpath), "--map-path", str(tmp_path / "world.json"),
                     "--actor-experience-path", str(tmp_path / "actor.jsonl"),
                     "--historian-critic-path", str(tmp_path / "critic.json"), "--output-dir", str(tmp_path / "logs")])
    assert len(Agent.calls) == 2
    # A historical +25 without a grounded transition cannot authorize "down".
    assert Agent.calls[0]["goal"]["kind"] != "score_action"
    assert Agent.calls[0]["current_location"] == "Living Room"
    records = [json.loads(row) for row in (tmp_path / "actor.jsonl").read_text().splitlines()]
    assert [row["action"] for row in records] == ["take lantern", "down"]
    assert records[0]["next_state"]["inventory_items"] == ["brass lantern"]
    assert json.loads((run / "metrics_summary.json").read_text())["scores"] == [25]


def test_completion_budgets_have_separate_validated_options():
    args = main.parse_args([])
    assert args.historian_max_tokens == args.evolution_max_tokens == 4096
    with pytest.raises(SystemExit):
        main.parse_args(["--evolution-max-tokens", "0"])
    with pytest.raises(SystemExit):
        main.parse_args(["--max-consecutive-episode-failures", "0"])


@pytest.mark.parametrize("status,expected", [(400, False), (401, False), (403, False),
                                            (404, False), (408, True), (429, True), (503, True)])
def test_only_transient_sdk_statuses_are_recoverable(status, expected):
    import httpx
    import openai

    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test/v1/chat/completions"))
    upstream = openai.APIStatusError("provider error", response=response, body=None)
    wrapped = LogitsUnavailable("scoring failed")
    wrapped.__cause__ = upstream
    assert recoverable_episode_error(wrapped) is expected


def test_configuration_errors_and_programming_errors_are_not_recoverable():
    assert recoverable_episode_error(IncompleteLabelScores("Missing C"))
    assert not recoverable_episode_error(LogitsUnavailable("no token log probabilities"))
    assert not recoverable_episode_error(ValueError("500 provider error"))
    assert not recoverable_episode_error(TimeoutError("local operation failed"))
    exc = RuntimeError("recursive cause")
    exc.__cause__ = exc
    assert not recoverable_episode_error(exc)


def failure_run(tmp_path, monkeypatch, *, always_fail=False, exception=None, extra=(), fail_every=False, terminal_moves=2):
    games = []

    class Game:
        def __init__(self, *args, **kwargs):
            self.moves, self.closed = 0, False
            games.append(self)

        def reset(self):
            return "A room.", {"score": 0, "moves": 0}

        def get_player_location(self):
            return NS(name="Gallery", num=1)

        def get_valid_actions(self, **kwargs):
            return ["look"]

        def step(self, action):
            self.moves += 1
            return "A room.", 1, self.moves == terminal_moves, {"score": self.moves, "moves": self.moves}

        def close(self):
            self.closed = True

    class Actor:
        episode, calls = 0, 0

        def reset_history(self):
            self.episode += 1
            self.calls = 0

        def apply_configuration(self, configuration):
            pass

        def add_history(self, **kwargs):
            pass

        def choose_decision(self, **kwargs):
            self.calls += 1
            if (fail_every or self.calls == 2) and (always_fail or self.episode == 1):
                raise exception or IncompleteLabelScores("Missing C after 3 attempts")
            return {"action": "look", "candidate_actions": ["look"], "base_logits": [0], "report": {}}

    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=Game))
    monkeypatch.setattr(main, "create_agent", lambda *args, **kwargs: Actor())
    argv = ["--episodes", "4" if always_fail else "2", "--max-steps", "2", "--target-score", "350",
            "--fresh-memory", "--no-evolve", "--no-historian-review", "--output-dir", str(tmp_path), *extra]
    return lambda: main.main(argv), games


def test_default_recovery_is_degraded_and_next_episode_finishes(tmp_path, monkeypatch):
    run, games = failure_run(tmp_path, monkeypatch)
    directory = run()
    status = json.loads((directory / "run_status.json").read_text())
    assert [row["status"] for row in status["outcomes"]] == ["completed", "completed"]
    assert status["completed_episodes"] == 2 and status["degraded_episodes"] == 1
    assert status["all_episodes_completed"]
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [row["episode"] for row in metrics] == [2]  # Actual identity, not completed count.
    assert [row["score"] for row in metrics] == [2]
    samples = [json.loads(line) for line in (directory / "memory/actor_experiences.jsonl").read_text().splitlines()]
    first = [sample for sample in samples if sample["episode_id"].endswith("episode_0001")]
    assert len(first) == 1  # Recovery action is not a native policy sample.
    assert first[0]["step"] == 0
    assert len(games) == 2 and all(game.closed for game in games)


@pytest.mark.parametrize("extra,expected_games", [(["--fail-fast"], 1)])
def test_repeated_recovery_failure_stops_at_limit_with_saved_status(tmp_path, monkeypatch, extra, expected_games):
    run, games = failure_run(tmp_path, monkeypatch, always_fail=True, extra=extra)
    with pytest.raises(IncompleteLabelScores):
        run()
    assert len(games) == expected_games and all(game.closed for game in games)
    status = json.loads(next(tmp_path.glob("*/run_status.json")).read_text())
    assert status["incomplete_episodes"] == expected_games
    assert status["completed_episodes"] == 0
    assert not status["outcomes"][-1]["continuing"]
    assert not list(tmp_path.glob("*/metrics.jsonl"))


def test_unexpected_code_error_is_saved_but_not_suppressed(tmp_path, monkeypatch):
    run, games = failure_run(tmp_path, monkeypatch, exception=ValueError("broken state"))
    with pytest.raises(ValueError, match="broken state"):
        run()
    assert len(games) == 1 and games[0].closed
    status = json.loads(next(tmp_path.glob("*/run_status.json")).read_text())
    assert status["outcomes"][0]["recoverable"] is False


def test_no_nsi_feasible_action_ends_cleanly_without_fabricating_moves(tmp_path, monkeypatch):
    run, games = failure_run(tmp_path, monkeypatch)
    monkeypatch.setattr(main.CognitiveMap, "action_feasibility",
                        lambda self, state, actions, *args: [0.0] * len(actions))
    directory = run()
    status = json.loads((directory / "run_status.json").read_text())
    assert [row["termination_reason"] for row in status["outcomes"]] == ["no_safe_actions"] * 2
    assert status["completed_episodes"] == 0
    assert not (directory / "metrics.jsonl").exists()
    assert all(game.moves == 0 and game.closed for game in games)
