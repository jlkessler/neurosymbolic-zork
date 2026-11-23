import json
import sys
from types import SimpleNamespace as NS

import main
from MANGORL.checkpoint import restore_arguments
from MANGORL.smart_agent import SmartAgent


def test_no_map_trial_scores_and_keeps_actor_in_control(tmp_path, monkeypatch):
    class Game:
        def __init__(self, *args, **kwargs):
            self.moves = 0

        def reset(self):
            return "Living Room", {"score": 0, "moves": 0}

        def get_player_location(self):
            return NS(name="Gallery", num=2) if self.moves in (1, 2) else NS(name="Living ", num=1)

        def get_state(self):
            return self.moves

        def set_state(self, moves):
            self.moves = moves

        def get_valid_actions(self, **kwargs):
            return [["east"], ["take painting"], ["west"], ["put painting in case"]][self.moves]

        def step(self, action):
            if action == "look":
                title = "Gallery" if self.moves in (1, 2) else "Living Room"
                return f"{title}\nA well-lit room.", 0, False, {}
            if action == "inventory":
                text = "You are carrying:\nA painting" if self.moves in (2, 3) else "You are empty-handed."
                return text, 0, False, {}
            assert action in self.get_valid_actions()
            self.moves += 1
            score = [0, 0, 4, 4, 10][self.moves]
            return "Done.", 0, self.moves == 4, {"score": score, "moves": self.moves}

        def close(self):
            pass

    def disabled(*args, **kwargs):
        raise AssertionError("Map or navigator was invoked in the map ablation")

    requests = []
    commands = iter(["east", "take painting", "west", "put painting in case"] * 2)

    def answer(**request):
        requests.append(request)
        action = next(commands)
        inventory = ["painting"] if action in ("west", "put painting in case") else []
        content = json.dumps({"action": action, "inventory_items": inventory})
        return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])

    client = NS(chat=NS(completions=NS(create=answer)))
    monkeypatch.setattr(main, "CognitiveMap", disabled)
    monkeypatch.setattr(main, "NavigationController", disabled)
    monkeypatch.setattr(main, "SmartAgent", lambda **kwargs: SmartAgent(client=client, **kwargs))
    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=Game))
    map_path = tmp_path / "existing_map.json"
    map_path.write_text("existing map bytes", encoding="utf-8")
    historian_path = tmp_path / "historian.json"
    run = main.main([
        "--no-map-navigation", "--nsi", "--no-evolve", "--no-historian-review",
        "--actor-policy-mode", "rank", "--target-score", "10", "--episodes", "1",
        "--max-steps", "4", "--map-path", str(map_path), "--historian-path", str(historian_path),
        "--actor-experience-path", str(tmp_path / "actor.jsonl"),
        "--historian-critic-path", str(tmp_path / "critic.json"), "--output-dir", str(tmp_path / "logs"),
    ])

    assert map_path.read_text(encoding="utf-8") == "existing map bytes"
    episode = json.loads(historian_path.read_text(encoding="utf-8"))["episodes"][0]
    assert [step["action_source"] for step in episode["action_log"]] == ["smart_actor"] * 4
    assert [step["goal_kind"] for step in episode["action_log"]][2:] == ["solve_return_route", "deposit_treasure"]
    assert episode["control_metrics"]["route_starts"] == 0
    assert len(requests) == 4
    for request in requests:
        system, user = (message["content"] for message in request["messages"])
        assert "symbolic navigation controller" not in system
        assert "CURRENT OBSERVATION" in user and "AVAILABLE ACTIONS" in user
        assert all(section not in user for section in ("NAVIGATION CONTROLLER", "KNOWN LOCATIONS", "KNOWN ROUTES", "Home route status"))
    metric = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8"))
    assert metric["map_navigation"] is False and metric["nsi"] is False
    assert metric["objectives_met"]["painting_taken"] == 2
    assert metric["objectives_met"]["painting_deposited"] == 4
    journal = [json.loads(line) for line in (run / "episode_0001_traversal.jsonl").read_text().splitlines()]
    assert journal[0]["start"] == "id:1"
    assert [r["delta"] for r in journal[1:]] == [0, 4, 0, 6]
    summary = json.loads((run / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["mean_score"] == 10 and summary["target_success_rate"] == 1
    resume = ["--resume-trial", str(run), "--episodes", "2"]
    assert restore_arguments(main.parse_args(resume), resume).map_navigation is False
    main.main(resume)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    first, second = [row["control_metrics"] for row in metrics]
    summary = json.loads((run / "metrics_summary.json").read_text())["objectives_met"]
    assert summary["episodes_with_evidence"] == 2
    assert summary["milestones"]["painting_deposited"]["first_step_by_episode"] == {"1": 4, "2": 4}
    assert first["memory_supported_decisions"] == 0
    assert second["memory_supported_decisions"] == second["policy_decisions"] == 4
    assert second["peak_score"] == 10 and second["steps_since_peak"] == 0
    assert second["nsi_decisions"] == 0
