import json
import math
import sys
from types import SimpleNamespace as NS

import pytest

from MANGORL.actor_experience import ActorExperienceMemory
from MANGORL.actor_rl_controller import ActorRLController
from MANGORL.cognitive_map import CognitiveMap
from MANGORL.evolution import ConfigurationArchive, default_configuration, validate_configuration
from MANGORL.historian_rl_controller import HistorianRLController
from MANGORL.navigation_controller import NavigationController
from MANGORL.nsi import NeuroSymbolicInference, softmax
from MANGORL.policy_logits import LabelLogprobScorer, LogitsUnavailable
from MANGORL.semantic_memory import FrozenStateEncoder
from MANGORL.smart_agent import SmartAgent
import main


@pytest.fixture(autouse=True)
def independent_mock_historian_connection(monkeypatch):
    # Integration tests use a separate mock provider rather than real credentials.
    monkeypatch.setattr(main, "create_historian_connection", lambda args: (GameClient(), None))


def test_nsi_matches_hand_calculated_integral_and_shared_atoms():
    nsi = NeuroSymbolicInference()
    # P(A or B) = .4 + .5 - .4*.5, not a sum of overlapping proof masses.
    assert nsi.integrate({"A": .4, "B": .5}, lambda w: w["A"] or w["B"]) == pytest.approx(.7)
    assert nsi.conjunction({"A": .4}, [("A", True), ("A", True)]) == pytest.approx(.4)
    assert nsi.conjunction({"A": .4}, [("A", True), ("A", False)]) == 0
    assert nsi.integrate({}, lambda w: True) == 1


def test_nsi_conditions_neural_action_belief():
    result = NeuroSymbolicInference().condition_actions([math.log(.8), math.log(.2)], [.1, 1])
    assert result["satisfaction_mass"] == pytest.approx(.28)
    assert result["probabilities"] == pytest.approx([.08 / .28, .2 / .28])
    with pytest.raises(ValueError, match="consistent"):
        NeuroSymbolicInference().condition_actions([0, 0], [0, 0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, 1.1])
def test_nsi_rejects_invalid_beliefs(bad):
    with pytest.raises(ValueError):
        NeuroSymbolicInference().integrate({"x": bad}, lambda w: w["x"])


def test_closed_form_logit_update_changes_probability_and_masks(tmp_path):
    memory = ActorExperienceMemory(tmp_path / "actor.jsonl")
    controller = ActorRLController(memory, beta=.5, exploration_coefficient=0)
    controller.estimate = lambda *_: {"actions": {
        "look": {"advantage": 0, "sample_count": 1},
        "north": {"advantage": 1, "sample_count": 1}}}
    selected = controller.select_action({}, ["look", "north"], base_logits=[0, 0], feasibility=[1, 1], sample=False)
    assert selected["probabilities"] == pytest.approx(softmax([0, .5]))
    assert selected["action"] == "north"
    masked = controller.select_action({}, ["look", "north"], base_logits=[0, 5], feasibility=[1, 0])
    assert masked["probabilities"] == [1, 0]
    assert masked["action"] == "look"
    assert masked["conditioned_logits"][1] is None


def test_no_actor_rl_preserves_base_distribution(tmp_path):
    policy = ActorRLController(ActorExperienceMemory(tmp_path / "x"), enabled=False)
    result = policy.select_action({}, ["a", "b"], base_logits=[-1, -2], feasibility=[1, 1])
    assert result["probabilities"] == pytest.approx(softmax([-1, -2]))


def test_online_critic_updates_before_episode_end_and_distinguishes_actions(tmp_path):
    critic = HistorianRLController(tmp_path / "critic.json", enabled=True, learning_rate=.1, gamma=.9)
    state = {"location_id": "1", "observation": "room", "score": 0}
    critic.start_episode("e1")
    result = critic.observe_transition(state=state, action="north", reward=10, next_state=state, terminal=True)
    assert result["target"] == 10
    assert critic.value(state) > 0
    critic.observe_transition(state=state, action="look", reward=0, next_state=state, terminal=True)
    values = critic.action_advantages(state, ["north", "look"])
    assert values["north"]["advantage"] > values["look"]["advantage"]
    before = dict(critic.weights)
    critic.finish_episode()
    assert critic.weights == before  # No duplicate episode-end training.
    loaded = HistorianRLController(tmp_path / "critic.json", enabled=True)
    assert len(loaded.td_memory) == 2


def test_bootstrap_applies_only_to_truncated_returns(tmp_path):
    memory = ActorExperienceMemory(tmp_path / "actor.jsonl")
    for terminal, expected in [(False, 11), (True, 2)]:
        memory.start_episode(str(terminal))
        memory.record_step(state={}, action="look", next_state={}, score_delta=2,
                           actor_decision=True, terminal=terminal, truncated=not terminal)
        memory.finish_episode(gamma=.9, bootstrap_value=10)
        assert memory.last_episode_records[0]["discounted_return"] == pytest.approx(expected)


def test_cognitive_map_learns_probabilities_and_resets_episode_evidence(tmp_path):
    world = CognitiveMap(tmp_path / "map.json")
    start = world.start_episode("A", 1)
    nav = NavigationController(world)
    nav.observe_action_result("north", "B", 2)
    edge = next(iter(world.graph.edges(keys=True)))
    assert world.edge_probability(edge) == pytest.approx(2 / 3)
    assert world.edge_probability(edge, confirmed=nav.confirmed_edges) == 1
    world.start_episode("A", 1)
    nav.observe_action_result("north", "A", 1)
    assert world.edge_probability(edge, blocked=nav.blocked_edges) == 0
    nav.invalidate_local_evidence(start)
    assert not nav.blocked_edges
    assert world.edge_probability(edge) == pytest.approx(.5)
    world.save()
    loaded = CognitiveMap(tmp_path / "map.json")
    loaded.load()
    assert loaded.edge_probability(edge) == pytest.approx(.5)


def test_route_risk_prefers_reliable_two_step_path(tmp_path):
    world = CognitiveMap(tmp_path / "map.json")
    a = world.start_episode("A", 1)
    b = world.add_standard_location("B", 2)
    c = world.add_standard_location("C", 3)
    world.add_transition(a, c, ["east"])
    world.add_transition(a, b, ["north"])
    world.add_transition(b, c, ["east"])
    for source, dest, key, data in world.graph.edges(keys=True, data=True):
        data.update(successes=1 if (source, dest) == (a, c) else 20,
                    failures=20 if (source, dest) == (a, c) else 0)
    route = NavigationController(world, risk_weight=2).plan_to(c, require_confirmed=False)
    assert route["actions"] == ["north", "east"]
    assert route["success_probability"] == pytest.approx((21 / 22) ** 2)


def fake_response(labels):
    entries = [NS(token=label, logprob=value) for label, value in labels.items()]
    return NS(choices=[NS(logprobs=NS(content=[NS(token=entries[0].token,
                 logprob=entries[0].logprob, top_logprobs=entries)]))])


def test_selector_uses_actual_label_logprobs_and_rejects_missing_entries():
    client = NS(chat=NS(completions=NS(create=lambda **_: fake_response({"A": -.2, "B": -2}))))
    result = LabelLogprobScorer().score(client=client, model="test", messages=[], actions=["look", "north"])
    assert result["base_logits"] == [-.2, -2]
    with pytest.raises(LogitsUnavailable, match="Missing exact"):
        LabelLogprobScorer().score(client=client, model="test", messages=[], actions=["look", "north", "south"])


def test_selector_does_not_strip_whitespace_or_accept_sentinel():
    for values in ({" A": -.1}, {"A": -9999}):
        client = NS(chat=NS(completions=NS(create=lambda **_: fake_response(values))))
        with pytest.raises(LogitsUnavailable):
            LabelLogprobScorer().score(client=client, model="test", messages=[], actions=["look"])


def test_luna_scoring_disables_reasoning_and_requires_logprob_route():
    requests = []

    def create(**request):
        requests.append(request)
        return fake_response({"A": -.2, "B": -2})

    client = NS(base_url="https://openrouter.ai/api/v1",
                chat=NS(completions=NS(create=create)))
    result = LabelLogprobScorer().score(
        client=client, model="openai/gpt-6-luna", messages=[], actions=["look", "north"],
        extra_body={"provider": {"allow_fallbacks": False}},
    )
    request, = requests
    assert request["reasoning_effort"] == "none"
    assert request["max_completion_tokens"] == 16
    assert "max_tokens" not in request
    assert request["logprobs"] is True
    assert request["extra_body"]["provider"] == {
        "allow_fallbacks": False, "require_parameters": True,
    }
    assert result["base_logits"] == [-.2, -2]


def test_selector_surfaces_upstream_parameter_error():
    def create(**_):
        error = RuntimeError("400 - Provider returned error")
        error.body = {"metadata": {
            "provider_name": "OpenAI",
            "raw": '{"error":{"param":"logprobs","message":"Reasoning must be disabled"}}',
        }}
        raise error

    client = NS(chat=NS(completions=NS(create=create)))
    with pytest.raises(LogitsUnavailable, match="OpenAI.*logprobs.*Reasoning must be disabled"):
        LabelLogprobScorer().score(client=client, model="test", messages=[], actions=["look"])


def test_evolution_archive_retains_parent_and_uses_observed_fitness(tmp_path):
    archive = ConfigurationArchive(tmp_path / "configs.json", exploration=.1)
    parent_id, parent = archive.select()
    archive.observe(350)
    child = {**parent, "temperature": .5}
    child_id = archive.add(child, parent_id=parent_id)
    assert archive.select()[0] == parent_id  # An untested mutation is not promoted blindly.
    archive.active_id = child_id
    archive.observe(0)
    assert archive.select()[0] == parent_id
    archive.save()
    loaded = ConfigurationArchive(archive.path)
    assert set(loaded.entries) == {parent_id, child_id}
    with pytest.raises(ValueError):
        archive.add({**child, "execute_python": "untrusted code"})
    with pytest.raises(ValueError):
        validate_configuration({**child, "section_order": ["state"] * 8})


def test_frozen_semantic_memory_caches_and_blends(tmp_path):
    calls = []
    encoder = FrozenStateEncoder(model=NS(encode=lambda text, **_: calls.append(text) or [1, 0]))
    memory = ActorExperienceMemory(tmp_path / "memory", state_encoder=encoder)
    a, b = {"location_id": "A"}, {"location_id": "B"}
    assert memory.similarity(a, b) == pytest.approx(.5 + .5 * memory.state_similarity(a, b))
    memory.similarity(a, b)
    assert len(calls) == 2


class GameClient:
    def __init__(self):
        self.chat = NS(completions=self)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if request.get("logprobs"):
            return fake_response({"A": -.4, "B": -1.2})
        if "Historian/Evolver" in request["messages"][0]["content"]:
            parent = json.loads(request["messages"][1]["content"])["parent"]
            child = {**parent, "temperature": parent["temperature"] + .1,
                     "section_order": list(reversed(parent["section_order"]))}
            payload = {"configuration": child, "rationale": "Test a different prompt order after this episode."}
        else:
            text = request["messages"][1]["content"]
            actions = text.split("AVAILABLE ACTIONS:\n")[1].split("\n\n")[0].splitlines()
            actions = [a.removeprefix("- ") for a in actions]
            payload = {"action": actions[0], "candidate_actions": actions,
                       "inventory_items": [], "puzzle_clues": []}
        return NS(choices=[NS(message=NS(content=json.dumps(payload)), finish_reason="stop")])


class TinyGame:
    instances = []

    def __init__(self, path, seed=None):
        self.step_count = 0
        self.closed = False
        self.instances.append(self)

    def reset(self):
        return "You are in an atrium.", {"score": 0, "moves": 0}

    def get_player_location(self):
        return NS(name="Atrium" if self.step_count == 0 else "Gallery", num=1 if self.step_count == 0 else 2)

    def get_valid_actions(self, **kwargs):
        return ["north", "look"] if self.step_count == 0 else ["look"]

    def step(self, action):
        self.step_count += 1
        return "A gallery.", 2, self.step_count == 2, {"score": self.step_count * 2, "moves": self.step_count}

    def close(self):
        self.closed = True


def test_two_episode_full_loop_uses_logits_td_nsi_and_evolution(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=TinyGame))
    client = GameClient()
    monkeypatch.setattr(main, "create_agent", lambda args, event_logger: SmartAgent(
        client=client, policy_mode=args.actor_policy_mode, event_logger=event_logger))
    run = main.main(["--episodes", "2", "--max-steps", "2", "--target-score", "350",
                     "--fresh-memory", "--no-historian-review", "--no-sample-actions",
                     "--output-dir", str(tmp_path), "--trial-name", "integration"])
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["score"] for row in metrics] == [4, 4]
    assert metrics[0]["configuration_id"] != metrics[1]["configuration_id"]
    critic = json.loads((run / "memory/historian_td.json").read_text())
    assert critic["state_visits"] == 4
    assert any(request.get("logprobs") for request in client.requests)
    records = [json.loads(line) for line in (run / "memory/actor_experiences.jsonl").read_text().splitlines()]
    assert records[0]["selection_info"]["method"] == "model_label_logits_plus_advantage_nsi"
    assert all(game.closed for game in TinyGame.instances)
    assert json.loads((run / "metrics_summary.json").read_text())["normalized_auc"] == pytest.approx(4 / 350)


def test_evolved_actor_reorders_sections_and_rewrites_strategy():
    client = GameClient()
    actor = SmartAgent(client=client)
    cfg = default_configuration()
    cfg["actor_strategy"] = "Compare observed outcomes before each new decision."
    cfg["section_order"] = list(reversed(cfg["section_order"]))
    actor.apply_configuration(cfg)
    actor.choose_decision(observation="A room", current_location="A", valid_actions=["look"])
    request = client.requests[0]
    assert cfg["actor_strategy"] in request["messages"][0]["content"]
    assert "Follow through on promising local interactions" in request["messages"][0]["content"]
    assert "ADDITIONAL LEARNED STRATEGY" in request["messages"][0]["content"]
    assert request["messages"][1]["content"].startswith("AVAILABLE ACTIONS:")


@pytest.mark.parametrize("first,finish,issue", [
    ('{"action":"open window"}', "stop", "action_outside_menu"),
    ('{"action":"north",', "length", "response_truncated"),
    ('this is not JSON', "stop", "malformed_json"),
])
def test_actor_repairs_response_before_requesting_native_scores(first, finish, issue):
    client = GameClient()
    normal_create = client.create
    calls = []

    def create(**request):
        calls.append(request)
        if len(calls) == 1:
            return NS(choices=[NS(message=NS(content=first), finish_reason=finish)])
        return normal_create(**request)

    client.create = create
    events = []
    actor = SmartAgent(client=client, policy_mode="logprobs",
                       event_logger=lambda event, payload: events.append((event, payload)))
    decision = actor.choose_decision(
        observation="A room.", current_location="Atrium", valid_actions=["north", "look"],
    )
    assert decision["action"] == "north"
    assert decision["base_logits"] == [-.4, -1.2]
    assert len(calls) == 3  # Rejected response, repair, genuine scoring.
    rejection = next(payload for event, payload in events if event == "actor_response_rejected")
    assert rejection["issue"] == issue
    assert rejection["valid_actions"] == ["north", "look"]
    assert issue in calls[1]["messages"][-1]["content"]
    if finish == "length":
        assert calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_episode_survives_exhausted_actor_repairs_with_genuine_scores(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=TinyGame))
    client = GameClient()
    normal_create = client.create
    failures = []

    def create(**request):
        if request.get("logprobs"):
            return normal_create(**request)
        failures.append(request)
        return NS(choices=[NS(message=NS(content='{"action":"unavailable"}'), finish_reason="stop")])

    client.create = create
    monkeypatch.setattr(main, "create_agent", lambda args, event_logger: SmartAgent(
        client=client, policy_mode="logprobs", event_logger=event_logger))
    run = main.main(["--episodes", "1", "--max-steps", "2", "--fresh-memory",
                     "--no-historian-review", "--no-evolve", "--output-dir", str(tmp_path)])
    records = [json.loads(line) for line in (run / "memory/actor_experiences.jsonl").read_text().splitlines()]
    assert len(failures) == 6  # Three attempts per turn, then score the legal menu.
    assert [row["action"] for row in records] == ["north", "look"]
    assert all(row["selection_info"]["method"] == "model_label_logits_plus_advantage_nsi"
               for row in records)
    assert json.loads((run / "metrics_summary.json").read_text())["final_score"] == 4


def test_snapshot_context_restores_game_even_when_query_fails():
    class SnapshotGame:
        moves = 7

        def get_state(self):
            return self.moves

        def set_state(self, state):
            self.moves = state

        def step(self, action):
            self.moves += 1
            if action == "inventory" and self.fail_inventory:
                raise RuntimeError("query failed")
            return action + " text", 0, False, {"moves": self.moves}

    game = SnapshotGame()
    game.fail_inventory = False
    text = main.current_environment_context(game, "last result")
    assert "look text" in text and "inventory text" in text and "last result" in text
    assert game.moves == 7
    game.fail_inventory = True
    with pytest.raises(RuntimeError, match="query failed"):
        main.current_environment_context(game, "last result")
    assert game.moves == 7


def test_terminal_relocation_is_not_learned_as_a_route(tmp_path, monkeypatch):
    class DeathGame(TinyGame):
        def step(self, action):
            self.step_count += 1
            return "You have died and reappeared elsewhere.", -10, True, {"score": -10, "moves": 1}

    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=DeathGame))
    monkeypatch.setattr(main, "create_agent", lambda args, event_logger: SmartAgent(
        client=GameClient(), event_logger=event_logger, policy_mode="logprobs"))
    run = main.main(["--episodes", "1", "--max-steps", "2", "--fresh-memory",
                     "--no-evolve", "--no-historian-review", "--output-dir", str(tmp_path)])
    world = CognitiveMap(run / "memory/world_map.json")
    world.load()
    assert world.graph.number_of_edges() == 0
    records = [json.loads(line) for line in (run / "memory/actor_experiences.jsonl").read_text().splitlines()]
    assert records[0]["terminal"] and records[0]["score_delta"] == -10
    assert records[0]["next_state"]["location"] == "Gallery"


def test_actor_grounds_directions_and_preserves_proposal_for_scoring():
    client = GameClient()
    actor = SmartAgent(client=client, policy_mode="logprobs")
    actor.choose_decision(
        observation="The passage opened.", current_location="Courtyard",
        valid_actions=["west", "east"],
        known_routes=[{"actions": ["W"], "destination": "Hall", "destination_label": "Hall",
                       "total_cost": 1}],
    )
    assert "west: historically led to Hall" in client.requests[0]["messages"][1]["content"]
    score_request = next(request for request in client.requests if request.get("logprobs"))
    assert '"action": "west"' in score_request["messages"][1]["content"]
    assert "ONLY as its menu letter" in score_request["messages"][0]["content"]


def test_action_oracle_failure_does_not_fabricate_movement_only_menu():
    def fail(**kwargs):
        raise RuntimeError("oracle failed")
    with pytest.raises(RuntimeError, match="command menu is unknown"):
        main.get_valid_actions(NS(get_valid_actions=fail))


@pytest.mark.parametrize("option,value", [("--actor-rl-beta", "nan"), ("--score-scale", "inf"),
                                         ("--route-risk-weight", "-1")])
def test_cli_rejects_invalid_numerical_settings(option, value):
    with pytest.raises(SystemExit):
        main.parse_args([option, value])


def test_all_learning_switches_off_leave_no_critic_or_archive(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=TinyGame))
    client = GameClient()
    monkeypatch.setattr(main, "create_agent", lambda args, event_logger: SmartAgent(
        client=client, policy_mode=args.actor_policy_mode, event_logger=event_logger))
    run = main.main(["--episodes", "1", "--max-steps", "2", "--target-score", "350", "--fresh-memory",
                     "--no-evolve", "--no-historian-critic", "--no-actor-rl", "--no-historian-review",
                     "--no-nsi", "--actor-policy-mode", "rank", "--output-dir", str(tmp_path)])
    assert not (run / "memory/historian_td.json").exists()
    assert not (run / "memory/configurations.json").exists()
    assert not any(request.get("logprobs") for request in client.requests)
    records = [json.loads(line) for line in (run / "memory/actor_experiences.jsonl").read_text().splitlines()]
    assert records and all(row["selection_info"]["corrections"] == [0.0] * len(row["candidate_actions"]) for row in records)


def test_fail_fast_logprob_failure_stops_without_review_evolution_or_fitness(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=TinyGame))
    client = GameClient()
    original_create = client.create
    def create(**kwargs):
        if kwargs.get("logprobs"):
            return fake_response({"A": -.1})  # Candidate B is missing.
        return original_create(**kwargs)
    client.create = create
    monkeypatch.setattr(main, "create_agent", lambda args, event_logger: SmartAgent(
        client=client, policy_mode="logprobs", event_logger=event_logger))
    with pytest.raises(LogitsUnavailable):
        main.main(["--episodes", "2", "--max-steps", "2", "--fresh-memory", "--fail-fast",
                   "--output-dir", str(tmp_path)])
    run = next(tmp_path.iterdir())
    assert not (run / "metrics.jsonl").exists()
    assert not (run / "memory/configurations.json").exists()
    assert len(client.requests) == 1
    assert TinyGame.instances[-1].closed


def test_terminal_game_without_location_is_recorded(tmp_path, monkeypatch):
    class TerminalGame(TinyGame):
        def get_player_location(self):
            return None if self.step_count == 2 else super().get_player_location()
    monkeypatch.setitem(sys.modules, "jericho", NS(FrotzEnv=TerminalGame))
    monkeypatch.setattr(main, "create_agent", lambda args, event_logger: SmartAgent(
        client=GameClient(), event_logger=event_logger, policy_mode="logprobs"))
    run = main.main(["--episodes", "1", "--max-steps", "2", "--target-score", "350", "--fresh-memory",
                     "--no-evolve", "--no-historian-review", "--output-dir", str(tmp_path)])
    records = [json.loads(line) for line in (run / "memory/actor_experiences.jsonl").read_text().splitlines()]
    assert records[-1]["terminal"]
    assert records[-1]["termination_reason"] == "environment_done"
    assert records[-1]["discounted_return"] == 2


def test_unknown_critic_schema_is_preserved(tmp_path):
    path = tmp_path / "critic.json"
    original = '{"schema_version": 999}'
    path.write_text(original)
    with pytest.raises(ValueError, match="original file preserved"):
        HistorianRLController(path, enabled=True)
    assert path.read_text() == original
    assert HistorianRLController(path, enabled=False).value({}) == 0


def test_legacy_map_links_field_loads(tmp_path):
    world = CognitiveMap(tmp_path / "map.json")
    world.start_episode("A", 1)
    world.record_step("north", "B", next_location_id=2)
    world.save()
    payload = json.loads(world.save_path.read_text())
    payload["links"] = payload.pop("edges")
    world.save_path.write_text(json.dumps(payload))
    reloaded = CognitiveMap(world.save_path)
    reloaded.load()
    assert reloaded.graph.number_of_edges() == 1


def test_logit_policy_is_stable_for_very_small_supported_prior(tmp_path):
    policy = ActorRLController(ActorExperienceMemory(tmp_path / "memory"), enabled=False)
    result = policy.select_action({}, ["a", "b"], base_logits=[0, -1000], feasibility=[0, 1])
    assert result["probabilities"] == [0, 1]
    assert result["kl_to_base"] == pytest.approx(1000)
    assert result["log_satisfaction_mass"] == pytest.approx(-1000)


def test_ablation_harness_uses_separate_memories_and_matched_budgets(capsys):
    from experiments import main as experiments_main
    experiments_main(["--dry-run", "--conditions", "fixed_policy", "hybrid", "--seeds", "1", "2",
                      "--episodes", "50", "--max-steps", "60"])
    commands = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 4
    assert all("--fresh-memory" in command for command in commands)
    assert all(command[command.index("--max-steps") + 1] == "60" for command in commands)
    assert len({command[command.index("--trial-name") + 1] for command in commands}) == 4


def test_neural_retrieval_bounds_encoding_work_and_accepts_legacy_records(tmp_path):
    calls = []
    encoder = FrozenStateEncoder(model=NS(encode=lambda text, **_: calls.append(text) or [1, 0]))
    memory = ActorExperienceMemory(tmp_path / "memory", state_encoder=encoder)
    memory.experiences = [{"state": {"location_id": str(i)}, "action": "look", "step": i}
                          for i in range(1000)]
    rows = memory.retrieve_similar({"location_id": "12"}, top_k=10, min_similarity=0)
    assert len(rows) == 10
    assert rows[0]["state"]["location_id"] == "12"
    assert len(calls) <= 65  # At most 64 prefiltered states plus query.


def test_cli_help_and_argument_repr_do_not_expose_environment_key(monkeypatch):
    monkeypatch.setenv("ACTOR_API_KEY", "private-test-value")
    assert "private-test-value" not in main.build_arg_parser().format_help()
    assert "private-test-value" not in repr(main.parse_args(["--api-key", "private-test-value"]))
