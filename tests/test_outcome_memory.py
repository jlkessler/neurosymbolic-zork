import json
from types import SimpleNamespace

import pytest

from MANGORL.actor_experience import ActorExperienceMemory
from MANGORL.actor_rl_controller import ActorRLController
from MANGORL.checkpoint import restore_arguments
from MANGORL.outcome_memory import OutcomeMemory
from MANGORL.smart_agent import SmartAgent
from main import parse_args


def setup_memory(tmp_path, *, horizon=2, gamma=0.9):
    outcomes = OutcomeMemory(tmp_path / "outcomes", horizon=horizon, gamma=gamma,
                             target=350, rom="test-rom", map_navigation=True)
    return ActorExperienceMemory(tmp_path / "actor.jsonl", outcomes=outcomes)


def state(score=75, peak=75, inventory="A lantern (providing light)", room="A trophy case.",
          remaining=2, feedback="Done."):
    return dict(location_id="living", score=score, peak_score=peak, steps_remaining=remaining,
                observation=(f"LAST ACTION RESULT:\n{feedback}\n\nCURRENT ROOM (look):\n{room}\n\n"
                             f"CURRENT INVENTORY:\nYou are carrying:\n{inventory}"))


def step(memory, before, action, after, *, source="smart_actor", terminal=False):
    memory.record_step(state=before, action=action, next_state=after,
                       score_delta=after["score"] - before["score"], action_source=source,
                       actor_decision=source == "smart_actor", terminal=terminal)


def estimates(memory, query, actions):
    return memory.outcomes.estimate_action_values(query, actions, top_k=40, min_similarity=0.65)


def test_closed_trajectories_use_signed_observed_returns_through_navigation(tmp_path):
    memory = setup_memory(tmp_path)
    original = state()
    carrying = state(69, inventory="A lantern (providing light)\nA painting", remaining=1)
    # Observed ideal-map episode 19 score changes on taking and redepositing the painting.
    memory.start_episode("e1")
    step(memory, original, "take painting", carrying)
    step(memory, carrying, "put painting in case", original)
    assert estimates(memory, original, ["take painting"])["samples"] == 0
    summary = memory.finish_episode(gamma=0.9, termination_reason="step_limit", bootstrap_value=99)
    assert summary["published_outcome_steps"] == 2
    memory.start_episode("e2")
    step(memory, original, "east", original)
    step(memory, original, "down", state(85, 85), source="symbolic_navigation", terminal=True)
    memory.finish_episode(gamma=0.9, termination_reason="environment_done")
    memory = setup_memory(tmp_path)
    memory.start_episode("next")
    actions = ["take painting", "east", "down", "wait"]
    values = estimates(memory, original, actions)
    assert values["state_value"] == pytest.approx((-0.6 + 9 + 10) / 3)
    assert values["actions"]["take painting"]["q_value"] == pytest.approx(-0.6)
    assert values["actions"]["east"]["q_value"] == pytest.approx(9)
    assert values["actions"]["down"]["example"]["source"] == "symbolic_navigation"
    assert "wait" not in values["actions"]
    for changed in (state(peak=80), state(score=69), state(inventory="A lantern"),
                    state(room="An open trophy case."), {"observation": "Taken."}):
        assert estimates(memory, changed, actions)["samples"] == 0
    assert estimates(memory, state(feedback="Taken."), actions)["samples"] == 3
    assert estimates(setup_memory(tmp_path, gamma=1), original, actions)["samples"] == 0


@pytest.mark.parametrize("reason,source", [
    ("episode_exception", "smart_actor"), ("interrupted", "smart_actor"),
    ("no_safe_actions", "smart_actor"), ("environment_done", "recovery_default"),
    ("step_limit", "smart_actor"),  # below the configured two-command horizon
])
def test_partial_or_degraded_episodes_never_enter_outcomes(tmp_path, reason, source):
    memory = setup_memory(tmp_path)
    memory.start_episode("failed")
    step(memory, state(), "wait", state(), source=source)
    summary = memory.finish_episode(termination_reason=reason, bootstrap_value=100)
    assert summary["published_outcome_steps"] == 0
    assert not list(memory.outcomes.directory.glob("*.json"))


def test_repeated_visits_supply_one_vote_per_action_per_episode(tmp_path):
    memory = setup_memory(tmp_path, horizon=5)
    memory.start_episode("loop")
    for remaining in range(5, 0, -1):
        step(memory, state(remaining=remaining), "look", state())
    memory.finish_episode(termination_reason="step_limit")
    memory.start_episode("other")
    evidence = estimates(memory, state(remaining=2), ["look"])
    assert evidence["samples"] == 1
    assert evidence["actions"]["look"]["example"]["step"] == 4


def test_observed_inventory_keeps_nested_objects_and_light_state():
    def key(pack):
        return OutcomeMemory.snapshot(state(inventory=pack))["key"]

    bottle = "  A bottle\n  The bottle contains:\n    Water"
    assert key(bottle + "\n  A lamp (providing light)") == key("  A lamp (providing light)\n" + bottle)
    assert key(bottle) != key("  A bottle\n  The bottle contains:\n  Water")
    assert key("  A lamp") != key("  A lamp (providing light)")


def test_repeat_cost_and_closed_loss_cooldown(tmp_path):
    memory = setup_memory(tmp_path)
    memory.start_episode("loop")
    original, carrying = state(), state(69, inventory="A painting")
    policy = ActorRLController(memory, memory_mode="concrete", repeat_penalty=0.5)
    actions = ["take painting", "east"]
    for _ in range(2):
        step(memory, original, "take painting", carrying)
        assert memory.outcomes.apply_cooldown(carrying, actions)[1] == []
        step(memory, carrying, "put painting in case", original)
    result = policy.select_action(original, actions, base_logits=[0.2, 0], sample=False)
    assert result["repeat_costs"] == [0.5, 0.0]
    assert result["action"] == "east" and not result["memory_changed_argmax"]
    assert memory.outcomes.apply_cooldown(original, ["take painting"])[1] == []
    assert memory.outcomes.apply_cooldown(state(peak=80), actions)[1] == []
    for _ in range(3):
        assert memory.outcomes.apply_cooldown(original, actions) == (["east"], ["take painting"])
    assert memory.outcomes.apply_cooldown(original, actions)[1] == []


def test_memory_and_nsi_switch_different_argmax_stages(tmp_path):
    memory = setup_memory(tmp_path, horizon=1)
    for episode, (action, delta) in enumerate([("east", 10), ("look", 0)]):
        memory.start_episode(episode)
        step(memory, state(), action, state(75 + delta, 75 + delta))
        memory.finish_episode(termination_reason="step_limit")
    memory.start_episode("test")
    policy = ActorRLController(memory, memory_mode="concrete", beta=1, exploration_coefficient=0)
    actions = ["look", "east"]
    assert policy.recall_candidates(policy.estimate(state(), actions)) == ["east"]
    for feas, expected in (([1, 1], "east"), ([1, 0], "look")):
        selection = policy.select_action(state(), actions, base_logits=[0.2, 0], feasibility=feas, sample=False)
        assert selection["without_memory_preferred_action"] == "look"
        assert selection["memory_changed_argmax"] and selection["adjusted_preferred_action"] == "east"
        assert selection["nsi_changed_argmax"] == (expected == "look")
        assert selection["action"] == expected
    off = ActorRLController(memory, memory_mode="off")
    assert not off.estimate(state(), actions)["memory_actions"]
    assert parse_args([]).actor_memory == "concrete"


def test_recalled_legal_alternative_gets_a_real_label_score(monkeypatch):
    proposed = ["look", "north", "west", "south", "wait"]
    legal = proposed + ["east"]
    agent = SmartAgent(client=object(), policy_mode="logprobs")
    monkeypatch.setattr(agent, "_request_decision", lambda *a: (
        {"action": "look", "candidate_actions": proposed}, "", False))
    menus = []
    def score(**kwargs):
        menus.append(kwargs["actions"])
        return {"base_logits": [0, -1, -2, -3, -4], "source": "test_model_labels"}
    monkeypatch.setattr("MANGORL.smart_agent.LabelLogprobScorer", lambda: SimpleNamespace(score=score))
    result = agent.choose_decision(observation="Room", current_location="Room", valid_actions=legal,
                                   recalled_actions=["invalid", "north", "east"])
    assert menus == [proposed[:4] + ["east"]]
    assert result["recalled_candidate"] == "east" and result["base_logits"][-1] == -4


def test_sole_observed_success_can_enter_menu_without_advantage(tmp_path):
    memory = setup_memory(tmp_path, horizon=1)
    memory.start_episode("one")
    step(memory, state(), "east", state(85, 85))
    memory.finish_episode(termination_reason="step_limit")
    policy = ActorRLController(memory, memory_mode="concrete")
    estimate = policy.estimate(state(), ["east", "look"])
    assert estimate["memory_actions"]["east"]["advantage"] == 0
    assert policy.recall_candidates(estimate) == ["east"]


def test_old_checkpoint_keeps_legacy_memory(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"schema_version": 1, "arguments": {}}), encoding="utf-8")
    argv = ["--resume-trial", str(tmp_path)]
    assert restore_arguments(parse_args(argv), argv).actor_memory == "legacy"


def test_unknown_score_change_is_excluded_from_observed_returns(tmp_path):
    memory = setup_memory(tmp_path, horizon=1)
    memory.start_episode("missing-score")
    memory.record_step(state=state(), action="down", next_state=state(), score_delta=None)
    assert memory.finish_episode(termination_reason="step_limit")["published_outcome_steps"] == 0
