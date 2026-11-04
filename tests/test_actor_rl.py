import json

import pytest

from MANGORL.actor_experience import ActorExperienceMemory
from MANGORL.actor_rl_controller import ActorRLController
from MANGORL.historian_rl_controller import HistorianRLController
from main import parse_args
from MANGORL.smart_agent import SmartAgent


def _state(location_id="cellar", observation="The cellar is dim."):
    return {
        "location_id": location_id,
        "location": "Cellar",
        "observation": observation,
        "score": 10,
        "inventory_items": ["brass lantern"],
        "puzzle_clues": ["door is locked"],
        "goal_kind": "solve_return_route",
        "goal_description": "Find a way back to the Living Room.",
        "home_route_status": "blocked",
    }


def test_actor_experience_attributes_later_navigation_reward_to_actor_decision(tmp_path):
    path = tmp_path / "actor.jsonl"
    memory = ActorExperienceMemory(path)
    memory.start_episode("trial_a/episode_0001")
    state = _state()
    middle = {**state, "observation": "The cellar door is locked."}
    after = {**state, "location_id": "living-room", "location": "Living Room"}

    memory.record_step(
        state=state,
        action="examine door",
        next_state=middle,
        score_delta=0,
        action_source="smart_actor",
        actor_decision=True,
        candidate_actions=["examine door", "open door"],
    )
    # A symbolic-navigation/environment step is not an Actor policy sample,
    # but its later reward is included in the earlier Actor decision's return.
    memory.record_step(
        state=middle,
        action="up",
        next_state=after,
        score_delta=5,
        action_source="symbolic_navigation",
        actor_decision=False,
    )

    summary = memory.finish_episode(gamma=0.9, final_score=15)
    assert summary["actor_experiences"] == 1
    stored = json.loads(path.read_text(encoding="utf-8").strip())
    assert stored["action"] == "examine door"
    assert stored["discounted_return"] == pytest.approx(4.5)
    assert stored["candidate_actions"] == ["examine door", "open door"]

    reloaded = ActorExperienceMemory(path)
    assert len(reloaded) == 1
    assert reloaded.experiences[0]["episode_id"] == "trial_a/episode_0001"


def test_actor_experience_similarity_and_action_value_estimation(tmp_path):
    memory = ActorExperienceMemory(tmp_path / "actor.jsonl")
    state = _state()
    memory.experiences = [
        {
            "record_type": "actor_experience",
            "episode_id": "e1",
            "state": state,
            "action": "open door",
            "discounted_return": 8.0,
            "step": 1,
        },
        {
            "record_type": "actor_experience",
            "episode_id": "e2",
            "state": state,
            "action": "look",
            "discounted_return": 0.0,
            "step": 2,
        },
    ]
    result = memory.estimate_action_values(state, ["open door", "look"])
    assert result["samples"] == 2
    assert result["actions"]["open door"]["q_value"] == pytest.approx(8.0)
    assert result["actions"]["open door"]["advantage"] > 0
    assert result["actions"]["look"]["advantage"] < 0


def test_actor_rl_reranks_candidates_using_experience(tmp_path):
    memory = ActorExperienceMemory(tmp_path / "actor.jsonl")
    state = _state()
    memory.experiences = [
        {
            "record_type": "actor_experience",
            "episode_id": f"e{i}",
            "state": state,
            "action": "open door",
            "discounted_return": 8.0,
            "step": i,
        }
        for i in range(4)
    ] + [
        {
            "record_type": "actor_experience",
            "episode_id": f"n{i}",
            "state": state,
            "action": "look",
            "discounted_return": 0.0,
            "step": i,
        }
        for i in range(4)
    ]
    controller = ActorRLController(
        memory, beta=1.0, rank_penalty=0.3, exploration_coefficient=0.0
    )
    result = controller.select_action(
        state, ["look", "open door"], base_action="look"
    )
    assert result["action"] == "open door"
    assert result["method"] == "candidate_rank_plus_experience_advantage"


def test_actor_rl_can_adjust_real_candidate_logits(tmp_path):
    memory = ActorExperienceMemory(tmp_path / "actor.jsonl")
    state = _state()
    memory.experiences = [
        {
            "record_type": "actor_experience",
            "episode_id": "e1",
            "state": state,
            "action": "open door",
            "discounted_return": 3.0,
            "step": 1,
        }
    ]
    controller = ActorRLController(memory, beta=0.5)
    result = controller.adjust_logits(
        state, ["look", "open door"], [0.7, 0.1]
    )
    assert result["method"] == "native_candidate_logit_adjustment"
    assert len(result["adjusted_logits"]) == 2


def test_historian_critic_disabled_by_default_and_has_no_updates(tmp_path):
    path = tmp_path / "critic.json"
    critic = HistorianRLController(path)
    assert critic.enabled is False
    assert critic.value(_state()) == 0.0
    critic.start_episode("e1")
    critic.observe_transition(
        state=_state(), action="look", reward=5, next_state=_state()
    )
    result = critic.finish_episode()
    assert result["enabled"] is False
    assert not path.exists()


def test_historian_critic_learns_and_persists_state_values(tmp_path):
    path = tmp_path / "critic.json"
    critic = HistorianRLController(path, enabled=True, learning_rate=0.1)
    state = _state()
    critic.start_episode("e1")
    critic.observe_transition(
        state=state,
        action="open door",
        reward=5,
        next_state={**state, "location_id": "living-room"},
        terminal=True,
    )
    result = critic.finish_episode(termination_reason="target_score_reached")
    assert result["updates"] == 1
    assert critic.value(state) > 0

    loaded = HistorianRLController(path, enabled=True)
    assert loaded.value(state) > 0
    assert loaded.episodes_trained == 1


def test_cli_exposes_actor_rl_and_disabled_historian_critic_defaults():
    args = parse_args([])
    assert args.actor_rl is True
    assert args.actor_rl_gamma == pytest.approx(0.995)
    assert args.historian_critic is False
    assert args.actor_experience_path.endswith("actor_experiences.jsonl")

    configured = parse_args([
        "--no-actor-rl",
        "--historian-critic",
        "--actor-rl-gamma", "0.97",
        "--actor-rl-beta", "0.5",
    ])
    assert configured.actor_rl is False
    assert configured.historian_critic is True
    assert configured.actor_rl_gamma == pytest.approx(0.97)
    assert configured.actor_rl_beta == pytest.approx(0.5)


def test_smart_agent_returns_ordered_candidate_actions_and_includes_rl_guidance():
    payload = json.dumps({
        "action": "look",
        "candidate_actions": ["look", "examine door", "open door"],
        "inventory_items": [],
        "carried_treasures": [],
        "puzzle_clues": [],
        "state_note": "Inspecting the room.",
    })

    class Message:
        content = payload

    class Choice:
        message = Message()
        finish_reason = "stop"

    class Response:
        choices = [Choice()]

    class Completions:
        request = None

        def create(self, **kwargs):
            self.request = kwargs
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    client = Client()
    agent = SmartAgent(client=client)
    result = agent.choose_decision(
        observation="The cellar is dim.",
        current_location="Cellar",
        valid_actions=["look", "examine door", "open door"],
        rl_guidance="Prior experience favors examining the door.",
    )
    assert result["action"] == "look"
    assert result["candidate_actions"] == ["look", "examine door", "open door"]
    assert "Prior experience favors examining the door" in client.chat.completions.request["messages"][1]["content"]
