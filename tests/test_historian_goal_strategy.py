import json

from MANGORL.historian import Historian
from MANGORL.smart_agent import SmartAgent


def _record_score_event(historian, *, node, location, action, before=0, after=5):
    historian.start_episode(node, location, initial_score=before)
    historian.observe_step(
        before_node=node,
        before_location=location,
        action=action,
        after_node=node,
        after_location=location,
        score_before=before,
        score_after=after,
        observation_before="There is something interesting here.",
        observation="The action succeeds.",
        action_source="smart_actor",
    )
    historian.finish_episode(final_score=after, moves=1)


def test_historian_prioritizes_early_return_route_after_entering_cellar(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Living Room", "Living Room", initial_score=0)
    historian.observe_step(
        before_node="Living Room",
        before_location="Living Room",
        action="down",
        after_node="Cellar",
        after_location="Cellar",
        score_before=0,
        score_after=0,
        observation_before="The trap door is open.",
        observation="You are in the cellar.",
        home_node="Living Room",
    )

    goal = historian.choose_goal(
        current_node="Cellar",
        current_location="Cellar",
        valid_actions=["up", "east", "look"],
        known_routes=[{
            "destination": "Living Room",
            "destination_label": "Living Room",
            "actions": ["up"],
            "total_cost": 1,
        }],
        current_score=0,
        home_node="Living Room",
        home_route_status="historical_unconfirmed",
    )

    assert goal["kind"] == "establish_return_route"
    assert goal["target_node"] == "Living Room"
    assert "early return trip" in goal["description"]
    assert [step["order"] for step in goal["sequence"]] == [1, 2, 3]
    assert "Establish a return route" in goal["sequence"][0]["goal"]


def test_historian_keeps_return_route_goal_when_no_route_is_known_even_without_treasure(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Cellar", "Cellar", initial_score=0)

    goal = historian.choose_goal(
        current_node="Cellar",
        current_location="Cellar",
        valid_actions=["up", "north", "look"],
        known_routes=[],
        current_score=0,
        home_node="Living Room",
        home_route_status="unknown",
    )

    assert goal["kind"] == "solve_return_route"
    assert "immediate strategic objective" in goal["description"]
    assert "before beginning a lengthy treasure-hunting sequence" in goal["description"]
    assert len(goal["sequence"]) == 3


def test_historian_marks_return_home_verified_only_after_reaching_living_room(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Living Room", "Living Room", initial_score=0)
    historian.observe_step(
        before_node="Living Room", before_location="Living Room", action="down",
        after_node="Cellar", after_location="Cellar", score_before=0, score_after=0,
        observation="The cellar is below the house.", home_node="Living Room",
    )
    assert historian.episode["underground_reached"] is True
    assert historian.episode["home_return_verified"] is False

    historian.observe_step(
        before_node="Cellar", before_location="Cellar", action="up",
        after_node="Living Room", after_location="Living Room", score_before=0, score_after=0,
        observation="You return to the Living Room.", home_node="Living Room",
    )
    assert historian.episode["home_return_verified"] is True
    assert historian.episode["home_return_step"] == 2


def test_historian_uses_reward_evidence_to_classify_treasure_vs_ordinary_items(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    _record_score_event(
        historian, node="Forest", location="Forest", action="take jeweled egg", after=5
    )
    historian.start_episode("Cellar", "Cellar", initial_score=5)
    historian.record_agent_report({
        "location": "Cellar",
        "inventory_items": [
            "jeweled egg", "leaflet", "bottle of water", "brass lantern", "small key"
        ],
        # The Actor omits the egg from its semantic candidates. Reward evidence
        # should still identify it as score-relevant when it appears in inventory.
        "treasure_candidates": [],
        "ordinary_items": ["leaflet", "bottle of water"],
        "useful_items": ["brass lantern", "small key"],
        "puzzle_clues": [],
    })

    state = historian.agent_state
    assert state["carried_treasures"] == ["jeweled egg"]
    assert "leaflet" in state["ordinary_items"]
    assert "bottle of water" in state["ordinary_items"]
    assert "brass lantern" in state["useful_items"]
    assert "small key" in state["useful_items"]

    goal = historian.choose_goal(
        current_node="Cellar", current_location="Cellar",
        valid_actions=["up", "look"], known_routes=[], current_score=5,
        home_node="Living Room", home_route_status="unknown",
    )
    assert goal["kind"] == "solve_return_route"
    assert goal["carried_treasures"] == ["jeweled egg"]
    assert "leaflet" in goal["ordinary_items"]
    assert "brass lantern" in goal["useful_items"]


def test_historian_filters_leaflet_and_water_from_treasure_candidates(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("West of House", "West of House", initial_score=0)
    historian.record_agent_report({
        "location": "West of House",
        "inventory_items": ["leaflet", "bottle of water", "gold coin"],
        "treasure_candidates": ["leaflet", "bottle of water", "gold coin"],
        "ordinary_items": ["leaflet", "bottle of water"],
        "useful_items": [],
    })

    assert historian.agent_state["carried_treasures"] == ["gold coin"]
    assert "leaflet" not in historian.agent_state["carried_treasures"]
    assert "bottle of water" not in historian.agent_state["carried_treasures"]


def test_historian_reuses_reward_positive_action_and_historical_path(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    _record_score_event(
        historian, node="Treasure Room", location="Treasure Room",
        action="open chest", before=0, after=12,
    )
    historian.start_episode("Cellar", "Cellar", initial_score=0)

    # The early return milestone is completed for this episode.
    historian.episode["underground_reached"] = True
    historian.episode["home_return_verified"] = True
    goal = historian.choose_goal(
        current_node="Cellar",
        current_location="Cellar",
        valid_actions=["north", "south", "look"],
        known_routes=[{
            "destination": "Treasure Room",
            "destination_label": "Treasure Room",
            "actions": ["east", "north", "down"],
            "total_cost": 3,
        }],
        current_score=0,
        home_node="Living Room",
        home_route_status="historical_unconfirmed",
    )

    assert goal["kind"] == "navigate_score_goal"
    assert goal["target_node"] == "Treasure Room"
    assert "open chest" in goal["description"]
    assert goal["sequence"][0]["goal"] == "Reach the previously productive location"
    assert "one transition at a time" in goal["sequence"][0]["details"]
    guidance = historian.build_guidance("Cellar", "Cellar", current_score=0)
    assert "Previously fruitful interactions" in guidance
    assert "open chest" in guidance


def test_smart_agent_prompt_explains_inventory_categories_and_goal_sequence():
    response = {
        "action": "inventory",
        "candidate_actions": ["inventory", "look"],
        "inventory_items": ["leaflet", "bottle of water", "jeweled egg", "brass lantern"],
        "treasure_candidates": ["jeweled egg"],
        "carried_treasures": ["jeweled egg"],
        "ordinary_items": ["leaflet", "bottle of water"],
        "useful_items": ["brass lantern"],
        "uncertain_items": [],
        "puzzle_clues": [],
        "state_note": "Inventory checked.",
    }

    class FakeCompletions:
        request = None

        def create(self, **kwargs):
            self.request = kwargs
            return type("Resp", (), {
                "choices": [type("Choice", (), {
                    "message": type("Message", (), {"content": json.dumps(response)})()
                })()]
            })()

    completions = FakeCompletions()
    agent = SmartAgent(
        client=type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})(),
        model="fake",
    )
    sequence = [
        {"order": 1, "goal": "Find a route home", "details": "Do this early."},
        {"order": 2, "goal": "Collect valuables", "details": "Then explore."},
    ]
    decision = agent.choose_decision(
        observation="You are in the cellar.",
        current_location="Cellar",
        valid_actions=["inventory", "look"],
        current_score=0,
        goal={
            "kind": "establish_return_route",
            "description": "Find the way home early.",
            "sequence": sequence,
            "ordinary_items": ["leaflet", "bottle of water"],
            "useful_items": ["brass lantern"],
        },
    )

    prompt = "\n".join(message["content"] for message in completions.request["messages"])
    assert "bottle of water" in prompt
    assert "leaflet" in prompt
    assert "useful_items" in prompt
    assert "Ordered strategic sequence" in prompt
    assert decision["report"]["treasure_candidates"] == ["jeweled egg"]
    assert decision["report"]["ordinary_items"] == ["leaflet", "bottle of water"]


def test_historian_updates_inventory_after_take_and_trophy_case_deposit(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Forest", "Forest", initial_score=0)
    historian.record_agent_report({
        "location": "Forest",
        "inventory_items": ["leaflet", "bottle of water"],
        "ordinary_items": ["leaflet", "bottle of water"],
    })

    historian.observe_step(
        before_node="Forest", before_location="Forest", action="take jeweled egg",
        after_node="Forest", after_location="Forest", score_before=0, score_after=5,
        observation_before="A nest holds a jeweled egg.", observation="Taken.",
        action_source="smart_actor",
    )
    assert "jeweled egg" in historian.agent_state["inventory_items"]
    assert "jeweled egg" in historian.agent_state["carried_treasures"]

    historian.observe_step(
        before_node="Living Room", before_location="Living Room",
        action="put jeweled egg in trophy case",
        after_node="Living Room", after_location="Living Room",
        score_before=5, score_after=7,
        observation_before="The trophy case is empty.",
        observation="You put the jeweled egg in the trophy case.",
        action_source="smart_actor",
    )
    assert "jeweled egg" not in historian.agent_state["inventory_items"]
    assert "jeweled egg" not in historian.agent_state["carried_treasures"]
    assert historian.memory["item_value_evidence"]["jeweled egg"]["confirmed_treasure"] is True


def test_surface_grating_observation_does_not_mark_player_underground(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Clearing", "Clearing", initial_score=0)
    historian.observe_step(
        before_node="Clearing", before_location="Clearing", action="examine grating",
        after_node="Clearing", after_location="Clearing", score_before=0, score_after=0,
        observation_before="A clearing.",
        observation="There is a metal grating in the ground.",
    )
    assert historian.episode["underground_reached"] is False
