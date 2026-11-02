import json

import pytest

from historian import Historian
from navigation_controller import NavigationController
from world_map import WorldMap


@pytest.fixture
def world_map(tmp_path):
    return WorldMap(save_path=tmp_path / "world_map.json")


def test_controller_executes_known_route_one_action_at_a_time(world_map):
    world_map.start_episode("A", 1)
    world_map.record_step("east", "B", next_location_id=2)
    world_map.record_step("north", "C", next_location_id=3)
    # Restore the episode's current pointer to A for this independent test.
    world_map.start_episode("A", 1)

    controller = NavigationController(world_map)
    controller.plan_to("C", destination_id=3)

    assert controller.next_action(["east", "west"]) == "east"
    result = controller.observe_action_result("east", "B", next_location_id=2)
    assert result["recorded_transition"] is True  # player changed rooms, even though edge exists already
    assert controller.next_action(["north", "south"]) == "north"
    result = controller.observe_action_result("north", "C", next_location_id=3)
    assert result["plan_status"] == "complete"
    assert controller.next_action(["north"]) is None


def test_controller_cancels_when_route_action_is_unavailable(world_map):
    world_map.start_episode("A", 1)
    world_map.record_step("east", "B", next_location_id=2)
    world_map.start_episode("A", 1)

    controller = NavigationController(world_map)
    controller.plan_to("B", destination_id=2)
    assert controller.next_action(["north", "south"]) is None
    assert controller.last_status == "planned_action_unavailable"


def test_controller_blocks_failed_edge_for_episode(world_map):
    world_map.start_episode("A", 1)
    world_map.record_step("east", "B", next_location_id=2)
    world_map.start_episode("A", 1)

    controller = NavigationController(world_map)
    route = controller.plan_to("B", destination_id=2)
    assert route["actions"] == ["east"]
    assert controller.next_action(["east"]) == "east"

    result = controller.observe_action_result("east", "A", next_location_id=1)
    assert result["plan_status"] == "route_edge_failed"
    assert len(controller.blocked_edges) == 1
    assert controller.plan_to("B", destination_id=2) is None


def test_controller_keeps_alternative_parallel_edge_after_failure(world_map):
    world_map.add_standard_location("A", 1)
    world_map.add_standard_location("B", 2)
    world_map.add_transition("A", "B", ["east"])
    world_map.add_transition("A", "B", ["north"])
    world_map.start_episode("A", 1)

    controller = NavigationController(world_map)
    route = controller.plan_to("B", destination_id=2)
    assert route["actions"] == ["east"]
    controller.next_action(["east", "north"])
    controller.observe_action_result("east", "A", next_location_id=1)

    alternative = controller.plan_to("B", destination_id=2)
    assert alternative["actions"] == ["north"]


def test_controller_records_unexpected_transition_and_cancels_plan(world_map):
    world_map.start_episode("A", 1)
    world_map.record_step("east", "B", next_location_id=2)
    world_map.add_standard_location("C", 3)
    world_map.start_episode("A", 1)

    controller = NavigationController(world_map)
    controller.plan_to("B", destination_id=2)
    controller.next_action(["east"])
    result = controller.observe_action_result("east", "C", next_location_id=3)

    assert result["deviation"] is True
    assert result["plan_status"] == "unexpected_transition"
    assert world_map.find_route("A", "C") == ["east"]


def test_historian_learns_score_positive_action_and_prioritizes_it(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Kitchen", "Kitchen", initial_score=0)
    historian.observe_step(
        before_node="Kitchen",
        before_location="Kitchen",
        action="take egg",
        after_node="Kitchen",
        after_location="Kitchen",
        score_before=0,
        score_after=5,
        observation="Taken.",
    )
    historian.finish_episode(final_score=5, moves=1)

    historian.start_episode("Kitchen", "Kitchen", initial_score=0)
    goal = historian.choose_goal(
        current_node="Kitchen",
        current_location="Kitchen",
        valid_actions=["take egg", "north"],
        known_routes=[],
    )

    assert goal["kind"] == "score_action"
    assert goal["suggested_action"] == "take egg"
    assert goal["expected_score_gain"] == 5


def test_historian_routes_to_remote_score_positive_location(world_map, tmp_path):
    world_map.start_episode("A", 1)
    world_map.record_step("east", "B", next_location_id=2)

    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("B", "B", initial_score=0)
    historian.observe_step(
        before_node="B",
        before_location="B",
        action="open chest",
        after_node="B",
        after_location="B",
        score_before=0,
        score_after=10,
    )
    historian.finish_episode(final_score=10, moves=1)

    historian.start_episode("A", "A", initial_score=0)
    routes = world_map.get_reachable_routes("A", max_destinations=10)
    goal = historian.choose_goal(
        current_node="A",
        current_location="A",
        valid_actions=["east", "look"],
        known_routes=routes,
    )

    assert goal["kind"] == "navigate_score_goal"
    assert goal["target_node"] == "B"
    assert goal["expected_score_gain"] == 10


def test_historian_warns_after_repeated_no_progress_actions(tmp_path):
    historian = Historian(tmp_path / "historian.json", repeat_threshold=2)
    historian.start_episode("A", "A", initial_score=0)
    for _ in range(3):
        historian.observe_step(
            before_node="A",
            before_location="A",
            action="north",
            after_node="A",
            after_location="A",
            score_before=0,
            score_after=0,
            observation_before="You can't go that way.",
            observation="You can't go that way.",
        )

    assert "north" in historian.get_avoid_actions("A")
    guidance = historian.build_guidance("A", "A")
    assert "north" in guidance


def test_historian_prefers_unmapped_exit_when_no_score_profile(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("A", "A", initial_score=0)

    goal = historian.choose_goal(
        current_node="A",
        current_location="A",
        valid_actions=["north", "south", "look"],
        known_routes=[],
        known_outgoing_actions=["north"],
    )

    assert goal["kind"] == "explore_frontier"
    assert goal["preferred_actions"] == ["south"]


def test_historian_persists_memory_between_instances(tmp_path):
    path = tmp_path / "historian.json"
    historian = Historian(path)
    historian.start_episode("A", "A", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="A", action="take coin",
        after_node="A", after_location="A", score_before=0, score_after=2,
    )
    historian.finish_episode(final_score=2, moves=1)

    loaded = Historian(path)
    assert len(loaded.memory["score_events"]) == 1
    assert loaded.memory["score_profiles"]
    with path.open() as file:
        assert json.load(file)["schema_version"] == 2


def test_historian_does_not_retry_failed_known_score_task_same_episode(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("A", "A", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="A", action="open chest",
        after_node="A", after_location="A", score_before=0, score_after=3,
    )
    # First action in this episode does not reproduce the previous score gain.
    historian.observe_step(
        before_node="A", before_location="A", action="open chest",
        after_node="A", after_location="A", score_before=0, score_after=0,
    )
    goal = historian.choose_goal(
        current_node="A", current_location="A",
        valid_actions=["open chest", "north"], known_routes=[],
    )
    assert goal["kind"] == "explore_frontier"
    assert goal["preferred_actions"] == ["north"]


def test_historian_does_not_permanently_suppress_exit_after_one_failure(tmp_path):
    historian = Historian(tmp_path / "historian.json", repeat_threshold=2)
    historian.start_episode("A", "A", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="A", action="north",
        after_node="A", after_location="A", score_before=0, score_after=0,
    )
    goal = historian.choose_goal(
        current_node="A", current_location="A",
        valid_actions=["north", "south"], known_routes=[],
    )
    assert goal["kind"] == "explore_frontier"
    assert goal["preferred_actions"] == ["north", "south"]


def test_controller_can_follow_identity_based_maze_cells(world_map):
    world_map.start_episode("Troll Room", 1)
    world_map.record_step("west", "Maze", next_location_id=100)
    world_map.record_step("west", "Maze", next_location_id=101)
    world_map.record_step("up", "Skeleton Room", next_location_id=200)
    world_map.start_episode("Troll Room", 1)

    controller = NavigationController(world_map)
    plan = controller.plan_to("Skeleton Room", destination_id=200)
    assert plan["actions"] == ["west", "west", "up"]

    action = controller.next_action(["west", "east"])
    assert action == "west"
    controller.observe_action_result("west", "Maze", next_location_id=100)
    assert controller.next_action(["west", "east"]) == "west"
    controller.observe_action_result("west", "Maze", next_location_id=101)
    assert controller.next_action(["up", "down"]) == "up"
    final = controller.observe_action_result("up", "Skeleton Room", next_location_id=200)
    assert final["plan_status"] == "complete"


def test_simple_agent_includes_historian_goal_and_navigation_in_prompt():
    from types import SimpleNamespace
    from smart_agent import SmartAgent

    class FakeCompletions:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"north","inventory_items":[],"carried_treasures":[],"puzzle_clues":[],"state_note":""}'))]
            )

    fake = FakeCompletions()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake)
    )
    agent = SmartAgent(client=fake_client, model="fake")

    action = agent.choose_action(
        observation="There is a wooden door to the north.",
        current_location="Hall",
        valid_actions=["north", "south", "look"],
        known_locations=["Hall", "Kitchen"],
        known_routes=[{
            "destination_label": "Kitchen",
            "total_cost": 2,
            "actions": ["south", "east"],
        }],
        goal={"kind": "explore_frontier", "description": "Explore an unmapped exit."},
        historian_guidance="Do not repeat north if it produced no progress.",
        avoid_actions=["north"],
        preferred_actions=["south"],
    )

    assert action == "north"
    prompt = fake.request["messages"][1]["content"]
    assert "Explore an unmapped exit" in prompt
    assert "Kitchen" in prompt
    assert "Do not repeat north" in prompt
    assert "north" in prompt


def test_historian_reduces_score_value_after_unsuccessful_replay(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("A", "A", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="A", action="open box",
        after_node="A", after_location="A", score_before=0, score_after=5,
    )
    # First episode had a +5 gain. In a later episode the same action no longer scores.
    historian.finish_episode(final_score=5, moves=1)
    historian.start_episode("A", "A", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="A", action="open box",
        after_node="A", after_location="A", score_before=0, score_after=0,
    )
    historian.finish_episode(final_score=0, moves=1)

    profile = next(iter(historian.memory["score_profiles"].values()))
    assert profile["attempts"] == 2
    assert profile["samples"] == 1
    assert profile["total_gain"] == 5

    historian.start_episode("A", "A", initial_score=0)
    goal = historian.choose_goal(
        current_node="A", current_location="A",
        valid_actions=["open box", "north"], known_routes=[],
    )
    # Average expected gain is now 2.5, not 5, after a zero-score attempt.
    assert goal["expected_score_gain"] == 2.5


def test_historian_does_not_treat_room_preserving_state_change_as_failure(tmp_path):
    historian = Historian(tmp_path / "historian.json", repeat_threshold=2)
    historian.start_episode("A", "A", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="A", action="open door",
        after_node="A", after_location="A", score_before=0, score_after=0,
        observation_before="There is a closed wooden door.",
        observation="The wooden door is now open.",
    )
    assert historian.memory["repeat_failures"][json.dumps(["A", "open door"], separators=(",", ":"))]["no_progress_count"] == 0
    assert "open door" not in historian.get_avoid_actions("A")


def test_historian_goal_is_to_exceed_75_not_optimize_speed(tmp_path):
    historian = Historian(tmp_path / "historian.json", target_score=76)
    historian.start_episode("A", "A", initial_score=0)
    goal = historian.choose_goal(
        current_node="A",
        current_location="A",
        valid_actions=["north", "look"],
        known_routes=[],
        current_score=75,
    )
    assert goal["kind"] != "target_reached"
    assert historian.target_score == 76
    guidance = historian.build_guidance("A", "A", current_score=75)
    assert "exceed 75 points" in guidance
    assert "Do not optimize for fewest moves" in guidance


def test_historian_stops_when_score_is_greater_than_75(tmp_path):
    historian = Historian(tmp_path / "historian.json", target_score=76)
    historian.start_episode("A", "A", initial_score=0)
    goal = historian.choose_goal(
        current_node="A",
        current_location="A",
        valid_actions=["look"],
        known_routes=[],
        current_score=76,
    )
    assert goal["kind"] == "target_reached"


def test_historian_prioritizes_returning_carried_treasure(world_map, tmp_path):
    world_map.start_episode("Underground", 1)
    world_map.record_step("up", "Living Room", next_location_id=2)
    world_map.start_episode("Underground", 1)
    routes = world_map.get_reachable_routes("Underground", max_destinations=20)
    home = world_map.find_nodes_by_label("Living Room")[0]

    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Underground", "Underground", initial_score=0)
    historian.record_agent_report({
        "location": "Underground",
        "inventory_items": ["jeweled egg"],
        "carried_treasures": ["jeweled egg"],
        "puzzle_clues": [],
    })
    goal = historian.choose_goal(
        current_node="Underground",
        current_location="Underground",
        valid_actions=["north", "look"],
        known_routes=routes,
        current_score=10,
        home_node=home,
    )
    assert goal["kind"] == "navigate_to_home"
    assert goal["target_node"] == home
    assert goal["carried_treasures"] == ["jeweled egg"]


def test_historian_prompts_return_route_puzzle_when_home_route_unknown(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Maze", "Maze", initial_score=0)
    historian.record_agent_report({
        "location": "Maze",
        "inventory_items": ["golden egg"],
        "carried_treasures": ["golden egg"],
        "puzzle_clues": ["A large guardian blocks the passage."],
    })
    goal = historian.choose_goal(
        current_node="Maze",
        current_location="Maze",
        valid_actions=["north", "south", "look"],
        known_routes=[],
        current_score=10,
        home_node="Living Room",
    )
    assert goal["kind"] == "solve_return_route"
    assert "Living Room" in goal["description"]
    assert "alternate routes" in goal["description"]


def test_historian_prioritizes_trophy_case_when_home_and_treasure_carried(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Living Room", "Living Room", initial_score=10)
    historian.record_agent_report({
        "location": "Living Room",
        "inventory_items": ["painting"],
        "carried_treasures": ["painting"],
        "puzzle_clues": [],
    })
    goal = historian.choose_goal(
        current_node="Living Room",
        current_location="Living Room",
        valid_actions=["put painting in trophy case", "look"],
        known_routes=[],
        current_score=10,
        home_node="Living Room",
    )
    assert goal["kind"] == "deposit_treasure"
    assert "trophy case" in goal["description"]


def test_historian_persists_actor_clues_without_walkthrough(tmp_path):
    path = tmp_path / "historian.json"
    historian = Historian(path)
    historian.start_episode("A", "A", initial_score=0)
    historian.record_agent_report({
        "location": "A",
        "inventory_items": ["small key"],
        "carried_treasures": [],
        "puzzle_clues": ["A locked door has a narrow keyhole."],
    })
    historian.save()

    loaded = Historian(path)
    assert any(
        "narrow keyhole" in item["clue"]
        for item in loaded.memory["puzzle_clues"]
    )


def test_smart_agent_returns_action_and_state_report():
    from types import SimpleNamespace
    from smart_agent import SmartAgent

    class FakeCompletions:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "action": "take egg",
                    "inventory_items": ["jeweled egg", "lantern"],
                    "carried_treasures": ["jeweled egg"],
                    "puzzle_clues": ["A locked door has a narrow keyhole."],
                    "state_note": "An item was acquired.",
                })))]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = SmartAgent(client=client, model="fake")
    decision = agent.choose_decision(
        observation="There is a nest with a jeweled egg.",
        current_location="Forest",
        valid_actions=["take egg", "look"],
        current_score=5,
        target_score=76,
        goal={"kind": "explore_frontier", "description": "Explore."},
    )
    assert decision["action"] == "take egg"
    assert decision["report"]["carried_treasures"] == ["jeweled egg"]
    assert "narrow keyhole" in decision["report"]["puzzle_clues"][0]

    prompt = completions.request["messages"][0]["content"]
    assert "greater than 75" in prompt
    assert "score feedback" in prompt
    # The Actor is not supplied the guide's named puzzle solution.
    assert "Odysseus" not in prompt
    assert "skeleton key" not in prompt
    assert "pray at the altar" not in prompt


def test_historian_does_not_discount_score_goal_by_route_length(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("B", "B", initial_score=0)
    historian.observe_step(
        before_node="B",
        before_location="B",
        action="open chest",
        after_node="B",
        after_location="B",
        score_before=0,
        score_after=12,
    )
    historian.finish_episode(final_score=12, moves=1)
    historian.start_episode("A", "A", initial_score=0)

    # Even a long route remains a valid learned score goal; route cost does
    # not divide or otherwise diminish its expected score gain.
    goal = historian.choose_goal(
        current_node="A",
        current_location="A",
        valid_actions=["east"],
        known_routes=[{
            "destination": "B",
            "destination_label": "B",
            "total_cost": 50,
            "actions": ["east"] * 50,
        }],
        current_score=0,
    )
    assert goal["kind"] == "navigate_score_goal"
    assert goal["expected_score_gain"] == 12
    assert "utility" not in goal


def test_historian_return_goal_when_home_not_discovered(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Maze", "Maze", initial_score=0)
    historian.record_agent_report({
        "location": "Maze",
        "inventory_items": ["silver object"],
        "carried_treasures": ["silver object"],
        "puzzle_clues": [],
    })
    goal = historian.choose_goal(
        current_node="Maze",
        current_location="Maze",
        valid_actions=["west", "look"],
        known_routes=[],
        current_score=5,
        home_node=None,
    )
    assert goal["kind"] == "solve_return_route"
    assert goal["target_label"] == "Living Room"


def test_navigation_controller_plan_to_home(world_map):
    world_map.start_episode("Underground", 1)
    world_map.record_step("up", "Living Room", next_location_id=2)
    world_map.start_episode("Underground", 1)
    home = world_map.find_nodes_by_label("Living Room")[0]

    controller = NavigationController(world_map)
    route = controller.plan_to_home(home)
    assert route["destination"] == home
    assert route["actions"] == ["up"]


def test_world_map_find_nodes_by_label_disambiguates_rooms(world_map):
    world_map.add_standard_location("Maze", 10)
    world_map.add_standard_location("Maze", 11)
    nodes = world_map.find_nodes_by_label("Maze")
    assert len(nodes) == 2
    assert world_map.display_location(nodes[0]) != world_map.display_location(nodes[1])
