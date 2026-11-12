"""Regressions from the tabula-rasa-2 room aliases and movement cycles."""
from MANGORL.historian import Historian
from MANGORL.world_map import WorldMap


def context(title, description="A familiar room.", inventory="A lantern."):
    return (f"LAST ACTION RESULT:\nDone.\n\nCURRENT ROOM (look):\n{title}\n{description}"
            f"\n\nCURRENT INVENTORY:\n{inventory}")


def test_observed_title_recovers_truncated_room_without_changing_identity(tmp_path):
    world = WorldMap(tmp_path / "world.json")
    key = world.start_episode("Living ", 193)
    assert world.find_nodes_by_label("Living Room") == []
    assert world.observe_room_title(key, context("Living Room"))
    assert world.find_nodes_by_label("Living Room") == [key]
    assert world.resolve_location("Living Room") == key
    assert world.resolve_location("Living ", 193) == key
    assert world.graph.nodes[key]["label"] == "Living "
    assert world.display_location(key) == "Living Room"
    world.save()
    loaded = WorldMap(world.save_path)
    loaded.load()
    assert loaded.find_nodes_by_label("Living Room") == [key]


def test_room_title_does_not_guess_alias_from_darkness_or_action_text(tmp_path):
    world = WorldMap(tmp_path / "world.json")
    key = world.start_episode("Living ", 193)
    assert not world.observe_room_title(key, "Living Room\nA trophy case.")
    assert not world.observe_room_title(key, context("It is pitch black."))
    assert not world.observe_room_title(key, context("Living things surround you."))
    assert not world.observe_room_title(key, context("Gallery"))
    assert world.find_nodes_by_label("Living Room") == []


def test_duplicate_observed_titles_do_not_merge_room_objects(tmp_path):
    world = WorldMap(tmp_path / "world.json")
    left = world.add_standard_location("Maze", 1)
    right = world.add_standard_location("Maze", 2)
    world.observe_room_title(left, context("Maze"))
    world.observe_room_title(right, context("Maze"))
    assert left != right
    assert world.resolve_location("Maze", 1) == left
    assert world.resolve_location("Maze", 2) == right
    assert len(world.find_nodes_by_label("Maze")) == 2
    assert "room #" in world.display_location(left)


def make_cycle(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Gallery", "Gallery", initial_score=0)
    for _ in range(4):
        for before, action, after in [("Gallery", "south", "Studio"), ("Studio", "north", "Gallery")]:
            historian.observe_step(
                before_node=before, before_location=before, action=action,
                after_node=after, after_location=after, score_before=0, score_after=0,
                observation=f"{after}\nThe room is unchanged.", inventory_snapshot=["lantern"],
            )
    return historian


def test_repeated_room_cycle_temporarily_excludes_old_route_and_prefers_new_exit(tmp_path):
    historian = make_cycle(tmp_path)
    recovery = historian.cycle_recovery("Gallery", ["south", "east", "look"])
    assert recovery["blocked_actions"] == ["south"]
    assert recovery["preferred_actions"] == ["east"]
    assert recovery["repeated_transitions"][0]["count"] == 3
    assert historian.episode["repeated_actions"][-1]["kind"] == "movement_cycle"
    # This is current-episode evidence, not a permanent map/command ban.
    assert "south" not in historian.get_avoid_actions("Gallery")


def test_cycle_guard_keeps_only_exit_and_purposeful_backtracking_available(tmp_path):
    historian = make_cycle(tmp_path)
    assert historian.cycle_recovery("Gallery", ["south", "look"])["blocked_actions"] == []
    assert historian.cycle_recovery("Gallery", ["south", "east"], protected_actions=["s"])["blocked_actions"] == ["south"]


def test_grounded_inventory_change_releases_temporary_cycle_exclusion(tmp_path):
    historian = make_cycle(tmp_path)
    historian.observe_step(
        before_node="Gallery", before_location="Gallery", action="take key",
        after_node="Gallery", after_location="Gallery", score_before=0, score_after=0,
        observation="Taken.", inventory_snapshot=["lantern", "key"],
    )
    assert historian.cycle_recovery("Gallery", ["south", "east"])["blocked_actions"] == []


def test_changed_room_snapshot_releases_cycle_but_movement_result_does_not(tmp_path):
    historian = make_cycle(tmp_path)
    before = context("Gallery", "The door is closed.")
    assert historian.cycle_recovery("Gallery", ["south", "east"], state_context=before)["blocked_actions"]
    other_action_result = before.replace("Done.", "You return north.")
    assert historian.cycle_recovery("Gallery", ["south", "east"], state_context=other_action_result)["blocked_actions"]
    after = context("Gallery", "The door is open.")
    assert not historian.cycle_recovery("Gallery", ["south", "east"], state_context=after)["blocked_actions"]


def test_new_episode_releases_cycle_exclusion(tmp_path):
    historian = make_cycle(tmp_path)
    historian.start_episode("Gallery", "Gallery", initial_score=0)
    assert not historian.cycle_recovery("Gallery", ["south", "east"])["blocked_actions"]


def test_home_discovery_is_distinct_from_unknown_return_route(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("Forest", "Forest", initial_score=0)
    historian.record_agent_report({"inventory_items": ["jeweled egg"], "carried_treasures": ["jeweled egg"]})
    goal = historian.choose_goal(
        current_node="Forest", current_location="Forest", valid_actions=["east"], known_routes=[], home_node=None,
    )
    assert goal["route_status"] == "home_undiscovered"
    assert goal["source"] == "home_discovery"
    assert "continuing discovery is necessary" in goal["description"]


def test_stale_return_goal_reframes_exploration_after_bounded_attempts(tmp_path):
    historian = make_cycle(tmp_path)
    historian.episode["underground_reached"] = True
    historian.episode["steps"] = 20
    historian.agent_state.update(inventory_items=["painting"], carried_treasures=["painting"])
    goal = historian.choose_goal(
        current_node="Gallery", current_location="Gallery", valid_actions=["south", "east"],
        known_routes=[], home_node="Living Room", home_route_status="blocked",
    )
    assert goal["kind"] == "explore_alternative"
    assert goal["preferred_actions"] == ["east"]
    # A viable destination-directed path retains its priority.
    routed = historian.choose_goal(
        current_node="Gallery", current_location="Gallery", valid_actions=["south", "east"],
        known_routes=[], home_node="Living Room", home_route_status="confirmed",
    )
    assert routed["kind"] == "navigate_to_home"
