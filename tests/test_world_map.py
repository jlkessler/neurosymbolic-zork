import pytest

from MANGORL.world_map import WorldMap


@pytest.fixture
def world_map(tmp_path):
    return WorldMap(save_path=tmp_path / "world_map.json")


def test_non_movement_interaction_does_not_add_transition(world_map):
    world_map.start_episode("Living Room", location_id=10)

    recorded = world_map.record_step(
        "put treasure in trophy case",
        "Living Room",
        observation="The treasure is safely inside the case.",
        next_location_id=10,
    )

    assert recorded is False
    assert world_map.graph.number_of_nodes() == 1
    assert world_map.graph.number_of_edges() == 0
    assert world_map.current_location == "Living Room"


def test_failed_directional_action_does_not_add_transition(world_map):
    world_map.start_episode("Living Room", location_id=10)

    recorded = world_map.record_step(
        "north",
        "Living Room",
        observation="You cannot go that way.",
        next_location_id=10,
    )

    assert recorded is False
    assert world_map.graph.number_of_edges() == 0


def test_same_display_name_different_room_ids_creates_distinct_nodes(world_map):
    world_map.start_episode("Maze", location_id=100)

    recorded = world_map.record_step(
        "west",
        "Maze",
        next_location_id=101,
    )

    destination = world_map.resolve_location("Maze", location_id=101)

    assert recorded is True
    assert destination is not None
    assert destination != "Maze"
    assert world_map.graph.nodes["Maze"]["room_id"] == 100
    assert world_map.graph.nodes[destination]["room_id"] == 101
    assert world_map.find_route("Maze", destination) == ["west"]


def test_pray_is_recorded_when_room_identity_changes(world_map):
    world_map.start_episode("Altar", location_id=20)

    recorded = world_map.record_step(
        "pray",
        "Forest",
        next_location_id=21,
    )

    assert recorded is True
    assert world_map.find_route("Altar", "Forest") == ["pray"]
    edge_data = world_map.graph.get_edge_data("Altar", "Forest")
    assert next(iter(edge_data.values()))["direction"] is None


def test_ordinary_room_transition_is_recorded(world_map):
    world_map.start_episode("West of House", location_id=1)

    recorded = world_map.record_step(
        "north",
        "North of House",
        next_location_id=2,
    )

    assert recorded is True
    assert world_map.find_route("West of House", "North of House") == ["north"]


def test_direction_alias_is_normalized(world_map):
    world_map.start_episode("A", location_id=1)
    world_map.record_step("n", "B", next_location_id=2)

    edge = world_map.graph.get_edge_data("A", "B")
    data = next(iter(edge.values()))
    assert data["direction"] == "north"
    # Preserve the actual command to replay in Jericho.
    assert data["actions"] == ["n"]


def test_shortest_route_minimizes_action_count(world_map):
    world_map.add_standard_location("A", 1)
    world_map.add_standard_location("B", 2)
    world_map.add_standard_location("D", 4)

    world_map.add_transition("A", "B", ["north"])
    world_map.add_transition("B", "D", ["east"])
    world_map.add_transition("A", "D", ["south", "south", "west"])

    assert world_map.find_route("A", "D") == ["north", "east"]
    assert world_map.find_route_details("A", "D")["total_cost"] == 2


def test_cheapest_parallel_edge_is_selected(world_map):
    world_map.add_standard_location("A", 1)
    world_map.add_standard_location("B", 2)

    world_map.add_transition("A", "B", ["north", "east"])
    world_map.add_transition("A", "B", ["west"])

    assert world_map.graph.number_of_edges("A", "B") == 2
    assert world_map.find_route("A", "B") == ["west"]


def test_exact_duplicate_transition_is_not_added(world_map):
    world_map.add_standard_location("A", 1)
    world_map.add_standard_location("B", 2)

    assert world_map.add_transition("A", "B", ["north"]) is True
    assert world_map.add_transition("A", "B", ["north"]) is False
    assert world_map.graph.number_of_edges("A", "B") == 1


def test_graph_respects_direction(world_map):
    world_map.start_episode("Living Room", location_id=1)
    world_map.record_step("west", "Kitchen", next_location_id=2)

    assert world_map.find_route("Living Room", "Kitchen") == ["west"]
    assert world_map.find_route("Kitchen", "Living Room") is None


def test_unreachable_destination_returns_none(world_map):
    world_map.add_standard_location("A", 1)
    world_map.add_standard_location("B", 2)
    world_map.add_standard_location("C", 3)
    world_map.add_transition("A", "B", ["north"])

    assert world_map.find_route("A", "C") is None


def test_unknown_location_returns_none(world_map):
    assert world_map.find_route("Unknown A", "Unknown B") is None


def test_same_start_and_destination_returns_empty_route(world_map):
    world_map.start_episode("West of House", location_id=1)

    assert world_map.find_route("West of House", "West of House") == []


def test_route_through_multiple_same_named_maze_rooms(world_map):
    world_map.start_episode("Troll Room", location_id=1)
    world_map.record_step("west", "Maze", next_location_id=100)
    world_map.record_step("west", "Maze", next_location_id=101)
    world_map.record_step("up", "Skeleton Room", next_location_id=200)

    route = world_map.find_route("Troll Room", "Skeleton Room")

    assert route == ["west", "west", "up"]
    assert len([n for n, d in world_map.graph.nodes(data=True)
                if d.get("label") == "Maze"]) == 2


def test_find_route_details_contains_labels_and_transitions(world_map):
    world_map.start_episode("A", location_id=1)
    world_map.record_step("north", "B", next_location_id=2)
    world_map.record_step("east", "C", next_location_id=3)

    details = world_map.find_route_details("A", "C")

    assert details is not None
    assert details["total_cost"] == 2
    assert details["actions"] == ["north", "east"]
    assert details["nodes"] == ["A", "B", "C"]
    assert len(details["transitions"]) == 2
    assert details["destination_label"] == "C"


def test_reachable_routes_include_maze_cells_and_readable_labels(world_map):
    world_map.start_episode("Troll Room", location_id=1)
    world_map.record_step("west", "Maze", next_location_id=100)
    world_map.record_step("west", "Maze", next_location_id=101)
    world_map.record_step("up", "Skeleton Room", next_location_id=200)

    routes = world_map.get_reachable_routes(
        "Troll Room", start_id=1, max_destinations=10
    )

    labels = [route["destination_label"] for route in routes]
    assert "Skeleton Room" in labels
    assert any(label.startswith("Maze (room #") for label in labels)

    skeleton_route = next(
        route for route in routes
        if route["destination_label"] == "Skeleton Room"
    )
    assert skeleton_route["actions"] == ["west", "west", "up"]


def test_duplicate_room_name_is_resolved_using_room_id(world_map):
    world_map.start_episode("Maze", location_id=100)
    world_map.record_step("west", "Maze", next_location_id=101)

    first = world_map.resolve_location("Maze", location_id=100)
    second = world_map.resolve_location("Maze", location_id=101)

    assert first == "Maze"
    assert second != first
    assert world_map.find_route(first, second) == ["west"]


def test_save_and_load_preserves_room_ids_and_routes(world_map):
    world_map.start_episode("Troll Room", location_id=1)
    world_map.record_step("west", "Maze", next_location_id=100)
    world_map.record_step("west", "Maze", next_location_id=101)
    world_map.record_step("up", "Skeleton Room", next_location_id=200)

    world_map.save()

    loaded = WorldMap(save_path=world_map.save_path)
    loaded.instantiate()

    assert loaded.resolve_location("Maze", location_id=100) is not None
    assert loaded.resolve_location("Maze", location_id=101) is not None
    assert loaded.find_route("Troll Room", "Skeleton Room") == [
        "west", "west", "up"
    ]


def test_legacy_maze_cleanup_is_opt_in(world_map):
    world_map.graph.add_node("old-maze", node_type="maze", label="Maze")
    world_map.add_standard_location("A", 1)
    world_map.add_transition("A", "old-maze", ["west"], edge_type="maze_entry")
    world_map.save()

    loaded = WorldMap(save_path=world_map.save_path)
    loaded.instantiate()
    assert "old-maze" in loaded.graph

    cleaned = WorldMap(save_path=world_map.save_path)
    cleaned.instantiate(discard_legacy_maze_nodes=True)
    assert "old-maze" not in cleaned.graph
