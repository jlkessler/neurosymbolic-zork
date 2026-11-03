import json
from pathlib import Path

import networkx as nx


class WorldMap:
    """Persistent, directed map of Jericho room-object transitions.

    Jericho's ZObject.num is used as the room identity when available. This
    allows multiple game rooms with the same printed name (for example,
    several rooms all called "Maze") to remain distinct graph nodes.
    """

    def __init__(self, save_path="memory/world_map.json"):
        self.save_path = Path(save_path)
        self.graph = nx.MultiDiGraph()
        self.current_location = None       # canonical graph node key
        self.current_location_name = None  # readable Jericho room name

        self.direction_aliases = {
            "n": "north", "north": "north",
            "s": "south", "south": "south",
            "e": "east", "east": "east",
            "w": "west", "west": "west",
            "ne": "northeast", "northeast": "northeast",
            "nw": "northwest", "northwest": "northwest",
            "se": "southeast", "southeast": "southeast",
            "sw": "southwest", "southwest": "southwest",
            "u": "up", "up": "up",
            "d": "down", "down": "down",
        }

    # ------------------------------------------------------------------
    # INITIALIZATION AND PERSISTENCE
    # ------------------------------------------------------------------

    def instantiate(self, discard_legacy_maze_nodes=False):
        """Load an existing graph, or initialize a new graph.

        Set discard_legacy_maze_nodes=True only if you intentionally want to
        remove synthetic maze nodes/edges created by the older map design.
        """
        if self.save_path.exists():
            self.load(discard_legacy_maze_nodes=discard_legacy_maze_nodes)
            print(f"Loaded world map from {self.save_path}")
        else:
            self.graph = nx.MultiDiGraph()
            print("Initialized a new world map.")

    def load(self, discard_legacy_maze_nodes=False):
        """Load a graph saved using NetworkX node-link JSON data."""
        with self.save_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.graph = nx.node_link_graph(data)

        # Removing legacy maze nodes is destructive, so it is opt-in.
        if discard_legacy_maze_nodes:
            legacy_maze_nodes = [
                node
                for node, attributes in self.graph.nodes(data=True)
                if attributes.get("node_type") == "maze"
            ]
            self.graph.remove_nodes_from(legacy_maze_nodes)

            legacy_maze_edges = [
                (source, target, key)
                for source, target, key, attributes
                in self.graph.edges(keys=True, data=True)
                if attributes.get("edge_type") in {"maze_entry", "maze_exit"}
            ]
            self.graph.remove_edges_from(legacy_maze_edges)

        # Episode state must always be initialized by start_episode().
        self.current_location = None
        self.current_location_name = None

    def save(self):
        """Persist the current graph as JSON."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)

        with self.save_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        # print(f"World map saved to {self.save_path}")

    # ------------------------------------------------------------------
    # ROOM IDENTITY AND NODE MANAGEMENT
    # ------------------------------------------------------------------

    def room_key(self, location, location_id=None):
        """Return a canonical graph key for a Jericho location.

        When location_id is available, it is the authoritative identity.
        Repeated display names are represented as e.g. "Maze" and
        "room:123", while each node retains the readable label "Maze".
        """
        if location_id is not None:
            for node, attributes in self.graph.nodes(data=True):
                if attributes.get("room_id") == location_id:
                    return node

        # Preserve readable keys for the first room with a given name.
        if location not in self.graph:
            return location

        attributes = self.graph.nodes[location]
        if attributes.get("node_type") == "location":
            existing_id = attributes.get("room_id")
            if existing_id is None or existing_id == location_id:
                return location

        # Same display name, but a different underlying Z-machine object.
        if location_id is None:
            return location

        candidate = f"room:{location_id}"
        if candidate in self.graph:
            candidate_id = self.graph.nodes[candidate].get("room_id")
            if candidate_id != location_id:
                candidate = f"room:{location_id}:{location}"

        return candidate

    def add_standard_location(self, location, location_id=None):
        """Add/update a room node and return its canonical graph key."""
        key = self.room_key(location, location_id)
        self.graph.add_node(
            key,
            node_type="location",
            label=location,
            room_id=location_id,
        )
        return key

    def resolve_location(self, location, location_id=None):
        """Resolve a readable location and optional Jericho ID to a node key.

        Supplying location_id is strongly recommended when names can repeat.
        A plain name resolves to the node with that exact key, if present;
        otherwise it resolves only when exactly one node has that label.
        """
        if location_id is not None:
            for node, attributes in self.graph.nodes(data=True):
                if attributes.get("room_id") == location_id:
                    return node

            # Permit an older node without a stored ID to be resolved by name.
            if location in self.graph:
                existing_id = self.graph.nodes[location].get("room_id")
                if existing_id is None:
                    return location
            return None

        if location in self.graph:
            return location

        matches = [
            node
            for node, attributes in self.graph.nodes(data=True)
            if attributes.get("node_type") == "location"
            and attributes.get("label") == location
        ]
        return matches[0] if len(matches) == 1 else None

    def display_location(self, node):
        """Return a readable label, disambiguating repeated room names."""
        if node not in self.graph:
            return str(node)

        attributes = self.graph.nodes[node]
        label = attributes.get("label", str(node))
        matches = [
            other
            for other, data in self.graph.nodes(data=True)
            if data.get("node_type") == "location"
            and data.get("label", str(other)) == label
        ]

        if len(matches) <= 1:
            return label

        room_id = attributes.get("room_id")
        suffix = f"room #{room_id}" if room_id is not None else str(node)
        return f"{label} ({suffix})"

    def find_nodes_by_label(self, label):
        """Return canonical nodes whose readable room label matches."""
        wanted = str(label).casefold()
        return [
            node for node, attributes in self.graph.nodes(data=True)
            if attributes.get("node_type") == "location"
            and str(attributes.get("label", node)).casefold() == wanted
        ]

    def get_known_locations(self):
        """Return human-readable labels for all known ordinary room nodes."""
        locations = [
            node
            for node, attributes in self.graph.nodes(data=True)
            if attributes.get("node_type") == "location"
        ]
        return sorted(self.display_location(node) for node in locations)

    # ------------------------------------------------------------------
    # EDGE MANAGEMENT AND OBSERVATION RECORDING
    # ------------------------------------------------------------------

    def _direction_for_action(self, action):
        normalized = action.strip().lower()
        if normalized.startswith("go "):
            normalized = normalized[3:].strip()
        return self.direction_aliases.get(normalized)

    def add_transition(self, source, destination, actions, edge_type="standard"):
        """Add a directed action-labeled edge, unless it already exists."""
        if isinstance(actions, str):
            actions = [actions]
        else:
            actions = list(actions)

        if not actions:
            return False

        existing_edges = self.graph.get_edge_data(
            source, destination, default={}
        )

        for data in existing_edges.values():
            if (
                data.get("edge_type") == edge_type
                and data.get("actions") == actions
            ):
                return False

        # A direction label is meaningful when this edge represents one
        # direction command. Multi-action macro edges need no single direction.
        direction = (
            self._direction_for_action(actions[0])
            if len(actions) == 1
            else None
        )

        self.graph.add_edge(
            source,
            destination,
            actions=actions,
            cost=len(actions),
            edge_type=edge_type,
            direction=direction,
        )
        return True

    def start_episode(self, start_location, location_id=None):
        """Set the current room at the beginning of an episode."""
        self.current_location = self.add_standard_location(
            start_location, location_id
        )
        self.current_location_name = start_location
        return self.current_location

    def record_step(
        self,
        action,
        next_location,
        observation=None,
        next_location_id=None,
    ):
        """Record a transition only when the underlying room identity changes.

        A failed command or an interaction within the same room normally keeps
        the same ZObject.num and is ignored. A different room object with the
        same printed name (such as another maze room) becomes a distinct node.
        """
        if self.current_location is None:
            raise RuntimeError("Call start_episode() before record_step().")

        old_location = self.current_location
        destination = self.room_key(next_location, next_location_id)

        # Same Z-machine room object: no navigation transition occurred.
        if destination == old_location:
            return False

        destination = self.add_standard_location(
            next_location, next_location_id
        )

        self.add_transition(
            source=old_location,
            destination=destination,
            actions=[action],
            edge_type="standard",
        )

        self.current_location = destination
        self.current_location_name = next_location
        return True

    # ------------------------------------------------------------------
    # ROUTE CALCULATION
    # ------------------------------------------------------------------

    def find_route_details(
        self,
        start,
        destination,
        start_id=None,
        destination_id=None,
    ):
        """Return the minimum-action route and its graph transitions.

        start/destination may be canonical node keys. For readable room names,
        pass the corresponding Jericho object number as start_id/destination_id
        whenever duplicate names are possible.
        """
        start_node = self.resolve_location(start, start_id)
        destination_node = self.resolve_location(destination, destination_id)

        if start_node is None or destination_node is None:
            return None

        try:
            total_cost, node_path = nx.single_source_dijkstra(
                self.graph,
                source=start_node,
                target=destination_node,
                weight="cost",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        transitions = []
        all_actions = []

        for source, target in zip(node_path, node_path[1:]):
            parallel_edges = self.graph.get_edge_data(source, target, default={})
            if not parallel_edges:
                return None

            edge_key, edge_data = min(
                parallel_edges.items(),
                key=lambda item: (
                    item[1].get("cost", len(item[1].get("actions", []))),
                    len(item[1].get("actions", [])),
                ),
            )

            actions = list(edge_data.get("actions", []))
            transitions.append({
                "source": source,
                "source_label": self.display_location(source),
                "actions": actions,
                "destination": target,
                "destination_label": self.display_location(target),
                "edge_type": edge_data.get("edge_type", "standard"),
                "edge_key": edge_key,
            })
            all_actions.extend(actions)

        return {
            "start": start_node,
            "start_label": self.display_location(start_node),
            "destination": destination_node,
            "destination_label": self.display_location(destination_node),
            "total_cost": total_cost,
            "nodes": node_path,
            "transitions": transitions,
            "actions": all_actions,
        }

    def find_route(self, start, destination, start_id=None, destination_id=None):
        """Return a flat action list, or None when no known route exists."""
        route = self.find_route_details(
            start,
            destination,
            start_id=start_id,
            destination_id=destination_id,
        )
        return None if route is None else route["actions"]

    def get_reachable_routes(self, start, max_destinations=8, start_id=None):
        """Return the cheapest known routes from start to known room nodes.

        Each result includes a canonical destination key, readable label,
        actions, cost, and transition details. Maze cells are ordinary location
        nodes in this identity-based representation, so they remain routable.
        """
        start_node = self.resolve_location(start, start_id)
        if start_node is None:
            return []

        routes = []
        for destination, attributes in self.graph.nodes(data=True):
            if destination == start_node:
                continue
            if attributes.get("node_type") != "location":
                continue

            details = self.find_route_details(start_node, destination)
            if details is None:
                continue

            routes.append({
                "destination": destination,
                "destination_label": self.display_location(destination),
                "total_cost": details["total_cost"],
                "actions": details["actions"],
                "transitions": details["transitions"],
            })

        routes.sort(
            key=lambda route: (
                route["total_cost"],
                route["destination_label"],
                str(route["destination"]),
            )
        )
        return routes[:max_destinations]

    # ------------------------------------------------------------------
    # DEBUGGING
    # ------------------------------------------------------------------

    def print_graph(self):
        """Print all known directed transitions."""
        print("\n===== WORLD MAP =====\n")
        for source, destination, data in self.graph.edges(data=True):
            direction = data.get("direction")
            relation = f"{direction}: " if direction else ""
            print(
                f"{self.display_location(source)} "
                f"--{relation}{data.get('actions', [])}--> "
                f"{self.display_location(destination)}"
            )
        print("\n===== END OF WORLD MAP =====\n")
