import json
from pathlib import Path

import networkx as nx

class WorldMap:
    def __init__(self, save_path="memory/world_map.json"):
        self.save_path = Path(save_path)

        # Directed multigraph: multiple edges between nodes are allowed, which is useful for modeling different actions leading to the same next state.
        self.graph = nx.MultiDiGraph()

        # State used while recording an episode
        self.current_location = None
        self.current_location_name = None

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

    # ----------------------------------------------------------------
    # INITIALIZATION AND PERSISTENCE
    # ----------------------------------------------------------------

    def instantiate(self):
        """Load a previously saved world map, or create a new one if it doesn't exist."""
        if self.save_path.exists():
            self.load()
            print(f"Loaded world map from {self.save_path}")
        else:
            self.graph = nx.MultiDiGraph()
            print("Initialized a new world map.")

    def load(self):
        """Load the world map from a JSON file."""
        with self.save_path.open("r") as f:
            data = json.load(f)

        self.graph = nx.node_link_graph(data)
        legacy_maze_nodes = [
            node for node, attributes in self.graph.nodes(data=True)
            if attributes.get("node_type") == "maze"
        ]
        self.graph.remove_nodes_from(legacy_maze_nodes)
        legacy_maze_edges = [
            (source, target, key)
            for source, target, key, attributes in self.graph.edges(keys=True, data=True)
            if attributes.get("edge_type") in {"maze_entry", "maze_exit"}
        ]
        self.graph.remove_edges_from(legacy_maze_edges)

    def save(self):
        """Save the world map to a JSON file."""
        self.save_path.parent.mkdir(
            parents=True, 
            exist_ok=True
        )
        
        data = nx.node_link_data(self.graph)
        
        with self.save_path.open("w") as f:
            json.dump(data, f, indent=4)

        print(f"World map saved to {self.save_path}")

    # ----------------------------------------------------------------
    # NODE AND EDGE MANAGEMENT
    # ----------------------------------------------------------------

    def room_key(self, location, location_id=None):
        """Use readable names until an identical name needs a distinct room key."""
        if location_id is None:
            return location

        for node, attributes in self.graph.nodes(data=True):
            if attributes.get("room_id") == location_id:
                return node

        if location not in self.graph:
            return location

        if location in self.graph and self.graph.nodes[location].get("node_type") == "location":
            existing_id = self.graph.nodes[location].get("room_id")
            if existing_id is None or existing_id == location_id:
                return location

        return f"room:{location_id}"

    def add_standard_location(self, location, location_id=None):
        """Add a room, retaining its readable name and optional engine identity."""
        key = self.room_key(location, location_id)
        self.graph.add_node(
            key,
            node_type="location",
            label=location,
            room_id=location_id
        )
        return key

    def add_transition(self, source, destination, actions, edge_type="standard"):
        """Add a directed transition to the graph"""
        if not actions:
            return

        # Avoid adding duplicate transitions
        existing = self.graph.get_edge_data(source, destination, default={})
        for _, data in existing.items():
            if data.get("edge_type") == edge_type and data.get("actions") == actions:
                return
            
        self.graph.add_edge(
            source,
            destination,
            actions=list(actions),
            cost=len(actions),
            edge_type=edge_type,
            direction=self._direction_for_action(actions[-1])
        )

    def _direction_for_action(self, action):
        normalized = action.strip().lower()
        if normalized.startswith("go "):
            normalized = normalized[3:].strip()
        return self.direction_aliases.get(normalized)

    # ----------------------------------------------------------------
    # RECORDING STEPS
    # ----------------------------------------------------------------

    def start_episode(self, start_location, location_id=None):
        """Initialize the world map for a new episode."""
        self.current_location_name = start_location
        self.current_location = self.add_standard_location(start_location, location_id)

    def record_step(self, action, next_location, observation=None, next_location_id=None):
        """Record an observed room change as a directed, action-labeled edge."""
        old_location = self.current_location
        destination = self.room_key(next_location, next_location_id)
        if destination == old_location:
            return

        self.add_standard_location(next_location, next_location_id)

        self.add_transition(
            source=old_location,
            destination=destination,
            actions=[action],
            edge_type="standard"
        )

        self.current_location = destination
        self.current_location_name = next_location


    # ----------------------------------------------------------------
    # ROUTE CALCULATION
    # ----------------------------------------------------------------

    def find_route(self, start, destination):
        """
        Calculate the lowest-cost route between two nodes.

        Returns:
            List of directions (actions) to take from start to destination, or None if no route exists.
        """

        if start not in self.graph:
            print(f"Start location '{start}' not found in the world map.")
            return None

        if destination not in self.graph:
            print(f"Destination location '{destination}' not found in the world map.")
            return None

        try:
            # Use Dijkstra's algorithm to find the shortest path based on edge costs
            _, node_path = nx.single_source_dijkstra(self.graph, source=start, target=destination, weight='cost')
        except nx.NetworkXNoPath:
            print(f"No path found from '{start}' to '{destination}'.")
            return None

        route_actions = []

        # Recover the action sequence from each edge
        for source, target in zip(node_path, node_path[1:]):
            parallel_edges = self.graph.get_edge_data(source, target)

            # Select the edge with the lowest cost (fewest actions)
            _, edge_data = min(parallel_edges.items(), key=lambda item: item[1]['cost'])

            route_actions.extend(edge_data['actions'])

        return route_actions

    # ----------------------------------------------------------------
    # DEBUGGING AND VISUALIZATION
    # ----------------------------------------------------------------

    def print_graph(self):
        """Print all known transitions in the world map."""
        print("\n===== WORLD MAP =====\n")

        labels = [data.get("label", str(node)) for node, data in self.graph.nodes(data=True)]
        label_counts = {label: labels.count(label) for label in set(labels)}

        def display(node):
            data = self.graph.nodes[node]
            label = data.get("label", str(node))
            if label_counts[label] > 1:
                return f"{label} [room {data.get('room_id', node)}]"
            return label

        for source, destination, data in self.graph.edges(data=True):
            direction = data.get("direction")
            relation = f"{direction}: " if direction else ""
            print(
                f"{display(source)} "
                f"--{relation}{data['actions']}--> "
                f"{display(destination)}"
            )

        print("\n===== END OF WORLD MAP =====\n")