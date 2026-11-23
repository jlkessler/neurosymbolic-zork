import json
import inspect
import hashlib
import re
from pathlib import Path

import networkx as nx

from .action_guards import snapshot_sections
from .room_identity import RoomIndex, observed_room_title, room_id_key


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

        edge_field = "edges" if "edges" in data else "links"
        parameter = "edges" if "edges" in inspect.signature(nx.node_link_graph).parameters else "link"
        self.graph = nx.node_link_graph(data, **{parameter: edge_field})
        self.legacy_rooms = RoomIndex(data["nodes"])
        # Rewrite machine references together, including non-movement experiences.
        mapping = {key: target for key, target in self.legacy_rooms.keys.items()
                   if key in self.graph and key != target}
        for key, target in mapping.items():
            aliases = self.graph.nodes[key].setdefault("legacy_keys", [])
            if key not in aliases:
                aliases.append(key)
        if mapping:
            self.graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        for row in [*self._experience_store(), *(self.graph.graph.get("walkthrough") or {}).get("steps", [])]:
            for field in ("source", "destination"):
                if field in row and row[field] in mapping:
                    row[field] = mapping[row[field]]
        self.graph.graph["room_identity_version"] = 2
        self._upgrade_legacy_walkthrough_experiences()

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

    def _normalize_command(self, action):
        command = " ".join(str(action or "").casefold().split())
        direction = self._direction_for_action(command)
        if direction:
            return direction
        if (self.graph.graph.get("rom_sha256") == "0ae5ac229e79094ff368b6669356444af0f35e21d862a1baaa546989085c15fd"
                and command == "odysseus"):
            return "odysse"
        return command

    def bind_rom(self, path):
        """Keep object IDs in the namespace of the story file that created them."""
        path = Path(path)
        if not path.is_file():
            return
        signature = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = self.graph.graph.get("rom_sha256")
        if previous and previous != signature:
            raise ValueError("This world map belongs to a different ROM. Select its matching map.")
        self.graph.graph["rom_sha256"] = signature

    @staticmethod
    def _context_features(context):
        """Extract observable room and inventory facts for experience retrieval."""
        text = str(context or "")
        room, inventory = snapshot_sections(text)

        def tokens(value):
            return sorted({
                token.casefold() for token in re.findall(r"[A-Za-z0-9']+", value)
                if len(token) > 1
            })

        return {
            "room_tokens": tokens(room),
            "inventory_tokens": tokens(inventory),
        }

    def _experience_store(self):
        return self.graph.graph.setdefault("action_experiences", [])

    def _upgrade_legacy_walkthrough_experiences(self):
        """Expose older saved walkthrough steps through the shared experience index."""
        if self.graph.graph.get("action_experience_schema", 0) >= 1:
            self._experience_store()
            return
        replay = self.graph.graph.get("walkthrough") or {}
        previous_score = 0
        previous_observation = ""
        for index, step in enumerate(replay.get("steps", []), 1):
            score_after = step.get("score")
            score_delta = (
                score_after - previous_score
                if isinstance(score_after, (int, float))
                and isinstance(previous_score, (int, float))
                else None
            )
            self.record_experience(
                source=step.get("source"),
                action=step.get("action"),
                destination=step.get("destination"),
                before_context=previous_observation,
                after_context=step.get("observation", ""),
                score_before=previous_score,
                score_after=score_after,
                reward=step.get("reward"),
                done=step.get("done", False),
                trajectory_id="legacy_reference_trajectory",
                step=index,
                provenance="reference_replay",
            )
            if isinstance(score_after, (int, float)):
                previous_score = score_after
            previous_observation = step.get("observation", "")
        self.graph.graph["action_experience_schema"] = 1

    def remove_trajectory_experiences(self, trajectory_id):
        """Replace a replayed trajectory without duplicating its observations."""
        wanted = str(trajectory_id)
        store = self._experience_store()
        store[:] = [row for row in store if str(row.get("trajectory_id")) != wanted]

    def record_experience(
        self, *, source, action, destination, before_context, after_context,
        score_before=None, score_after=None, reward=None, done=False,
        trajectory_id=None, step=None, provenance="runtime",
    ):
        """Record any executed command in the world's common experience bank.

        Navigation and room-preserving puzzle interactions use the same schema.
        Provenance remains available for auditing but is never used to rank or
        filter experiences during play.
        """
        action = " ".join(str(action or "").split()).strip()
        if not action:
            return False
        score_delta = (
            score_after - score_before
            if isinstance(score_before, (int, float))
            and isinstance(score_after, (int, float))
            else None
        )
        self.observe_room_title(source, before_context)
        self.observe_room_title(destination, after_context)
        self._experience_store().append({
            "trajectory_id": str(trajectory_id or "unidentified_trajectory"),
            "step": step,
            "source": source,
            "source_label": self.display_location(source),
            "action": action,
            "destination": destination,
            "destination_label": self.display_location(destination) if destination is not None else None,
            "before_features": self._context_features(before_context),
            "after_features": self._context_features(after_context),
            "result": " ".join(str(after_context or "").split())[:500],
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_delta,
            "reward": reward,
            "done": bool(done),
            "provenance": str(provenance),
        })
        self.graph.graph["action_experience_schema"] = 1
        return True

    @staticmethod
    def _jaccard(left, right):
        left, right = set(left or ()), set(right or ())
        if not left and not right:
            return 0.5
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def retrieve_action_experiences(
        self, *, current_node, state_context, valid_actions, top_k=6, blocked_edges=(),
    ):
        """Retrieve legal commands observed in comparable world states.

        The retrieval policy deliberately ignores whether a transition came
        from a reference replay, the Actor, or symbolic navigation.
        """
        available = {
            self._normalize_command(action): action for action in valid_actions
        }
        current_node = self.resolve_location(current_node)
        if current_node is None:
            return []
        current = self._context_features(state_context)
        matches = []
        for row in self._experience_store():
            normalized = self._normalize_command(row.get("action"))
            if str(row.get("source")) != str(current_node) or normalized not in available:
                continue
            previous = row.get("before_features") or {}
            room_similarity = self._jaccard(
                current["room_tokens"], previous.get("room_tokens", [])
            )
            inventory_similarity = self._jaccard(
                current["inventory_tokens"], previous.get("inventory_tokens", [])
            )
            # Location and current parser legality are hard gates. Observable
            # facts rank competing experiences within that grounded set.
            similarity = 0.7 * room_similarity + 0.3 * inventory_similarity
            score_delta = row.get("score_delta")
            if self._direction_for_action(normalized) and isinstance(score_delta, (int, float)) and score_delta > 0:
                # A movement reward belongs to a specific directed transition and puzzle state.
                edges = self.graph.get_edge_data(current_node, row.get("destination"), default={})
                usable = any((current_node, row["destination"], k) not in blocked_edges
                             and len(e.get("actions", [])) == 1
                             and self._normalize_command(e["actions"][0]) == normalized
                             for k, e in edges.items())
                if not usable or not current["room_tokens"] or room_similarity != 1:
                    continue
            matches.append({
                **row,
                "action": available[normalized],
                "state_similarity": similarity,
                "productive": isinstance(score_delta, (int, float)) and score_delta > 0,
            })
        matches.sort(key=lambda row: (
            row["state_similarity"],
            not bool(row.get("done")),
            bool(row["productive"]),
            row.get("score_delta") if isinstance(row.get("score_delta"), (int, float)) else 0,
            -(row.get("step") or 0),
        ), reverse=True)
        # Return one strongest observation for each command. Repeated Actor
        # visits and reference visits therefore compete on state relevance.
        result, seen = [], set()
        for row in matches:
            key = self._normalize_command(row["action"])
            if key in seen:
                continue
            result.append(row)
            seen.add(key)
            if len(result) >= max(1, int(top_k)):
                break
        return result

    def format_experience_guidance(self, experiences):
        """Render concise, observable solution evidence for Actor reasoning."""
        if not experiences:
            return ""
        lines = [
            "WORLD-MODEL ACTION EXPERIENCE:",
            "These commands were executed in comparable observed states. Recheck current prerequisites and result feedback.",
        ]
        for row in experiences:
            delta = row.get("score_delta")
            delta_text = "unknown" if delta is None else f"{delta:+g}"
            destination = row.get("destination_label") or row.get("source_label") or "same/unknown room"
            result = str(row.get("result") or "")[:180]
            lines.append(
                f"- {row['action']}: score change {delta_text}; resulting location {destination}; "
                f"state match {row.get('state_similarity', 0.0):.2f}; result: {result or 'not recorded'}"
            )
        return "\n".join(lines)

    def save(self):
        """Persist the current graph as JSON."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        parameter = "edges" if "edges" in inspect.signature(nx.node_link_data).parameters else "link"
        data = nx.node_link_data(self.graph, **{parameter: "edges"})

        temporary = self.save_path.with_suffix(self.save_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        temporary.replace(self.save_path)

        print(f"World map saved to {self.save_path}")

    # ------------------------------------------------------------------
    # ROOM IDENTITY AND NODE MANAGEMENT
    # ------------------------------------------------------------------

    def room_key(self, location, location_id=None):
        """Prefer the stable ROM object number, even when its short name is empty."""
        if location_id is not None:
            key = room_id_key(location_id)
            if key is None:
                raise ValueError(f"Invalid Jericho room number {location_id!r}")
            return key
        if not str(location or "").strip():
            return None
        known = self.resolve_location(location)
        if known is not None or self.find_nodes_by_label(location):
            return known
        return location

    def add_standard_location(self, location, location_id=None):
        key = self.room_key(location, location_id)
        if key is None:
            return None
        if key not in self.graph:
            numeric_key = room_id_key(key) if str(key).startswith("room:") else None
            self.graph.add_node(key, node_type="location", label=location or "",
                                room_id=int(numeric_key[5:]) if numeric_key else None)
        elif location and not self.graph.nodes[key].get("label"):
            self.graph.nodes[key]["label"] = location
        self.graph.graph["room_identity_version"] = 2
        return key

    def resolve_location(self, location, location_id=None):
        if location_id is not None:
            key = room_id_key(location_id)
            return key if key in self.graph else None
        if not str(location or "").strip():
            return None
        if location in self.graph and (str(location).startswith("room:")
                                       or self.graph.nodes[location].get("node_type") != "location"):
            return location
        numbered = re.fullmatch(r"(?:.+ \(room #(\d+)\)|Room #(\d+))", str(location), re.I)
        if numbered:
            key = room_id_key(numbered[1] or numbered[2])
            if key in self.graph and self.display_location(key).casefold() == str(location).casefold():
                return key
        matches = self.find_nodes_by_label(location)
        return matches[0] if len(matches) == 1 else None

    def display_location(self, node):
        """Return a readable label, disambiguating repeated room names."""
        if node not in self.graph:
            return str(node) if str(node or "").strip() else "Unknown room"

        attributes = self.graph.nodes[node]
        label = attributes.get("observed_title") or str(attributes.get("label") or "").strip()
        if not label:
            return f"Room #{attributes['room_id']}" if attributes.get("room_id") is not None else "Unknown room"
        matches = [
            other
            for other, data in self.graph.nodes(data=True)
            if data.get("node_type") == "location"
            and (data.get("observed_title") or str(data.get("label") or "").strip()) == label
        ]

        if len(matches) <= 1:
            return label

        room_id = attributes.get("room_id")
        suffix = f"room #{room_id}" if room_id is not None else str(node)
        return f"{label} ({suffix})"

    def find_nodes_by_label(self, label):
        """Return canonical nodes whose readable room label matches."""
        wanted = " ".join(str(label).casefold().split())
        if not wanted:
            return []
        return [
            node for node, attributes in self.graph.nodes(data=True)
            if attributes.get("node_type") == "location"
            and wanted in {
                " ".join(str(name).casefold().split()) for name in
                [attributes.get("label", node), attributes.get("observed_title", ""),
                 *attributes.get("aliases", [])]
            }
        ]

    def observe_room_title(self, node, state_context):
        """Learn a room's printed title from a restored-state ``look`` snapshot.

        Jericho object short names can be truncated (e.g. ``Living ``). Keep
        the object's identity/key and original label; add the observed full
        title as an alias instead of merging rooms by a guessed name prefix.
        Raw action responses are not accepted as room snapshots.
        """
        if node not in self.graph:
            return False
        attributes = self.graph.nodes[node]
        title = observed_room_title(attributes.get("label", node), state_context)
        if title is None:
            return False
        aliases = attributes.setdefault("aliases", [])
        if title not in aliases:
            aliases.append(title)
        attributes["observed_title"] = title
        return True

    def routing_graph(self, start_node):
        """An ID-grounded route can only cross rooms with known identities."""
        if room_id_key(start_node) is not None:
            return nx.subgraph_view(self.graph, filter_node=lambda n: room_id_key(n) is not None)
        return nx.subgraph_view(self.graph, filter_node=lambda n: bool(str(n).strip()))

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

        source = self.resolve_location(source)
        destination = self.resolve_location(destination)
        if source is None or destination is None or not actions:
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
        if self.current_location is None and self.current_location_name is None:
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
        return old_location is not None and destination is not None

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

        graph = self.routing_graph(start_node)
        try:
            total_cost, node_path = nx.single_source_dijkstra(
                graph,
                source=start_node,
                target=destination_node,
                weight="cost",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        return self._route_details(graph, node_path, total_cost)

    def _route_details(self, graph, node_path, total_cost):
        start_node, destination_node = node_path[0], node_path[-1]
        transitions = []
        all_actions = []

        for source, target in zip(node_path, node_path[1:]):
            parallel_edges = graph.get_edge_data(source, target, default={})
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

    def get_reachable_routes(self, start, max_destinations=8, start_id=None, *, valid_actions=None, blocked_edges=()):
        """Return the cheapest known routes from start to known room nodes.

        Each result includes a canonical destination key, readable label,
        actions, cost, and transition details. Maze cells are ordinary location
        nodes in this identity-based representation, so they remain routable.
        """
        start_node = self.resolve_location(start, start_id)
        if start_node is None:
            return []

        legal = None if valid_actions is None else {self._normalize_command(a) for a in valid_actions}

        def feasible(source, destination, key):
            if (source, destination, key) in blocked_edges:
                return False
            if source != start_node or legal is None:
                return True
            actions = self.graph[source][destination][key].get("actions", [])
            return bool(actions) and self._normalize_command(actions[0]) in legal

        graph = nx.subgraph_view(self.routing_graph(start_node), filter_edge=feasible)
        costs, paths = nx.single_source_dijkstra(graph, start_node, weight="cost")
        routes = []
        for destination, path in paths.items():
            if destination == start_node or graph.nodes[destination].get("node_type") != "location":
                continue
            details = self._route_details(graph, path, costs[destination])

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
