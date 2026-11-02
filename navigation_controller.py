"""Symbolic navigation controller for the Zork WorldMap.

The controller owns deterministic route execution. It does not choose puzzle
solutions; it plans on the observed WorldMap, proposes the next route action,
and checks each environment result against the expected transition.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


class NavigationController:
    def __init__(self, world_map):
        self.world_map = world_map
        # Failed edges are blocked for the current episode only. A route can be
        # retried in a later episode because doors/inventory/state may differ.
        self.blocked_edges: set[tuple[Any, Any, Any]] = set()
        self.active_plan: dict | None = None
        self.last_status = "idle"

    @staticmethod
    def _normalize_action(action: str) -> str:
        value = str(action).strip().lower()
        if value.startswith("go "):
            value = value[3:].strip()
        aliases = {
            "n": "north", "s": "south", "e": "east", "w": "west",
            "ne": "northeast", "nw": "northwest",
            "se": "southeast", "sw": "southwest",
            "u": "up", "d": "down",
        }
        return aliases.get(value, value)

    def start_episode(self):
        """Reset transient navigation state without deleting the learned map."""
        self.blocked_edges.clear()
        self.active_plan = None
        self.last_status = "idle"

    def _resolve_node(self, location, location_id=None):
        if location in self.world_map.graph:
            return location
        return self.world_map.resolve_location(location, location_id)

    def _route_avoiding_blocked_edges(self, start_node, target_node):
        """Run Dijkstra on a graph copy with this episode's failed edges removed."""
        graph = self.world_map.graph.copy()
        for source, destination, key in self.blocked_edges:
            if graph.has_edge(source, destination, key):
                graph.remove_edge(source, destination, key)

        try:
            total_cost, node_path = nx.single_source_dijkstra(
                graph, source=start_node, target=target_node, weight="cost"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        transitions = []
        all_actions = []
        for source, target in zip(node_path, node_path[1:]):
            parallel = graph.get_edge_data(source, target, default={})
            if not parallel:
                return None
            edge_key, data = min(
                parallel.items(),
                key=lambda item: (
                    item[1].get("cost", len(item[1].get("actions", []))),
                    len(item[1].get("actions", [])),
                ),
            )
            actions = list(data.get("actions", []))
            transitions.append({
                "source": source,
                "source_label": self.world_map.display_location(source),
                "destination": target,
                "destination_label": self.world_map.display_location(target),
                "actions": actions,
                "edge_type": data.get("edge_type", "standard"),
                "edge_key": edge_key,
            })
            all_actions.extend(actions)

        return {
            "start": start_node,
            "start_label": self.world_map.display_location(start_node),
            "destination": target_node,
            "destination_label": self.world_map.display_location(target_node),
            "total_cost": total_cost,
            "nodes": node_path,
            "transitions": transitions,
            "actions": all_actions,
        }

    def plan_to(self, destination, destination_id=None):
        """Plan a minimum-action route from the map's current node to a target.

        `destination` can be a canonical graph key or a room label. Supply the
        Jericho room number when resolving an ambiguous display name.
        """
        start_node = self.world_map.current_location
        target_node = self._resolve_node(destination, destination_id)
        if start_node is None or target_node is None:
            self.active_plan = None
            self.last_status = "unknown_location"
            return None

        route = self._route_avoiding_blocked_edges(start_node, target_node)
        if route is None:
            self.active_plan = None
            self.last_status = "no_route"
            return None

        self.active_plan = {
            **route,
            "transition_index": 0,
            "action_index": 0,
            "status": "active" if route["transitions"] else "complete",
        }
        self.last_status = self.active_plan["status"]
        return route

    def ensure_plan(self, destination, destination_id=None):
        """Keep an existing plan to the same target; otherwise calculate one."""
        target_node = self._resolve_node(destination, destination_id)
        if (
            self.active_plan is not None
            and self.active_plan.get("destination") == target_node
            and self.active_plan.get("status") == "active"
        ):
            return self.active_plan
        return self.plan_to(target_node) if target_node is not None else None

    def plan_to_home(self, home_node):
        """Plan a route to the Living Room/home node, if it is known."""
        if home_node is None:
            self.active_plan = None
            self.last_status = "home_unknown"
            return None
        return self.ensure_plan(home_node)

    def can_reach(self, destination, destination_id=None):
        """Return whether the current map has a route to destination."""
        return self._route_avoiding_blocked_edges(
            self.world_map.current_location,
            self._resolve_node(destination, destination_id),
        ) is not None

    def cancel_plan(self, reason="cancelled"):
        self.active_plan = None
        self.last_status = reason

    def next_action(self, valid_actions=None):
        """Return the next route command, mapped to an available action string.

        If `valid_actions` is supplied and the planned command is unavailable,
        the plan is cancelled rather than issuing a stale command.
        """
        plan = self.active_plan
        if not plan or plan.get("status") != "active":
            return None

        index = plan["transition_index"]
        if index >= len(plan["transitions"]):
            plan["status"] = "complete"
            self.last_status = "complete"
            return None

        transition = plan["transitions"][index]
        actions = transition["actions"]
        action_index = plan["action_index"]
        if action_index >= len(actions):
            self.cancel_plan("invalid_plan")
            return None

        planned_action = actions[action_index]
        if valid_actions is None:
            return planned_action

        by_normalized = {
            self._normalize_action(candidate): candidate
            for candidate in valid_actions
        }
        actual_action = by_normalized.get(self._normalize_action(planned_action))
        if actual_action is None:
            self.cancel_plan("planned_action_unavailable")
            return None
        return actual_action

    def observe_action_result(
        self,
        action,
        next_location,
        next_location_id=None,
        observation=None,
    ):
        """Record the observed step and reconcile it with the active route.

        Returns a status dictionary. Unexpected transitions are still added to
        the WorldMap; the active plan is then cancelled so the Actor/Historian
        can reassess from the newly observed state.
        """
        before_node = self.world_map.current_location
        recorded = self.world_map.record_step(
            action=action,
            next_location=next_location,
            observation=observation,
            next_location_id=next_location_id,
        )
        after_node = self.world_map.current_location

        result = {
            "recorded_transition": recorded,
            "before_node": before_node,
            "after_node": after_node,
            "plan_status": self.last_status,
            "deviation": False,
            "failed_edge": None,
        }

        plan = self.active_plan
        if not plan or plan.get("status") != "active":
            return result

        transition_index = plan["transition_index"]
        if transition_index >= len(plan["transitions"]):
            plan["status"] = "complete"
            self.last_status = "complete"
            result["plan_status"] = "complete"
            return result

        transition = plan["transitions"][transition_index]
        expected_actions = transition["actions"]
        action_index = plan["action_index"]
        expected_action = expected_actions[action_index]

        if self._normalize_action(action) != self._normalize_action(expected_action):
            self.cancel_plan("actor_overrode_navigation")
            result.update(plan_status=self.last_status, deviation=True)
            return result

        source = transition["source"]
        destination = transition["destination"]

        if after_node == destination:
            # The expected transition succeeded. Move to the next edge.
            plan["transition_index"] += 1
            plan["action_index"] = 0
            if plan["transition_index"] >= len(plan["transitions"]):
                plan["status"] = "complete"
                self.last_status = "complete"
            else:
                self.last_status = "active"
            result["plan_status"] = self.last_status
            return result

        if after_node == source:
            # A multi-command macro edge can contain preparatory commands that
            # leave the player in the same room. Continue only if another
            # command remains in that same edge; otherwise mark it blocked.
            if action_index + 1 < len(expected_actions):
                plan["action_index"] += 1
                self.last_status = "active"
                result["plan_status"] = "active"
                return result

            failed_edge = (source, destination, transition["edge_key"])
            self.blocked_edges.add(failed_edge)
            self.cancel_plan("route_edge_failed")
            result.update(
                plan_status=self.last_status,
                failed_edge=failed_edge,
                deviation=True,
            )
            return result

        # The environment moved somewhere other than the planned destination.
        self.cancel_plan("unexpected_transition")
        result.update(plan_status=self.last_status, deviation=True)
        return result

    def remaining_actions(self):
        """Return the unexecuted actions in the active plan."""
        plan = self.active_plan
        if not plan or plan.get("status") != "active":
            return []
        actions = []
        for index in range(plan["transition_index"], len(plan["transitions"])):
            edge_actions = plan["transitions"][index]["actions"]
            start = plan["action_index"] if index == plan["transition_index"] else 0
            actions.extend(edge_actions[start:])
        return actions

    def navigation_context(self):
        """Compact, prompt-friendly description of the active route."""
        if not self.active_plan or self.active_plan.get("status") != "active":
            return "No active symbolic navigation plan."
        return (
            f"Active destination: {self.active_plan['destination_label']}\n"
            f"Remaining directions: {' -> '.join(self.remaining_actions())}"
        )
