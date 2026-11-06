"""Symbolic navigation controller for the Zork WorldMap.

The controller owns deterministic route execution. It does not choose puzzle
solutions; it plans on the persistent WorldMap, treats old paths as candidates,
and checks each transition against current-episode evidence. Confirmed and
blocked edge sets are transient; the learned WorldMap is never erased on failure.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


class NavigationController:
    def __init__(self, world_map):
        self.world_map = world_map
        # Failed edges are blocked for the current episode only. A route can be
        # retried in a later episode because doors/inventory/state may differ.
        self.confirmed_edges: set[tuple[Any, Any, Any]] = set()
        self.blocked_edges: set[tuple[Any, Any, Any]] = set()
        self.active_plan: dict | None = None
        self.last_status = "idle"
        self.last_plan_summary: dict | None = None

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
        self.confirmed_edges.clear()
        self.blocked_edges.clear()
        self.active_plan = None
        self.last_status = "idle"
        self.last_plan_summary = None

    def _resolve_node(self, location, location_id=None):
        if location in self.world_map.graph:
            return location
        return self.world_map.resolve_location(location, location_id)

    def _route_avoiding_episode_edges(
        self,
        start_node,
        target_node,
        *,
        require_confirmed=True,
        exclude_blocked=True,
    ):
        """Plan on a graph copy filtered by this episode's route evidence.

        The persistent WorldMap remains untouched. With require_confirmed=True,
        every edge in the route must have been traversed successfully this
        episode. Historical edges can still be inspected as candidate routes.
        """
        if start_node is None or target_node is None:
            return None

        graph = self.world_map.graph.copy()
        for source, destination, key in list(graph.edges(keys=True)):
            edge = (source, destination, key)
            if exclude_blocked and edge in self.blocked_edges:
                graph.remove_edge(source, destination, key)
            elif require_confirmed and edge not in self.confirmed_edges:
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

    def plan_to(self, destination, destination_id=None, *, require_confirmed=True):
        """Plan a route, requiring current-episode confirmation by default."""
        start_node = self.world_map.current_location
        target_node = self._resolve_node(destination, destination_id)
        if start_node is None or target_node is None:
            self.active_plan = None
            self.last_status = "unknown_location"
            return None

        route = self._route_avoiding_episode_edges(
            start_node,
            target_node,
            require_confirmed=require_confirmed,
            exclude_blocked=True,
        )
        if route is None:
            self.active_plan = None
            self.last_status = "no_confirmed_route" if require_confirmed else "no_route"
            return None

        self.active_plan = {
            **route,
            "transition_index": 0,
            "action_index": 0,
            "require_confirmed": require_confirmed,
            "status": "active" if route["transitions"] else "complete",
        }
        self.last_status = self.active_plan["status"]
        return route

    def ensure_plan(
        self, destination, destination_id=None, *, require_confirmed=True
    ):
        """Keep an existing plan only when target and evidence policy match."""
        target_node = self._resolve_node(destination, destination_id)
        if (
            self.active_plan is not None
            and self.active_plan.get("destination") == target_node
            and self.active_plan.get("status") == "active"
            and self.active_plan.get("require_confirmed", True) == require_confirmed
        ):
            return self.active_plan
        return (
            self.plan_to(target_node, require_confirmed=require_confirmed)
            if target_node is not None else None
        )

    def route_status(self, destination, destination_id=None):
        """Classify a destination route using this episode's evidence.

        Returns one of: confirmed, historical_unconfirmed, blocked, unknown.
        """
        start_node = self.world_map.current_location
        target_node = self._resolve_node(destination, destination_id)
        if start_node is None or target_node is None:
            return "unknown"
        if start_node == target_node:
            return "confirmed"

        confirmed = self._route_avoiding_episode_edges(
            start_node, target_node,
            require_confirmed=True,
            exclude_blocked=True,
        )
        if confirmed is not None:
            return "confirmed"

        unblocked_historical = self._route_avoiding_episode_edges(
            start_node, target_node,
            require_confirmed=False,
            exclude_blocked=True,
        )
        if unblocked_historical is not None:
            return "historical_unconfirmed"

        any_historical = self._route_avoiding_episode_edges(
            start_node, target_node,
            require_confirmed=False,
            exclude_blocked=False,
        )
        if any_historical is not None:
            return "blocked"
        return "unknown"

    def plan_to_home(self, home_node):
        """Plan a confirmed route or cautiously probe a historical candidate.

        The plan advances one transition at a time. A failed or unavailable
        transition is blocked for this episode and cancels the active plan.
        """
        if home_node is None:
            self.active_plan = None
            self.last_status = "home_unknown"
            return None

        status = self.route_status(home_node)
        if status not in {"confirmed", "historical_unconfirmed"}:
            self.active_plan = None
            self.last_status = f"home_route_{status}"
            return None

        # A historical route is a candidate, not a guarantee. It is still
        # usable as a sequence of one-step probes: every transition is checked
        # before the plan advances, and a failed edge is blocked this episode.
        return self.ensure_plan(
            home_node,
            require_confirmed=(status == "confirmed"),
        )

    def can_reach(self, destination, destination_id=None, *, require_confirmed=True):
        """Return whether a route exists under this episode's evidence policy."""
        return self._route_avoiding_episode_edges(
            self.world_map.current_location,
            self._resolve_node(destination, destination_id),
            require_confirmed=require_confirmed,
            exclude_blocked=True,
        ) is not None

    def _matching_single_action_edges(self, source, destination, action):
        """Find direct map edges matching one actually executed command."""
        if source is None or destination is None:
            return []
        parallel = self.world_map.graph.get_edge_data(source, destination, default={})
        matches = []
        for key, data in parallel.items():
            actions = list(data.get("actions", []))
            if (
                len(actions) == 1
                and self._normalize_action(actions[0]) == self._normalize_action(action)
            ):
                matches.append((source, destination, key))
        return matches

    def _matching_single_action_out_edges(self, source, action):
        """Find historical direct edges disproved by a no-movement attempt."""
        if source is None or source not in self.world_map.graph:
            return []
        matches = []
        for _, destination, key, data in self.world_map.graph.out_edges(
            source, keys=True, data=True
        ):
            actions = list(data.get("actions", []))
            if (
                len(actions) == 1
                and self._normalize_action(actions[0]) == self._normalize_action(action)
            ):
                matches.append((source, destination, key))
        return matches

    def cancel_plan(self, reason="cancelled"):
        if self.active_plan is not None:
            plan = self.active_plan
            self.last_plan_summary = {
                "destination": plan.get("destination_label", plan.get("destination")),
                "start": plan.get("start_label", plan.get("start")),
                "remaining_actions": self.remaining_actions(),
                "transition_index": plan.get("transition_index", 0),
                "transition_count": len(plan.get("transitions", [])),
                "route_type": (
                    "historical candidate"
                    if not plan.get("require_confirmed", True)
                    else "episode-confirmed route"
                ),
                "result": str(reason),
            }
        elif reason and reason != "actor_or_puzzle_decision":
            self.last_plan_summary = {
                "destination": None,
                "start": self.world_map.display_location(self.world_map.current_location)
                if self.world_map.current_location is not None else None,
                "remaining_actions": [],
                "transition_index": 0,
                "transition_count": 0,
                "route_type": "none",
                "result": str(reason),
            }
        self.active_plan = None
        # The main loop routinely hands control to the Actor even when no
        # plan was active. Do not overwrite a more informative status such as
        # home_route_blocked or planned_action_unavailable in that case.
        if reason != "actor_or_puzzle_decision" or self.last_status in {"idle", "active", "complete"}:
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
            # A command unavailable in the current parser state cannot safely
            # be assumed executable. Exclude this edge for the episode.
            failed_edge = (
                transition["source"],
                transition["destination"],
                transition["edge_key"],
            )
            self.blocked_edges.add(failed_edge)
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
            "confirmed_edges": [],
        }

        # Actor-driven exploration also supplies episode evidence. A successful
        # one-command transition confirms only the exact directed edge used.
        if after_node != before_node:
            matched = self._matching_single_action_edges(
                before_node, after_node, action
            )
            for edge in matched:
                self.confirmed_edges.add(edge)
            result["confirmed_edges"] = matched
            if matched and not self.active_plan:
                self.last_status = "transition_confirmed"
        else:
            # A historical direct edge that fails to move the player is not
            # trusted for the rest of this episode (e.g. a locked cellar door).
            failed_matches = self._matching_single_action_out_edges(
                before_node, action
            )
            for edge in failed_matches:
                self.blocked_edges.add(edge)
            if failed_matches:
                result["failed_edge"] = failed_matches[0]
                if not self.active_plan:
                    self.last_status = "edge_blocked"

        result["plan_status"] = self.last_status
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
            # The expected transition succeeded. Confirm the map edge itself,
            # including a multi-command edge that cannot be matched as one action.
            confirmed_edge = (source, destination, transition["edge_key"])
            self.confirmed_edges.add(confirmed_edge)
            result["confirmed_edges"].append(confirmed_edge)
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
        """Describe active or recently abandoned routing evidence for the Actor.

        The main loop can hand control back to the Actor after a route is
        unavailable or blocked. Keeping the last plan summary lets the Actor
        see what the navigator tried instead of losing that context when the
        active plan is cancelled.
        """
        if self.active_plan and self.active_plan.get("status") == "active":
            plan = self.active_plan
            route_type = (
                "historical candidate; not yet verified in this episode"
                if not plan.get("require_confirmed", True)
                else "confirmed during this episode"
            )
            remaining = self.remaining_actions()
            return (
                f"ACTIVE SYMBOLIC ROUTE\n"
                f"Destination: {plan['destination_label']}\n"
                f"Route type: {route_type}\n"
                f"Progress: transition {plan.get('transition_index', 0) + 1} "
                f"of {len(plan.get('transitions', []))}\n"
                f"Remaining commands: {' -> '.join(remaining) or 'none'}\n"
                "The navigator executes only the next command and validates its result."
            )

        summary = self.last_plan_summary or {}
        blocked_descriptions = []
        for source, destination, key in list(self.blocked_edges)[-5:]:
            edge_data = self.world_map.graph.get_edge_data(source, destination, key) or {}
            edge_actions = edge_data.get("actions", [])
            blocked_descriptions.append(
                f"{self.world_map.display_location(source)} --"
                f"[{', '.join(edge_actions)}]--> "
                f"{self.world_map.display_location(destination)}"
            )

        lines = [
            "NO ACTIVE SYMBOLIC ROUTE IS CURRENTLY BEING EXECUTED.",
            f"Last navigation status: {summary.get('result', self.last_status)}.",
        ]
        if summary.get("destination"):
            lines.extend([
                f"Last route destination: {summary['destination']}.",
                f"Last route type: {summary.get('route_type', 'unknown')}.",
                f"Unexecuted route commands at handoff: "
                f"{' -> '.join(summary.get('remaining_actions', [])) or 'none'}.",
            ])
        if blocked_descriptions:
            lines.append("Edges blocked during this episode: " + "; ".join(blocked_descriptions))
        lines.append(
            "Use this information as evidence, not as a guarantee that a historical "
            "passage is currently open. Investigate blocked prerequisites or an alternate route."
        )
        return "\n".join(lines)
