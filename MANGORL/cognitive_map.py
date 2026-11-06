"""Probabilistic cognitive map over observed directed room transitions."""
from __future__ import annotations

from .nsi import NeuroSymbolicInference
from .world_map import WorldMap


class CognitiveMap(WorldMap):
    """Backward-compatible WorldMap JSON with empirical edge beliefs.

    Successful and failed traversals update Beta(1,1) counts. Historical edges
    without counts have one observed success. Episode evidence takes priority.
    No reverse edge, prerequisite, or puzzle solution is invented.
    """

    def edge_probability(self, edge, *, confirmed=(), blocked=()):
        if edge in blocked:
            return 0.0
        if edge in confirmed:
            return 1.0
        data = self.graph.get_edge_data(*edge)
        if data is None:
            return 0.5
        successes = max(0, int(data.get("successes", 1)))
        failures = max(0, int(data.get("failures", 0)))
        return (successes + 1) / (successes + failures + 2)

    def record_step(self, action, next_location, observation=None, next_location_id=None):
        source = self.current_location
        old_edges = set(self.graph.edges(keys=True))
        moved = super().record_step(action, next_location, observation, next_location_id)
        destination = self.current_location
        for left, right, key, data in self.graph.out_edges(source, keys=True, data=True):
            if data.get("actions") != [action]:
                continue
            if moved and right == destination:
                # An old edge had at least one historical observation; a new
                # edge gets exactly the success observed on this call.
                data["successes"] = int(data.get("successes", 1 if (left, right, key) in old_edges else 0)) + 1
            else:
                data["failures"] = int(data.get("failures", 0)) + 1
        return moved

    def route_probability(self, route, *, confirmed=(), blocked=()):
        atoms, literals = {}, []
        for step in route.get("transitions", []):
            edge = (step["source"], step["destination"], step["edge_key"])
            name = repr(edge)
            atoms[name] = self.edge_probability(edge, confirmed=confirmed, blocked=blocked)
            literals.append((name, True))
        return NeuroSymbolicInference().conjunction(atoms, literals)

    def action_feasibility(self, state, actions, navigator, valid_actions):
        """Marginal evidence for direct traversals; unknown actions remain open.

        Several observed destinations for one command are treated as alternative
        hypotheses using their maximum support, not independent simultaneous
        destinations. Multi-command macros constrain routes, not individual acts.
        """
        source = state.get("location_id")
        result = []
        valid = {navigator._normalize_action(a) for a in valid_actions}
        for action in actions:
            normalized = navigator._normalize_action(action)
            if normalized not in valid:
                result.append(0.0)
                continue
            edges = []
            if source in self.graph:
                for left, right, key, data in self.graph.out_edges(source, keys=True, data=True):
                    commands = data.get("actions", [])
                    if len(commands) == 1 and navigator._normalize_action(commands[0]) == normalized:
                        edges.append((left, right, key))
            result.append(max((self.edge_probability(
                edge, confirmed=navigator.confirmed_edges, blocked=navigator.blocked_edges
            ) for edge in edges), default=1.0))
        return result
