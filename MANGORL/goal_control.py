"""Episode-local goal persistence. No cancellation edits the learned map."""
from copy import deepcopy


def carried_treasures(historian, inventory):
    if inventory is None:
        return []
    return [item for item in historian.agent_state.get('carried_treasures', [])
            if any(historian._items_match(item, present) for present in inventory)]


def authorize_navigation(goal, historian, inventory, current_node, home_node, route_status):
    return bool(goal.get('kind') == 'navigate_to_home' and home_node is not None
                and str(goal.get('target_node')) == str(home_node)
                and str(current_node) != str(home_node)
                and carried_treasures(historian, inventory)
                and route_status in {'confirmed', 'historical_unconfirmed'})


class GoalControl:
    def __init__(self):
        self.active = None
        self.age = 0
        self.visits = {}
        self.suppressed = set()
        self.rooms = {}
        self.facts = None
        self.events = []
        self.origin = None

    @staticmethod
    def identity(goal):
        return (goal.get('kind'), str(goal.get('target_node')), goal.get('productive_action'))

    def cancel(self, reason):
        if self.active:
            self.suppressed.add(self.identity(self.active))
            self.events.append({'status': 'abandoned', 'reason': reason, 'goal': self.active})
        self.active, self.age, self.visits = None, 0, {}

    def select(self, proposed, *, node, inventory, context, score, map_size, treasures):
        # Movement alone does not reset stagnation. Changes to observations at
        # the SAME room, inventory, score, or discovered topology release bans.
        room = str(context).split('CURRENT ROOM (look):\n')[-1].split('\n\nCURRENT INVENTORY:')[0]
        facts = (tuple(sorted(inventory or [])), score, map_size)
        changed = self.facts != facts or (str(node) in self.rooms and self.rooms[str(node)] != room)
        self.facts = facts
        self.rooms[str(node)] = room
        if changed:
            self.suppressed.clear()
            self.visits.clear()
            self.age = 0
        if self.active:
            old = self.active
            deposit = old.get('kind') in {'navigate_to_home', 'solve_return_route', 'deposit_treasure'}
            invalid = (deposit and (not treasures or self.identity(proposed) != self.identity(old)) or
                       old.get('target_node') is None and self.origin != str(node) or
                       old.get('target_node') is not None and str(old['target_node']) == str(node)
                       and proposed.get('kind') != old.get('kind') or
                       treasures and proposed.get('kind') in {'navigate_to_home', 'deposit_treasure'}
                       and self.identity(proposed) != self.identity(old))
            if invalid:
                self.events.append({'status': 'released', 'reason': 'prerequisites_or_destination_changed', 'goal': old})
                self.active = None
                self.age, self.visits = 0, {}
            elif old.get('kind') == 'score_action' and self.identity(old) != self.identity(proposed):
                self.active = None  # No longer legal or already attempted.
        if self.active:
            self.age += 1
            state = (str(node), room, facts)
            self.visits[state] = self.visits.get(state, 0) + 1
            if self.visits[state] >= 3 or self.age >= self.active.get('progress_budget', 24):
                self.cancel('goal_repeated_state_or_budget_exhausted')
        if self.active is None and self.identity(proposed) not in self.suppressed:
            self.active = deepcopy(proposed)
            self.origin = str(node)
            self.events.append({'status': 'active', 'reason': 'eligible_goal', 'goal': self.active})
        return deepcopy(self.active) if self.active else {
            'kind': 'explore_alternative', 'target_node': None, 'preferred_actions': [],
            'description': 'The previous objective stalled. Preserve valuables and investigate a new exit or local prerequisite. Reconsider it when observed evidence changes.',
            'sequence': [], 'source': 'goal_recovery',
        }
