import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import main
from MANGORL.checkpoint import TrialCheckpoint, trial_lock
from MANGORL.goal_control import GoalControl, authorize_navigation
from MANGORL.historian import Historian
from MANGORL.provider_requests import MalformedProviderResponse, provider_request
from MANGORL.provider_requests import ProviderBudgetExhausted, request_budget
from MANGORL.run_recovery import recoverable_episode_error
from MANGORL.world_map import WorldMap
from test_run_reliability import failure_run


@pytest.mark.parametrize('inventory,goal_kind,current,route,expected', [
    (['sword', 'lantern'], 'navigate_to_home', 'Cellar', 'confirmed', False),
    (['painting', 'lantern'], 'navigate_to_home', 'Cellar', 'confirmed', True),
    ([], 'navigate_to_home', 'Cellar', 'confirmed', False),
    (None, 'navigate_to_home', 'Cellar', 'confirmed', False),
    (['painting'], 'navigate_score_goal', 'Cellar', 'confirmed', False),
    (['painting'], 'navigate_to_home', 'Living Room', 'confirmed', False),
    (['painting'], 'navigate_to_home', 'Cellar', 'blocked', False),
])
def test_inventory_gates_handoff_independently(tmp_path, inventory, goal_kind, current, route, expected):
    h = Historian(tmp_path / 'h.json')
    h.agent_state['carried_treasures'] = ['painting']  # Deliberately stale report.
    goal = {'kind': goal_kind, 'target_node': 'Living Room'}
    assert authorize_navigation(goal, h, inventory, current, 'Living Room', route) is expected


def test_old_deposit_reward_without_painting_cannot_create_home_goal(tmp_path):
    h = Historian(tmp_path / 'h.json')
    h.start_episode('Gallery', 'Gallery', initial_score=35)
    h.agent_state.update(inventory_items=['sword', 'lantern'], carried_treasures=[])
    h.memory['score_profiles']['old_painting'] = {
        'node': 'Living Room', 'location': 'Living Room', 'action': 'put painting in case',
        'total_gain': 6, 'attempts': 1, 'samples': 1,
    }
    route = {'destination': 'Living Room', 'actions': ['north', 'up', 'west'], 'total_cost': 3}
    goal = h.choose_goal(current_node='Gallery', current_location='Gallery', valid_actions=['north', 'look'],
                         known_routes=[route], home_node='Living Room', home_route_status='confirmed')
    assert goal.get('target_node') != 'Living Room'
    assert not goal.get('productive_action')


def test_recorded_episode8_empty_inventory_circuit_has_no_authorized_takeover(tmp_path):
    recorded = json.loads((Path(__file__).parent / 'fixtures/episode8_loop.json').read_text())
    assert len(recorded['transitions']) == 13 and recorded['inventories']
    h = Historian(tmp_path / 'h.json')
    h.agent_state['carried_treasures'] = ['painting']  # Stronger than the real log: stale claim.
    for inventory in recorded['inventories']:
        for row in recorded['transitions']:
            assert not authorize_navigation({'kind': 'navigate_to_home', 'target_node': 'Living Room'},
                                            h, inventory, row['location'], 'Living Room', 'confirmed')


def select(control, node='Gallery', inventory=None, room='A room.', proposed=None, map_size=(6, 6)):
    return control.select(proposed or {'kind': 'navigate_to_home', 'target_node': 'Living Room'},
                          node=node, inventory=['painting'] if inventory is None else inventory,
                          context=f'CURRENT ROOM (look):\n{node}\n{room}\n\nCURRENT INVENTORY:\npainting',
                          score=35, map_size=map_size, treasures=['painting'] if inventory is None else inventory)


def test_route_cycle_suppresses_objective_until_observed_evidence_changes(tmp_path):
    world = WorldMap(tmp_path / 'map.json')
    a = world.add_standard_location('Gallery', 1)
    b = world.add_standard_location('Studio', 2)
    world.add_transition(a, b, ['north'])
    original = list(world.graph.edges(data=True))
    control = GoalControl()
    for _ in range(4):
        select(control, 'Gallery')
        goal = select(control, 'Studio')
    assert goal['kind'] == 'explore_alternative'
    assert control.suppressed
    assert list(world.graph.edges(data=True)) == original
    assert select(control, 'Studio', room='A newly opened door.')['kind'] == 'navigate_to_home'


def test_active_goal_survives_reranking_but_releases_on_arrival_and_deposit():
    control = GoalControl()
    first = select(control)
    assert select(control, 'Studio', proposed={'kind': 'navigate_to_home', 'target_node': 'Living Room',
                                             'description': 'Reworded'}) == first
    goal = select(control, 'Living Room', proposed={'kind': 'deposit_treasure', 'target_node': 'Living Room'})
    assert goal['kind'] == 'deposit_treasure'
    goal = select(control, 'Living Room', inventory=[], proposed={'kind': 'explore_frontier'})
    assert goal['kind'] == 'explore_frontier'


def test_long_acyclic_zero_reward_route_is_not_a_movement_cycle():
    control = GoalControl()
    for n in range(15):
        assert select(control, f'Room {n}')['kind'] == 'navigate_to_home'
    assert not control.suppressed


def fake_client(outcomes):
    replies, calls, options = iter(outcomes), [], []
    def create(**kwargs):
        calls.append(kwargs)
        result = next(replies)
        if isinstance(result, Exception):
            raise result
        return result
    client = NS(chat=NS(completions=NS(create=create)))
    client.with_options = lambda **kwargs: options.append(kwargs) or client
    return client, calls, options


def test_http_json_decode_retries_same_request_without_model_repair(monkeypatch):
    monkeypatch.setattr('MANGORL.provider_requests.time.sleep', lambda _: None)
    client, calls, options = fake_client([json.JSONDecodeError('bad', 'PRIVATE_BODY', 0), {'ok': True}])
    events = []
    request = {'model': 'test', 'messages': [{'role': 'user', 'content': 'choose'}]}
    assert provider_request(client, request, event_logger=lambda *x: events.append(x)) == {'ok': True}
    assert calls == [request, request]
    assert options == [{'max_retries': 0, 'timeout': 60.0}]
    assert 'PRIVATE_BODY' not in json.dumps(events)


def test_exhausted_http_json_retries_are_typed_and_bounded(monkeypatch):
    monkeypatch.setattr('MANGORL.provider_requests.time.sleep', lambda _: None)
    client, calls, _ = fake_client([json.JSONDecodeError('bad', '', 0)] * 3)
    with pytest.raises(MalformedProviderResponse) as error:
        provider_request(client, {'model': 'test'})
    assert len(calls) == 3
    assert recoverable_episode_error(error.value)
    assert not recoverable_episode_error(json.JSONDecodeError('local file', '', 0))


def test_shared_request_budget_prevents_nested_retry_multiplication():
    client, calls, _ = fake_client([{}, {}, {}])
    with request_budget(2):
        provider_request(client, {})
        provider_request(client, {})
        with pytest.raises(ProviderBudgetExhausted):
            provider_request(client, {})
    assert len(calls) == 2


def test_local_exploration_goal_revalidates_after_room_change():
    control = GoalControl()
    select(control, 'Gallery', inventory=[], proposed={'kind': 'explore_frontier', 'preferred_actions': ['north']})
    goal = select(control, 'Studio', inventory=[], proposed={'kind': 'explore_frontier', 'preferred_actions': ['up']})
    assert goal['preferred_actions'] == ['up']


def test_default_action_budget_ends_episode_without_native_samples(tmp_path, monkeypatch):
    run, games = failure_run(tmp_path, monkeypatch, always_fail=True, fail_every=True, terminal_moves=100,
                            extra=['--max-steps', '10', '--max-recovery-actions', '5'])
    directory = run()
    outcomes = json.loads((directory / 'run_status.json').read_text())['outcomes']
    assert all(x['termination_reason'] == 'recovery_action_limit' for x in outcomes)
    assert all(game.moves == 5 and game.closed for game in games)
    assert not (directory / 'metrics.jsonl').exists()
    assert not (directory / 'memory/actor_experiences.jsonl').exists()
    assert 'RECOVERY_DEFAULT_ACTION' in (directory / 'episode_0001_chat.txt').read_text()


def test_resume_preserves_old_logs_and_episode_sample_identities(tmp_path, monkeypatch):
    run, _ = failure_run(tmp_path, monkeypatch)
    directory = run()
    old_log = (directory / 'episode_0001_chat.txt').read_bytes()
    original = (directory / 'memory/actor_experiences.jsonl').read_bytes()
    assert main.main(['--resume-trial', str(directory), '--episodes', '3']) == directory
    assert (directory / 'episode_0001_chat.txt').read_bytes() == old_log
    assert (directory / 'memory/actor_experiences.jsonl').read_bytes().startswith(original)
    rows = [json.loads(line) for line in (directory / 'memory/actor_experiences.jsonl').read_text().splitlines()]
    keys = [(r['episode_id'], r['step']) for r in rows]
    assert len(keys) == len(set(keys))
    assert [x['episode'] for x in json.loads((directory / 'run_status.json').read_text())['outcomes']] == [1, 2, 3]


def test_crash_reservation_is_consumed_without_replaying_partial_episode(tmp_path):
    args = main.parse_args(['--no-evolve', '--no-historian-review'])
    checkpoint = TrialCheckpoint(tmp_path, args)
    checkpoint.begin(1)
    restored = TrialCheckpoint(tmp_path, args)
    restored.publish(2)
    assert restored.data['next_episode'] == 2
    assert restored.data['outcomes'] == [{'episode': 1, 'status': 'interrupted', 'termination_reason': 'process_interrupted'}]
    assert len(TrialCheckpoint(tmp_path, args).data['outcomes']) == 1
    assert 'api_key' not in restored.data['arguments']


def test_same_trial_cannot_be_resumed_concurrently(tmp_path):
    with trial_lock(tmp_path):
        with pytest.raises(OSError):
            with trial_lock(tmp_path):
                pytest.fail('Second writer acquired the lock')


def test_environment_closes_even_when_final_persistence_fails(tmp_path, monkeypatch):
    run, games = failure_run(tmp_path, monkeypatch)
    def fail(*a, **k):
        raise OSError('disk unavailable')
    monkeypatch.setattr(main.ActorExperienceMemory, 'finish_episode', fail)
    with pytest.raises(OSError, match='disk unavailable'):
        run()
    assert games[0].closed
