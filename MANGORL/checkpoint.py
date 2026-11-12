"""Atomic episode-boundary resume; never replay a potentially committed episode."""
import json
import os
import hashlib
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def trial_lock(directory):
    """OS locks are released on process death; no stale-PID guessing needed."""
    with (Path(directory) / '.trial.lock').open('a+b') as file:
        if os.name == 'nt':
            import msvcrt
            file.write(b'0'); file.flush(); file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                file.seek(0); msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file, fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as file:
        json.dump(value, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


class TrialCheckpoint:
    def __init__(self, directory, args):
        self.path = Path(directory) / 'checkpoint.json'
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if self.data.get('schema_version') != 1:
                raise ValueError('Unsupported checkpoint version; original preserved.')
            rom = Path(self.data['arguments']['rom_path'])
            expected = self.data.get('rom_sha256')
            if expected and (not rom.exists() or hashlib.sha256(rom.read_bytes()).hexdigest() != expected):
                raise ValueError('Checkpoint ROM changed; refusing to mix incompatible map identities.')
            active = self.data.get('active_episode')
            if active is not None:
                # A killed process may already have appended learning records.
                # Consume that identity instead of replaying or overwriting it.
                self.data['outcomes'].append({'episode': active, 'status': 'interrupted',
                                             'termination_reason': 'process_interrupted'})
                self.data['active_episode'] = None
                self.save()
        else:
            safe = {k: v for k, v in vars(args).items() if 'api_key' not in k and 'secret' not in k}
            for key in ('map_path', 'historian_path', 'actor_experience_path', 'historian_critic_path', 'evolution_path', 'rom_path'):
                safe[key] = str(Path(safe[key]).resolve())
            self.data = {'schema_version': 1, 'arguments': safe, 'outcomes': [],
                         'next_episode': 1, 'active_episode': None}
            rom = Path(safe['rom_path'])
            self.data['rom_sha256'] = hashlib.sha256(rom.read_bytes()).hexdigest() if rom.exists() else None
            self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def begin(self, number):
        self.data.update(active_episode=number, next_episode=number + 1)
        self.save()  # Reserve identity before opening logs or executing commands.

    def finish(self, outcome):
        self.data['outcomes'].append(outcome)
        self.data['active_episode'] = None
        self.save()

    def publish(self, requested):
        outcomes = self.data['outcomes']
        completed = sum(x['status'] == 'completed' for x in outcomes)
        atomic_json(self.path.parent / 'run_status.json', {
            'requested_episodes': requested, 'attempted_episodes': len(outcomes),
            'completed_episodes': completed, 'incomplete_episodes': len(outcomes) - completed,
            'degraded_episodes': sum(bool(x.get('degraded')) for x in outcomes),
            'all_episodes_completed': completed == requested, 'outcomes': outcomes,
        })
        # The checkpoint is authoritative. Rebuild the readable ledger after a
        # crash rather than appending the same episode's outcome a second time.
        ledger = self.path.parent / 'episode_outcomes.jsonl'
        temp = ledger.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(x) + '\n' for x in outcomes), encoding='utf-8')
        temp.replace(ledger)


def restore_arguments(args, argv):
    if not args.resume_trial:
        return args
    if args.fresh_memory:
        raise ValueError('--resume-trial cannot be combined with --fresh-memory.')
    path = Path(args.resume_trial).resolve() / 'checkpoint.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint version.')
    total = args.episodes
    resume = str(path.parent)
    # Resume the original experimental configuration, with current credentials.
    # Only the requested total episode count may be extended.
    for key, value in saved['arguments'].items():
        if 'api_key' not in key and 'secret' not in key:
            setattr(args, key, value)
    if any(x in {'--episodes', '--num-episodes'} or x.startswith(('--episodes=', '--num-episodes=')) for x in argv):
        args.episodes = total
    args.resume_trial, args.fresh_memory = resume, False
    return args
