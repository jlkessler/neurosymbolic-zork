"""Evaluator-only counterfactual probes; no APIs or trial-memory writes.

Run from the WSL project root. Uses recorded commands and seed 0. These
counterfactual observations are deliberately not installed as agent memories.
"""
from pathlib import Path
import json
import re
from jericho import FrotzEnv

trial = Path("logs/tabula-rasa-5_big_20261111_172225")
destination = Path("/mnt/c/Users/jlkes/Code/Thesis/my_solution/reports/trial5-data/replay_probes.json")
blocks = re.split(r"^========== STEP (\d+)/\d+ =+\s*$", (trial / "episode_0044.txt").read_text(), flags=re.M)
commands = []
for i in range(1, len(blocks), 2):
    m = re.search(r"^Action: (.*)$", blocks[i+1], re.M)
    if m:
        commands.append((int(blocks[i]), m[1]))
env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5", seed=0)
results = []
try:
    env.reset()
    for step, command in commands:
        if step in (118, 195):
            snapshot = env.get_state()
            before = env.get_player_location()
            probes = ([['up'], ['drop manual', 'drop sword', 'drop garlic', 'drop lunch', 'up']]
                      if step == 118 else [['southeast'], ['west']])
            for sequence in probes:
                env.set_state(snapshot)
                observed = []
                for action in sequence:
                    text, reward, done, info = env.step(action)
                    after = env.get_player_location()
                    observed.append({"action": action, "observation": text, "reward": reward,
                                     "done": done, "info": info, "room_id": after.num, "room_name": after.name})
                    if done:
                        break
                results.append({"episode": 44, "before_displayed_step": step,
                                "before_room_id": before.num, "probe_sequence": observed})
            env.set_state(snapshot)
        if step == 195:
            break
        observation, reward, done, info = env.step(command)
        if done:
            raise RuntimeError(f"Replay terminated unexpectedly at step {step}")
finally:
    env.close()
destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
