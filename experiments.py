"""Run matched-budget Zork ablations with isolated memories per seed/session.

Example: python experiments.py --dry-run --seeds 0 1 2 -- --model MODEL
The harness runs sequentially and stops on failure. It does not claim paper
reproduction: the current runner uses privileged Jericho action/room access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CONDITIONS = {
    "fixed_policy": ["--no-evolve", "--no-actor-rl", "--no-historian-critic", "--no-historian-review", "--no-nsi"],
    "evolution": ["--evolve", "--no-actor-rl", "--no-historian-critic", "--no-historian-review", "--no-nsi"],
    "actor_critic": ["--no-evolve", "--actor-rl", "--historian-critic", "--no-historian-review", "--no-nsi"],
    "hybrid": ["--evolve", "--actor-rl", "--historian-critic", "--no-historian-review", "--nsi"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=110)
    parser.add_argument("--output-dir", default="logs/ablations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("Episodes and max steps must be positive.")
    extra = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    root = Path(__file__).resolve().parent
    for condition in args.conditions:
        for seed in args.seeds:
            command = [sys.executable, str(root / "main.py"), *extra,
                       "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
                       "--target-score", "350", "--score-scale", "350", "--seed", str(seed),
                       "--output-dir", args.output_dir, "--trial-name", f"{condition}_seed{seed}",
                       "--fresh-memory", *CONDITIONS[condition]]
            # JSON preserves Windows argument boundaries and avoids shell quoting.
            displayed = list(command)
            for index, value in enumerate(displayed):
                if value in {"--api-key", "--historian-api-key"} and index + 1 < len(displayed):
                    displayed[index + 1] = "<redacted>"
                elif value.startswith(("--api-key=", "--historian-api-key=")):
                    displayed[index] = value.split("=", 1)[0] + "=<redacted>"
            print(json.dumps(displayed))
            if not args.dry_run:
                subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
