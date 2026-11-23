"""Build an offline, interactive traversal figure from a Zork trial directory."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re

from MANGORL.checkpoint import atomic_json, trial_lock
from MANGORL.episode_analysis import (
    OBJECTIVES, has_complete_steps, measured_objectives, objective_summary, objectives, read_episode,
)


def layout(nodes, edges):
    """Place rooms once per trial. Frequencies never change the geometry."""
    keys = sorted(nodes)
    if not keys:
        return {}
    rng = random.Random(41)
    pos = {k: [rng.uniform(-1, 1), rng.uniform(-1, 1)] for k in keys}
    springs = sorted(edges)
    for iteration in range(150):
        force = {k: [0., 0.] for k in keys}
        for i, a in enumerate(keys):
            for b in keys[:i]:
                dx, dy = (pos[a][j] - pos[b][j] for j in (0, 1))
                scale = 1 / max(.0001, dx * dx + dy * dy)
                force[a][0] += dx * scale
                force[a][1] += dy * scale
                force[b][0] -= dx * scale
                force[b][1] -= dy * scale
        for a, b in springs:
            dx, dy = (pos[a][j] - pos[b][j] for j in (0, 1))
            length = math.hypot(dx, dy)
            force[a][0] -= dx * length
            force[a][1] -= dy * length
            force[b][0] += dx * length
            force[b][1] += dy * length
        for k in keys:
            fx, fy = force[k]
            fx -= .05 * pos[k][0]
            fy -= .05 * pos[k][1]
            scale = min(1, (.25 * (1 - iteration / 150)) / max(.0001, math.hypot(fx, fy)))
            pos[k][0] += fx * scale
            pos[k][1] += fy * scale
    return {k: [round(x * 145), round(y * 116)] for k, (x, y) in pos.items()}


def metrics(directory):
    path = directory / "metrics.jsonl"
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["episode"]: row for row in rows}


def backfill(directory):
    with trial_lock(directory):
        path = directory / "metrics_summary.json"
        if not path.exists() or not (directory / "metrics.jsonl").exists():
            raise ValueError("Updating objectives requires both metrics_summary.json and metrics.jsonl.")
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["objectives_met"] = objective_summary({
            n: measured_objectives(directory, row) for n, row in metrics(directory).items()
        })
        atomic_json(path, summary)


def report_data(directory):
    evaluated = metrics(directory)
    summary_path = directory / "metrics_summary.json"
    saved = (json.loads(summary_path.read_text(encoding="utf-8")).get("objectives_met")
             if summary_path.exists() else None)
    path = directory / "run_status.json"
    status = json.loads(path.read_text(encoding="utf-8")).get("outcomes", []) if path.exists() else []
    outcomes = {s["episode"]: s for s in status}
    numbers = set(outcomes) | set(evaluated)
    saved_by_episode = {}
    if saved is not None:
        saved_by_episode = {int(n): {} for n in saved.get("distinct_by_episode", {})}
        for key, stat in saved["milestones"].items():
            for n, step in stat["first_step_by_episode"].items():
                saved_by_episode.setdefault(int(n), {})[key] = step
        numbers.update(saved_by_episode)
        numbers.update(saved["episodes_without_evidence"])
    for file in directory.iterdir():
        match = re.fullmatch(r"episode_(\d+)(?:\.txt|_traversal\.jsonl)", file.name)
        if match:
            numbers.add(int(match[1]))
    if not numbers:
        raise ValueError(f"No episode logs or outcomes in {directory}")
    episodes, labels, edges = [], {}, set()
    cohort = {}
    have_metrics = (directory / "metrics.jsonl").exists()
    for number in sorted(numbers):
        trace = read_episode(directory, number)
        ep = trace or {"episode": number, "start": None, "start_label": None,
                                                  "steps": [], "identity": "No traversal evidence"}
        state, metric = outcomes.get(number, {}), evaluated.get(number, {})
        ep["status"] = state.get("status", "evaluated" if metric else "status unavailable")
        if state.get("degraded"):
            ep["status"] += " / degraded"
        ep["reason"] = state.get("termination_reason") or metric.get("termination_reason", "")
        observed = objectives(ep["steps"])
        recorded = saved_by_episode.get(number, metric.get("objectives_met"))
        ep["objectives"] = observed if recorded is None else recorded
        if saved is None:
            if number in evaluated:
                cohort[number] = recorded if recorded is not None else (
                    observed if has_complete_steps(trace, metric.get("steps")) else None)
            elif not have_metrics and state.get("status") == "completed" and not state.get("degraded"):
                expected = state.get("steps", ep.get("console_end_step"))
                cohort[number] = observed if has_complete_steps(trace, expected) else None
        if ep["start"]:
            labels.setdefault(ep["start"], ep["start_label"] or ep["start"].replace("id:", "Room #"))
        for row in ep["steps"]:
            for side in ("before", "after"):
                node, title = row[side], row.pop(side + "_label")
                if node:
                    previous = labels.setdefault(node, node.replace("id:", "Room #"))
                    if title and (previous.startswith("Room #") or
                                  (not title.startswith("Room #") and len(title) >= len(previous))):
                        labels[node] = title
            a, b = row["before"], row["after"]
            if a and b and a != b and not row["relocation"]:
                edges.add(tuple(sorted((a, b))))
            row.pop("inventory_added", None)
            row.pop("inventory_removed", None)
        if "start_score" not in ep and ep["steps"]:
            first = ep["steps"][0]
            if first["step"] == 1 and first["score"] is not None and first["delta"] is not None:
                ep["start_score"] = first["score"] - first["delta"]
        episodes.append(ep)
    repeated = Counter(labels.values())
    positions = layout(labels, edges)
    rooms = {key: {"name": label, "label": f"{label} · {key.split(':', 1)[-1]}"
                   if repeated[label] > 1 else label, "pos": positions[key]} for key, label in labels.items()}
    if saved is not None:
        source = "metrics_summary.json"
    elif have_metrics:
        source = "metrics.jsonl and episode records"
    elif outcomes:
        source = "Completed, non-degraded outcomes in run_status.json and episode records"
    else:
        source = "Evaluation cohort unavailable"
    summary = saved if saved is not None else objective_summary(cohort)
    labels_by_objective = {**OBJECTIVES, **{k: v["label"] for k, v in summary["milestones"].items()}}
    return {"trial": directory.name, "episodes": episodes, "rooms": rooms, "objectives": labels_by_objective,
            "milestone_summary": summary, "milestone_source": source}


def build_report(directory, output):
    data = report_data(directory)
    template = (Path(__file__).parent / "MANGORL/traversal_viewer.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("<!--TRIAL_DATA-->", payload), encoding="utf-8")
    return len(data["episodes"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="Trial directory containing episode logs")
    parser.add_argument("--output", type=Path, help="HTML destination (default: TRIAL/traversal.html)")
    parser.add_argument("--update-summary", action="store_true", help="Backfill milestone statistics in an existing summary")
    args = parser.parse_args(argv)
    if not args.trial.is_dir():
        parser.error(f"Trial directory not found: {args.trial.resolve()}\n"
                     f"Current directory: {Path.cwd()}\n"
                     "From the project root, use logs/... and select the directory containing the episode files.")
    output = args.output or args.trial / "traversal.html"
    if output.suffix.lower() not in {".htm", ".html"}:
        parser.error("The report output must be an HTML file.")
    try:
        try:
            if args.update_summary:
                backfill(args.trial)
                print(f"Updated {args.trial / 'metrics_summary.json'}")
        finally:
            # Archives without the original metrics files still support replay.
            count = build_report(args.trial, output)
            print(f"Wrote {output} ({count} episodes)")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
