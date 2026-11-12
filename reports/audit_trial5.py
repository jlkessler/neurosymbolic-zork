"""Read-only, reproducible audit of the supplied trial. Writes derived reports only."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import re
import statistics


def events(text):
    parts = re.split(r"^=+ ([A-Z_]+) =+\s*$", text, flags=re.M)
    for i in range(1, len(parts), 2):
        body = re.sub(r"^\s*Timestamp:[^\n]*\n", "", parts[i + 1]).strip()
        try:
            yield parts[i], json.loads(body)
        except json.JSONDecodeError:
            yield "UNPARSED_" + parts[i], {"chars": len(body)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    p = args.trial
    manifest = []
    # Hash every supplied file, including redundant narrative views. Nothing in
    # the source trial is opened for writing; the manifest also records coverage.
    for f in sorted(p.rglob("*")):
        if f.is_file():
            raw = f.read_bytes()
            manifest.append({"path": str(f.relative_to(p)), "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest()})
    status = json.loads((p / "run_status.json").read_text())
    outcomes = {r["episode"]: r for r in status["outcomes"]}
    rows, traces, errors, snapshots, config_runs = [], [], [], [], []
    all_events, all_actions, all_locations, all_sources = Counter(), Counter(), Counter(), Counter()
    rewards, positive_actions, negative_actions = [], Counter(), Counter()
    chat_sizes, prompt_chars, safe_menu_counts = [], [], Counter()
    controls, goal_kinds, cancellations = Counter(), Counter(), Counter()
    logprob_attempts, logprob_missing, response_finishes = Counter(), Counter(), Counter()
    provider_errors, review_counts, mutation_counts = [], Counter(), Counter()
    maze_frontiers = []
    experience_rows, policy_rows, menu_coverage = [], [], []
    for ep in range(1, 51):
        stem = f"episode_{ep:04d}"
        txt = (p / f"{stem}.txt").read_text()
        parts = re.split(r"^========== STEP (\d+)/\d+ ==========\s*$", txt, flags=re.M)
        steps = []
        for i in range(1, len(parts), 2):
            n, block = int(parts[i]), parts[i + 1]
            def field(key):
                m = re.search(r"^" + re.escape(key) + r": (.*)$", block, re.M)
                return m[1] if m else None
            action = field("Action")
            if action is None:
                continue
            # End-of-episode material after this block must not contaminate its observation.
            obs = block.split("Observation: ", 1)[-1].split("\nNew location:", 1)[0].strip()
            m = re.search(r"^Score: (-?[\d.]+) \(delta: (-?[\d.]+)\)", block, re.M)
            score, delta = (float(m[1]), float(m[2])) if m else (None, None)
            step = {"episode": ep, "step": n, "location": field("Location"),
                    "new_location": field("New location"), "action": action,
                    "source": field("Action source"), "score": score, "delta": delta,
                    "observation": obs, "goal": field("Goal")}
            steps.append(step)
            all_actions[action] += 1
            all_locations[step["location"]] += 1
            all_sources[step["source"]] += 1
            if delta:
                rewards.append(step)
                (positive_actions if delta > 0 else negative_actions)[action] += 1
            if re.search(r"curse|banish|died|no longer|too.*load|too.*carry", obs, re.I):
                traces.append(step)
        last = steps[-1] if steps else {}
        scored = [s["score"] for s in steps if s["score"] is not None]
        gains = [s for s in steps if s["delta"] and s["delta"] > 0]
        tail = len(steps) - gains[-1]["step"] if gains else len(steps)
        row = {"episode": ep, **outcomes.get(ep, {}), "logged_actions": len(steps),
               "peak_score": max(scored) if scored else None,
               "last_positive_step": gains[-1]["step"] if gains else None,
               "tail_actions_without_gain": tail, "last_action": last.get("action"),
               "last_location": last.get("location"), "last_observation": last.get("observation"),
               "unique_locations": len({s["location"] for s in steps}),
               "maze_actions": sum((s["location"] or "").startswith("Maze") for s in steps),
               "symbolic_actions": sum(s["source"] == "symbolic_navigation" for s in steps)}
        rows.append(row)
        for line in txt.splitlines():
            if re.search(r"failed|no_safe_actions|no safe|retry|error", line, re.I):
                errors.append({"episode": ep, "text": line[:1800]})
        chat = (p / f"{stem}_chat.txt").read_text()
        chat_sizes.append(len(chat))
        context = {}
        for name, obj in events(chat):
            all_events[name] += 1
            if name == "ACTOR_ENVIRONMENT_CONTEXT":
                context = obj
                menu = obj.get("valid_actions", [])
                for a in ("look", "inventory", "wait"):
                    safe_menu_counts[a] += a in menu
                if (obj.get("location") or "").startswith("Maze"):
                    maze_frontiers.append({"episode": ep, "location": obj["location"], "valid_actions": menu})
                text = obj.get("observation", "")
                if "CURRENT INVENTORY:" in text:
                    inv = text.split("CURRENT INVENTORY:", 1)[1]
                    if re.search(r"troll|thief|died|glow|swing|finishing|carrying a large bag", inv, re.I):
                        snapshots.append({"episode": ep, "location": obj.get("location"), "inventory_text": inv})
            elif name == "ACTIVE_CONFIGURATION":
                config_runs.append({"episode": ep, **obj})
                row["configuration_id"] = obj.get("id")
            elif name == "REQUEST":
                prompt_chars.append(sum(len(str(m.get("content", ""))) for m in obj.get("messages", [])))
            elif name == "RESPONSE":
                response_finishes[str(obj.get("finish_reason"))] += 1
            elif name == "CONTROL_METRICS":
                controls.update(obj)
                row["control_metrics"] = obj
            elif name == "ACTOR_LOGPROB_ATTEMPT":
                logprob_attempts[str(obj.get("uniform_label_bias"))] += 1
                for label in obj.get("missing_labels", []):
                    logprob_missing[label] += 1
            elif name == "ACTOR_FINAL_SELECTION":
                selected = obj.get("selection", {})
                if context.get("valid_actions"):
                    menu_coverage.append(len(selected.get("candidates", [])) / len(context["valid_actions"]))
            elif name == "GOAL_TRANSITION":
                goal_kinds[obj.get("goal", {}).get("kind")] += 1
                if obj.get("status") == "abandoned":
                    cancellations[obj.get("reason")] += 1
            elif name == "PROVIDER_REQUEST_FAILURE":
                provider_errors.append({"episode": ep, **obj})
            elif "HISTORIAN_REVIEW" in name:
                review_counts[name] += 1
            elif "CONFIGURATION" in name:
                mutation_counts[name] += 1
        experience_path = p / f"{stem}_experience.jsonl"
        if experience_path.exists():
            for line in experience_path.read_text().splitlines():
                x = json.loads(line)
                experience_rows.append(x)
                if x.get("selection_info", {}).get("base_probabilities"):
                    policy_rows.append(x)
    complete = [r for r in rows if r.get("status") == "completed"]
    scored = [r for r in rows if "score" in r]
    def argmax(v):
        return max(range(len(v)), key=v.__getitem__)
    def gap(v):
        ordered = sorted(v, reverse=True)
        return ordered[0] - ordered[1] if len(ordered) > 1 else None
    selection = [x["selection_info"] for x in policy_rows]
    memory_rows = [json.loads(s) for s in (p / "memory/actor_experiences.jsonl").read_text().splitlines() if s.strip()]
    death_causes = Counter()
    for r in rows:
        if r.get("termination_reason") != "environment_done":
            continue
        obs = str(r["last_observation"]).lower()
        cause = ("grue/darkness" if "grue" in obs else
                 "jump" if r["last_action"] == "jump" else
                 "rusty knife" if r["last_action"] == "hit chimney with knife" else
                 "troll" if "troll" in obs else "thief" if "thief" in obs else "unclassified")
        death_causes[cause] += 1
    result = {
        "source_trial": str(p), "file_count": len(manifest), "bytes": sum(m["bytes"] for m in manifest),
        "outcome_counts": Counter(r.get("termination_reason") for r in rows),
        "completed_mean": statistics.mean(r["score"] for r in complete),
        "all_scored_mean": statistics.mean(r["score"] for r in scored),
        "ten_episode_blocks": [{"start": a, "end": a + 9,
            "mean_recorded_score": statistics.mean(r["score"] for r in rows[a-1:a+9] if "score" in r),
            "n_scored": sum("score" in r for r in rows[a-1:a+9])} for a in range(1,51,10)],
        "total_actions": sum(r["logged_actions"] for r in rows),
        "tail_actions_without_gain": sum(r["tail_actions_without_gain"] for r in rows),
        "events": all_events, "actions": all_actions, "locations": all_locations, "sources": all_sources,
        "positive_actions": positive_actions, "negative_actions": negative_actions,
        "control_metrics": controls, "goal_kinds": goal_kinds, "goal_cancellations": cancellations,
        "logprob_attempt_bias": logprob_attempts, "missing_label_counts": logprob_missing,
        "provider_errors": provider_errors, "review_events": review_counts, "mutation_events": mutation_counts,
        "safe_actions_in_oracle": safe_menu_counts, "inventory_with_ambient_text_count": len(snapshots),
        "prompt_chars": {"count": len(prompt_chars), "median": statistics.median(prompt_chars), "max": max(prompt_chars)},
        "death_causes_from_final_observation": death_causes,
        "experience": {"episode_records": len(experience_rows), "memory_records": len(memory_rows),
            "episode_files_equal_memory": experience_rows == memory_rows},
        "policy": {"records": len(selection),
            "median_max_base_probability": statistics.median(max(x["base_probabilities"]) for x in selection),
            "base_max_over_99_percent": sum(max(x["base_probabilities"]) > .99 for x in selection),
            "median_top_two_gap": statistics.median(v for x in selection if (v := gap(x["base_logits"])) is not None),
            "median_kl_to_base": statistics.median(x["kl_to_base"] for x in selection),
            "rl_changes": sum(argmax(x["base_logits"]) != argmax(x["adjusted_logits"]) for x in selection),
            "nsi_changes": sum(argmax(x["adjusted_logits"]) != argmax(x["probabilities"]) for x in selection),
            "combined_changes": sum(argmax(x["base_logits"]) != argmax(x["probabilities"]) for x in selection),
            "label_a_preferred": sum(argmax(x["base_logits"]) == 0 for x in selection),
            "median_candidate_menu_coverage": statistics.median(menu_coverage)},
        "episodes": rows,
    }
    for name, obj in [("audit.json", result), ("manifest.json", manifest), ("reward_events.json", rewards),
                      ("diagnostic_events.json", traces), ("errors.json", errors),
                      ("inventory_snapshots.json", snapshots), ("active_configurations.json", config_runs),
                      ("maze_menus.json", maze_frontiers)]:
        (args.output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [k for k in rows[0] if k not in {"last_observation", "control_metrics"}]
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items() if k not in {
        "episodes", "actions", "events", "locations", "provider_errors", "goal_kinds"}}, indent=2))
    print("EVENTS", all_events)
    print("TOP_ACTIONS", all_actions.most_common(20))
    print("LOCATIONS", all_locations.most_common())


if __name__ == "__main__":
    main()
