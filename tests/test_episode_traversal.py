import json

from MANGORL.episode_analysis import measured_objectives, objective_summary, objectives, read_episode, transition
from visualize_trial import backfill, build_report, report_data


def step(n, before, after, command, response, **kwargs):
    return transition(n, before, after, before, after, command, "smart_actor", response, 0, 0, **kwargs)


def test_milestones_require_observed_outcomes_and_count_first_success_only():
    rows = [
        step(1, "Gallery", "Gallery", "take painting", "You can't see any painting here."),
        step(2, "Gallery", "Gallery", "take art", "Taken."),
        step(3, "Gallery", "Gallery", "take painting", "Taken."),
        step(4, "Studio", "Kitchen", "up", "Kitchen"),
        step(5, "Kitchen", "Living Room", "west", "Living Room"),
        step(6, "Living Room", "Living Room", "put painting in case", "The case is closed."),
        step(7, "Living Room", "Living Room", "put painting in sack", "Done."),
        step(8, "Living Room", "Living Room", "put painting in trophy case", "Done."),
        step(9, "Living Room", "Living Room", "take bag", "Taken.",
             inventory_before=[], inventory_after=["bag"]),
        step(10, "Living Room", "Living Room", "take bag", "Taken.",
             inventory_before=[], inventory_after=["bag of coins"]),
        step(11, "Living Room", "Living Room", "put all in case", "Bag of coins: Done.\nPlatinum bar: You don't have it."),
        step(12, "Cellar", "Troll Room", "north", "The troll swings his axe."),
        step(13, "Troll Room", "Troll Room", "attack troll", "Almost as soon as the troll breathes his last breath, fog envelops him."),
        step(14, "Maze (room #12)", "Treasure Room", "east", "Treasure Room"),
    ]
    assert objectives(rows) == {"painting_taken": 2, "kitchen_reached": 4, "return_route_found": 4,
                                "returned_home": 5, "painting_deposited": 8, "coins_taken": 10,
                                "coins_deposited": 11, "troll_killed": 13, "treasure_room_reached": 14}


def test_reset_and_log_gaps_break_return_route_evidence():
    rows = [step(1, "Living Room", "Cellar", "down", "Cellar"),
            step(2, "Cellar", "Forest", "east", "**** You have died ****\nForest"),
            step(3, "Kitchen", "Living Room", "west", "Living Room"),
            step(4, "Living Room", "Cellar", "down", "Cellar"),
            step(9, "Kitchen", "Living Room", "west", "Living Room")]
    assert objectives(rows) == {"cellar_reached": 1}


def test_signed_scores_ids_and_incomplete_backfill(tmp_path):
    journal = tmp_path / "episode_0001_traversal.jsonl"
    header = {"start": "id:1", "start_label": "Maze", "start_score": 0}
    rows = [transition(1, "id:1", "id:2", "Maze", "Maze", "down", "symbolic_navigation", "Maze", 5, 5),
            transition(2, "id:2", "id:1", "Maze", "Maze", "up", "smart_actor", "Maze", 3, -2),
            transition(3, "id:1", "id:1", "Maze", "Maze", "take bar", "smart_actor", "Taken. </script>", 3, 0)]
    journal.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]), encoding="utf-8")
    metrics = [{"episode": 1, "score": 3, "steps": 3}, {"episode": 2, "score": 0, "steps": 8}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, metrics)), encoding="utf-8")
    (tmp_path / "metrics_summary.json").write_text('{"mean_score": 1.5}', encoding="utf-8")
    assert measured_objectives(tmp_path, {"episode": 1, "steps": 5}) is None
    backfill(tmp_path)
    summary = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert summary["mean_score"] == 1.5
    summary = summary["objectives_met"]
    assert summary["evaluated_episodes"] == 2 and summary["episodes_without_evidence"] == [2]
    assert summary["milestones"]["bar_taken"]["first_step_by_episode"] == {"1": 3}
    assert summary["milestones"]["maze_reached"]["rate"] == 1.0
    report = report_data(tmp_path)
    assert report["milestone_summary"] == summary
    assert report["milestone_source"] == "metrics_summary.json"
    assert report["rooms"]["id:1"]["label"] != report["rooms"]["id:2"]["label"]
    assert [r["delta"] for r in report["episodes"][0]["steps"]] == [5, -2, 0]
    assert report["episodes"][1]["steps"] == []
    output = tmp_path / "map.html"
    assert build_report(tmp_path, output) == 2
    assert r"Taken. \u003c/script>" in output.read_text(encoding="utf-8")
    assert objective_summary({2: None})["milestones"]["egg_taken"]["rate"] is None


def test_legacy_trace_keeps_navigator_and_excludes_unexecuted_commands(tmp_path):
    text = """========== STEP 1/4 ==========
Location: Maze (room #41)
Action source: symbolic_navigation
Action: down
Observation: Maze
New location: Maze (room #42)
Score: 4 (delta: 4) | reward: 4
========== STEP 2/4 ==========
Location: Maze (room #42)
Action source: smart_actor
Action: take all
Observation: Jewel-encrusted egg: Taken.
Painting: Too heavy.
New location: Maze (room #42)
Score: 4 (delta: 0) | reward: 0
========== STEP 3/4 ==========
Location: Maze (room #42)
Action: look
"""
    (tmp_path / "episode_0001.txt").write_text(text, encoding="utf-8")
    ep = read_episode(tmp_path, 1)
    assert len(ep["steps"]) == 2
    assert (ep["steps"][0]["before"], ep["steps"][0]["after"]) == ("id:41", "id:42")
    assert ep["steps"][0]["source"] == "symbolic_navigation"
    assert objectives(ep["steps"]) == {"maze_reached": 1, "egg_taken": 2}


def test_report_uses_saved_milestones_when_replay_is_missing(tmp_path):
    saved = objective_summary({5: {"troll_killed": 79}, 6: {}, 8: None})
    (tmp_path / "metrics_summary.json").write_text(json.dumps({"objectives_met": saved}), encoding="utf-8")
    report = report_data(tmp_path)
    assert report["milestone_summary"] == saved
    assert [ep["episode"] for ep in report["episodes"]] == [5, 6, 8]
    assert report["episodes"][0]["objectives"] == {"troll_killed": 79}
    assert report["episodes"][0]["steps"] == []


def test_archived_trial_rates_exclude_incomplete_degraded_and_missing_traces(tmp_path):
    outcomes = [{"episode": 1, "status": "completed"}, {"episode": 2, "status": "incomplete"},
                {"episode": 3, "status": "completed", "degraded": True}, {"episode": 4, "status": "completed"}]
    (tmp_path / "run_status.json").write_text(json.dumps({"outcomes": outcomes}), encoding="utf-8")
    transcript = "\n".join(["========== STEP 1/5 ==========", "Location: Up a Tree", "Action: take egg",
                            "Observation: Taken.", "New location: Up a Tree", "Score: 5 (delta: 5)",
                            "Episode finished: score=5, moves=1, reason=environment_done"])
    for n in (1, 2, 3):
        (tmp_path / f"episode_{n:04d}.txt").write_text(transcript, encoding="utf-8")
    (tmp_path / "episode_0004.txt").write_text(transcript.replace("STEP 1/", "STEP 2/"), encoding="utf-8")
    report = report_data(tmp_path)
    summary = report["milestone_summary"]
    assert len(report["episodes"]) == 4
    assert summary["evaluated_episodes"] == 2 and summary["episodes_with_evidence"] == 1
    assert summary["episodes_without_evidence"] == [4]
    assert summary["milestones"]["egg_taken"]["first_step_by_episode"] == {"1": 1}
    assert report["episodes"][1]["objectives"]["egg_taken"] == 1

    # The explicit metrics cohort takes precedence over reconstructed completion statuses.
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"episode": 3, "steps": 1}), encoding="utf-8")
    report = report_data(tmp_path)
    assert report["milestone_summary"]["milestones"]["egg_taken"]["first_step_by_episode"] == {"3": 1}
