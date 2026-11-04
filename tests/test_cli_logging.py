import json

import pytest

from MANGORL.historian import Historian
from main import parse_args
from MANGORL.smart_agent import SmartAgent
from MANGORL.trial_logger import TrialLogger


def test_cli_parses_agent_model_and_runtime_options():
    args = parse_args([
        "--agent-type", "smart",
        "--model", "test-model",
        "--base-url", "http://localhost:11434/v1",
        "--rom-path", "zork1.z5",
        "--max-steps", "123",
        "--target-score", "80",
        "--episodes", "3",
        "--historian-model", "review-model",
        "--historian-temperature", "0.15",
        "--historian-max-tokens", "700",
        "--map-path", "tmp/map.json",
        "--historian-path", "tmp/historian.json",
        "--output-dir", "tmp/logs",
        "--trial-name", "baseline A",
        "--history-length", "7",
        "--temperature", "0.4",
        "--max-tokens", "450",
        "--no-openwebui-parent-id-null",
    ])

    assert args.agent_type == "smart"
    assert args.model == "test-model"
    assert args.base_url == "http://localhost:11434/v1"
    assert args.rom_path == "zork1.z5"
    assert args.max_steps == 123
    assert args.target_score == 80
    assert args.episodes == 3
    assert args.historian_model == "review-model"
    assert args.historian_temperature == 0.15
    assert args.historian_max_tokens == 700
    assert args.map_path == "tmp/map.json"
    assert args.historian_path == "tmp/historian.json"
    assert args.output_dir == "tmp/logs"
    assert args.trial_name == "baseline A"
    assert args.history_length == 7
    assert args.temperature == 0.4
    assert args.max_tokens == 450
    assert args.openwebui_parent_id_null is False


def test_cli_rejects_nonpositive_episode_count():
    with pytest.raises(SystemExit):
        parse_args(["--episodes", "0"])


def test_cli_supports_num_episodes_alias_and_review_disable():
    args = parse_args(["--num-episodes", "4", "--no-historian-review"])
    assert args.episodes == 4
    assert args.historian_review is False


def test_trial_logger_writes_console_chat_and_safe_metadata(tmp_path):
    logger = TrialLogger(tmp_path, "test trial")
    logger.write_metadata({"model": "m", "api_key": "secret"})

    with logger.episode(1) as episode:
        print("A visible episode output")
        episode.log_chat_event("request", {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        })
        episode.log_chat_event("response", {"content": "world"})

    output = episode.output_path.read_text(encoding="utf-8")
    chat = episode.chat_path.read_text(encoding="utf-8")
    metadata = (logger.run_directory / "trial_metadata.txt").read_text(encoding="utf-8")

    assert "A visible episode output" in output
    assert "REQUEST" in chat
    assert '"content": "hello"' in chat
    assert "RESPONSE" in chat
    assert "secret" not in metadata
    assert "<redacted>" not in metadata


def test_trial_logger_uses_unique_folder_for_same_trial_name(tmp_path):
    first = TrialLogger(tmp_path, "repeat")
    second = TrialLogger(tmp_path, "repeat")
    assert first.run_directory != second.run_directory
    assert first.run_directory.exists()
    assert second.run_directory.exists()


def test_episode_logger_writes_historian_review_and_trial_lessons(tmp_path):
    logger = TrialLogger(tmp_path, "review-test")
    review = {
        "report": "The Actor found a useful clue.\n\nIt should avoid repeating blocked actions.",
        "review_source": "llm",
        "lessons": {
            "positives": ["Inspect unfamiliar objects.", "Use score deltas.", "Verify routes."],
            "negatives": ["Do not repeat failures.", "Do not assume doors are open.", "Do not wander without a goal."],
        },
    }
    with logger.episode(1) as episode:
        episode.write_historian_review(review)

    assert "The Actor found a useful clue" in episode.review_path.read_text(encoding="utf-8")
    assert "Inspect unfamiliar objects" in episode.lessons_snapshot_path.read_text(encoding="utf-8")
    assert "Do not repeat failures" in (logger.run_directory / "lessons_learned.txt").read_text(encoding="utf-8")
    assert "Historian episode review" in episode.output_path.read_text(encoding="utf-8")
    assert "HISTORIAN_REVIEW_AND_LESSONS_SAVED" in episode.chat_path.read_text(encoding="utf-8")


def test_historian_reviews_episode_and_updates_next_prompt_lessons(tmp_path):
    answer = json.dumps({
        "report": (
            "The Actor explored several new rooms and used score feedback to identify "
            "a productive interaction. It made progress toward the score target.\n\n"
            "It should have changed strategy sooner after repeated no-progress responses. "
            "The next episode should use clues and map evidence more deliberately."
        ),
        "positives": [
            "Use score increases to identify actions worth pursuing.",
            "Inspect newly discovered rooms before leaving them.",
            "Record successful room transitions for later navigation.",
        ],
        "negatives": [
            "Do not repeat commands after identical no-progress responses.",
            "Do not assume historical routes are currently open.",
            "Do not carry an unverified puzzle assumption into later decisions.",
        ],
    })

    class FakeMessage:
        content = answer

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            self.request = kwargs
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("A", "Living Room", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="Living Room", action="look",
        after_node="A", after_location="Living Room", score_before=0,
        score_after=2, observation_before="before", observation="after",
        goal={"kind": "explore", "description": "Inspect the room."},
        action_source="smart_actor",
        actor_report={"state_note": "Found a potentially useful object."},
    )
    finished = historian.finish_episode(final_score=2, moves=1, termination_reason="step_limit")
    events = []
    review = historian.review_episode(
        finished,
        client=FakeClient(),
        model="review-model",
        event_logger=lambda event, payload: events.append((event, payload)),
    )

    assert review["review_source"] == "llm"
    assert len(review["positives"]) == 3
    assert len(review["negatives"]) == 3
    assert finished["historian_review"] == review["report"]
    assert [event for event, _ in events] == [
        "historian_review_request", "historian_review_response", "historian_review_result"
    ]
    guidance = historian.build_guidance("A", "Living Room", current_score=2)
    assert "LESSONS LEARNED FROM PREVIOUS EPISODES" in guidance
    assert "Use score increases to identify actions worth pursuing." in guidance


def test_historian_uses_three_by_three_fallback_without_review_client(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    historian.start_episode("A", "Room A", initial_score=0)
    historian.observe_step(
        before_node="A", before_location="Room A", action="north",
        after_node="A", after_location="Room A", score_before=0,
        score_after=0, observation_before="same", observation="same",
    )
    finished = historian.finish_episode(final_score=0, moves=1, termination_reason="step_limit")

    review = historian.review_episode(
        finished, client=None, model="unused-model"
    )
    assert review["review_source"] == "deterministic_fallback"
    assert len(review["positives"]) == 3
    assert len(review["negatives"]) == 3
    assert len(review["report"].split("\n\n")) == 2



def test_cli_configures_historian_lessons_per_category():
    args = parse_args(["--episodes", "4", "--historian-lessons-per-category", "5"])
    assert args.historian_lessons_per_category == 5

    alias_args = parse_args(["--historian-lessons", "7"])
    assert alias_args.historian_lessons_per_category == 7

    with pytest.raises(SystemExit):
        parse_args(["--historian-lessons-per-category", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--historian-lessons-per-category", "11"])


def test_historian_generates_configured_number_of_lessons_and_updates_prompt(tmp_path):
    lesson_count = 5
    answer = json.dumps({
        "report": (
            "The Actor made useful progress by inspecting objects and following score feedback. "
            "It should continue grounding actions in observed responses.\n\n"
            "It also repeated an ineffective action and should change strategy sooner when "
            "the environment does not change."
        ),
        "positives": [f"Positive lesson {index}" for index in range(1, lesson_count + 1)],
        "negatives": [f"Negative lesson {index}" for index in range(1, lesson_count + 1)],
    })

    class FakeMessage:
        content = answer

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        request = None

        def create(self, **kwargs):
            self.request = kwargs
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    historian = Historian(
        tmp_path / "historian.json",
        lessons_per_category=lesson_count,
    )
    historian.start_episode("A", "Living Room", initial_score=0)
    finished = historian.finish_episode(
        final_score=0, moves=0, termination_reason="step_limit"
    )
    review = historian.review_episode(
        finished, client=FakeClient(), model="review-model"
    )

    assert review["review_source"] == "llm"
    assert review["lessons_per_category"] == lesson_count
    assert len(review["lessons"]["positives"]) == lesson_count
    assert len(review["lessons"]["negatives"]) == lesson_count
    assert "exactly 5" in FakeChat.completions.request["messages"][1]["content"]

    next_prompt_guidance = historian.build_guidance("A", "Living Room", current_score=0)
    for lesson in review["lessons"]["positives"] + review["lessons"]["negatives"]:
        assert lesson in next_prompt_guidance


def test_historian_fallback_respects_configured_lesson_count(tmp_path):
    lesson_count = 7
    historian = Historian(
        tmp_path / "historian.json",
        lessons_per_category=lesson_count,
    )
    historian.start_episode("A", "Room A", initial_score=0)
    finished = historian.finish_episode(
        final_score=0, moves=1, termination_reason="step_limit"
    )

    review = historian.review_episode(
        finished, client=None, model="unused-model"
    )

    assert review["review_source"] == "deterministic_fallback"
    assert len(review["lessons"]["positives"]) == lesson_count
    assert len(review["lessons"]["negatives"]) == lesson_count
    assert len(set(review["lessons"]["positives"])) == lesson_count
    assert len(set(review["lessons"]["negatives"])) == lesson_count

def test_smart_agent_logs_full_request_and_response_and_resets_history():
    events = []

    class FakeMessage:
        content = json.dumps({
            "action": "look",
            "inventory_items": [],
            "carried_treasures": [],
            "puzzle_clues": [],
            "state_note": "Checking the room.",
        })

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    agent = SmartAgent(
        client=FakeClient(),
        model="unit-test-model",
        temperature=0.6,
        max_tokens=123,
        event_logger=lambda event, payload: events.append((event, payload)),
    )
    agent.add_history("old observation", "north", "Forest")

    decision = agent.choose_decision(
        observation="You are in a room.",
        current_location="Living Room",
        valid_actions=["look", "north"],
    )

    assert decision["action"] == "look"
    assert [event for event, _ in events] == ["request","response","actor_decision_rationale"]
    rationale_event = events[2][1]
    assert rationale_event["rationale"] == decision["decision_rationale"]
    assert rationale_event["rationale_source"] == "model_emitted_json_field"
    assert rationale_event["base_action"] == "look"
    request = events[0][1]
    assert request["model"] == "unit-test-model"
    assert request["temperature"] == 0.6
    assert request["max_tokens"] == 123
    assert "CURRENT OBSERVATION" in request["messages"][1]["content"]
    assert events[1][1]["content"] == FakeMessage.content

    agent.reset_history()
    assert agent.history == []


def test_episode_logger_writes_actor_experience_dataset(tmp_path):
    logger = TrialLogger(tmp_path, "actor-rl-log-test")
    sample = {
        "record_type": "actor_experience",
        "episode_id": "trial/episode_0001",
        "step": 3,
        "action": "examine door",
        "discounted_return": 4.5,
    }
    with logger.episode(1) as episode:
        path = episode.write_actor_experiences([sample])
        episode_log = episode.chat_path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == sample
    assert "ACTOR_EXPERIENCE_DATASET_SAVED" in episode_log.read_text(encoding="utf-8")
