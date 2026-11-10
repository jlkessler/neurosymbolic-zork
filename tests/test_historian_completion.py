import json
from types import SimpleNamespace as NS

import pytest

from MANGORL.evolution import default_configuration, propose_configuration
from MANGORL.historian import Historian
from MANGORL.llm_connections import AnthropicHistorian, HistorianResponseError


def native_client(responses):
    calls = []
    pending = iter(responses)

    def create(**kwargs):
        calls.append(kwargs)
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        text, reason = response
        return NS(content=[NS(type="text", text=text)] if text else [], stop_reason=reason)

    return AnthropicHistorian(NS(messages=NS(create=create))), calls


def review_json():
    return json.dumps({
        "report": "Repeated transitions did not increase the score. Try an untested exit.",
        "positives": ["Retain verified transitions.", "Check score feedback.", "Inspect unknown objects."],
        "negatives": ["Avoid repeated loops.", "Do not discard useful tools.", "Do not invent clues."],
    })


def test_native_review_recovers_empty_truncation_and_logs_stop_reason(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    client, calls = native_client([("", "max_tokens"), (review_json(), "end_turn")])
    events = []
    episode = {}
    result = historian.review_episode(
        episode, client=client, model="claude-test", max_tokens=4096,
        event_logger=lambda *event: events.append(event),
    )
    assert [call["max_tokens"] for call in calls] == [4096, 8192]
    assert result["lessons"] == historian.trial_lessons
    assert episode["review_source"] == "llm"
    responses = [payload for name, payload in events if name == "historian_review_response"]
    assert [response["finish_reason"] for response in responses] == ["max_tokens", "end_turn"]
    assert responses[0]["content"] == ""
    assert "truncated" in calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("bad", [
    "{\"report\":",
    json.dumps({"report": ["not a string"], "positives": [], "negatives": []}),
    json.dumps({"report": "A review", "positives": [1, 2, 3], "negatives": []}),
])
def test_review_retries_malformed_artifact_without_fabricating_lessons(tmp_path, bad):
    historian = Historian(tmp_path / "historian.json")
    client, calls = native_client([(bad, "end_turn"), (review_json(), "end_turn")])
    result = historian.review_episode({}, client=client, model="claude-test", max_tokens=4096)
    assert [call["max_tokens"] for call in calls] == [4096, 4096]
    assert result["report"] == json.loads(review_json())["report"]


def test_exhausted_review_retry_preserves_previous_lessons_and_complete_error(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    previous = {"positives": ["Retain prior evidence"], "negatives": ["Retain prior warning"]}
    historian.trial_lessons = previous.copy()
    # Even syntactically complete JSON is not accepted after a truncation stop.
    client, calls = native_client([(review_json(), "max_tokens"), ("unfinished", "max_tokens")])
    episode = {}
    with pytest.raises(HistorianResponseError, match="after 2 attempts.*truncated"):
        historian.review_episode(episode, client=client, model="claude-test")
    assert len(calls) == 2
    assert historian.trial_lessons == previous
    assert "historian_review" not in episode
    assert episode["historian_review_error"]["raw_response"] == "unfinished"
    assert episode["historian_review_error"]["finish_reason"] == "max_tokens"
    assert episode["historian_review_error"]["fallback_used"] is False


def test_authentication_or_model_failure_is_not_retried_as_bad_json(tmp_path):
    historian = Historian(tmp_path / "historian.json")
    client, calls = native_client([RuntimeError("404 model not found")])
    with pytest.raises(RuntimeError, match="404 model not found"):
        historian.review_episode({}, client=client, model="claude-test")
    assert len(calls) == 1


def test_evolution_uses_its_own_budget_and_logs_truncation():
    parent = default_configuration()
    child = {**parent, "history_length": 12}
    answer = json.dumps({"configuration": child, "rationale": "The recent room cycle needs more context."})
    client, calls = native_client([('{"configuration":', "max_tokens"), (answer, "end_turn")])
    events = []
    result, rationale = propose_configuration(
        client=client, model="claude-test", parent=parent, episode={}, max_tokens=5000,
        event_logger=lambda *event: events.append(event),
    )
    assert [call["max_tokens"] for call in calls] == [5000, 10000]
    assert result == child
    assert rationale == json.loads(answer)["rationale"]
    responses = [payload for name, payload in events if name == "configuration_evolution_response"]
    assert [response["finish_reason"] for response in responses] == ["max_tokens", "end_turn"]
    assert parent["history_length"] == 10


def test_invalid_evolution_never_mutates_parent():
    parent = default_configuration()
    original = json.dumps(parent, sort_keys=True)
    bad = json.dumps({"configuration": {**parent, "actor_beta": 999}, "rationale": "bad"})
    client, calls = native_client([(bad, "end_turn"), (bad, "end_turn")])
    with pytest.raises(HistorianResponseError, match="actor_beta"):
        propose_configuration(client=client, model="claude-test", parent=parent, episode={})
    assert len(calls) == 2
    assert json.dumps(parent, sort_keys=True) == original
