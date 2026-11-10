"""Regression coverage for top-k censoring without rank-prior substitution."""
from copy import deepcopy
import json
import math
import sys
from types import SimpleNamespace as NS

import pytest

from MANGORL import policy_logits
from MANGORL.actor_experience import ActorExperienceMemory
from MANGORL.actor_rl_controller import ActorRLController
from MANGORL.nsi import softmax
from MANGORL.policy_logits import LabelLogprobScorer, LogitsUnavailable
from MANGORL.smart_agent import SmartAgent


def response(values, finish_reason="length"):
    entries = [{"token": key, "logprob": value, "bytes": list(key.encode())}
               for key, value in values.items()]
    return {"id": "score-test", "model": "gpt-4o-mini", "choices": [{
        "finish_reason": finish_reason,
        "logprobs": {"content": [{**entries[0], "top_logprobs": entries}]},
    }]}


def client_for(create):
    return NS(base_url="https://openrouter.ai/api/v1", chat=NS(completions=NS(create=create)))


@pytest.fixture
def known_ids(monkeypatch):
    ids = {label: str(32 + index) for index, label in enumerate("ABCDE")}
    monkeypatch.setattr(policy_logits, "_label_token_ids", lambda model: ids)
    return ids


def test_censored_labels_recovered_without_changing_conditional_policy(known_ids):
    logits = {"A": 0, "B": -5, "C": -40, "D": -60, "E": -20}
    logits.update({f"other{i}": -.2 for i in range(21)})
    requests, events = [], []

    def create(**request):
        requests.append(deepcopy(request))
        bias = request.get("logit_bias", {})
        biased = {token: value + bias.get(known_ids.get(token), 0) for token, value in logits.items()}
        log_z = max(biased.values()) + math.log(sum(math.exp(x - max(biased.values())) for x in biased.values()))
        top = sorted(biased, key=biased.get, reverse=True)[:20]
        return response({token: biased[token] - log_z for token in top})

    messages = [{"role": "user", "content": "The light is out; choose a safe command."}]
    result = LabelLogprobScorer().score(
        client=client_for(create), model="openai/gpt-4o-mini", messages=messages,
        actions=["light lantern", "north", "drop manual", "drop all", "look"],
        event_logger=lambda kind, payload: events.append((kind, payload)),
    )
    assert len(requests) == 3
    assert "logit_bias" not in requests[0]
    assert requests[1]["logit_bias"] == {token: 20 for token in known_ids.values()}
    assert requests[2]["logit_bias"] == {token: 80 for token in known_ids.values()}
    assert all(request["messages"] == requests[0]["messages"] for request in requests)
    assert all(request["extra_body"]["provider"]["require_parameters"] for request in requests)
    assert softmax(result["base_logits"]) == pytest.approx(softmax([logits[k] for k in "ABCDE"]))
    assert [value - result["base_logits"][0] for value in result["base_logits"]] == pytest.approx(
        [logits[k] - logits["A"] for k in "ABCDE"]
    )
    assert result["source"] == "model_label_logprobs"
    assert result["uniform_label_bias"] == 80
    attempts = [payload for kind, payload in events if kind == "actor_logprob_attempt"]
    assert "C" in attempts[0]["missing_labels"]
    assert attempts[-1]["missing_labels"] == []
    assert attempts[-1]["finish_reason"] == "length"
    assert attempts[0]["tokens"]


def test_complete_response_uses_one_call_without_resolving_tokenizer(monkeypatch):
    def unexpected(model):
        pytest.fail("A complete native distribution should not need a tokenizer.")
    monkeypatch.setattr(policy_logits, "_label_token_ids", unexpected)
    result = LabelLogprobScorer().score(
        client=client_for(lambda **_: response({"A": -.1, "B": -3})),
        model="other/provider", messages=[], actions=["look", "north"],
    )
    assert result["base_logits"] == [-.1, -3]
    assert result["scoring_attempts"] == 1


def test_recovery_is_bounded_if_provider_ignores_bias(known_ids):
    requests = []
    def create(**request):
        requests.append(request)
        return response({"A": -.1, " C": -4, "C": -9999})
    with pytest.raises(LogitsUnavailable, match="after 3 attempts"):
        LabelLogprobScorer().score(client=client_for(create), model="gpt-4o-mini",
                                 messages=[], actions=["look", "north", "south"])
    assert len(requests) == 3


def test_missing_logprob_support_does_not_retry_bias(known_ids):
    calls = []
    def create(**request):
        calls.append(request)
        return {"choices": [{"finish_reason": "stop", "logprobs": None}]}
    with pytest.raises(LogitsUnavailable, match="no token log probabilities"):
        LabelLogprobScorer().score(client=client_for(create), model="gpt-4o-mini",
                                 messages=[], actions=["look", "north"])
    assert len(calls) == 1


def test_recovery_does_not_combine_logits_from_different_requests(known_ids):
    replies = iter([response({"A": -.1}), response({"B": -.1}), response({"C": -.1})])
    with pytest.raises(LogitsUnavailable, match="after 3 attempts"):
        LabelLogprobScorer().score(client=client_for(lambda **_: next(replies)), model="gpt-4o-mini",
                                 messages=[], actions=["look", "north", "south"])


@pytest.mark.parametrize("override", [
    {"logit_bias": {"32": -100}}, {"top_logprobs": 1}, {"logprobs": False},
    {"temperature": .1}, {"messages": []}, {"models": ["another/model"]},
])
def test_scoring_parameters_cannot_be_overridden_by_extra_body(override):
    with pytest.raises(LogitsUnavailable, match="extra_body"):
        LabelLogprobScorer().score(client=None, model="gpt-4o-mini", messages=[],
                                 actions=["look"], extra_body=override)


def test_preparation_warms_known_ids(known_ids):
    assert policy_logits.prepare_model("openai/gpt-4o-mini") == {
        "model": "openai/gpt-4o-mini", "recovery_available": True, "label_token_ids": known_ids,
    }


def test_preparation_reports_unknown_model_without_guessing(monkeypatch):
    def unsupported(model):
        raise ValueError(f"No verified label tokenizer for {model!r}.")
    monkeypatch.setattr(policy_logits, "_label_token_ids", unsupported)
    status = policy_logits.prepare_model("unrecognized/model")
    assert status["recovery_available"] is False
    assert "No verified label tokenizer" in status["reason"]


def test_preparation_fails_early_for_missing_known_model_dependency(monkeypatch):
    def missing(model):
        raise ValueError("Missing-label recovery needs tiktoken; install requirements.txt.")
    monkeypatch.setattr(policy_logits, "_label_token_ids", missing)
    with pytest.raises(LogitsUnavailable, match="install requirements.txt"):
        policy_logits.prepare_model("openai/gpt-4o-mini")
    assert policy_logits.prepare_model("custom-model")["recovery_available"] is False


def test_episode_10_dark_cellar_menu_recovers_and_persists_native_selection(tmp_path, known_ids):
    # Replay the actual failing menu through proposal -> scoring -> RL/NSI ->
    # experience persistence. Missing C is simulated; its real raw score was
    # not recorded by the old run, so this does not invent measured run data.
    actions = ["turn on lantern", "put down lantern", "put down manual",
               "put down all", "throw lantern at manual"]
    requests, events = [], []
    def create(**request):
        requests.append(request)
        if not request.get("logprobs"):
            return NS(choices=[NS(finish_reason="stop", message=NS(content=json.dumps({
                "action": actions[0], "candidate_actions": actions,
                "decision_rationale": "The cellar is pitch black; restore the light.",
                "inventory_items": ["lantern", "manual"],
            })))])
        if "logit_bias" not in request:
            return response({"A": -.01, "B": -8, "D": -9, "E": -10})
        return response({"A": -.01, "B": -8, "C": -25, "D": -9, "E": -10})

    actor = SmartAgent(client=client_for(create), model="openai/gpt-4o-mini", policy_mode="logprobs",
                       event_logger=lambda kind, payload: events.append((kind, payload)))
    state = {"location": "Cellar", "location_id": "cellar", "observation": "It is pitch black.",
             "score": 39, "inventory_items": ["lantern", "manual"]}
    decision = actor.choose_decision(observation=state["observation"], current_location="Cellar",
                                     current_score=39, valid_actions=actions)
    memory = ActorExperienceMemory(tmp_path / "experience.jsonl")
    memory.start_episode("episode_0010_replay")
    selection = ActorRLController(memory).select_action(
        state, decision["candidate_actions"], base_action=decision["action"],
        base_logits=decision["base_logits"], feasibility=[1, 0, 1, 0, 0], sample=False,
    )
    memory.record_step(state=state, action=selection["action"],
                       next_state={**state, "observation": "The brass lantern is now on."},
                       score_delta=0, action_source="smart_actor", actor_decision=True,
                       candidate_actions=actions, selection_info=selection)
    memory.finish_episode(final_score=39, termination_reason="step_limit")
    stored = json.loads(memory.path.read_text(encoding="utf-8").strip())
    assert len(requests) == 3  # One proposal, censored scores, recovered scores.
    assert stored["action"] == "turn on lantern"
    assert stored["selection_info"]["method"] == "model_label_logits_plus_advantage_nsi"
    assert len(stored["selection_info"]["base_logits"]) == 5
    attempts = [payload for kind, payload in events if kind == "actor_logprob_attempt"]
    assert [attempt["missing_labels"] for attempt in attempts] == [["C"], []]


def test_tokenizer_verifies_each_label_and_normalizes_only_openai_prefix(monkeypatch):
    models = []
    def get_encoding(name):
        models.append(name)
        return NS(encode=lambda label, **_: [ord(label) - 33],
                  decode_single_token_bytes=lambda token: bytes([token + 33]))
    monkeypatch.setitem(sys.modules, "tiktoken", NS(encoding_for_model=get_encoding))
    policy_logits._label_token_ids.cache_clear()
    try:
        assert policy_logits._label_token_ids("openai/gpt-4o-mini") == {
            "A": "32", "B": "33", "C": "34", "D": "35", "E": "36",
        }
        assert models == ["gpt-4o-mini"]
        with pytest.raises(ValueError, match="No verified label tokenizer"):
            policy_logits._label_token_ids("another/gpt-4o-mini")
    finally:
        policy_logits._label_token_ids.cache_clear()


@pytest.mark.parametrize("bad_encoding", [
    NS(encode=lambda label, **_: [1, 2]),
    NS(encode=lambda label, **_: [1], decode_single_token_bytes=lambda _: b"wrong"),
])
def test_invalid_tokenizer_never_sends_guessed_bias(monkeypatch, bad_encoding):
    monkeypatch.setitem(sys.modules, "tiktoken", NS(encoding_for_model=lambda _: bad_encoding))
    policy_logits._label_token_ids.cache_clear()
    try:
        with pytest.raises(ValueError, match="not one exact token"):
            policy_logits._label_token_ids("gpt-4o-mini")
    finally:
        policy_logits._label_token_ids.cache_clear()
