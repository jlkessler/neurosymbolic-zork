"""Obtain real model log probabilities for a finite command menu.

Each command maps to one exact token A..E. We condition on those tokens,
not on all possible natural-language command strings. Missing top-k entries
are unknown, never fabricated as zero probability or reconstructed from ranks.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from urllib.parse import urlsplit


class LogitsUnavailable(RuntimeError):
    pass


class IncompleteLabelScores(LogitsUnavailable):
    """A supported scorer still censored a menu label after bounded recovery."""


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _provider_error(exc):
    body = getattr(exc, "body", None)
    error = _field(body, "error", body)
    metadata = _field(error, "metadata")
    raw = _field(metadata, "raw")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            pass
    upstream = _field(raw, "error", raw)
    detail = raw if isinstance(raw, str) else _field(upstream, "message")
    provider = _field(metadata, "provider_name")
    param = _field(upstream, "param") or _field(error, "param")
    return " ".join(
        f"{label}: {value[:500]}"
        for label, value in (("Provider", provider), ("Parameter", param), ("Detail", detail))
        if isinstance(value, str) and value
    )


@lru_cache(maxsize=16)
def _label_token_ids(model):
    """Resolve IDs from a known tokenizer, never guess IDs for another model.

    OpenRouter's ``openai/`` prefix is a routing namespace. Arbitrary provider
    names and unknown OpenAI models must not inherit a convenient tokenizer.
    This is lazy: complete native distributions need no recovery dependency.
    """
    name = model.removeprefix("openai/")
    if "/" in name:
        raise ValueError(f"No verified label tokenizer for {model!r}.")
    try:
        import tiktoken
    except ImportError as exc:
        raise ValueError(
            "Missing-label recovery needs tiktoken; install requirements.txt."
        ) from exc
    try:
        encoding = tiktoken.encoding_for_model(name)
    except KeyError as exc:
        raise ValueError(f"No verified label tokenizer for {model!r}.") from exc
    result = {}
    for label in "ABCDE":
        tokens = encoding.encode(label, disallowed_special=())
        if len(tokens) != 1 or encoding.decode_single_token_bytes(tokens[0]) != label.encode():
            raise ValueError(f"Label {label!r} is not one exact token for {model!r}.")
        result[label] = str(tokens[0])
    return result


def prepare_model(model):
    """Warm the optional recovery tokenizer before spending on an episode.

    Unknown providers may still return complete label scores without recovery.
    Known OpenAI models must have the declared dependency installed; report
    that setup error now instead of waiting for a rare censored label.
    """
    try:
        ids = _label_token_ids(model)
    except Exception as exc:
        detail = str(exc)
        known_model = re.fullmatch(
            r"(?:openai/)?(?:gpt-4o(?:-mini)?|gpt-4\.1(?:-mini|-nano)?|"
            r"gpt-4(?:-turbo)?|gpt-3\.5-turbo)(?:-\d{4}(?:-\d{2}-\d{2})?)?", model,
        )
        unsupported = isinstance(exc, ValueError) and detail.startswith("No verified label tokenizer")
        missing_optional = (isinstance(exc, ValueError) and "needs tiktoken" in detail and not known_model)
        if unsupported or missing_optional:
            return {"model": model, "recovery_available": False, "reason": detail}
        raise LogitsUnavailable(f"Cannot prepare label-logprob recovery for {model}: {detail}") from exc
    return {"model": model, "recovery_available": True, "label_token_ids": dict(ids)}


def _read_scores(response, labels, *, attempt, bias, event_logger):
    choices = _field(response, "choices", []) or []
    choice = choices[0] if choices else None
    content = _field(_field(choice, "logprobs"), "content", []) or []
    first = content[0] if content else None
    entries = [*(_field(first, "top_logprobs", []) or []), first] if first else []
    scores = {}
    for entry in entries:
        token = _field(entry, "token")
        value = _field(entry, "logprob")
        if token in labels and isinstance(value, (int, float)) and not isinstance(value, bool):
            # -9999 is the API's unavailable/very-unlikely sentinel.
            if math.isfinite(value) and -9999 < value <= 0:
                scores[token] = float(value)
    missing = sorted(set(labels) - scores.keys())
    if event_logger:
        event_logger("actor_logprob_attempt", {
            "attempt": attempt, "uniform_label_bias": bias,
            "response_id": _field(response, "id"), "model": _field(response, "model"),
            "finish_reason": _field(choice, "finish_reason"),
            "missing_labels": missing,
            "tokens": [{"token": _field(entry, "token"),
                        "logprob": _field(entry, "logprob"),
                        "bytes": _field(entry, "bytes")} for entry in entries],
        })
    if not content:
        raise ValueError("Endpoint returned no token log probabilities.")
    if _field(choice, "finish_reason") == "content_filter":
        raise ValueError("Endpoint filtered the scoring response.")
    return scores, missing


class LabelLogprobScorer:
    def score(self, *, client, model, messages, actions, extra_body=None, event_logger=None):
        if not 1 <= len(actions) <= 5 or len(set(actions)) != len(actions):
            raise ValueError("The logprob selector requires 1-5 distinct commands.")
        labels = list("ABCDE"[:len(actions)])
        menu = "\n".join(f"{label}: {action}" for label, action in zip(labels, actions))
        request = {
            "model": model,
            "messages": [*messages, {"role": "user", "content": (
                "Select the best next command from this menu. Output exactly one uppercase "
                "letter token, with no whitespace, explanation, or JSON.\n" + menu
            )}],
            "temperature": 1.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
        }
        if model in {"gpt-6-luna", "openai/gpt-6-luna"}:
            request["reasoning_effort"] = "none"
            del request["max_tokens"]
            request["max_completion_tokens"] = 16

        body = dict(extra_body or {})
        # OpenAI's SDK merges extra_body over the regular arguments. A hidden
        # override here could disable scores, truncate the distribution, or
        # change the prompt/model while recovery still uses the original IDs.
        controlled = {
            "model", "models", "messages", "temperature", "top_p", "top_k", "min_p",
            "frequency_penalty", "presence_penalty", "repetition_penalty",
            "logit_bias", "logprobs", "top_logprobs", "max_tokens", "max_completion_tokens",
            "reasoning_effort", "stop", "n", "stream", "response_format", "tools", "tool_choice",
        }
        overrides = sorted(controlled.intersection(body))
        if overrides:
            raise LogitsUnavailable(
                f"Do not override Actor scoring parameters through extra_body: {overrides}. "
                "Native label scoring controls its model, prompt, sampling, and equal label bias."
            )
        if urlsplit(str(getattr(client, "base_url", ""))).hostname == "openrouter.ai":
            # Require a route that can honor the scoring parameters.
            body["provider"] = {**(body.get("provider") or {}), "require_parameters": True}
        if body or extra_body is not None:
            request["extra_body"] = body
        try:
            # top_logprobs is a censored top-k list, not a score for every menu
            # item. Promote ALL labels equally when one falls out of that list.
            # For any two labels i,j, (z_i+b)-(z_j+b)=z_i-z_j, so conditioning
            # on the menu gives the same policy when the provider honors bias.
            # Never merge different responses or replace unknown scores by 0.
            for attempt, bias in enumerate((0, 20, 80), start=1):
                attempt_request = dict(request)
                if bias:
                    try:
                        token_ids = _label_token_ids(model)
                    except Exception as exc:
                        raise ValueError(
                            f"Missing exact single-token labels: {missing}. {exc}"
                        ) from exc
                    attempt_request["logit_bias"] = {token_ids[label]: bias for label in labels}
                if event_logger:
                    event_logger("actor_logprob_request", attempt_request)
                response = client.chat.completions.create(**attempt_request)
                scores, missing = _read_scores(
                    response, labels, attempt=attempt, bias=bias, event_logger=event_logger,
                )
                if not missing:
                    break
            else:
                raise IncompleteLabelScores(
                    f"Missing exact single-token labels: {missing} after {attempt} attempts. "
                    "The endpoint did not return a complete distribution even with equal "
                    "label bias; inspect actor_logprob_attempt events for raw tokens."
                )
        except IncompleteLabelScores:
            raise
        except Exception as exc:
            detail = _provider_error(exc)
            raise LogitsUnavailable(
                f"Native label logprobs are unavailable for {model}. "
                f"{exc}" + (f" {detail}" if detail else "")
            ) from exc
        result = {"actions": list(actions), "base_logits": [scores[x] for x in labels],
                  "labels": labels, "source": "model_label_logprobs",
                  "scoring_attempts": attempt, "uniform_label_bias": bias}
        if event_logger:
            event_logger("actor_logprob_response", result)
        return result
