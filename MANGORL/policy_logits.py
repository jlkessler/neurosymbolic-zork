"""Obtain real model log probabilities for a finite command menu.

Each command maps to one exact token A..E. We condition on those tokens,
not on all possible natural-language command strings. Missing top-k entries
are unknown, never fabricated as zero probability or reconstructed from ranks.
"""
from __future__ import annotations

import math


class LogitsUnavailable(RuntimeError):
    pass


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


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
        if extra_body is not None:
            request["extra_body"] = extra_body
        if event_logger:
            event_logger("actor_logprob_request", request)
        try:
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            content = _field(_field(choice, "logprobs"), "content", []) or []
            if not content:
                raise ValueError("Endpoint returned no token log probabilities.")
            first = content[0]
            entries = [*(_field(first, "top_logprobs", []) or []), first]
            scores = {}
            for entry in entries:
                token = _field(entry, "token")
                value = _field(entry, "logprob")
                if token in labels and isinstance(value, (int, float)):
                    # -9999 is the API's unavailable/very-unlikely sentinel.
                    if math.isfinite(value) and -9999 < value <= 0:
                        scores[token] = float(value)
            missing = set(labels) - scores.keys()
            if missing:
                raise ValueError(f"Missing exact single-token labels: {sorted(missing)}")
        except Exception as exc:
            raise LogitsUnavailable(
                "Native label logprobs are unavailable. Use a compatible endpoint, or explicitly "
                "select --actor-policy-mode rank for the rank-prior ablation. " + str(exc)
            ) from exc
        result = {"actions": list(actions), "base_logits": [scores[x] for x in labels],
                  "labels": labels, "source": "model_label_logprobs"}
        if event_logger:
            event_logger("actor_logprob_response", result)
        return result
