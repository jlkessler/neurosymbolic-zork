"""Classify operational failures without hiding configuration or code errors."""
from __future__ import annotations

from .policy_logits import IncompleteLabelScores


def recoverable_episode_error(exc):
    """SDK retries have already run; only transient errors can skip an episode.

    Scoring wraps upstream errors, so inspect explicit causes. Never classify
    failures from their human-readable message (e.g. a 404 mentioning a retry).
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, IncompleteLabelScores):
            return True
        # These are the SDKs used for the Actor and Historian respectively.
        # Local Python timeouts and arbitrary status_code attributes are not
        # sufficient evidence to suppress a potential programming error.
        if type(exc).__module__.split(".")[0] in {"openai", "anthropic"}:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and (status in {408, 409, 429} or 500 <= status < 600):
                return True
            if any(cls.__name__ in {"APIConnectionError", "APITimeoutError"}
                   for cls in type(exc).__mro__):
                return True
        exc = exc.__cause__
    return False
