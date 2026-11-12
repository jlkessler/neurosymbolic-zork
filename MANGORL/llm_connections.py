"""Independent credentials and transport settings for the Historian role."""
from __future__ import annotations

import os
from urllib.parse import urlsplit

from .provider_requests import provider_request
from .policy_logits import _provider_error


class AnthropicHistorian:
    def __init__(self, client):
        self.client = client

    def complete(self, *, model, messages, max_tokens, event_logger=None):
        system = "\n\n".join(msg["content"] for msg in messages if msg["role"] == "system")
        conversation = [msg for msg in messages if msg["role"] != "system"]
        response = provider_request(self.client, dict(model=model, system=system, messages=conversation, max_tokens=max_tokens), native=True, event_logger=event_logger)
        content = "".join(block.text for block in response.content if block.type == "text")
        return content, response.stop_reason

    def validate_model(self, model):
        try:
            self.client.models.retrieve(model)
        except Exception as exc:
            raise RuntimeError(
                f"Historian startup check failed for Anthropic model {model!r} "
                f"at https://api.anthropic.com/v1/models. "
                f"Verify HISTORIAN_MODEL and access with HISTORIAN_API_KEY. "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def complete_historian(client, request, event_logger=None):
    native = isinstance(client, AnthropicHistorian)
    try:
        if native:
            # Recent Claude models reject temperature. Use native sampling defaults.
            return client.complete(
                model=request["model"], messages=request["messages"], max_tokens=request["max_tokens"], event_logger=event_logger,
            )
        choice = provider_request(client, request, event_logger=event_logger).choices[0]
        return choice.message.content or "", getattr(choice, "finish_reason", None)
    except Exception as exc:
        endpoint = urlsplit(str(getattr(client, "base_url", "")))
        target = ("https://api.anthropic.com/v1/messages" if native else
                  f"{endpoint.scheme}://{endpoint.hostname}{endpoint.path.rstrip('/')}/chat/completions")
        raise RuntimeError(
            f"Historian model={request['model']!r}, endpoint={target}. "
            f"{type(exc).__name__}: {exc} {_provider_error(exc)}"
        ) from exc


class HistorianResponseError(ValueError):
    """A bounded set of completed calls did not produce a valid artifact."""

    def __init__(self, message, *, raw_response, finish_reason, attempts):
        super().__init__(message)
        self.raw_response = raw_response
        self.finish_reason = finish_reason
        self.attempts = attempts


def complete_historian_validated(client, request, *, validate, event_prefix, event_logger=None):
    """Validate before publishing; retry a malformed/truncated artifact once.

    Truncated text is never accepted, even when it happens to parse as JSON.
    The retry restarts generation with the original evidence and a compact
    correction. Only a token-limit stop doubles the initial token budget.
    Transport failures retain the SDK's own retry behavior and fail directly.
    """
    current = {**request, "messages": list(request["messages"])}
    for attempt in (1, 2):
        if event_logger:
            event_logger(f"{event_prefix}_request", {**current, "attempt": attempt})
        raw, finish_reason = complete_historian(client, current, event_logger=event_logger)
        if event_logger:
            event_logger(f"{event_prefix}_response", {
                "model": current["model"], "content": raw,
                "finish_reason": finish_reason, "attempt": attempt,
                "max_tokens": current["max_tokens"],
            })
        truncated = finish_reason in {"length", "max_tokens"}
        try:
            if truncated:
                raise ValueError(f"Response was truncated (stop reason: {finish_reason}).")
            return validate(raw), raw, finish_reason
        except (ValueError, TypeError, KeyError) as exc:
            if event_logger:
                event_logger(f"{event_prefix}_response_rejected", {
                    "attempt": attempt, "error": str(exc),
                    "finish_reason": finish_reason, "retrying": attempt == 1,
                })
            if attempt == 2:
                raise HistorianResponseError(
                    f"Historian response invalid after {attempt} attempts: {exc}",
                    raw_response=raw, finish_reason=finish_reason, attempts=attempt,
                ) from exc
            if truncated:
                current["max_tokens"] = 2 * int(request["max_tokens"])
            current["messages"] = [*request["messages"], {
                "role": "user", "content": (
                    "The previous generation was rejected: " + str(exc)[:500] +
                    " Regenerate the complete required JSON object from the evidence above. "
                    "Be concise, include every required field, and omit commentary or markdown."
                ),
            }]


def create_historian_connection(args):
    """Never send Actor credentials to a Historian endpoint implicitly.

    Both reviews and configuration evolution use this connection. A keyless
    local service can be configured with an explicit placeholder API key.
    """
    base_url = (args.historian_base_url or "").strip()
    api_key = args.historian_api_key or os.getenv("HISTORIAN_API_KEY")
    if not base_url or not api_key or not api_key.strip():
        raise RuntimeError(
            "Historian LLM calls require HISTORIAN_BASE_URL and HISTORIAN_API_KEY "
            "(or --historian-base-url and --historian-api-key). Actor credentials "
            "are separate. For a keyless local service, set an explicit placeholder key. "
            "Use --no-historian-review --no-evolve to disable Historian LLM calls."
        )
    endpoint = urlsplit(base_url)
    if endpoint.scheme == "https" and endpoint.hostname == "api.anthropic.com":
        if not args.historian_model:
            raise RuntimeError("Set HISTORIAN_MODEL to a direct Claude model ID.")
        from anthropic import Anthropic
        return AnthropicHistorian(Anthropic(api_key=api_key, base_url="https://api.anthropic.com")), None

    if endpoint.path.rstrip("/").endswith(("/messages", "/responses", "/chat/completions")):
        raise RuntimeError(
            "HISTORIAN_BASE_URL must be an API base, not an operation URL. "
            "For direct Anthropic use https://api.anthropic.com; "
            "for OpenRouter use https://openrouter.ai/api/v1."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_body = ({"parent_id": None} if args.historian_openwebui_parent_id_null
                  and base_url.rstrip("/").endswith("/api") else None)
    return client, extra_body
