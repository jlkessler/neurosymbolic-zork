"""Bounded transport recovery, separate from model-output repair and scoring."""
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar


class MalformedProviderResponse(RuntimeError):
    pass


class ProviderBudgetExhausted(RuntimeError):
    pass


_budget = ContextVar('provider_request_budget', default=None)


@contextmanager
def request_budget(limit=12):
    """Shared ceiling across proposal repairs, scoring retries, and NSI retry."""
    token = _budget.set([limit])
    try:
        yield
    finally:
        _budget.reset(token)


def transient_error(exc):
    if isinstance(exc, (MalformedProviderResponse, ProviderBudgetExhausted)):
        return True
    if type(exc).__module__.split('.')[0] not in {'openai', 'anthropic'}:
        return False
    status = getattr(exc, 'status_code', None)
    return (isinstance(status, int) and (status in {408, 409, 429} or 500 <= status < 600)
            or any(c.__name__ in {'APIConnectionError', 'APITimeoutError'} for c in type(exc).__mro__))


def provider_request(client, request, *, native=False, event_logger=None, attempts=3):
    # Disable SDK retries so this layer owns the transport budget. Output repair
    # and equal-bias scoring retain their own small, finite budgets.
    configured = client.with_options(max_retries=0, timeout=60.0) if hasattr(client, 'with_options') else client
    create = configured.messages.create if native else configured.chat.completions.create
    for attempt in range(1, attempts + 1):
        budget = _budget.get()
        if budget is not None:
            if budget[0] <= 0:
                raise ProviderBudgetExhausted('Per-decision provider request budget exhausted.')
            budget[0] -= 1
        try:
            return create(**request)
        except Exception as exc:
            # Catch decoding only inside the SDK call, never around our own
            # memory/configuration readers or model-output validation.
            error = MalformedProviderResponse('Provider returned an undecodable HTTP JSON response.') if isinstance(exc, json.JSONDecodeError) else exc
            retry = transient_error(error) and attempt < attempts
            if event_logger:
                response = getattr(exc, 'response', None)
                headers = getattr(response, 'headers', {})
                event_logger('provider_request_failure', {
                    'model': request.get('model'), 'transport': 'anthropic' if native else 'chat',
                    'attempt': attempt, 'retrying': retry, 'error_type': type(error).__name__,
                    'status_code': getattr(response, 'status_code', None),
                    'request_id': headers.get('x-request-id'),
                    'content_type': headers.get('content-type'),
                })
            if not retry:
                if error is not exc:
                    raise error from exc
                raise
            time.sleep(min(2 ** (attempt - 1), 4))
