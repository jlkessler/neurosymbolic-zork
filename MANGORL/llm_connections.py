"""Independent credentials and transport settings for the Historian role."""
from __future__ import annotations

import os


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
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_body = ({"parent_id": None} if args.historian_openwebui_parent_id_null
                  and base_url.rstrip("/").endswith("/api") else None)
    return client, extra_body
