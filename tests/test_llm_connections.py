import json
import sys
from types import SimpleNamespace as NS

import pytest

from main import parse_args, build_arg_parser
from MANGORL.llm_connections import AnthropicHistorian, complete_historian, create_historian_connection
from MANGORL.trial_logger import TrialLogger


def test_native_model_check_fails_before_trial_starts(monkeypatch, tmp_path):
    import main

    def missing(model):
        assert model == "unavailable-model"
        raise RuntimeError("404 model not found")

    client = AnthropicHistorian(NS(models=NS(retrieve=missing)))
    monkeypatch.setattr(main, "create_historian_connection", lambda args: (client, None))
    with pytest.raises(RuntimeError, match="Historian startup check.*unavailable-model"):
        main.main(["--historian-model", "unavailable-model", "--output-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_historian_error_identifies_transport_model_and_operation():
    def fail(**kwargs):
        raise RuntimeError("404")

    client = AnthropicHistorian(NS(messages=NS(create=fail)))
    with pytest.raises(RuntimeError, match="model='claude-test'.*anthropic.com/v1/messages.*404"):
        complete_historian(client, {"model": "claude-test", "messages": [], "max_tokens": 100})


def test_native_historian_uses_anthropic_messages(monkeypatch):
    monkeypatch.setenv("ACTOR_API_KEY", "router-key")
    monkeypatch.setenv("HISTORIAN_API_KEY", "claude-key")
    constructor_calls, message_calls = [], []

    def create(**kwargs):
        message_calls.append(kwargs)
        return NS(content=[NS(type="thinking"), NS(type="text", text="{}")],
                  stop_reason="end_turn")

    def anthropic(**kwargs):
        constructor_calls.append(kwargs)
        return NS(messages=NS(create=create))

    monkeypatch.setitem(sys.modules, "anthropic", NS(Anthropic=anthropic))
    args = parse_args(["--historian-base-url", "https://api.anthropic.com/v1/messages",
                       "--historian-model", "claude-sonnet-5-5"])
    client, extra_body = create_historian_connection(args)
    conversation = [{"role": "user", "content": "Review this episode."}]
    assert complete_historian(client, {
        "model": args.historian_model,
        "messages": [{"role": "system", "content": "Return JSON."}, *conversation],
        "max_tokens": 900, "temperature": 0.2,
    }) == ("{}", "end_turn")
    assert constructor_calls == [{"api_key": "claude-key", "base_url": "https://api.anthropic.com"}]
    assert extra_body is None
    assert message_calls == [{"model": "claude-sonnet-5-5", "system": "Return JSON.",
                              "messages": conversation, "max_tokens": 900}]


def test_historian_uses_only_its_own_credentials_and_endpoint(monkeypatch):
    monkeypatch.setenv("ACTOR_API_KEY", "actor-secret")
    monkeypatch.setenv("HISTORIAN_API_KEY", "historian-secret")
    args = parse_args(["--actor-base-url", "https://actor.example/v1",
                       "--historian-base-url", "https://historian.example/api",
                       "--historian-openwebui-parent-id-null"])
    calls = []
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: calls.append(kwargs) or object())
    _, body = create_historian_connection(args)
    assert calls == [{"api_key": "historian-secret", "base_url": "https://historian.example/api"}]
    assert body == {"parent_id": None}


def test_historian_cli_overrides_env_and_does_not_inherit_actor_options(monkeypatch):
    monkeypatch.setenv("HISTORIAN_API_KEY", "env-secret")
    calls = []
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: calls.append(kwargs) or object())
    args = parse_args(["--historian-api-key", "cli-secret", "--historian-base-url", "https://history.example/v1",
                       "--openwebui-parent-id-null", "--no-historian-openwebui-parent-id-null"])
    _, body = create_historian_connection(args)
    assert calls[0]["api_key"] == "cli-secret"
    assert body is None


@pytest.mark.parametrize("missing", ["key", "endpoint"])
def test_missing_historian_settings_never_fall_back_to_actor(monkeypatch, missing):
    monkeypatch.setenv("ACTOR_API_KEY", "actor-secret")
    monkeypatch.setenv("HISTORIAN_API_KEY", "" if missing == "key" else "history-secret")
    args = parse_args([])
    args.historian_base_url = None if missing == "endpoint" else "https://history.example/v1"
    with pytest.raises(RuntimeError, match="HISTORIAN_BASE_URL and HISTORIAN_API_KEY"):
        create_historian_connection(args)


def test_role_keys_are_absent_from_help_repr_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HISTORIAN_API_KEY", "history-secret")
    args = parse_args(["--actor-api-key", "actor-secret", "--historian-api-key", "history-secret"])
    logger = TrialLogger(tmp_path, "credentials")
    metadata = logger.write_metadata(vars(args)).read_text()
    for secret in ("actor-secret", "history-secret"):
        assert secret not in metadata
        assert secret not in repr(args)
        assert secret not in build_arg_parser().format_help()


def test_evolution_uses_dedicated_historian_client(monkeypatch, tmp_path):
    from MANGORL.adaptive_system import AdaptiveSystem
    from MANGORL.evolution import default_configuration
    args = parse_args(["--evolution-path", str(tmp_path / "archive.json")])
    actor_client, historian_client = object(), object()
    actor = NS(client=actor_client, extra_body={"actor_only": True})
    historian = NS(_review_payload=lambda episode: episode)
    system = AdaptiveSystem(args, actor=actor, historian=historian, critic=None,
                            policy=None, navigator=None, output_dir=tmp_path,
                            historian_client=historian_client, historian_extra_body={"historian_only": True})
    system.configuration_id, system.configuration = system.archive.select()
    calls = []
    def proposal(**kwargs):
        calls.append(kwargs)
        return default_configuration(), "Observed outcome"
    monkeypatch.setattr("MANGORL.adaptive_system.propose_configuration", proposal)
    system.finish_episode({}, info={"score": 1}, reason="step_limit", event_logger=lambda *args: None)
    assert calls[0]["client"] is historian_client
    assert calls[0]["extra_body"] == {"historian_only": True}
