# MANGORL: evolving neuro-symbolic Zork agent

A frozen LLM Actor and Historian learn at deployment time through experience,
an external actor-critic loop, evolving prompt configurations, and a probabilistic
Cognitive Map. This is an experimental hybrid inspired by EvoTest, JitRL and
Defining Neurosymbolic AI. It has not established a state-of-the-art Zork score.

## What changes during a run

- **Actor policy:** actual model label log probabilities receive action-specific
  advantage corrections and are conditioned on symbolic feasibility. The model
  weights stay frozen. An explicitly labelled rank-prior ablation is available.
- **Historian:** an external sparse TD critic learns from every transition;
  experience and optional prose reviews improve future guidance. The Historian
  also proposes new Actor/Historian configurations after episodes.
- **Prompt structure:** strategy text, section order, history, memory retrieval,
  temperature, exploration and route caution evolve. A bounded UCB archive
  selects configurations using their measured episode scores.
- **Cognitive Map:** directed room transitions accumulate traversal evidence.
  NSI computes logic-times-belief integrals, while the navigator plans with
  route uncertainty and checks each executed transition.
- **Memory:** completed Actor returns and all critic transitions persist.
  Optional frozen neural embeddings support semantic retrieval; the default
  retrieval index is lexical and state-based.

Read [the architecture and equations](docs/architecture.md) and
[the paper mapping, framework comparison and evaluation protocol](docs/research.md).

## Install

Use Python 3.10+ and a Jericho-supported environment. Upstream Jericho loads
`libfrotz.so`; use Linux/WSL for game execution on a Windows host. The model
server can run elsewhere. Mocked tests and CLI help do not require Jericho.
[Jericho source/platform details](https://github.com/microsoft/jericho)

```bash
pip install -r requirements.txt
```

For a test-only installation without Jericho:

```bash
pip install pytest networkx python-dotenv openai
python -m pytest -q
```

Optional neural memory retrieval:

```bash
pip install -r requirements-semantic.txt
```

Copy `.env.example` to `.env` if setting up a new checkout. Existing `.env`
settings are preserved. Configuration precedence is CLI, shell environment,
project `.env`, then application defaults. The active Actor uses `ACTOR_MODEL`,
`ACTOR_BASE_URL` and `ACTOR_API_KEY`. The Historian has its own independent
connection. Configure both roles explicitly:

| Setting | Actor | Historian |
| --- | --- | --- |
| API key | `ACTOR_API_KEY` | `HISTORIAN_API_KEY` |
| Endpoint | `ACTOR_BASE_URL` | `HISTORIAN_BASE_URL` |
| Model | `ACTOR_MODEL` | `HISTORIAN_MODEL` |
| Calls | Game decisions and label scoring | Episode reviews and configuration evolution |

The Historian never inherits the Actor API key or endpoint. Set its endpoint
and key whenever reviews or evolution are enabled; missing settings produce a
startup error. For a local service without authentication, provide an explicit
placeholder key. To deliberately use the same provider for both roles, set
both sets of variables. No credentials belong in source control.

CLI equivalents are `--actor-api-key`, `--actor-base-url`, `--actor-model`,
and `--historian-api-key`, `--historian-base-url`, `--historian-model`.
The older `--api-key`, `--base-url`, and `--model` remain Actor-only aliases.
Prefer environment variables for keys so secrets do not enter shell history.
If `HISTORIAN_MODEL` is omitted, its model name defaults to the Actor model
name; the connection and credentials remain independent. The Historian's
Open WebUI compatibility flag is `--historian-openwebui-parent-id-null`
or `HISTORIAN_OPENWEBUI_PARENT_ID_NULL`.

## Run the hybrid

```bash
python main.py --fresh-memory --episodes 50 --max-steps 110 --target-score 350 --trial-name hybrid
```

The default enables Actor RL, the Historian TD critic, NSI, configuration evolution,
and prose reviews. `--fresh-memory` puts all new memories inside that trial's
directory and leaves existing `memory/` files intact. Omit it to continue using
the configured persistent memory paths. To run with frozen neural retrieval,
add `--memory-embedding-model MODEL_OR_LOCAL_PATH`.

**Endpoint requirement:** default `--actor-policy-mode logprobs` requires real
first-token `logprobs`/`top_logprobs` for all menu labels. It makes a second
Actor call to score the shortlist. If the endpoint omits any required label,
the run fails with `LogitsUnavailable`; it does not silently call ranks logits.
OpenAI-compatible API shape does not guarantee this capability. For an endpoint
without it, explicitly use `--actor-policy-mode rank` and report that ablation
as a rank prior, not native model-logit adaptation.

The selected policy is over up to five labelled commands, not every possible
command string or the model's full vocabulary. The remote model's internal
weights and token tensors are not modified. `--no-sample-actions` selects the
largest posterior probability; otherwise sampling uses `--seed`.

## Controlled comparisons

```bash
python experiments.py --dry-run --seeds 0 1 2
python experiments.py --seeds 0 1 2 --episodes 50 --max-steps 110 -- --model YOUR_MODEL
python experiments.py --seeds 0 1 2 --episodes 50 --max-steps 60 -- --model YOUR_MODEL
```

The harness isolates all memories for every condition and seed. It compares
fixed-policy control, evolution, actor-critic, and the hybrid sequentially.
Ordinary prose reviews are off in all four conditions to isolate the mechanisms.
It stops on an error instead of hiding a failed run. The fixed-policy control
still maintains observed map and Historian bookkeeping.

Useful switches: `--no-evolve`, `--no-actor-rl`, `--no-historian-critic`,
`--no-historian-review`, `--no-nsi`, `--actor-policy-mode rank`.
`--no-historian-review` does not disable the separate Evolver; use `--no-evolve`
as well to freeze configuration mutation. `--no-actor-rl` still collects
experience and can retain NSI. `python main.py --help` lists all settings.

Do not compare a 500-step peak score to a paper's 60-step average. The runner
uses Jericho room IDs, world-change hashes and valid-action generation, and
retains existing Zork-specific return/deposit heuristics. These must be matched
or disclosed in benchmark comparisons. No walkthrough is loaded.

## Files and logs

| File | Role |
| --- | --- |
| `main.py` | Environment loop and CLI |
| `MANGORL/smart_agent.py` | Active Actor, evolving prompt rendering, validated command proposals |
| `MANGORL/policy_logits.py` | Actual menu-label log probability acquisition |
| `MANGORL/actor_rl_controller.py` | Advantage correction, posterior and action sampling |
| `MANGORL/historian_rl_controller.py` | Online TD critic and action-conditioned transition memory |
| `MANGORL/actor_experience.py` | Delayed returns, persistence and state retrieval |
| `MANGORL/semantic_memory.py` | Optional frozen neural encoder |
| `MANGORL/nsi.py` | Weighted model counting and posterior conditioning |
| `MANGORL/cognitive_map.py` | Probabilistic extension of the persistent WorldMap |
| `MANGORL/navigation_controller.py` | Route planning, execution and verification |
| `MANGORL/evolution.py` | Validated configurations, archive and UCB |
| `MANGORL/adaptive_system.py` | Episode lifecycle, evolution and evaluation metrics |
| `MANGORL/historian.py` | Evidence, inventory categories, goals and optional prose reviews |
| `experiments.py` | Matched-budget ablation runner |

`MANGORL/actor.py` is a retained legacy implementation; the runner uses
`smart_agent.py`. Existing inventory categorization, goal sequences, review
lesson counts, route traces and explicit model rationale logging are preserved.
Missing rationales are recorded as missing. Failed reviews retain previous
lessons and never invent replacement text.

Each trial contains console/chat/decision/review logs, per-episode experience
JSONL, nonsecret metadata, `metrics.jsonl`, and `metrics_summary.json`. The summary
reports the score sequence, mean, final, best, target success rate, and normalized
AUC (`sum(scores) / (episodes * score_scale)`). Configuration snapshots and
mutations, native/adjusted probabilities, NSI mass, KL and TD errors are in chat
events. `api_requests_before_evolution` excludes the subsequent mutation call;
it is not a token/dollar-cost estimate.

Persistent defaults are `memory/world_map.json`, `memory/historian.json`,
`memory/actor_experiences.jsonl`, `memory/historian_td.json`, and
`memory/configurations.json`. Existing map and Actor-experience formats remain
readable. The former `historian_value.json` is not overwritten by default.
Do not share memory paths across independent ablations or simultaneous runs.

Tests cover the finite integral, masked policy and numerical stability, native
logprob failures, online credit updates, terminal versus truncated returns,
route risk, memory reloads, prompt mutation, parent retention and a mocked
two-episode integration. Live model/ROM performance remains an empirical test.
