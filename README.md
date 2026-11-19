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

For direct Anthropic billing, use these Historian settings in your local `.env`.
Keep your OpenRouter key in `ACTOR_API_KEY` and use
`ACTOR_BASE_URL=https://openrouter.ai/api/v1` for the Actor.

```dotenv
HISTORIAN_BASE_URL=https://api.anthropic.com
HISTORIAN_API_KEY=your-anthropic-api-key
HISTORIAN_MODEL=claude-sonnet-5-5
```

The Anthropic host selects the native Messages API for both reviews and prompt
evolution. Direct Claude requests use the model's sampling defaults.
Before starting a trial, direct Anthropic connections verify the configured
model with the Models API. A missing model or access error stops the run before
any game episodes. Shell environment values take precedence over the local
`.env`; check both if the reported model differs from your intended setting.

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
weights and token tensors are not modified. The default selects the largest
posterior probability. `--sample-actions` enables sampling using `--seed`.

The Actor gets current room and inventory text through snapshot queries that
restore the emulator state, plus the remaining command budget. These queries
are part of this runner's privileged Jericho access and should be disclosed in
benchmark comparisons. Learned strategy augments the base strategy.

Malformed, truncated, or out-of-menu Actor responses receive up to two repair
requests. If repair fails, the scorer evaluates a menu of up to five available
commands, prioritizing the current goal's preferred commands. This recovery
provides no inventory report. Native-logprob mode still requires actual model
probabilities. Chat logs record the full responses, rejection reasons,
available commands, recovery menus, and the final action selection.

When a label is missing from the API's top-20 list, native scoring retries with
equal positive token bias on every menu label (20, then 80). Equal additive bias
preserves pairwise logit differences when the provider honors it. `tiktoken`
verifies the model's exact label IDs; unknown tokenizers are never guessed.
Each retry must return the entire distribution in one response. Raw token
diagnostics and the applied bias are logged. Missing scores are never replaced
by invented values or a silent rank prior. See the
[Chat Completions parameter reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Historian reviews and configuration evolution each start with 4096 output
tokens, configured separately with `--historian-max-tokens` and
`--evolution-max-tokens`. Invalid output receives one repair attempt;
truncation doubles that attempt's budget. Both attempts log their stop reason.
If they still fail, previous lessons/configurations survive.

The runner learns full room-title aliases from snapshot observations while
retaining Jericho room IDs. It temporarily excludes repeated movement cycles
when alternatives exist, and revisits stalled return goals. Current inventory
grounds the Historian's item classifications. A historical score-positive
command must be reassessed by the Actor against the present state. Observation
guards preserve the sole active carried light and prioritize restoring it in
observed darkness; these are heuristics, not a guarantee of survival.

Automatic symbolic navigation now requires a home-deposit goal and treasures
confirmed in the current inventory. Historical acquisition/deposit rewards
remain Actor advice; they cannot trigger an empty-handed trip. Active goals
are retained while eligible, with bounded stagnation detection that also
applies to planned actions. Cancellation never deletes the learned map.

Malformed HTTP JSON and transient transport errors retry the same request up
to three times, with 60-second request timeouts and SDK retries disabled.
The Actor has a shared ceiling of 12 actual calls per decision, including
proposal repairs, label recovery, and NSI re-selection. Exhausted recoverable
errors select an explicitly logged, guarded observation/light command when
one is available. `--max-recovery-actions 5` caps these actions per episode
(allowed range 1-10); exhaustion ends an incomplete episode. Recovery commands
never receive fabricated logits or enter Actor policy samples. Observed
transitions still train the external critic. Episodes using recovery actions
are flagged `degraded` and excluded from benchmark scores/configuration fitness.
`--fail-fast` disables default-action recovery. Three consecutive propagated
recoverable errors stop the trial (`--max-consecutive-episode-failures`).
Authentication, endpoint, unsupported-scoring, and unexpected code errors stop
immediately. This does not guarantee remote-service availability.

`episode_outcomes.jsonl` and `run_status.json` record every attempted episode,
including failures. `metrics.jsonl` retains actual episode numbers and excludes
incomplete episodes; inspect the completion count alongside score averages.
Interrupted experience is bootstrapped at the last recorded state when the
critic is enabled, otherwise excluded from return-based learning. Existing
interrupted records without a valid bootstrap remain on disk but are not
retrieved for training.

New trials write an atomic `checkpoint.json`. Resume with
`python main.py --resume-trial /absolute/path/to/trial --episodes TOTAL`.
Resume restores the original configuration and memory paths, using current
credentials; TOTAL can extend the planned number of episodes. It starts at
the next unused episode identity, not the exact interrupted emulator move.
A process-killed partial episode is marked interrupted and never replayed;
already persisted learning is retained, while unflushed state may be lost.
Concurrent writers to the same checkpointed trial are rejected. Older trials
without checkpoints cannot be resumed by this option; their learned memories
remain usable through the existing five memory-path arguments. Do not use
`--fresh-memory` when intending to reuse learned routes.

`control_metrics` chat events report unauthorized navigation (must stay zero),
goal cancellations, route starts/restarts, observed treasure deposits, recovery
actions, and provider retries. Incomplete/degraded episodes remain visible in
`run_status.json`, even though they are excluded from score averages.

## Controlled comparisons

```bash
python experiments.py --dry-run --seeds 0 1 2
python experiments.py --seeds 0 1 2 --episodes 50 --max-steps 110 -- --model YOUR_MODEL
python experiments.py --seeds 0 1 2 --episodes 50 --max-steps 60 -- --model YOUR_MODEL
```

The harness isolates all memories for every condition and seed. It compares
fixed-policy control, evolution, actor-critic, and the hybrid sequentially.
Written Historian reviews are enabled in both `hybrid` and `hybrid_no_map`.
The fixed-policy, evolution-only, and actor-critic conditions keep reviews off.
It stops on an error instead of hiding a failed run. The fixed-policy control
still maintains observed map and Historian bookkeeping.

Run the paired hybrid conditions on the same seeds and budget to measure the
effect of the map and navigator.

```bash
python experiments.py --conditions hybrid hybrid_no_map --seeds 0 1 2 --episodes 10 --max-steps 500 -- --model YOUR_MODEL
```

For a single run, pass `python main.py --no-map-navigation --fresh-memory`.
The switch skips loading, updating, and saving the map, removes map and route
sections from the Actor prompt, and disables autonomous route execution and
map-based NSI feasibility. The Actor still chooses movement commands from
Jericho's available actions. The Historian retains observations, score evidence,
inventory, and cycle detection from actual play. Other settings apply as configured.
Use fresh memory for both conditions to avoid carrying lessons from one condition
into the other. Compare each trial's mean score and rate of reaching the
harness's 350-point target in `metrics_summary.json`. The condition is recorded as `map_navigation` in the
trial metadata and per-episode metrics.

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
# Seed a map from the reference walkthrough

Run `python main.py --walkthrough` in your Jericho environment to replay the
ROM's built-in reference walkthrough without calling either LLM. This saves to
`memory/world_model.json`; choose another destination with
`--walkthrough-map-path PATH`. Existing map nodes and edges are preserved.
The replay runs independently of `--max-steps`, `--target-score`, and `--episodes`.

The JSON contains the observed room transitions plus the full ordered command
sequence and executed steps under `graph.walkthrough`, including puzzle and
inventory commands that do not change rooms. This is Jericho's reference solution,
not a verified shortest path. Interrupted replays save their observed progress.
To use the seeded map in an agent run:
`python main.py --map-path memory/world_model.json`.
