# Zork I score-seeking neuro-symbolic agent

This project is a research prototype combining:

- `SmartAgent`: LLM reasoning, puzzle-solving, exploration, and compact state reporting.
- `NavigationController`: deterministic execution and validation of known graph routes.
- `Historian`: score feedback, inventory/treasure reports, episode lessons, goal selection.
- `WorldMap`: persistent Jericho room-object graph.

## Objective

The default target is **76 points**, which means the agent must beat 75. Configure with:

```bash
export ZORK_TARGET_SCORE=76
export ZORK_MAX_STEPS=500
```

The agent stops once Jericho reports a score at or above the configured target.

## Information separation

The attached Zork guide was used to understand the overall research objective and validate the score-seeking premise. Its walkthrough, treasure table, point values, and specific puzzle solutions are **not included in runtime files or prompts**.

The Historian does not load the guide. It learns score-producing actions from Jericho's observed score deltas. The Actor sees the target score and broad strategic heuristics, but must infer useful actions and puzzle solutions from game observations.

## How the components interact

1. Jericho returns an observation, current room identity, score, and available actions.
2. The Historian chooses a broad goal:
   - meet the score threshold;
   - return with Actor-reported likely valuables;
   - deposit at the Living Room when there is a known route;
   - solve the return route if no known route exists;
   - revisit an observed score-positive action;
   - explore an unmapped exit or investigate.
3. The Navigation Controller treats persistent paths as candidates. It issues one command at a time and verifies each transition before advancing; a failed or unavailable edge is blocked for the rest of that episode.
4. Otherwise the SmartAgent selects one action and reports inventory candidates and newly observed puzzle clues. When all known return paths are blocked, the Actor investigates prerequisites and alternate routes.
5. Jericho's score delta and room transition update the Historian and WorldMap.

The map's Dijkstra route is action-count minimizing; the Historian does **not** use travel length to discount a score goal. Long exploration is allowed.

## Install and run

```bash
pip install -r requirements.txt
```

The project reads configuration from a project-local `.env` file using `python-dotenv`. Edit `.env` before running and replace `PASTE_YOUR_OPEN_WEBUI_API_KEY_HERE` with your actual API key. The included `.gitignore` excludes `.env` so the real key is not accidentally committed. `.env.example` is a safe template to share.

Configuration priority is: **command-line arguments > shell environment variables > `.env` values > application defaults**. Existing shell variables are not overwritten by `.env`.

The included `.env` is preformatted for your Open WebUI endpoint and current defaults (`gemma4:31b`, target score 350, 500 steps, 10 episodes). Historian credentials are optional: leave `HISTORIAN_API_KEY` and `HISTORIAN_BASE_URL` blank to reuse the Actor connection, or fill them in to use a separate compatible service. `HISTORIAN_MODEL` selects the review model and defaults to the Actor model if unset.

```bash
python main.py \
  --agent-type smart \
  --model gemma4:31b \
  --base-url http://cs-steverw.umflint.edu:8080/api \
  --rom-path z-machine-games-master/jericho-game-suite/zork1.z5 \
  --target-score 76 \
  --max-steps 500 \
  --episodes 3 \
  --historian-review \
  --historian-model gemma4:31b \
  --historian-temperature 0.2 \
  --historian-max-tokens 900 \
  --map-path memory/world_map.json \
  --historian-path memory/historian.json \
  --output-dir logs \
  --trial-name baseline
```

The equivalent settings can also be supplied through environment variables (`ZORK_ROM_PATH`, `OPENAI_BASE_URL`, `LLM_MODEL`, `ZORK_TARGET_SCORE`, `ZORK_MAX_STEPS`, `ZORK_EPISODES`, `ZORK_MAP_PATH`, `ZORK_HISTORIAN_PATH`, `OPENAI_API_KEY`, `HISTORIAN_API_KEY`, `HISTORIAN_BASE_URL`, and `HISTORIAN_MODEL`). Use `python main.py --help` to see every option, including temperature, response token limit, Actor history length, Historian review settings, and the Open WebUI `parent_id` compatibility switch.

`--episodes 3` (also accepted as `--num-episodes 3`) runs three fresh games in one trial folder. Its default can also be set with `ZORK_EPISODES`. Episode reviews are enabled by default; use `--no-historian-review` to disable them. The Historian uses `--historian-model` when supplied, otherwise it uses the Actor's `--model`. Review sampling temperature and token budget are controlled independently by `--historian-temperature` and `--historian-max-tokens`.

`OPENWEBUI_PARENT_ID_NULL=1` (or `--openwebui-parent-id-null`) sends `parent_id: null` for the Open WebUI `/api` endpoint workaround discussed in the previous conversation. Use `--no-openwebui-parent-id-null` for endpoints that do not accept that field.

### Trial and episode logs

Every invocation creates a unique trial folder beneath `--output-dir` (default: `logs/`). For example:

```text
logs/
└── baseline_20261102_031300/
    ├── trial_metadata.txt
    ├── episode_0001.txt
    ├── episode_0001_chat.txt
    ├── episode_0001_review.txt
    ├── episode_0001_lessons.txt
    ├── episode_0002.txt
    ├── episode_0002_chat.txt
    ├── episode_0002_review.txt
    ├── episode_0002_lessons.txt
    └── lessons_learned.txt
```

- `episode_NNNN.txt` captures the terminal output and errors for that episode, including observations, actions, score changes, and navigation status. Output is also echoed to the terminal.
- `episode_NNNN_chat.txt` records each full Actor and Historian system/user prompt, raw model response, and request/fallback errors. It is useful for auditing what context the Actor actually received and how it responded, as well as what the Historian considered during its review.
- `episode_NNNN_review.txt` contains the Historian's 1-2 paragraph post-episode assessment of Actor performance, including advice and critiques. The Historian uses runtime observations, score changes, recorded actions, and previous trial lessons; it is not given the Zork guide or walkthrough.
- `episode_NNNN_lessons.txt` snapshots the configured number of positive behaviors to repeat and negative behaviors to avoid after that episode.
- `lessons_learned.txt` is overwritten after each review with the latest configured set of positive and negative trial lessons. At the beginning of the next episode, these lessons are inserted into the Historian guidance included in the Actor's prompt. Current game feedback takes precedence over an old lesson when the evidence warrants it.
- `trial_metadata.txt` records the non-secret run configuration. API keys are deliberately excluded.

`--episodes` runs multiple fresh games in one trial. The WorldMap and Historian's persistent observations carry across those episodes, while the Actor's short-term history and navigation controller's episode-local route evidence are reset for each new game. The Historian also retains a separate coaching set within the current trial; each completed episode reviews and refines that set before the next episode begins. Configure the number with `--historian-lessons-per-category N` (also accepted as `--historian-lessons N` or `--historian-lesson-count N`). This creates N positive lessons and N negative lessons, for 2N total. The default is 3 per category; valid values are 1 through 10. The setting can also be supplied as `ZORK_HISTORIAN_LESSONS_PER_CATEGORY` in `.env`; the default `.env` and `.env.example` set it to 3. The first episode starts without trial-specific lessons. Each episode receives its own console, Actor/Historian chat, review, and lesson snapshot files. Use `--trial-name` to label the folder; a timestamp and collision-safe suffix keep runs separate.

For example, to ask for five positive and five negative lessons after every episode:

```bash
python main.py --episodes 10 --historian-review --historian-lessons-per-category 5
```

The count is part of `trial_metadata.txt`, included in each review result, and reflected in the episode and current-trial lesson files. It changes only the number of coaching items; it does not change the Historian's value-function critic, which remains independently controlled by `--historian-critic`.

The Actor reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from `.env`. The Historian review uses `HISTORIAN_API_KEY` and `HISTORIAN_BASE_URL` when provided; otherwise it reuses the Actor's connection. The Historian model is read from `HISTORIAN_MODEL`. Avoid passing API keys directly with `--api-key` or `--historian-api-key` on shared machines because command-line arguments can appear in shell history or process listings.

## Tests

```bash
python -m pytest -v
```

Tests use synthetic graphs and a mocked LLM client. They do not contact the model server or run a live Zork episode.

## Persistent memory

- `memory/world_map.json`: historically observed directed room transitions, with Jericho room IDs to disambiguate repeated room names. Failed attempts do not delete historical edges.
- `memory/historian.json`: observed score events, repeated no-progress actions, episode summaries, and Actor-reported clues.
- `NavigationController.confirmed_edges` and `blocked_edges`: transient episode-local evidence. A successful traversal confirms that exact directed edge; a failed or unavailable transition blocks it until the next episode. Both sets reset each episode while the persistent graph remains intact.

Back up these files before experimenting with new schemas or reset them intentionally for a fresh condition.

## Research cautions

- Actor-reported `carried_treasures` are uncertain estimates, not a trusted inventory oracle.
- A historical map route is not proof that a door or passage is currently open. `plan_to_home()` may probe a historical candidate one transition at a time, but immediately abandons a failed/unavailable edge. Once every known candidate route is blocked, the Actor must infer prerequisites or discover an alternate route.
- Score deltas are the strongest runtime signal. Reward and textual clues are retained as supporting evidence.
- A score target does not imply the game has been completed. It is the experiment's stopping criterion.

## Actor reinforcement-learning prototype

The current experimental Actor-learning path is enabled by default. It keeps the
LLM weights frozen and stores completed Actor decisions in a separate,
append-only JSONL file:

- `memory/actor_experiences.jsonl`: one record for each Actor-selected action,
  including the pre-action state, action candidates, score/environment reward,
  next state, and the episode's discounted return.
- `episode_NNNN_experience.jsonl` in the trial output folder: a per-episode
  snapshot of the exact Actor training samples, including calculated returns.
- All environment transitions are used while calculating returns. This means
  score gained after one or more symbolic-navigation commands can still be
  attributed to the earlier Actor decision. Symbolic-navigation commands are
  not themselves stored as Actor policy decisions.
- The state representation includes the Jericho room ID/name, observation,
  inventory report, puzzle clues, Historian goal, home-route status, and score.
  Experience retrieval compares these fields to avoid relying on room name
  alone.
- `SmartAgent` is asked for its preferred action and up to four alternatives.
  `ActorRLController` combines that candidate order with action advantages
  estimated from similar historical states and discounted returns.

**Important implementation distinction:** the current OpenAI-compatible API
path does not expose native model logits to this controller. Thus the live
remote-API implementation performs a candidate-rank adjustment, not true
LLM-token-logit adjustment. `ActorRLController.adjust_logits()` is provided as
a separate interface for a future local inference backend that can supply
genuine candidate logits. The experiment logs the exact candidate ranking,
experience estimates, and adjusted selection for later analysis.

Initial settings can be changed on the command line:

```bash
python main.py \
  --actor-rl \
  --actor-experience-path memory/actor_experiences.jsonl \
  --actor-rl-gamma 0.995 \
  --actor-rl-beta 0.35 \
  --actor-rl-exploration 0.08 \
  --actor-rl-top-k 40
```

Use `--no-actor-rl` to disable action adaptation. Experience collection still
runs so the same trial can collect data for a later learning condition. Use a
separate `--actor-experience-path` for independent experimental conditions.
The default score reward is the Jericho score delta. Jericho's separate reward
is logged but has zero contribution by default because it may overlap with
score. Optional environment reward weighting and a no-progress penalty are
available, but both default to zero to keep the initial reward definition
simple.

### Optional Historian value-function critic

`historian_rl_controller.py` is a first, intentionally conservative critic
prototype. It learns a sparse linear state-value function from completed
episode Monte-Carlo returns. Features include room identity, goal type, route
status, inventory terms, puzzle-clue terms, and observation/goal words. When
enabled, this value estimate can replace the experience-memory average as the
baseline used to form Actor action advantages.

The critic is **off by default**. Its weights are stored separately in
`memory/historian_value.json`; it is not the same as the Historian's textual
post-episode review or its configurable coaching lessons. To enable the numerical
critic in a future experiment:

```bash
python main.py --historian-critic
```

For the initial Actor-learning / frozen-critic condition, do not pass
`--historian-critic`. To also freeze the Historian's generated cross-episode
prompt lessons, use `--no-historian-review`. Note that this first critic integration
is a baseline-style approximation, not a full action-conditioned actor-critic
algorithm: V(s) is the same state baseline for all candidate actions, and its
influence on their relative ordering can be limited when their confidence values
are similar. A stronger next step is to estimate action-specific TD advantages
from stored transitions, `r + gamma * V(s') - V(s)`, and aggregate those by
candidate action. The Historian's ordinary runtime
bookkeeping (observed score events, repeated failures, and map-informed goals)
continues as part of the existing neuro-symbolic baseline; disabling the
review only disables its LLM-generated episode critiques and lesson updates.

### Interpretation and limitations

This is an initial, online, small-data action-value estimator—not a trained
neural Q-network and not an unbiased off-policy estimator. It uses weighted
similarity retrieval and empirical discounted returns. Candidate rank is only
a proxy for the frozen LLM's actual probability distribution. In particular,
the Actor is not yet requesting or consuming OpenAI `logprobs`, and the
reported candidate ordering should not be described as native logits in
research results. The next methodological improvement would be to obtain
real candidate-selection log probabilities from a supported endpoint or run
the model through a local inference backend, then use `adjust_logits()`.

The first experiments should compare the frozen baseline, prompt/retrieval
only, and candidate-rank-plus-advantage conditions while holding Historian
review settings constant. Use a fresh experience file for each condition if
you want clean ablations. Inspect `episode_NNNN_chat.txt` for the Actor's
candidate list and `actor_rl_selection` event, and inspect
`memory/actor_experiences.jsonl` for the completed training samples.
