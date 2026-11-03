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
- `episode_NNNN_lessons.txt` snapshots the three positive behaviors to repeat and three negative behaviors to avoid after that episode.
- `lessons_learned.txt` is overwritten after each review with the latest six trial lessons. At the beginning of the next episode, these lessons are inserted into the Historian guidance included in the Actor's prompt. Current game feedback takes precedence over an old lesson when the evidence warrants it.
- `trial_metadata.txt` records the non-secret run configuration. API keys are deliberately excluded.

`--episodes` runs multiple fresh games in one trial. The WorldMap and Historian's persistent observations carry across those episodes, while the Actor's short-term history and navigation controller's episode-local route evidence are reset for each new game. The Historian also retains a separate six-item coaching set within the current trial; each completed episode reviews and refines that set before the next episode begins. The first episode starts without trial-specific lessons. Each episode receives its own console, Actor/Historian chat, review, and lesson snapshot files. Use `--trial-name` to label the folder; a timestamp and collision-safe suffix keep runs separate.

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
