**Experimentation evaluation: `tabula-rasa-5_big_20261111_172225`**

Prepared 12 November 2026. Objective: reliably obtain a Zork I score of at least 80.

The agent has learned useful early-game behavior and a real underground return route. Its main limitation is now the connection between observation, memory, goal selection, and exploration. It repeatedly recognizes a problem in prose without changing the decisions that reproduce it. A more elaborate geometric map would leave most of these failures intact.

The strongest evidence is episode 44. At step 195, the available action `southeast` would have discovered the Cyclops Room. The agent instead selected the familiar `south` exit. I reproduced this using the recorded commands and seed in the local ROM. Earlier in that same episode, the return route failed because the agent carried too much for the chimney; a separate replay confirmed that reducing its load made the route work. These are concrete policy and prerequisite problems.

**1. What was examined and what the experiment achieved**

The audit indexed and SHA-256 hashed all 310 supplied files, totaling 543,558,210 bytes. It parsed all 50 console and chat transcripts, all 49 finalized episode-experience files, the aggregate Actor experience memory, outcomes, metrics, configurations, and final memories. Historian reviews, lessons, selected rationale traces, and the implementation were examined to explain the quantitative patterns. The rationale files largely duplicate structured chat events; I do not treat model-emitted rationales as reliable accounts of hidden reasoning.

This was a programmatic full-trial audit with detailed inspection of diagnostic passages, not a claim to have manually read every repeated prompt word. The `.trial.lock` file was inventoried, not removed. The checkpoint ends with `next_episode=51` and `active_episode=null`. The current Windows and WSL copies of the 13 requested modules, `main.py`, `action_guards.py`, and `goal_control.py` match byte-for-byte. The trial does not record a complete source revision manifest, so matching today's checkouts cannot independently establish the exact source bytes used for every historical episode.

All original logs and memories were left unchanged. No paid model requests were made for this evaluation. The local replay observations are evaluator evidence, stored separately; they were not inserted into the agent's memories.

| Measure | Result | Interpretation |
|---|---:|---|
| Scheduled/attempted episode slots | 50/50 | The trial continued through the schedule. |
| Completed episodes | 44 | 43 ended on a death; one reached the step limit. “Completed” is operational, not task success. |
| Incomplete episodes | 6 | Five `no_safe_actions`; episode 33 was interrupted and skipped on resume. |
| Episodes reaching 80 | 0/50 | Neither completed nor partial traces show target attainment. |
| Best final score | 55 | Episodes 42 and 44. |
| Best observed score before death | 65 | Episodes 42 and 44 subsequently lost 10. |
| Mean of completed episodes | 35.34 | The existing summary excludes incomplete episodes. |
| Mean of all 49 recorded final scores | 33.16 | Episode 33 has no finalized score; it is not imputed as zero. |
| Executed commands visible in logs | 7,153 | Includes 91 commands from interrupted episode 33. |
| Actor / symbolic / recovery commands | 6,947 / 203 / 3 | Symbolic navigation accounts for only 2.84% of commands. |
| Persisted Actor records | 6,856 | Exactly equal to the concatenated 49 episode-experience files. |
| Persisted critic transitions | 7,062 | Actor, navigation, and recovery transitions from finalized episodes. |
| Final learned map | 28 rooms, 56 directed edges | Eight distinct Maze rooms and two distinct Dead Ends. |
| Successful Historian reviews / mutations | 49 / 44 | The earlier review/evolution transport failures are not the main problem in this trial. |
| Logged automatic navigation without treasure | 0 | The treasure gate appears to be working. |

The recorded-score means by episode block were 37.4, 32.6, 27.9, 27.78, and 39.6. The fourth block contains nine finalized scores because episode 33 was interrupted. This is a decline followed by partial recovery, not consistent improvement. These are sequential adaptive episodes with seed 0 and shared memory, not 50 independent evaluations of one fixed agent.

The agent discovered a narrow set of repeatable rewards: egg pickup, entry through the window, egg deposit, first Cellar entry, painting pickup/deposit, and occasional coin pickup. Positive painting deposits occurred nine times; positive coin pickups occurred five times; no positive coin deposit appears. The coin profile records six attempts, which is different from five reward-bearing pickups. Successful early actions are real progress, but repeating them cannot by itself expand the achievable score.

Across the console traces, 5,033 commands occurred after their episode's last positive score change: 70.36% of all logged commands. That is a stagnation indicator, not proof that every such command was useless; travel and puzzle preparation can be valuable without immediate reward. Here it aligns with extensive repeated movement and failure to discover new areas.

The 43 death endings classify from their final observations as 17 grue/darkness, 14 troll, five thief, six jumping, and one rusty-knife attack. The immediate command is not always the cause: a harmless item action can consume a turn during enemy combat. Deaths must therefore be represented as contextual transitions, not universal bans on whichever verb happened last.

Only episodes 1, 3, 5, 6, 42, and 44 actually entered Maze rooms, totaling 836 Maze commands. Repeated mentions of a distant Maze goal in prompts do not indicate visits. No Cyclops, Grating Room, or dam-region node exists in the final learned map.

**2. Global themes and their causes**

The first theme is a mismatch between instructions and available actions. `main.py:296` gets a Jericho valid-action list, and `SmartAgent._match_available` rejects anything outside it. Across 7,159 logged environment contexts, `look` and `inventory` never appear in that list; `wait` appears only 124 times. The audit also found no commands beginning with `examine`, `read`, or `say` in the raw menus. Some aliases such as `look around thief` do appear, so this does not mean every inspection or combat action is impossible. It does mean the prompts prescribe capabilities the action gate frequently prevents.

Jericho documents this list as a best-guess set of actions that affect game state. It is not an exhaustive grammar of meaningful commands or information-gathering actions. A command can reveal useful information without producing the state change required by the oracle's filter. [Jericho's valid-action documentation](https://github.com/microsoft/jericho/blob/master/docs/source/tutorial_quick.rst)

This is especially damaging for critical thinking: an Actor cannot read a clue, test an unsuccessful mechanism, or propose a meaningful utterance if the proposal is rejected solely for being absent from that menu. The same restriction prevents the recovery handler from using its preferred `look` or `inventory` defaults.

The second theme is an extremely concentrated policy. Among 6,856 persisted Actor decisions:

| Policy diagnostic | Result |
|---|---:|
| Median maximum base probability | 0.99999916 |
| Decisions with maximum base probability above 0.99 | 5,577 / 6,856, or 81.34% |
| Median gap between the top two label log probabilities | 14 nats |
| RL correction changes the base argmax | 46, or 0.67% |
| NSI changes the already-adjusted argmax | 2 |
| Combined RL+NSI changes the base argmax | 46 |
| Median KL divergence from base distribution | 8.89 × 10^-9 |
| Scorer prefers label A | 6,263 / 6,856, or 91.35% |
| Median candidate coverage of raw valid-action menu | 35.7% |

Label A is normally the Actor's proposed favorite, and the scoring request includes that proposal and its rationale with an instruction to reconsider only when warranted. The A preference is consistent with anchoring, but does not isolate positional bias; a shuffled-label experiment is needed. It would be incorrect to attribute all A selections to label bias.

Most evolved configurations use beta 0.3, while normalized advantages are clipped to ±2. Their direct advantage contribution is therefore at most ±0.6. Small exploration bonuses cannot ordinarily overcome a 14-nat preference gap under greedy selection. Increasing the exploration parameter without calibrating the prior or allocating actual exploration decisions leaves behavior nearly unchanged. Simply enabling sampling at the present concentration is unlikely to solve that problem either.

The third theme is that prose learning does not reliably alter operational decisions. Historian reviews repeatedly criticize stale coin goals, but `historian.py:1166` continues ranking historical score opportunities by average gain without a complete prerequisite or route-cost model. `AdaptiveSystem.begin_episode` installs an evolved Historian strategy, but that strategy is consumed by the review prompt, not the deterministic `choose_goal` scheduler. This creates a control mismatch: the reviewer says to stop pursuing an objective while the scheduler keeps proposing it.

At the end of the trial, the historical coin profile averages 50/6 = 8.33 points per recorded attempt, while Cellar `down` averages 1075/131 = 8.21. These aggregate averages describe past outcomes across incompatible states. They are not expected marginal rewards from the current state, and they do not account for reaching the room, a live troll, depleted light, or treasure already taken this episode.

The fourth theme is overgeneralization of danger. Learned strategies include bans on the Maze, north from the Cellar, or the skeleton key. These conflict with exploration goals and confuse neighboring objects or different conditions. The logs show successful `take key` actions followed, in some episodes, by the distinct and harmful `take skeleton` action. A blanket prohibition on the key removes an enabling tool because another object nearby was dangerous.

The fifth theme is corrupted state evidence. `inventory_from_text` in `action_guards.py:24` treats every nonempty line after “You are carrying:” as an item. Ambient enemy narration and even death/reincarnation text can consequently become inventory entries. Examples include episode 5's thief descriptions and episode 44's final death narrative. There are 163 logged inventory snapshot sections containing the audit's ambient/combat markers; this is a screening count, not a claim that all 163 contain exactly the same parsing defect.

The snapshot implementation restores emulator state, but `look` and `inventory` can emit time-dependent events during the temporary probe. Restoration does not make those emitted events true observations of the committed trajectory. Feeding them back as current possessions, damage, or objective progress can contaminate both learning and goal control.

The sixth theme is weak measurement of progress. Opening/closing a sack, rearranging objects, or incidental changes in room narration can reset stagnation counters. `GoalControl` clears suppression after broad changes to inventory, score, map size, or same-room text. That is too broad to protect a specific objective from repeated restarts. There are 334 finalized-episode goal cancellations and seven repeated route starts; the full event stream has 340 abandonment events because it also includes the interrupted episode. The logs demonstrate detection, but detection alone does not ensure the next action is different or useful.

**3. The most informative episode-level evidence**

Episode 44 separates successful navigation from unsuccessful planning particularly well:

| Displayed step | Observation | Implication |
|---|---|---|
| 31 | Symbolic `up` travels Studio → Kitchen carrying painting. | The return route is real and useful. |
| 109 | Coin pickup raises score to 65. | The agent reaches a new useful item. |
| 110–117 | Symbolic navigation carries the coins from the Maze back to the Studio. | The route executor can traverse a substantial learned route correctly. |
| 118 | `up` is absent from the oracle menu; the agent chooses `south`. | The route depends on state, not merely location. |
| 182 | The keys are taken. | The enabling tool is obtainable. |
| 187 | `take skeleton` curses and removes valuables; score stays 65. | A major loss can have zero immediate score penalty. |
| 195 | In Maze room 52, available moves include south, west, southeast; the Actor chooses south. | Exploration fails despite explicit alternatives. |
| 320 | Death reduces score to 55. | Returning safely and depositing remained unresolved. |

I replayed the recorded episode 44 actions using Jericho 3.3.1 and seed 0, with no LLM calls. Before step 118, a direct `up` probe returned “You can't get up there with what you're carrying.” In an isolated counterfactual branch, dropping the manual, sword, garlic, and lunch while retaining coins and lantern allowed `up` to reach the Kitchen. This demonstrates the missing load prerequisite; it is not a general recommendation to discard the sword whenever navigating. A real agent needs safe item staging, preservation, and retrieval plans.

Before step 195, the replay was in room 52 at score 65. `southeast` reached the Cyclops Room, while `west` reached another Maze room, number 53. The original policy assigned the known `south` a log probability near zero, versus approximately -11 for west and -10.25 for southeast; all three received the same +0.5 correction. The return to the known room was therefore almost inevitable.

Other useful cases are episode 11, where the RL correction changed the scorer's preferred `north` to `jump` and the agent died; episode 28, which spent 500 commands above ground and repeatedly removed/redeposited the egg; and episodes 8, 14, and 16, which repeatedly failed while trying to score a menu containing a straightforward egg deposit. These demonstrate respectively unsafe novelty incentives, reversible reward cycling, and a failure of score recovery unrelated to puzzle difficulty.

**4. Assessment of each requested module**

The table describes the current implementation and observed contribution. This single adaptive trial cannot establish causal benefit for a module; that requires ablation.

| Module | What helps | What limits performance | Recommended change |
|---|---|---|---|
| `smart_agent.py` | Produces structured decisions, inventories, candidate menus, repairable JSON, and explicit policy telemetry. Learned early-game sequences are often sensible. | Candidate menu is capped at five; scorer sees the favored proposal and rationale; hard oracle matching prevents useful investigative commands. Prompts are long and contain conflicting instructions. | Separate action proposal from neutral scoring; reserve candidate slots for a frontier or hypothesis test; permit grounded information actions; use a compact typed state and evidence-ranked memory. |
| `actor_rl_controller.py` | Corrects actual label scores without changing model weights; logs base, adjusted, and conditioned distributions; combines MC and TD evidence. | Changes only 0.67% of argmax decisions. Novelty can favor dangerous unseen actions. Historical returns and present advantage are mixed heuristically under a changing policy. | Calibrate prior strength separately from correction strength; add a bounded exploration policy over eligible safe tests; keep actual behavior probability and correction diagnostics; evaluate each component separately. |
| `policy_logits.py` | Requires exact labels and finite scores; never invents missing logits or merges responses; equal-bias recovery has a sound conditional-probability rationale when the endpoint honors it. | Eight scoring operations exhaust recovery and lead to five incomplete episodes. Returned scores barely change under bias in the repeated egg-deposit failures. Tokenizer readiness is not an endpoint behavior test. | Add a cached endpoint capability check and precise failure classification. Use bounded, explicitly marked recovery; verify any fallback scientifically. Do not turn missing scores into fabricated probabilities. |
| `adaptive_system.py` | Coordinates settings and excludes incomplete/degraded episodes from configuration fitness; logs actual configuration identities. | Settings, memory, and critic all change together. Repeated single-episode mutations are hard to compare. The existing average describes evaluated episodes only. | Separate development, configuration evaluation, and held-out testing. Freeze memory snapshots for paired parent/child evaluation; report all scheduled outcomes alongside conditional score summaries. |
| `world_map.py` | Directed multigraph, stable room IDs, aliases, and preserved paths solve real navigation problems. Identical Maze titles remain distinct. | Stores moved-room transitions; same-room effects and movement self-loops are lost. No persistent ledger of available-but-untried exits or conditional prerequisites. | Preserve topology and add frontier and action-effect records keyed by room, command, and relevant state. |
| `cognitive_map.py` | Adds empirical edge beliefs and episode-local confirmation/blocking. Gives the planner a way to express uncertainty. | All 56 final edges lack failure counts. Traversal frequency dominates confidence; temporary blockage and inventory dependence are not explained. Unknown valid actions receive support 1. | Condition evidence on relevant facts, distinguish unknown from confirmed, and store failed/unavailable probes with explanations. Treat alternative destinations as exclusive outcomes where appropriate. |
| `semantic_memory.py` | Provides an optional frozen encoder and cache without weight training. | It was disabled in this trial (`memory_embedding_model=null`), so it cannot explain success or failure here. Whole-state JSON embeddings would still risk conflating different prerequisites. | First fix structured state and retrieval filters; then compare lexical and frozen-embedding retrieval under matched conditions. Preserve exact room/item/negation facts alongside embeddings. |
| `nsi.py` | Implements a finite logic-times-belief sum and conjunction specialization; masks zero-feasibility actions and exposes satisfaction mass. | Runtime logic is mostly room-edge feasibility, not puzzle prerequisites. Independent edge beliefs do not model shared conditions or mechanisms. NSI changed the adjusted argmax twice. | Supply meaningful facts and formulas first. Represent shared light/load/door/mechanism variables once; test dependency assumptions and calibration before adopting a heavier solver. |
| `historian.py` | Preserves reward evidence, goals, inventory classifications, and successful LLM reviews; several reviews correctly identify wasted loops and conflicts. | Goal scheduler does not follow evolved review policy. Aggregate reward profiles are state-insensitive. Reviews see a truncated trajectory and fixed five-positive/five-negative lists; key and skeleton get conflated; resumed active lessons are not restored. | Use typed conditional lessons with evidence IDs, signed/loss events, prerequisite-aware goals, relevance retrieval, and durable lesson restoration. Allow fewer lessons when evidence is insufficient. |
| `historian_rl_controller.py` | Maintains a small online TD value model and action-conditioned transition memory; final memory contains 7,062 transitions from 49 finalized episodes. | Sparse text features do not explicitly represent remaining horizon, lamp state, puzzle flags, or one-time reward completion. Retrieval takes a global top-k before filtering by action; rare actions can receive no evidence. | Add grounded state features and per-action retrieval quotas; log prediction calibration and uncertainty; preserve truncation semantics; avoid interpreting low TD error as strategic competence. |
| `evolution.py` | Validates mutation structure, retains an archive, and successfully generated 44 proposals. | Prose mutations often embed stale episode facts or broad bans. There were 44 distinct active configurations across 50 slots; the final 12-entry archive has eleven entries evaluated once and one unevaluated. This is weak evidence for selection quality. | Mutate one policy hypothesis at a time, keep factual memory separate, and compare children against parents across matched seeds and memory snapshots before promotion. |
| `navigation_controller.py` | Executes and verifies learned transitions, yields when unavailable, preserves the map, and respects treasure-only automatic return. | Cannot solve a blocked route's prerequisite; route confidence has little causal detail. The chimney case repeatedly hands back an unresolved problem. | Keep executor role and treasure gate; emit a structured blocked-edge task with prior successful conditions, current differences, and a bounded investigation budget. |
| `trial_logger.py` | Full prompts, raw label tokens, selection details, reviews, outcomes, and separate experience files made this diagnosis possible. Exclusive episode files protect earlier logs. | Approximately 544 MB of repetitive text lacks universal step/request IDs, efficient event joins, comprehensive usage/latency records, and normalized action-effect events. Some embedded Historian episode numbers shift after interruption. | Add append-only structured event JSONL with explicit trial/episode/step/request IDs and schema versions, compressed raw payload references, token usage, latency, actual provider metadata, and source/environment hashes. |

This architecture currently adapts external policy scores, memory, prompts, configuration, and a small TD critic. It does not train the Actor or Historian LLM weights. The Historian LLM's prose evolution and the numerical critic are distinct mechanisms. Likewise, “neural memory” was an available option, not an active component of this run. These distinctions matter when describing the thesis method or comparing with JitRL/EvoTest.

**5. Reliability findings that still need correction**

The earlier infrastructure work helped: both observed 503 failures were retried, all 49 available episode reviews succeeded, 44 mutations succeeded, and the trial reached slot 50. The failures that remain are specific.

First, missing label C is still not recovered reliably. There were 6,955 initial scoring attempts, 13 attempts with equal bias 20, and ten with equal bias 80. Thirteen initial requests needed recovery; five recovered and eight exhausted it. Those eight errors caused `no_safe_actions` in episodes 8, 14, 16, 35, and 47. In three of these episodes a default lantern action ran before the next failure. No episode reached the configured five-default limit.

In episode 8, biased and unbiased responses retain very similar label/non-label gaps and still omit C. That is inconsistent with the behavior the recovery mechanism expects. The logs alone cannot distinguish an upstream route ignoring bias, log probabilities reported before bias, or another serving mismatch. The request includes the expected token IDs and `require_parameters`; the code's intention is visible, but successful parameter enforcement is not demonstrated. A controlled capability probe should test observable effects and record the serving route when available, not merely whether the API accepts a field.

Second, `no_safe_actions` is misleading here. Legal game actions remain; what is missing is a fully scored policy or an eligible recovery default. Introduce separate reasons such as `policy_scores_incomplete`, `provider_recovery_exhausted`, and `action_space_empty`. Retain the original cause in the outcome, rather than training the Historian to blame exploration for an infrastructure stop.

Third, maintain the user-requested bounded recovery behavior, but make defaults independent of the state-change oracle. A grammar-valid `look` can be available even when it is omitted by the oracle. It is not universally risk-free in an active combat scene, and repeated observations should not become a substitute for planning. Track both consecutive and total degraded actions; stop or pause after the configured bound and preserve partial work. Never record a recovery action as a native-scored Actor sample. Authentication/configuration errors should remain distinct from transient failures.

Fourth, resume preserves important memory but not all coaching state. `Historian.__init__` initializes `trial_lessons` empty and `load()` restores `memory` without reconstructing that field. Episode 34's review receives empty previous lessons after episode 33's interruption. Separately, internal episode IDs derived from `len(memory['episodes']) + 1` shift after the skipped episode. Use the explicit trial episode ID everywhere and restore the last committed lesson set. The 6,856 persisted Actor records are internally consistent, but episode 33's 91 logged Actor decisions were not finalized into that dataset. A transition journal would make that loss explicit and support safe reconciliation without silently inventing returns.

**6. Prompt changes for critical thinking**

The next prompts should request small, checkable decisions. Asking for “more reasoning” while retaining the same action gate, state corruption, and dominant prior would not address the observed failures.

Use one authoritative state block, followed by the current objective, a few relevant lessons, the frontier/obstacle ledger, and a diverse candidate menu. Remove repeated historical route listings and stale episode narratives from the scoring prompt. Proposal/repair requests currently contain a median 17,675 characters and a maximum 22,455; their length is a cost and conflict risk, although this audit does not establish a context-limit failure.

The Actor's core instruction should say:

> Choose an executable action that advances the current objective or tests a specific unresolved question. Distinguish observed facts, remembered conditional facts, and hypotheses. An information-producing action may be useful without immediate score. When a known action fails, compare its current prerequisites with a recorded success. Change one relevant condition where practical, predict an observable result, and record what actually happens. Do not treat movement, repeated inspection, or reversible object shuffling as progress unless it resolves a named uncertainty. Preserve carried valuables, necessary light, and useful tools. A danger lesson applies to its observed conditions; it is not a ban on an entire region.

A compact proposal contract could be:

```json
{
  "objective_id": "investigate_blocked_return",
  "observed_fact_ids": ["e44:s117:room", "e44:s117:inventory"],
  "hypothesis": "The narrow passage may restrict carried load.",
  "action": "examine chimney",
  "expected_observation": "A passage description or a carrying restriction.",
  "alternative_if_disconfirmed": "Compare another known return route.",
  "candidate_actions": ["examine chimney", "inventory", "south"]
}
```

This is a proposed schema, not an existing supported command contract. Implement action-space expansion before relying on it. Keep explanations short and externally checkable; a long private reasoning transcript is neither required nor useful as an experiment measure.

The scorer should receive the same grounded state and candidate consequences without being told that candidate A is the previous best choice. Assign labels using a logged seeded permutation, then map scores back to actions. Include at least one contextually appropriate untried exit or information test when available. A candidate generator cannot discover actions excluded from its own menu, so menu coverage must be measured separately from scoring quality.

The Historian prompt should replace “do not disclose puzzle solutions” with: “Do not import unobserved walkthrough facts. Preserve solutions discovered in this trial, with their prerequisites and evidence references.” Its current wording discourages exactly the reusable solution memory we want. Store a lesson as a conditional assertion with supporting and conflicting observations, rather than as a global slogan. For example, the evidence should distinguish `take key` from `take skeleton`, and `enter dark room without carried light` from `never go north`.

The Evolver should change decision rules, retrieval settings, or prompt organization, while keeping discovered game facts in the memory layer. Require each mutation to name a measurable failure, its proposed mechanism, and a falsification test. A claim such as “never enter the Maze” should not become a system-level instruction merely because one episode died there.

For the user's puzzle examples, the underlying skills are analogy, cross-room causal memory, prerequisite testing, and delayed-effect tracking. The yellow button is in the Maintenance Room; the bolt/control panel is at the dam. The useful behavior is to record a switch change, revisit the affected mechanism, observe its indicator, and test the tool under the new condition. [Infocom's dam hints](https://raw.githubusercontent.com/historicalsource/zork1-gold/master/hint-txt.txt)

For the Cyclops, a language model can propose a mythological association as a hypothesis, and in-game textual clues can strengthen it. A name must remain executable even if absent from the oracle menu. The source supports the Odysseus interaction and a passage connecting the Cyclops area with the Living Room. Keep this evaluator-known example out of a claimed discovery-only agent's initial memory; alternatively run an explicitly labeled knowledge-assisted condition. [Infocom action logic](https://raw.githubusercontent.com/historicalsource/zork1/master/1actions.zil), [room connections](https://raw.githubusercontent.com/historicalsource/zork1/master/1dungeon.zil)

**7. The case for and against the Cognitive Map**

The case for retaining it is strong. It preserves successful routes across episodes, distinguishes repeated room descriptions using stable IDs, represents directed and one-way transitions, and supports safe handoff between planning and action execution. The Studio → Kitchen edge has 88 recorded successes. Episode 44's successful eight-command coin-return segment demonstrates value that a prompt-only history would have to reconstruct repeatedly.

The case against the present form is that it is much closer to a route graph with confidence estimates than a general world model. It does not explain why an edge is available, what a switch changes, which item is inside a container, whether a treasure has already been secured, or how to recover from losing a tool. Large success counts can make a conditional edge look generally reliable. No persisted failure count on any final edge is a warning about the evidence collected, not proof every historical route is always usable.

The current NSI computation is mathematically legitimate for its stated finite assumptions. In abstract form it computes a sum of logic satisfaction times belief mass; action selection then weights the neural prior by feasibility. Its practical value depends on the supplied logic and beliefs. Edge-existence beliefs alone cannot infer a chimney load limit or a remotely enabled mechanism. More solver machinery would not manufacture those missing observations and rules.

Keep the graph, but make the world model factored:

```text
state = room identity
      + carried objects and containment
      + observed light / threat / load facts
      + local and cross-room mechanism facts
      + treasure status: seen, carried, cached, deposited, lost
      + current objective and remaining budget

action model = relevant preconditions
             + possible outcomes and uncertainty
             + observation/effect evidence
             + cost, danger, and reversibility
```

Store only relevant factors on an edge or action rule. Avoid materializing every combination of rooms, inventories, switches, and goals as a separate graph node. This preserves the compact topology while allowing planning through preparation actions and conditional routes.

A light-source condition shared across several route segments should be represented once, rather than as independent new evidence on every edge. Mutually exclusive destinations should not be treated as independent simultaneous events. Initially, explicit conditional rules and the existing NSI code are sufficient; compiled probabilistic logic becomes worthwhile only when measured rule complexity demands it.

**8. Would a 3D or n-dimensional map help, and can it live on a new branch?**

Yes, a branch such as `codex/conditional-world-model` is appropriate once the current validated reliability fixes are represented in the chosen base. Branching from an older `main` without those changes would confound the experiment. This evaluation created no branch and changed no production source.

A 3D visual layout would help a human inspect regions, one-way transitions, and unresolved frontiers. It should not impose Euclidean geometry on Zork. The saved graph includes Maze 70 south → 69 and Maze 69 south → 70. Such transitions cannot both be interpreted as ordinary equal-length southward displacement in a consistent coordinate map. Up/down and compass labels are action labels; reverse exits must be observed, not inferred.

The useful “n dimensions” are state variables: inventory, light, threat, mechanism configuration, treasure status, and goal progress. The same room can have several actionable states. This should improve conditional route planning and reduce irrelevant retrieval if implemented well. It also increases modeling and debugging complexity. Its effect on score is a hypothesis to test, not a guaranteed improvement.

NetworkX remains suitable at 28 rooms. `MultiDiGraph` already supports directed parallel edges, self-loops, and attributes, so there is no need to replace the core library to add conditional transitions. Use typed Python records and versioned JSON initially; consider SQLite for growing event/fact history. Graphviz or a Plotly view could provide human visualization later. There is no evidence here that switching to LTNs, DeepProbLog, or another neural-symbolic framework would be the highest-value first change. [NetworkX graph capabilities](https://networkx.org/documentation/stable/reference/classes/multidigraph.html)

**9. A maze exploration procedure that can discover both exits**

Maintain a persistent frontier ledger for each room ID and normalized command. Its states should include untried, traversed-to-destination, self-loop, currently unavailable, blocked-by-known-prerequisite, and failed-under-recorded-conditions. Store successes and counterexamples with episode/step references. Do not discard a successful route when a particular load or mechanism state blocks it.

Choose a frontier by information gain and estimated cost from the current state, with constraints for light, immediate threats, carried treasure, and recoverability. Follow known paths toward it using Actor-selected commands unless a separately authorized navigation policy permits more automation. Preserve the current rule that automatic homeward takeover requires currently carried treasure; frontier suggestions do not need to bypass it.

At the selected frontier, allocate an actual decision to an untried move. A small bonus against a 10–18-nat base preference is not enough. Record the resulting destination even when its name is another “Maze,” and record same-room transitions as evidence rather than silently dropping them. Re-plan from observed directed edges; never assume that the opposite compass direction returns to the preceding room.

When carrying treasure, prefer a verified state-compatible exit. If blocked, investigate its prerequisite or choose a different discovered exit. Avoid spending the entire lamp budget proving that a known loop still exists. In a text-only experiment without Jericho IDs, room identity would require a belief over locations, landmarks, and carefully chosen probes. The current experiment already has privileged room IDs, so it should first exploit that advantage correctly.

The agent was one offered action away from the Cyclops Room at episode 44 step 195. This is now verified by local replay, not just inferred from a walkthrough. The original agent did not discover that edge; it must remain labeled evaluator-only until a new run discovers it or the experiment explicitly supplies it.

The other requested route is the Grating Room with a skull-and-crossbones lock. The skeleton key unlocks the grating from below; the exit reaches the forest, after which an outdoor route can lead back to the house. It is not a direct passage into the Living Room. The key and skeleton must remain distinct objects in memory and action restrictions. [Infocom's grating hints](https://raw.githubusercontent.com/historicalsource/zork1-gold/master/hint-txt.txt)

A single live traversal cannot guarantee maze completion under enemy encounters and limited light. The evaluation should measure how many eligible frontiers are tested before resource loss, whether new rooms persist across episodes, and whether each new connection improves an executable escape route.

**10. Prioritized implementation plan for the next iteration**

| Priority | Change | Files / interface | Acceptance evidence |
|---|---|---|---|
| P0 | Separate committed observations from temporary probe narration; ground inventory in objects or a strict parser. | `main.py`, `action_guards.py`, state-building code | Combat/death prose never becomes inventory; snapshots restore state and RNG; lamp possession is distinct from lamp being lit in a room. |
| P0 | Expand admissible actions beyond the oracle's state-changing suggestions. | New action-admission helper, `main.py`, `smart_agent.py` | Context-grounded read/examine/speech and selected prerequisite probes can execute and produce normal logged outcomes. |
| P0 | Repair scoring-exhaustion recovery and outcome classification. | `policy_logits.py`, `provider_requests.py`, `run_recovery.py`, `main.py` | Repeated episode-8 failures consume bounded retries, use marked recovery if appropriate, retain the cause, and never invent native scores. |
| P0 | Restore lesson state and stable identity on resume. | `historian.py`, `checkpoint.py`, logger | Resume after interrupted 33 retains episode-32 lessons, starts the intended next slot, and uses actual trial IDs without duplication. |
| P1 | Add conditional goal eligibility and meaningful progress. | `historian.py`, `goal_control.py` | Already-completed one-time rewards stay retired; irrelevant object shuffles do not release a stalled objective; relevant new evidence does. |
| P1 | Persist frontiers and action effects. | `world_map.py`, `cognitive_map.py`, new typed records | Maze52's untried exits receive an exploration opportunity; self-loops and blocked transitions are represented; old paths remain intact. |
| P1 | Explain blocked routes with prerequisite tasks. | `navigation_controller.py`, Historian interface | A blocked Studio chimney produces an inventory-comparison task and can become usable again without erasing its historical edge. |
| P1 | Make policy changes behaviorally meaningful and safe. | `smart_agent.py`, `actor_rl_controller.py` | Neutral/permuted scoring, diverse candidates, calibrated prior strength, and explicit exploration have measurable selection effects without favoring known fatal tests. |
| P1 | Replace global prose bans with conditional evidence. | `historian.py`, `evolution.py` | Key/skeleton distinction retained; learned solutions can be reused; contradictory lessons are revised or quarantined with provenance. |
| P2 | Improve retrieval and critic state. | `actor_experience.py`, `semantic_memory.py`, `historian_rl_controller.py` | Per-action evidence coverage, correct prerequisite matching, finite calibrated values, explicit score/horizon/progress features. |
| P2 | Evaluate evolution instead of continuously trusting mutations. | `adaptive_system.py`, `evolution.py`, experiment runner | A child is promoted only after a predeclared matched comparison; a frozen baseline remains available. |
| P2 | Add an evaluator event schema and compact reports. | `trial_logger.py`, `checkpoint.py`, audit tooling | Every decision joins to its goal, prompt, scores, effect, learning record, and outcome; costs and provenance are inspectable. |

For action admission, use a union of oracle suggestions, known parser-compatible inspection templates grounded in visible objects, and a small number of hypothesis-backed proposals. Validate syntax, referenced objects, and known hazards. Do not pre-test all proposed actions against hidden future rewards and then give only successful ones to the Actor; that would introduce a different and stronger oracle. Failed tests are useful observations and should normally occur as actual counted actions.

For policy calibration, compare a tempered prior and an explicit exploration mixture rather than blindly increasing beta. One possible research policy is a safety-constrained mixture of the corrected neural policy and a distribution over eligible frontier/tests. Log the mixture's actual action probability. Retain the unmodified native scores separately. If selection is greedy, log the probability under the score distribution separately from the deterministic behavior probability, which is one for the selected action. Do not present this heuristic adaptation as a demonstrated unbiased off-policy policy-gradient algorithm.

For rewards, keep official Zork score as the primary benchmark. Add separate event measures for newly discovered rooms/exits, validated prerequisites, treasure deposited/lost, light depletion, and irreversible mistakes. If using intrinsic reward or potential shaping, make it an explicit ablation and prevent repeated open/close or take/redeposit cycles from earning unlimited bonuses. Episode 44's zero-score curse is a key regression fixture.

For preservation, create versioned copies before migration; never run `fresh-memory` against the original trial as a cleanup step. Import all existing nodes, aliases, transitions, and evidence. Unknown prerequisites should remain unknown. Quarantine demonstrably unreliable prose lessons with their original text and a revision reason instead of deleting the trial record. Keep evaluator counterfactuals and source-derived puzzle facts outside discovery-only runtime memory.

**11. Tests and experiments that would make the next result convincing**

Start with deterministic regression tests from these logs:

1. Inventory text with thief/combat/death suffixes produces only actual carried items; contained objects preserve containment.
2. A lit lantern left in a room does not satisfy carried-light prerequisites for a dark destination.
3. `look`, `inventory`, and a grounded inspection command can be admitted without oracle membership; the recovery count remains bounded and excludes Actor-native training samples.
4. Missing C at every bias level preserves the scoring cause and never combines different responses into a complete distribution.
5. Transient 503 or malformed HTTP bodies retry without consuming game moves; authentication and programming defects remain distinct.
6. Sword/lantern without treasure cannot authorize automatic homeward navigation; treasure deposited, dropped, or stolen revokes authorization.
7. Objective-specific progress ignores irrelevant sack toggling and item shuffling; a changed prerequisite legitimately reopens the objective.
8. Two Maze rooms with identical titles stay distinct; directional cycles and self-loops remain valid graph structure.
9. The episode-44 Studio state supports a blocked-load explanation and retains the previously successful edge.
10. Maze52 with unexplored west/southeast gets a budgeted frontier decision despite a concentrated south prior.
11. Taking the key does not inherit the skeleton's curse rule; zero-score treasure loss generates a material loss event.
12. A one-time reward already obtained in the episode cannot dominate goals indefinitely.
13. Resume after interrupted episode 33 preserves lessons and explicit IDs; repeated recovery does not duplicate metrics or experience.
14. Prompt mutation cannot turn a single contextual failure into an unconditional regional ban without supporting evidence and review.

Then use short isolated ROM replays to confirm mechanics and action admission. Keep them distinct from scored live experiments. The two replay cases supplied with this report are a starting point, not a complete navigation test suite.

Run a staged evaluation rather than another immediately evolving 50-episode trial. First compare the existing baseline with P0 fixes under fixed prompts. Next add frontiers and conditional goals. Then test neutral/calibrated scoring and RL. Finally evaluate conditional-map NSI and controlled evolution. Use at least several paired environment seeds for initial screening, cloned starting memories, fixed command/API budgets, and a documented model/provider route. Expand promising variants to a larger held-out seed set before making performance claims.

Ablations should include base Actor; Actor plus map/frontier planner; Actor plus RL correction; conditional-map NSI on/off; and controlled evolution on/off. Some components need the same corrected state/action interface to make a fair comparison. Randomly turning off the map while retaining its facts in prompts would not be a genuine no-map ablation.

Report target-80 attainment over all scheduled runs, final and peak score, time/commands to first deposit and new region, death cause, frontier coverage, repeated-transition rate, treasure loss, policy argmax changes, scoring recovery rate, incomplete/degraded counts, and actual token cost/latency. Keep the current conditional mean for continuity, but never report it alone. A low TD error or a completed Python process is not evidence of puzzle-solving success.

The most promising next milestone is reproducible retention and deposit of the coins, followed by discovery of a new Maze exit under the same bounded resource budget. Those milestones directly test whether the architecture can preserve knowledge, solve a conditional route, and explore beyond familiar rewards. They are stronger evidence of progress toward 80 than simply running longer episodes.

**Audit artifacts and reproducibility**

The companion directory `trial5-data` contains the 50-row episode table, full file manifest, aggregate metrics, signed reward events, diagnostic observations, inventory screens, active configurations, Maze menus, and isolated replay results. `audit_trial5.py` reproduces the counts from the supplied trial; `replay_trial5_probes.py` reproduces the two episode-44 counterfactual cases. These scripts read the source trial and write derived files only.

Absolute artifact directory: `C:\Users\jlkes\Code\Thesis\my_solution\reports\trial5-data`.

Original trial directory: `/home/jlkess/projects/thesis/my_solution/logs/tabula-rasa-5_big_20261111_172225` in Ubuntu WSL.
