"""Zork I: LLM Actor + Symbolic Navigator + persistent Historian."""
from __future__ import annotations

# Load project-local .env before reading configuration defaults.
import config  # noqa: F401

import argparse
import os
import hashlib
import math
import json
import sys
from pathlib import Path
from multiprocessing import freeze_support

from MANGORL.actor_experience import ActorExperienceMemory
from MANGORL.actor_rl_controller import ActorRLController
from MANGORL.historian import Historian
from MANGORL.historian_rl_controller import HistorianRLController
from MANGORL.navigation_controller import NavigationController
from MANGORL.smart_agent import SmartAgent
from MANGORL.trial_logger import TrialLogger
from MANGORL.world_map import observed_room_title
from MANGORL.cognitive_map import CognitiveMap
from MANGORL.adaptive_system import AdaptiveSystem
from MANGORL.semantic_memory import FrozenStateEncoder
from MANGORL.policy_logits import LogitsUnavailable, prepare_model
from MANGORL.llm_connections import AnthropicHistorian, create_historian_connection
from MANGORL.action_guards import guard_actions, ground_inventory, inventory_from_text, snapshot_sections
from MANGORL.provider_requests import request_budget
from MANGORL.goal_control import GoalControl, authorize_navigation, carried_treasures
from MANGORL.checkpoint import TrialCheckpoint, restore_arguments, trial_lock
from MANGORL.run_recovery import recoverable_episode_error

DEFAULT_BASE_URL = "http://cs-steverw.umflint.edu:8080/api"


class RunArguments(argparse.Namespace):
    def __repr__(self):
        safe = {key: value for key, value in vars(self).items() if "api_key" not in key}
        return f"RunArguments({safe!r})"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for a reproducible trial run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the neuro-symbolic Zork I agent. Each invocation creates a "
            "unique trial folder, with separate console/chat logs, a Historian "
            "review, and a configurable set of lessons for every completed episode."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent-type", choices=("smart",), default="smart",
        help="Actor implementation to use (only 'smart' is currently implemented).",
    )
    parser.add_argument(
        "--actor-model", "--model", dest="model", default=os.getenv("ACTOR_MODEL", "gemma4:31b"),
        help="LLM model identifier passed to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--actor-base-url", "--base-url", dest="base_url", default=os.getenv("ACTOR_BASE_URL", DEFAULT_BASE_URL),
        help="Actor-only OpenAI-compatible API base URL (ACTOR_BASE_URL).",
    )
    parser.add_argument(
        "--actor-api-key", "--api-key", dest="api_key", default=None,
        help=(
            "Actor-only API key (prefer ACTOR_API_KEY; command-line values may be visible "
            "in shell history and process listings)."
        ),
    )
    parser.add_argument(
        "--rom-path", default=os.getenv(
            "ZORK_ROM_PATH",
            "z-machine-games-master/jericho-game-suite/zork1.z5",
        ),
        help="Path to the Zork I .z5 game file.",
    )
    parser.add_argument(
        "--max-steps", type=int,
        default=int(os.getenv("ZORK_MAX_STEPS", "500")),
        help="Maximum commands per episode.",
    )
    parser.add_argument(
        "--target-score", type=int,
        default=int(os.getenv("ZORK_TARGET_SCORE", "76")),
        help="Stop an episode when this score is reached.",
    )
    parser.add_argument(
        "--episodes", "--num-episodes", dest="episodes", type=int,
        default=int(os.getenv("ZORK_EPISODES", "1")),
        help="Number of fresh Zork episodes to run in this trial.",
    )
    parser.add_argument(
        "--map-path", default=os.getenv("ZORK_MAP_PATH", "memory/world_map.json"),
        help="Persistent WorldMap JSON path.",
    )
    parser.add_argument(
        "--map-navigation", action=argparse.BooleanOptionalAction, default=True,
        help="Use the persistent map, symbolic routing, and map-based feasibility. Disable all three with --no-map-navigation.",
    )
    parser.add_argument("--walkthrough", action="store_true",
                        help="Replay Jericho's walkthrough without LLMs and save its path, then exit.")
    parser.add_argument("--walkthrough-map-path", default="memory/world_model.json",
                        help="WorldMap JSON to populate when --walkthrough is enabled.")
    parser.add_argument(
        "--historian-path",
        default=os.getenv("ZORK_HISTORIAN_PATH", "memory/historian.json"),
        help="Persistent Historian JSON path.",
    )
    parser.add_argument(
        "--output-dir", "--log-dir", dest="output_dir",
        default=os.getenv("ZORK_OUTPUT_DIR", "logs"),
        help="Parent directory in which a unique trial folder is created.",
    )
    parser.add_argument(
        "--trial-name", default=None,
        help="Optional readable trial name (a timestamp is always appended).",
    )
    parser.add_argument(
        "--history-length", type=int, default=10,
        help="Number of recent Actor transitions retained in its prompt context.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.25,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=900,
        help="Maximum response tokens requested from the LLM.",
    )
    parser.add_argument(
        "--historian-review", action=argparse.BooleanOptionalAction,
        default=True,
        help="Have the Historian review each finished episode and update the configured trial lessons.",
    )
    parser.add_argument(
        "--historian-model", default=os.getenv("HISTORIAN_MODEL"),
        help="Model used for episode reviews (defaults to --model).",
    )
    parser.add_argument("--historian-base-url", default=os.getenv("HISTORIAN_BASE_URL"),
                        help="Historian-only endpoint for reviews and evolution (HISTORIAN_BASE_URL).")
    parser.add_argument("--historian-api-key", default=None,
                        help="Historian-only API key; prefer HISTORIAN_API_KEY. Never inherits ACTOR_API_KEY.")
    parser.add_argument("--historian-openwebui-parent-id-null", action=argparse.BooleanOptionalAction,
                        default=os.getenv("HISTORIAN_OPENWEBUI_PARENT_ID_NULL", "0") == "1",
                        help="Historian-only Open WebUI compatibility setting.")
    parser.add_argument(
        "--historian-lessons-per-category", "--historian-lesson-count", "--historian-lessons",
        dest="historian_lessons_per_category",
        type=int,
        default=int(os.getenv("ZORK_HISTORIAN_LESSONS_PER_CATEGORY", "3")),
        help=(
            "Number of positive lessons to repeat AND negative lessons to avoid "
            "generated after each episode (1-10 per category)."
        ),
    )
    parser.add_argument(
        "--historian-temperature", type=float, default=0.2,
        help="Sampling temperature for OpenAI-compatible Historian episode reviews.",
    )
    parser.add_argument(
        "--historian-max-tokens", type=int, default=4096,
        help="Initial Historian review token budget; a truncated response may retry once with twice this budget.",
    )
    parser.add_argument(
        "--actor-rl", action=argparse.BooleanOptionalAction, default=True,
        help="Enable experience-based Actor action adaptation (disable with --no-actor-rl).",
    )
    parser.add_argument(
        "--actor-experience-path",
        default=os.getenv("ZORK_ACTOR_EXPERIENCE_PATH", "memory/actor_experiences.jsonl"),
        help="Append-only JSONL file containing completed Actor experiences.",
    )
    parser.add_argument(
        "--actor-rl-gamma", type=float, default=0.995,
        help="Discount factor used to assign delayed episode rewards to Actor decisions.",
    )
    parser.add_argument(
        "--actor-rl-beta", type=float, default=0.35,
        help="Strength of experience advantages in candidate action reranking.",
    )
    parser.add_argument(
        "--actor-rl-exploration", type=float, default=0.08,
        help="Small bonus favoring less-tried candidate actions.",
    )
    parser.add_argument(
        "--actor-rl-rank-penalty", type=float, default=0.30,
        help="Prior penalty per position in the Actor's candidate ranking.",
    )
    parser.add_argument(
        "--actor-rl-top-k", type=int, default=40,
        help="Maximum similar Actor experiences used to estimate action values.",
    )
    parser.add_argument(
        "--actor-rl-min-similarity", type=float, default=0.65,
        help="Minimum state similarity required when retrieving Actor experiences.",
    )
    parser.add_argument(
        "--actor-rl-environment-reward-weight", type=float, default=0.0,
        help="Optional weight for Jericho reward in addition to score delta (default 0).",
    )
    parser.add_argument(
        "--actor-rl-no-progress-penalty", type=float, default=0.0,
        help="Optional reward penalty for repeated no-progress steps (default 0).",
    )
    parser.add_argument(
        "--historian-critic", action=argparse.BooleanOptionalAction, default=True,
        help="Update the external Historian TD critic online; LLM weights remain frozen.",
    )
    parser.add_argument(
        "--historian-critic-path",
        default=os.getenv("ZORK_HISTORIAN_CRITIC_PATH", "memory/historian_td.json"),
        help="Persistent sparse value-function weights for the optional Historian critic.",
    )
    parser.add_argument(
        "--historian-critic-learning-rate", type=float, default=0.025,
        help="Online TD value-function update rate when --historian-critic is enabled.",
    )
    parser.add_argument(
        "--openwebui-parent-id-null",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("OPENWEBUI_PARENT_ID_NULL", "1") == "1",
        help="Send parent_id: null for compatible Open WebUI /api endpoints.",
    )
    parser.add_argument("--actor-policy-mode", choices=("logprobs", "rank"), default="logprobs",
                        help="Actual model label probabilities, or explicit rank-prior ablation.")
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=False,
                        help="Sample the corrected policy instead of choosing its highest-probability command.")
    parser.add_argument("--nsi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-risk-weight", type=float, default=2.0)
    parser.add_argument("--evolve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evolution-path", default="memory/configurations.json")
    parser.add_argument("--evolution-max-tokens", type=int, default=4096,
                        help="Initial evolution token budget; a truncated response may retry once with twice this budget.")
    parser.add_argument("--fresh-memory", action="store_true",
                        help="Use empty memories inside this trial directory; keep existing memory files intact.")
    parser.add_argument("--resume-trial", default=None, help="Resume at the next unused episode identity in an existing checkpointed trial.")
    parser.add_argument("--max-recovery-actions", type=int, default=5,
                        help="Maximum explicit default actions per episode after exhausted recoverable calls (1-10).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on the first exhausted recoverable episode error, instead of saving it and continuing.")
    parser.add_argument("--max-consecutive-episode-failures", type=int, default=3,
                        help="Stop after this many consecutive recoverable episode failures; no rank fallback is used.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-embedding-model", default=None,
                        help="Optional frozen SentenceTransformer model/path; omitted means lexical retrieval.")
    parser.add_argument("--score-scale", type=float, default=350.0,
                        help="Maximum game score used for normalized AUC and configuration fitness.")
    return parser


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv, namespace=RunArguments())
    if args.walkthrough and not args.map_navigation:
        raise SystemExit("--walkthrough requires map storage and cannot be used with --no-map-navigation")
    args.nsi = args.nsi and args.map_navigation
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be greater than zero")
    if args.target_score < 0:
        raise SystemExit("--target-score cannot be negative")
    if not 1 <= args.max_recovery_actions <= 10:
        raise SystemExit("--max-recovery-actions must be between 1 and 10")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than zero")
    if args.max_consecutive_episode_failures <= 0:
        raise SystemExit("--max-consecutive-episode-failures must be greater than zero")
    if args.history_length <= 0:
        raise SystemExit("--history-length must be greater than zero")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be greater than zero")
    if args.historian_max_tokens <= 0:
        raise SystemExit("--historian-max-tokens must be greater than zero")
    if args.evolution_max_tokens <= 0:
        raise SystemExit("--evolution-max-tokens must be greater than zero")
    if not 1 <= args.historian_lessons_per_category <= 10:
        raise SystemExit("--historian-lessons-per-category must be between 1 and 10")
    if args.temperature < 0:
        raise SystemExit("--temperature cannot be negative")
    if args.historian_temperature < 0:
        raise SystemExit("--historian-temperature cannot be negative")
    if not 0.0 <= args.actor_rl_gamma <= 1.0:
        raise SystemExit("--actor-rl-gamma must be between 0 and 1")
    if args.actor_rl_beta < 0 or args.actor_rl_exploration < 0:
        raise SystemExit("Actor RL beta and exploration coefficient cannot be negative")
    if args.actor_rl_rank_penalty < 0:
        raise SystemExit("--actor-rl-rank-penalty cannot be negative")
    if args.actor_rl_top_k <= 0:
        raise SystemExit("--actor-rl-top-k must be greater than zero")
    if not 0.0 <= args.actor_rl_min_similarity <= 1.0:
        raise SystemExit("--actor-rl-min-similarity must be between 0 and 1")
    if args.historian_critic_learning_rate <= 0 or args.historian_critic_learning_rate > 1:
        raise SystemExit("--historian-critic-learning-rate must be in (0, 1]")
    if args.score_scale <= 0 or not 0 <= args.route_risk_weight <= 10:
        raise SystemExit("Score scale must be positive and route risk weight must be in [0, 10].")
    if args.history_length > 30 or args.actor_rl_top_k > 100 or args.temperature > 2:
        raise SystemExit("Configuration limits: history <=30, retrieval top-k <=100, temperature <=2.")
    if args.actor_rl_beta > 3 or args.actor_rl_exploration > 1 or args.historian_critic_learning_rate > 0.5:
        raise SystemExit("Configuration limits: beta <=3, exploration <=1, critic learning rate <=0.5.")
    return args


def get_valid_actions(env):
    """Get Jericho parser actions without invoking multiprocessing."""
    try:
        actions = env.get_valid_actions(use_parallel=False)
        return list(dict.fromkeys(actions or []))
    except Exception as exc:
        raise RuntimeError("Jericho action generation failed; the available command menu is unknown.") from exc


def current_environment_context(env, observation):
    """Read room/inventory text from a snapshot without advancing the game."""
    if not callable(getattr(env, "get_state", None)) or not callable(getattr(env, "set_state", None)):
        return observation
    snapshot = env.get_state()
    try:
        room_text, _, _, _ = env.step("look")
        env.set_state(snapshot)
        inventory_text, _, _, _ = env.step("inventory")
    finally:
        env.set_state(snapshot)
    return (f"LAST ACTION RESULT:\n{observation}\n\n"
            f"CURRENT ROOM (look):\n{room_text}\n\n"
            f"CURRENT INVENTORY:\n{inventory_text}")


def get_known_outgoing_actions(world_map, current_node):
    actions = set()
    for _, _, data in world_map.graph.out_edges(current_node, data=True):
        actions.update(data.get("actions", []))
    return actions


def find_living_room_node(world_map):
    """Return the canonical Living Room node, if already discovered."""
    matches = world_map.find_nodes_by_label("Living Room")
    return matches[0] if matches else None


def _match_action(candidate, valid_actions):
    if candidate is None:
        return None
    candidate = str(candidate).strip()
    if candidate in valid_actions:
        return candidate
    normalized = NavigationController._normalize_action(candidate)
    for action in valid_actions:
        if NavigationController._normalize_action(action) == normalized:
            return action
    return None


def create_agent(args, event_logger=None):
    """Construct the selected Actor implementation."""
    if args.agent_type == "smart":
        if args.actor_policy_mode == "logprobs":
            readiness = prepare_model(args.model)
            if event_logger:
                event_logger("actor_logprob_recovery_readiness", readiness)
        base_url = args.base_url
        extra_body = (
            {"parent_id": None}
            if args.openwebui_parent_id_null
            and base_url.rstrip("/").endswith("/api")
            else None
        )
        return SmartAgent(
            model=args.model,
            api_key=args.api_key,
            base_url=base_url,
            extra_body=extra_body,
            history_length=args.history_length,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            event_logger=event_logger,
            policy_mode=args.actor_policy_mode,
            seed=args.seed,
            map_navigation=args.map_navigation,
        )
    raise ValueError(f"Unsupported agent type: {args.agent_type}")


def build_actor_rl_state(
    *, current_node, current_label, observation, current_score,
    historian, goal, home_route_status,
):
    """Build a compact state snapshot shared by Actor RL and the value critic."""
    agent_state = getattr(historian, "agent_state", {}) or {}
    return {
        "location_id": str(current_node),
        "location": str(current_label),
        "observation": str(observation or ""),
        "score": current_score,
        "inventory_items": list(agent_state.get("inventory_items", [])),
        "treasure_candidates": list(agent_state.get("treasure_candidates", [])),
        "ordinary_items": list(agent_state.get("ordinary_items", [])),
        "useful_items": list(agent_state.get("useful_items", [])),
        "puzzle_clues": list(agent_state.get("puzzle_clues", [])),
        "goal_kind": str((goal or {}).get("kind", "unknown")),
        "goal_description": str((goal or {}).get("description", "")),
        "home_route_status": str(home_route_status or "unknown"),
    }


def run_episode(
    args, episode_number, world_map, historian, navigator, agent,
    episode_logger, actor_experience, actor_rl_controller,
    historian_rl_controller, trial_id, map_initialized=False, adaptive_system=None,
    historian_client=None, historian_extra_body=None,
):
    """Run one fresh Jericho episode while preserving trial-level memory."""
    # Import lazily so CLI help and parser tests work without Jericho installed.
    try:
        from jericho import FrotzEnv
    except ImportError as exc:
        raise RuntimeError(
            "Jericho is required to run the game. Install it with `pip install jericho`."
        ) from exc

    env = FrotzEnv(args.rom_path, seed=args.seed)
    info = {}
    last_recorded_state = None
    goal_control = GoalControl()
    recovery_count = 0
    control_metrics = {"navigation_without_treasure": 0, "goal_cancellations": 0, "route_starts": 0, "recovery_actions": 0, "treasure_deposits": 0}
    episode_started = False
    termination_reason = "initialization_failed"
    home_node = None
    home_label = "Living Room"
    use_nsi = args.nsi and world_map is not None

    try:
        observation, info = env.reset()
        room = env.get_player_location()
        if room is None:
            raise RuntimeError("Jericho did not provide the starting room.")

        if world_map is not None:
            if not map_initialized:
                world_map.instantiate()
            current_node = world_map.start_episode(room.name, room.num)
            current_label = world_map.display_location(current_node)
            navigator.start_episode()
        else:
            current_node, current_label = str(room.num), room.name
        historian.start_episode(
            start_node=current_node,
            start_location=current_label,
            initial_score=info.get("score"),
        )
        agent.reset_history()
        agent.event_logger = episode_logger.log_chat_event
        episode_started = True
        episode_key = f"{trial_id}/episode_{episode_number:04d}"
        actor_experience.start_episode(episode_key)
        historian_rl_controller.start_episode(episode_key)
        if adaptive_system is not None:
            adaptive_system.begin_episode(episode_logger.log_chat_event)
        termination_reason = "step_limit"

        print(f"Episode: {episode_number}/{args.episodes}")
        print(f"Initial observation: {observation}")
        print(f"Starting location: {current_label}")
        print(f"Target score: {args.target_score}")
        print(f"Model: {args.model} | Agent type: {args.agent_type}")
        print(f"Maximum steps: {args.max_steps}")
        print("\n" + historian.format_trial_lessons())

        for step in range(args.max_steps):
            print(f"\n========== STEP {step + 1}/{args.max_steps} ==========")
            room = env.get_player_location()
            if room is None:
                raise RuntimeError("Jericho did not provide the current room.")

            if world_map is not None:
                current_node = world_map.resolve_location(room.name, room.num)
                if current_node is None:
                    current_node = world_map.add_standard_location(room.name, room.num)
                    world_map.current_location = current_node
                    world_map.current_location_name = room.name
            else:
                current_node = str(room.num)
            current_score = info.get("score")
            if isinstance(current_score, (int, float)) and current_score >= args.target_score:
                termination_reason = "target_score_reached"
                break

            valid_actions = get_valid_actions(env)
            if not valid_actions:
                termination_reason = "no_valid_actions"
                break
            decision_observation = current_environment_context(env, observation)
            if world_map is not None:
                world_map.observe_room_title(current_node, decision_observation)
                current_label = world_map.display_location(current_node)
            else:
                current_label = observed_room_title(room.name, decision_observation) or room.name
                if current_label.casefold() == home_label.casefold():
                    home_node = current_node
            _, inventory_text = snapshot_sections(decision_observation)
            observed_inventory = inventory_from_text(inventory_text)
            ground_inventory(historian, observed_inventory)
            episode_logger.log_chat_event("actor_environment_context", {
                "location": current_label, "valid_actions": valid_actions,
                "observation": decision_observation, "steps_remaining": args.max_steps - step,
            })
            valid_actions, unsafe_actions = guard_actions(valid_actions, decision_observation)
            if unsafe_actions:
                episode_logger.log_chat_event("action_guard", {"rejected": unsafe_actions,
                                                               "permitted": valid_actions})
            if not valid_actions:
                termination_reason = "no_safe_actions"
                break

            known_routes, known_outgoing = [], ()
            home_route_status = "disabled"
            map_size = None
            if world_map is not None:
                map_size = (world_map.graph.number_of_nodes(), world_map.graph.number_of_edges())
                known_routes = world_map.get_reachable_routes(
                    start=current_node, max_destinations=max(1, map_size[0])
                )
                known_outgoing = get_known_outgoing_actions(world_map, current_node)
                home_node = find_living_room_node(world_map)
                home_route_status = navigator.route_status(home_node) if home_node is not None else "home_undiscovered"

            # Refresh progress evidence before deciding whether a goal is stale.
            historian.cycle_recovery(current_node, valid_actions, state_context=decision_observation)
            goal = historian.choose_goal(
                current_node=current_node,
                current_location=current_label,
                valid_actions=valid_actions,
                known_routes=known_routes,
                known_outgoing_actions=known_outgoing,
                current_score=current_score,
                home_node=home_node,
                home_label=home_label,
                home_route_status=home_route_status,
            )
            goal["progress_budget"] = max(24, 2 * max(
                (len(r.get("actions", [])) for r in known_routes if str(r.get("destination")) == str(goal.get("target_node"))),
                default=0) + 4)
            proposed_goal = goal
            goal = goal_control.select(
                proposed_goal, node=current_node, inventory=observed_inventory,
                context=decision_observation, score=current_score,
                map_size=map_size,
                treasures=carried_treasures(historian, observed_inventory),
            )
            for event in goal_control.events:
                episode_logger.log_chat_event("goal_transition", event)
                control_metrics["goal_cancellations"] += int(event["status"] == "abandoned")
            goal_control.events.clear()
            guidance = historian.build_guidance(
                current_node,
                current_label,
                current_score=current_score,
            )
            avoid_actions = historian.get_avoid_actions(current_node)
            actor_rl_state = build_actor_rl_state(
                current_node=current_node,
                current_label=current_label,
                observation=decision_observation,
                current_score=current_score,
                historian=historian,
                goal=goal,
                home_route_status=home_route_status,
            )
            rl_guidance = actor_rl_controller.build_guidance(
                actor_rl_state, valid_actions
            )
            if adaptive_system is not None and not adaptive_system.configuration["memory_guidance"]:
                rl_guidance = "Memory prompt guidance disabled by the active configuration."

            action = None
            action_source = "smart_actor"

            # A historical score event has no verified precondition contract.
            # The Actor must reassess it against the current room and inventory.
            # Defense in depth: goal proposals cannot bypass grounded inventory.
            if navigator is not None and authorize_navigation(
                goal, historian, observed_inventory, current_node, home_node, home_route_status
            ):
                had_plan = navigator.active_plan is not None
                plan = navigator.plan_to_home(home_node)
                if plan is not None:
                    control_metrics["route_starts"] += int(not had_plan)
                    action = navigator.next_action(valid_actions)
                    if action:
                        action_source = "symbolic_navigation"
                if action is None:
                    goal_control.cancel("route_unavailable_or_failed")
                    guidance += "\nThe deposit route failed; investigate a local prerequisite before trying again."
            elif navigator is not None:
                navigator.cancel_plan("treasure_handoff_not_authorized")

            recovery = historian.cycle_recovery(
                current_node, valid_actions, known_routes=known_routes,
                state_context=decision_observation,
                protected_actions=[],
            )
            if recovery["blocked_actions"]:
                blocked = set(recovery["blocked_actions"])
                if navigator is not None and action in blocked:
                    goal_control.cancel("movement_cycle")
                    navigator.cancel_plan("movement_cycle")
                    action, action_source = None, "smart_actor"
                valid_actions = [a for a in valid_actions if a not in blocked]
                avoid_actions = list(dict.fromkeys([*avoid_actions, *recovery["blocked_actions"]]))
                guidance += "\n" + recovery["reason"]
                goal["preferred_actions"] = recovery["preferred_actions"]
                episode_logger.log_chat_event("cycle_recovery", recovery)

            if action is not None and use_nsi and not world_map.action_feasibility(
                actor_rl_state, [action], navigator, valid_actions
            )[0]:
                episode_logger.log_chat_event("symbolic_action_rejected", {"action": action, "reason": "zero NSI feasibility"})
                action = None
                action_source = "smart_actor"

            if action is None:
                if navigator is not None:
                    navigator.cancel_plan("actor_or_puzzle_decision")
                try:
                    with request_budget():
                        decision = agent.choose_decision(
                            observation=decision_observation,
                            current_location=current_label,
                            valid_actions=valid_actions,
                            current_score=current_score,
                            target_score=args.target_score,
                            known_locations=world_map.get_known_locations() if world_map is not None else [],
                            known_routes=known_routes,
                            goal=goal,
                            historian_guidance=guidance,
                            avoid_actions=avoid_actions,
                            preferred_actions=goal.get("preferred_actions", []),
                            navigation_context=navigator.navigation_context() if navigator is not None else None,
                            rl_guidance=rl_guidance,
                            steps_remaining=args.max_steps - step,
                        )
                        base_action = decision["action"]
                        candidates = decision.get("candidate_actions", [base_action])
                        if args.actor_policy_mode == "logprobs" and decision.get("base_logits") is None:
                            raise LogitsUnavailable("The Actor did not supply the required native label logprobs.")
                        feasibility = world_map.action_feasibility(
                            actor_rl_state, candidates, navigator, valid_actions
                        ) if use_nsi else [1.0] * len(candidates)
                        if not any(feasibility):
                            # Ask again over feasible commands rather than executing a
                            # masked action or fabricating a model probability.
                            permitted = [a for a, p in zip(valid_actions, world_map.action_feasibility(
                                actor_rl_state, valid_actions, navigator, valid_actions)) if p > 0]
                            if not permitted:
                                termination_reason = "no_safe_actions"
                                episode_logger.log_chat_event("episode_no_feasible_action", {
                                    "location": current_label, "valid_actions": valid_actions,
                                    "reason": "NSI found no consistent available command.",
                                })
                                print("No feasible action remains; saving an incomplete episode.")
                                break
                            decision = agent.choose_decision(
                                observation=decision_observation, current_location=current_label,
                                valid_actions=permitted, current_score=current_score, target_score=args.target_score,
                                goal=goal, historian_guidance=guidance, rl_guidance=rl_guidance,
                                navigation_context=navigator.navigation_context(),
                                steps_remaining=args.max_steps - step,
                            )
                            base_action = decision["action"]
                            candidates = decision["candidate_actions"]
                            feasibility = world_map.action_feasibility(actor_rl_state, candidates, navigator, permitted)
                        rl_selection = actor_rl_controller.select_action(
                            actor_rl_state,
                            candidates,
                            base_action=base_action,
                            base_logits=decision.get("base_logits"),
                            feasibility=feasibility,
                            sample=args.sample_actions,
                        )
                        action = rl_selection.get("action") or base_action
                        decision["action"] = action
                        decision["rl_selection"] = rl_selection
                        decision["action_logprob"] = rl_selection.get("action_logprob")
                        episode_logger.log_chat_event("actor_rl_selection", {
                            "state": actor_rl_state,
                            "base_action": base_action,
                            "candidate_actions": decision.get("candidate_actions", []),
                            "selection": rl_selection,
                        })
                        episode_logger.log_chat_event("actor_final_selection", {
                            "location": current_label,
                            "goal": goal,
                            "decision_rationale": decision.get("decision_rationale", ""),
                            "base_model_action": base_action,
                            "final_action_after_rl": action,
                            "candidate_actions": decision.get("candidate_actions", []),
                            "selection": rl_selection,
                        })
                        historian.record_agent_report(decision.get("report", {}))
                        ground_inventory(historian, observed_inventory)
                except Exception as exc:
                    if args.fail_fast or not recoverable_episode_error(exc):
                        raise
                    # This is explicitly NOT a model policy sample. Use only a
                    # legal guarded observation/light command, never random travel.
                    feasible = world_map.action_feasibility(actor_rl_state, valid_actions, navigator, valid_actions) if use_nsi else [1] * len(valid_actions)
                    permitted = [a for a, weight in zip(valid_actions, feasible) if weight > 0]
                    action = next((a for preferred in ("look", "inventory", "turn on lantern", "turn on lamp", "wait")
                                   for a in permitted if a.casefold() == preferred), None)
                    if action is None:
                        termination_reason = "no_safe_actions"
                        break
                    recovery_count += 1
                    control_metrics["recovery_actions"] = recovery_count
                    action_source = "recovery_default"
                    decision = {"action": action, "report": {}, "candidate_actions": [],
                                "decision_rationale": "Bounded default after exhausted provider recovery; no model probability is available."}
                    episode_logger.log_chat_event("recovery_default_action", {
                        "action": action, "count": recovery_count, "limit": args.max_recovery_actions,
                        "error_type": type(exc).__name__, "policy_sample": False,
                    })

            else:
                decision = {
                    "action": action,
                    "report": {},
                    "decision_rationale": "",
                    "raw_response": "symbolic navigation",
                }
                episode_logger.log_chat_event("non_llm_action_selection", {
                    "location": current_label,
                    "goal": goal,
                    "action_source": action_source,
                    "selected_action": action,
                    "navigation_context": navigator.navigation_context(),
                    "note": "No LLM rationale exists because this command was selected by the symbolic navigator.",
                })

            print(f"Location: {current_label}")
            print(f"Goal: {goal.get('description')}")
            for sequence_step in goal.get("sequence", [])[:5]:
                print(
                    f"  {sequence_step.get('order', '?')}. "
                    f"{sequence_step.get('goal', 'Next objective')}: "
                    f"{sequence_step.get('details', '')}"
                )
            if navigator is not None and home_node is not None:
                print(f"Home route status this episode: {home_route_status}")
            print(f"Action source: {action_source}")
            print(f"Action: {action}")
            if action_source == "smart_actor" and decision.get("rl_selection"):
                print(
                    "Actor RL selection method: "
                    f"{decision['rl_selection'].get('method')}"
                )

            observation_before = observation
            score_before = info.get("score")
            before_node = current_node
            before_label = current_label

            state_hash = getattr(env, "get_world_state_hash", None) if navigator is not None else None
            world_before = state_hash() if callable(state_hash) else None
            observation, reward, done, next_info = env.step(action)
            info = next_info
            world_changed = world_before is not None and world_before != state_hash()
            next_room = env.get_player_location()
            if next_room is None and not done:
                raise RuntimeError("Jericho did not provide the room after the action.")
            navigation_result = None
            if world_map is not None:
                if done:
                    # Death/terminal relocation must not become a navigable route.
                    if next_room is not None:
                        world_map.current_location = world_map.add_standard_location(next_room.name, next_room.num)
                        world_map.current_location_name = next_room.name
                    navigator.cancel_plan("terminal_transition")
                    navigation_result = {"recorded_transition": False, "before_node": before_node,
                                         "after_node": world_map.current_location, "plan_status": "terminal_transition",
                                         "deviation": False, "failed_edge": None, "confirmed_edges": []}
                else:
                    navigation_result = navigator.observe_action_result(
                        action=action,
                        next_location=next_room.name,
                        next_location_id=next_room.num,
                        observation=observation,
                    )
                if action_source == "symbolic_navigation":
                    if navigation_result.get("failed_edge") or navigation_result.get("deviation"):
                        goal_control.cancel("route_failed_or_deviated")
                    if not carried_treasures(historian, observed_inventory):
                        control_metrics["navigation_without_treasure"] += 1
                after_node = world_map.current_location
                if world_changed and before_node == after_node and action.casefold() not in {"look", "inventory"}:
                    navigator.invalidate_local_evidence(after_node)
                after_label = world_map.display_location(after_node)
            score_after = next_info.get("score")
            score_delta = (
                score_after - score_before
                if isinstance(score_before, (int, float))
                and isinstance(score_after, (int, float))
                else None
            )
            next_context = observation if done else current_environment_context(env, observation)
            if world_map is None:
                if next_room is None:
                    after_node, after_label = before_node, before_label
                else:
                    after_node = str(next_room.num)
                    after_label = observed_room_title(next_room.name, next_context) or (
                        before_label if after_node == before_node else next_room.name
                    )
                    if after_label.casefold() == home_label.casefold():
                        home_node = after_node
            _, next_inventory_text = snapshot_sections(next_context)
            next_inventory = inventory_from_text(next_inventory_text)

            agent.add_history(
                observation=observation,
                action=action,
                location=after_label,
                score_delta=score_delta,
            )
            treasures_before = carried_treasures(historian, observed_inventory)
            historian.observe_step(
                before_node=before_node,
                before_location=before_label,
                action=action,
                after_node=after_node,
                after_location=after_label,
                score_before=score_before,
                score_after=score_after,
                reward=reward,
                observation_before=observation_before,
                observation=observation,
                navigation_result=navigation_result,
                goal=goal,
                action_source=action_source,
                actor_report=decision.get("report", {}),
                home_node=home_node,
                inventory_snapshot=next_inventory,
            )
            if (goal.get("kind") == "deposit_treasure" and next_inventory is not None
                    and score_delta is not None and score_delta > 0
                    and any(not any(historian._items_match(t, present) for present in next_inventory) for t in treasures_before)):
                control_metrics["treasure_deposits"] += 1
            ground_inventory(historian, next_inventory)

            next_rl_state = {
                **actor_rl_state,
                "location_id": str(after_node),
                "location": str(after_label),
                "observation": str(next_context or ""),
                "score": score_after,
                "home_route_status": (
                    navigator.route_status(home_node)
                    if navigator is not None and home_node is not None else home_route_status
                ),
                "inventory_items": list(
                    historian.agent_state.get("inventory_items", [])
                ),
                "treasure_candidates": list(
                    historian.agent_state.get("treasure_candidates", [])
                ),
                "ordinary_items": list(
                    historian.agent_state.get("ordinary_items", [])
                ),
                "useful_items": list(
                    historian.agent_state.get("useful_items", [])
                ),
                "puzzle_clues": list(
                    (decision.get("report") or {}).get(
                        "puzzle_clues", historian.agent_state.get("puzzle_clues", [])
                    )
                ),
            }
            no_progress = (
                str(before_node) == str(after_node)
                and (score_delta is None or score_delta <= 0)
                and " ".join(str(observation_before).split())
                    == " ".join(str(observation).split())
            )
            step_terminal = bool(done) or (
                isinstance(score_after, (int, float)) and score_after >= args.target_score
            )
            recorded_step = actor_experience.record_step(
                state=actor_rl_state,
                action=action,
                next_state=next_rl_state,
                score_delta=score_delta,
                environment_reward=reward,
                action_source=action_source,
                actor_decision=(action_source == "smart_actor"),
                candidate_actions=decision.get("candidate_actions", []),
                selection_info=decision.get("rl_selection", {}),
                action_logprob=decision.get("action_logprob"),
                no_progress=no_progress,
                terminal=step_terminal,
                truncated=(step + 1 == args.max_steps and not step_terminal),
            )
            last_recorded_state = next_rl_state
            critic_reward = actor_experience.step_reward(recorded_step)
            td_update = historian_rl_controller.observe_transition(
                state=actor_rl_state,
                action=action,
                reward=critic_reward,
                next_state=next_rl_state,
                terminal=step_terminal,
            )
            episode_logger.log_chat_event("historian_td_update", td_update)

            info = next_info
            print(f"Observation: {observation}")
            print(f"New location: {after_label}")
            print(f"Score: {score_after} (delta: {score_delta}) | reward: {reward}")
            if navigation_result is not None:
                print(f"Navigation status: {navigation_result['plan_status']}")
                world_map.save()
            historian.save()

            if isinstance(score_after, (int, float)) and score_after >= args.target_score:
                termination_reason = "target_score_reached"
                print(f"Target achieved: score {score_after} >= {args.target_score}")
                break
            if done:
                termination_reason = "environment_done"
                print("Environment ended.")
                break
            if recovery_count >= args.max_recovery_actions:
                termination_reason = "recovery_action_limit"
                print("Recovery action budget exhausted; saving an incomplete episode.")
                break

        if world_map is not None:
            world_map.save()
            world_map.print_graph()
        print(
            f"Episode finished: score={info.get('score')}, "
            f"moves={info.get('moves')}, reason={termination_reason}"
        )
        return info, termination_reason

    except KeyboardInterrupt:
        termination_reason = "interrupted"
        raise
    except Exception:
        termination_reason = "episode_exception"
        raise

    finally:
        try:
            final_score = info.get("score") if isinstance(info, dict) else None
            final_moves = info.get("moves") if isinstance(info, dict) else None
            if episode_started:
                control_metrics['provider_retries'] = getattr(episode_logger, 'provider_retries', 0)
                control_metrics['repeated_route_starts'] = (
                    sum(max(0, n - 1) for n in navigator.route_start_counts.values())
                    if navigator is not None else 0
                )
                control_metrics['goal_cancellations'] += sum(e['status'] == 'abandoned' for e in goal_control.events)
                for event in goal_control.events:
                    episode_logger.log_chat_event('goal_transition', event)
                historian.episode["control_metrics"] = control_metrics
                episode_logger.log_chat_event("control_metrics", control_metrics)
                finished_episode = historian.finish_episode(
                    final_score=final_score,
                    moves=final_moves,
                    termination_reason=termination_reason,
                )
                actor_rl_summary = actor_experience.finish_episode(
                    gamma=args.actor_rl_gamma,
                    termination_reason=termination_reason,
                    final_score=final_score,
                    bootstrap_value=(historian_rl_controller.value(last_recorded_state)
                                     if termination_reason in {"step_limit", "episode_exception", "interrupted",
                                                               "no_valid_actions", "no_safe_actions", "recovery_action_limit"}
                                     and last_recorded_state is not None and historian_rl_controller.enabled else None),
                )
                critic_summary = historian_rl_controller.finish_episode(
                    gamma=args.actor_rl_gamma,
                    termination_reason=termination_reason,
                )
                print("Actor experience update:", actor_rl_summary)
                print("Historian critic update:", critic_summary)
                episode_logger.write_actor_experiences(
                    actor_experience.last_episode_records
                )
                episode_logger.log_chat_event("actor_experience_episode_finalized", actor_rl_summary)
                episode_logger.log_chat_event("historian_critic_episode_finalized", critic_summary)
                if args.historian_review and finished_episode is not None and termination_reason not in {"episode_exception", "interrupted", "recovery_action_limit"}:
                    try:
                        review = historian.review_episode(
                            finished_episode,
                            client=historian_client,
                            model=args.historian_model or args.model,
                            temperature=args.historian_temperature,
                            max_tokens=args.historian_max_tokens,
                            event_logger=episode_logger.log_chat_event,
                            extra_body=historian_extra_body,
                            trial_episode_number=episode_number,
                        )
                        episode_logger.write_historian_review(review)
                    except Exception as review_exc:
                        # A review/logging problem should not erase a completed
                        # game episode or prevent the next episode from running.
                        print(
                            "Historian LLM review failed (no fallback used): "
                            f"{type(review_exc).__name__}: {review_exc}"
                        )
                    historian.save()
                if adaptive_system is not None and finished_episode is not None:
                    adaptive_system.finish_episode(finished_episode, info=info, reason=termination_reason,
                                                   event_logger=episode_logger.log_chat_event,
                                                   api_requests=episode_logger.api_requests,
                                                   episode_number=episode_number)
            if world_map is not None and (map_initialized or world_map.current_location is not None):
                world_map.save()
        finally:
            try:
                env.close()
            except Exception as close_exc:
                print(f"Could not close Jericho environment cleanly: {close_exc}")


def main(argv=None):
    args = parse_args(argv)
    if args.walkthrough:
        from MANGORL.walkthrough import run_walkthrough
        return run_walkthrough(args.rom_path, args.walkthrough_map_path, seed=args.seed)
    args = restore_arguments(args, list(argv if argv is not None else sys.argv[1:]))
    historian_client, historian_extra_body = (
        create_historian_connection(args) if args.historian_review or args.evolve else (None, None)
    )
    if isinstance(historian_client, AnthropicHistorian):
        historian_client.validate_model(args.historian_model)
    logger = TrialLogger(args.output_dir, args.trial_name, resume_directory=args.resume_trial)
    if args.fresh_memory:
        directory = logger.run_directory / "memory"
        for field, filename in (("map_path", "world_map.json"), ("historian_path", "historian.json"),
                                ("actor_experience_path", "actor_experiences.jsonl"),
                                ("historian_critic_path", "historian_td.json"),
                                ("evolution_path", "configurations.json")):
            setattr(args, field, str(directory / filename))
    metadata = vars(args).copy()
    metadata["api_key"] = "<redacted>" if args.api_key else None
    metadata["effective_historian_model"] = args.historian_model or args.model
    metadata["historian_transport"] = ("anthropic_messages" if isinstance(historian_client, AnthropicHistorian)
                                      else "openai_chat" if historian_client is not None else "disabled")
    rom = Path(args.rom_path)
    metadata["rom_sha256"] = hashlib.sha256(rom.read_bytes()).hexdigest() if rom.exists() else None
    metadata["observation_access"] = (
        "Jericho room IDs, " + ("world-change hashes, " if args.map_navigation else "")
        + "valid-action oracle, snapshot look and inventory"
    )
    metadata["action_guards"] = "preserve sole active carried light; restore light in observed darkness; temporary cycle exclusions"
    metadata["score_replay"] = "Actor reassesses historical score opportunities using current observations"
    if not args.resume_trial:
        logger.write_metadata(metadata)
    with trial_lock(logger.run_directory):
        checkpoint = TrialCheckpoint(logger.run_directory, args)
        if args.episodes < checkpoint.data['next_episode'] - 1:
            raise ValueError('Requested episode total is smaller than the already attempted count.')
        checkpoint.data['arguments']['episodes'] = args.episodes
        checkpoint.save()
        checkpoint.publish(args.episodes)

        print(f"Trial output directory: {logger.run_directory}")
        world_map = CognitiveMap(args.map_path) if args.map_navigation else None
        historian = Historian(
            args.historian_path,
            target_score=args.target_score,
            lessons_per_category=args.historian_lessons_per_category,
            map_navigation=args.map_navigation,
        )
        navigator = (
            NavigationController(world_map, risk_weight=args.route_risk_weight if args.nsi else 0)
            if world_map is not None else None
        )
        state_encoder = FrozenStateEncoder(args.memory_embedding_model) if args.memory_embedding_model else None
        actor_experience = ActorExperienceMemory(
            args.actor_experience_path,
            environment_reward_weight=args.actor_rl_environment_reward_weight,
            no_progress_penalty=args.actor_rl_no_progress_penalty,
            state_encoder=state_encoder,
        )
        historian_critic = HistorianRLController(
            args.historian_critic_path,
            enabled=args.historian_critic,
            gamma=args.actor_rl_gamma,
            learning_rate=args.historian_critic_learning_rate,
            state_encoder=state_encoder,
        )
        actor_rl_controller = ActorRLController(
            actor_experience,
            enabled=args.actor_rl,
            beta=args.actor_rl_beta,
            exploration_coefficient=args.actor_rl_exploration,
            rank_penalty=args.actor_rl_rank_penalty,
            top_k=args.actor_rl_top_k,
            min_similarity=args.actor_rl_min_similarity,
            value_controller=historian_critic if args.historian_critic else None,
            seed=args.seed,
        )
        agent = None
        map_initialized = False
        adaptive_system = None
        outcomes = checkpoint.data["outcomes"]
        consecutive_failures = 0

        def record_outcome(episode_number, status, **details):
            outcome = {"episode": episode_number, "status": status, **details}
            checkpoint.finish(outcome)
            checkpoint.publish(args.episodes)

        for episode_number in range(checkpoint.data["next_episode"], args.episodes + 1):
            checkpoint.begin(episode_number)
            with logger.episode(episode_number) as episode_logger:
                try:
                    if agent is None:
                        agent = create_agent(args, event_logger=episode_logger.log_chat_event)
                        adaptive_system = AdaptiveSystem(args, actor=agent, historian=historian,
                            critic=historian_critic, policy=actor_rl_controller, navigator=navigator,
                            output_dir=logger.run_directory, historian_client=historian_client,
                            historian_extra_body=historian_extra_body)
                    result, reason = run_episode(
                        args=args,
                        episode_number=episode_number,
                        world_map=world_map,
                        historian=historian,
                        navigator=navigator,
                        agent=agent,
                        episode_logger=episode_logger,
                        actor_experience=actor_experience,
                        actor_rl_controller=actor_rl_controller,
                        historian_rl_controller=historian_critic,
                        trial_id=logger.run_directory.name,
                        map_initialized=map_initialized,
                        adaptive_system=adaptive_system,
                        historian_client=historian_client,
                        historian_extra_body=historian_extra_body,
                    )
                    map_initialized = True
                    consecutive_failures = 0
                    record_outcome(episode_number,
                                   "incomplete" if reason in {"no_safe_actions", "no_valid_actions", "recovery_action_limit"} else "completed",
                                   termination_reason=reason, score=result.get("score"),
                                   degraded=bool(historian.memory["episodes"][-1].get("control_metrics", {}).get("recovery_actions")))
                except KeyboardInterrupt:
                    record_outcome(episode_number, "interrupted", termination_reason="interrupted")
                    raise
                except Exception as exc:
                    print(f"Episode {episode_number} failed: {type(exc).__name__}: {exc}")
                    consecutive_failures += 1
                    recoverable = recoverable_episode_error(exc)
                    can_continue = (recoverable and not args.fail_fast and
                                    consecutive_failures < args.max_consecutive_episode_failures)
                    details = {"termination_reason": "episode_exception", "error_type": type(exc).__name__,
                               "error": str(exc), "recoverable": recoverable,
                               "continuing": can_continue and episode_number < args.episodes}
                    record_outcome(episode_number, "failed", **details)
                    episode_logger.log_chat_event("episode_failure", details)
                    if not can_continue:
                        raise
                    # run_episode saved observed transitions before propagating.
                    # Keep the in-memory graph; loading again would lose episode state.
                    map_initialized = world_map is not None and world_map.current_location is not None
                    print("Incomplete episode saved and excluded from score metrics. "
                          + ("Continuing with the next episode." if details["continuing"] else "No episodes remain."))

        completed = sum(item["status"] == "completed" for item in outcomes)
        print(f"Trial finished: {completed}/{args.episodes} completed episodes; "
              f"{len(outcomes) - completed} incomplete. Logs saved in: {logger.run_directory}")
        return logger.run_directory


if __name__ == "__main__":
    freeze_support()
    main()
