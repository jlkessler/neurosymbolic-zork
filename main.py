"""Zork I: LLM Actor + Symbolic Navigator + persistent Historian."""
from __future__ import annotations

# Load project-local .env before reading configuration defaults.
import config  # noqa: F401

import argparse
import os
import hashlib
import math
from pathlib import Path
from multiprocessing import freeze_support

from MANGORL.actor_experience import ActorExperienceMemory
from MANGORL.actor_rl_controller import ActorRLController
from MANGORL.historian import Historian
from MANGORL.historian_rl_controller import HistorianRLController
from MANGORL.navigation_controller import NavigationController
from MANGORL.smart_agent import SmartAgent
from MANGORL.trial_logger import TrialLogger
from MANGORL.world_map import WorldMap
from MANGORL.cognitive_map import CognitiveMap
from MANGORL.adaptive_system import AdaptiveSystem
from MANGORL.semantic_memory import FrozenStateEncoder
from MANGORL.policy_logits import LogitsUnavailable

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
        "--model", default=os.getenv("ACTOR_MODEL", "gemma4:31b"),
        help="LLM model identifier passed to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url", default=os.getenv("ACTOR_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help=(
            "API key (prefer ACTOR_API_KEY; command-line values may be visible "
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
        help="Sampling temperature for the Historian's episode-review request.",
    )
    parser.add_argument(
        "--historian-max-tokens", type=int, default=900,
        help="Maximum completion tokens for each Historian review.",
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
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nsi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-risk-weight", type=float, default=2.0)
    parser.add_argument("--evolve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evolution-path", default="memory/configurations.json")
    parser.add_argument("--fresh-memory", action="store_true",
                        help="Use empty memories inside this trial directory; keep existing memory files intact.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-embedding-model", default=None,
                        help="Optional frozen SentenceTransformer model/path; omitted means lexical retrieval.")
    parser.add_argument("--score-scale", type=float, default=350.0,
                        help="Maximum game score used for normalized AUC and configuration fitness.")
    return parser


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv, namespace=RunArguments())
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be greater than zero")
    if args.target_score < 0:
        raise SystemExit("--target-score cannot be negative")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than zero")
    if args.history_length <= 0:
        raise SystemExit("--history-length must be greater than zero")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be greater than zero")
    if args.historian_max_tokens <= 0:
        raise SystemExit("--historian-max-tokens must be greater than zero")
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
        if actions:
            return actions
    except Exception as exc:
        print(f"Jericho action generation failed: {exc}")

    return [
        "north", "south", "east", "west", "northeast", "northwest",
        "southeast", "southwest", "up", "down", "look", "inventory", "pray",
    ]


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
    episode_started = False
    termination_reason = "initialization_failed"

    try:
        observation, info = env.reset()
        room = env.get_player_location()
        if room is None:
            raise RuntimeError("Jericho did not provide the starting room.")

        if not map_initialized:
            world_map.instantiate()
        current_node = world_map.start_episode(room.name, room.num)
        navigator.start_episode()
        historian.start_episode(
            start_node=current_node,
            start_location=world_map.display_location(current_node),
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
        print(f"Starting location: {world_map.display_location(current_node)}")
        print(f"Target score: {args.target_score}")
        print(f"Model: {args.model} | Agent type: {args.agent_type}")
        print(f"Maximum steps: {args.max_steps}")
        print("\n" + historian.format_trial_lessons())

        for step in range(args.max_steps):
            print(f"\n========== STEP {step + 1}/{args.max_steps} ==========")
            room = env.get_player_location()
            if room is None:
                raise RuntimeError("Jericho did not provide the current room.")

            current_node = world_map.resolve_location(room.name, room.num)
            if current_node is None:
                current_node = world_map.add_standard_location(room.name, room.num)
                world_map.current_location = current_node
                world_map.current_location_name = room.name

            current_label = world_map.display_location(current_node)
            current_score = info.get("score")
            if isinstance(current_score, (int, float)) and current_score >= args.target_score:
                termination_reason = "target_score_reached"
                break

            valid_actions = get_valid_actions(env)
            if not valid_actions:
                termination_reason = "no_valid_actions"
                break

            known_routes = world_map.get_reachable_routes(
                start=current_node,
                max_destinations=max(1, world_map.graph.number_of_nodes()),
            )
            known_outgoing = get_known_outgoing_actions(world_map, current_node)
            home_node = find_living_room_node(world_map)
            home_route_status = (
                navigator.route_status(home_node)
                if home_node is not None else "unknown"
            )

            goal = historian.choose_goal(
                current_node=current_node,
                current_location=current_label,
                valid_actions=valid_actions,
                known_routes=known_routes,
                known_outgoing_actions=known_outgoing,
                current_score=current_score,
                home_node=home_node,
                home_label="Living Room",
                home_route_status=home_route_status,
            )
            guidance = historian.build_guidance(
                current_node,
                current_label,
                current_score=current_score,
            )
            avoid_actions = historian.get_avoid_actions(current_node)
            actor_rl_state = build_actor_rl_state(
                current_node=current_node,
                current_label=current_label,
                observation=observation,
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

            if goal.get("kind") == "score_action":
                action = _match_action(goal.get("suggested_action"), valid_actions)
                if action:
                    action_source = "historian_observed_score"

            elif goal.get("kind") in {
                "navigate_to_home", "establish_return_route", "navigate_score_goal"
            }:
                target_node = goal.get("target_node")
                if target_node is not None:
                    if goal.get("kind") in {"navigate_to_home", "establish_return_route"}:
                        plan = navigator.plan_to_home(target_node)
                    else:
                        # Previously productive rooms are reached through
                        # historical candidate paths. The navigator probes
                        # each transition and cancels on the first failure.
                        plan = navigator.ensure_plan(
                            target_node, require_confirmed=False
                        )
                    if plan is not None:
                        action = navigator.next_action(valid_actions)
                        if action:
                            action_source = "symbolic_navigation"

            if action is not None and args.nsi and not world_map.action_feasibility(
                actor_rl_state, [action], navigator, valid_actions
            )[0]:
                episode_logger.log_chat_event("symbolic_action_rejected", {"action": action, "reason": "zero NSI feasibility"})
                action = None
                action_source = "smart_actor"

            if action is None:
                navigator.cancel_plan("actor_or_puzzle_decision")
                decision = agent.choose_decision(
                    observation=observation,
                    current_location=current_label,
                    valid_actions=valid_actions,
                    current_score=current_score,
                    target_score=args.target_score,
                    known_locations=world_map.get_known_locations(),
                    known_routes=known_routes,
                    goal=goal,
                    historian_guidance=guidance,
                    avoid_actions=avoid_actions,
                    preferred_actions=goal.get("preferred_actions", []),
                    navigation_context=navigator.navigation_context(),
                    rl_guidance=rl_guidance,
                )
                base_action = decision["action"]
                candidates = decision.get("candidate_actions", [base_action])
                if args.actor_policy_mode == "logprobs" and decision.get("base_logits") is None:
                    raise LogitsUnavailable("The Actor did not supply the required native label logprobs.")
                feasibility = world_map.action_feasibility(
                    actor_rl_state, candidates, navigator, valid_actions
                ) if args.nsi else [1.0] * len(candidates)
                if not any(feasibility):
                    # Ask again over feasible commands rather than executing a
                    # masked action or fabricating a model probability.
                    permitted = [a for a, p in zip(valid_actions, world_map.action_feasibility(
                        actor_rl_state, valid_actions, navigator, valid_actions)) if p > 0]
                    if not permitted:
                        raise RuntimeError("NSI found no consistent available command.")
                    decision = agent.choose_decision(
                        observation=observation, current_location=current_label,
                        valid_actions=permitted, current_score=current_score, target_score=args.target_score,
                        goal=goal, historian_guidance=guidance, rl_guidance=rl_guidance,
                        navigation_context=navigator.navigation_context(),
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
            else:
                decision = {
                    "action": action,
                    "report": {},
                    "decision_rationale": "",
                    "raw_response": "symbolic navigation or Historian-observed score action",
                }
                episode_logger.log_chat_event("non_llm_action_selection", {
                    "location": current_label,
                    "goal": goal,
                    "action_source": action_source,
                    "selected_action": action,
                    "navigation_context": navigator.navigation_context(),
                    "note": "No LLM rationale exists because this command was selected by the symbolic navigator or deterministic Historian score policy.",
                })

            print(f"Location: {current_label}")
            print(f"Goal: {goal.get('description')}")
            for sequence_step in goal.get("sequence", [])[:5]:
                print(
                    f"  {sequence_step.get('order', '?')}. "
                    f"{sequence_step.get('goal', 'Next objective')}: "
                    f"{sequence_step.get('details', '')}"
                )
            if home_node is not None:
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

            state_hash = getattr(env, "get_world_state_hash", None)
            world_before = state_hash() if callable(state_hash) else None
            observation, reward, done, next_info = env.step(action)
            info = next_info
            world_changed = world_before is not None and world_before != state_hash()
            next_room = env.get_player_location()
            if next_room is None and not done:
                raise RuntimeError("Jericho did not provide the room after the action.")
            if next_room is None:
                # A terminal observation may have no player object. Preserve
                # the last known room without inventing a failed map edge.
                navigator.cancel_plan("terminal_location_unavailable")
                navigation_result = {"recorded_transition": False, "before_node": before_node,
                                     "after_node": before_node, "plan_status": "terminal_location_unavailable",
                                     "deviation": False, "failed_edge": None, "confirmed_edges": []}
            else:
                navigation_result = navigator.observe_action_result(
                    action=action,
                    next_location=next_room.name,
                    next_location_id=next_room.num,
                    observation=observation,
                )
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

            agent.add_history(
                observation=observation,
                action=action,
                location=after_label,
                score_delta=score_delta,
            )
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
            )

            next_rl_state = {
                **actor_rl_state,
                "location_id": str(after_node),
                "location": str(after_label),
                "observation": str(observation or ""),
                "score": score_after,
                "home_route_status": navigator.route_status(home_node) if home_node is not None else "unknown",
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

        world_map.save()
        world_map.print_graph()
        print(
            f"Episode finished: score={info.get('score')}, "
            f"moves={info.get('moves')}, reason={termination_reason}"
        )
        return info, termination_reason

    except Exception:
        termination_reason = "episode_exception"
        raise

    finally:
        final_score = info.get("score") if isinstance(info, dict) else None
        final_moves = info.get("moves") if isinstance(info, dict) else None
        if episode_started:
            finished_episode = historian.finish_episode(
                final_score=final_score,
                moves=final_moves,
                termination_reason=termination_reason,
            )
            actor_rl_summary = actor_experience.finish_episode(
                gamma=args.actor_rl_gamma,
                termination_reason=termination_reason,
                final_score=final_score,
                bootstrap_value=(historian_rl_controller.value(next_rl_state)
                                 if termination_reason == "step_limit" and "next_rl_state" in locals() else 0.0),
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
            if args.historian_review and finished_episode is not None and termination_reason != "episode_exception":
                try:
                    review = historian.review_episode(
                        finished_episode,
                        client=getattr(agent, "client", None),
                        model=args.historian_model or args.model,
                        temperature=args.historian_temperature,
                        max_tokens=args.historian_max_tokens,
                        event_logger=episode_logger.log_chat_event,
                        extra_body=getattr(agent, "extra_body", None),
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
                                               api_requests=episode_logger.api_requests)
        if map_initialized or world_map.current_location is not None:
            world_map.save()
        try:
            env.close()
        except Exception as close_exc:
            print(f"Could not close Jericho environment cleanly: {close_exc}")


def main(argv=None):
    args = parse_args(argv)
    logger = TrialLogger(args.output_dir, args.trial_name)
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
    rom = Path(args.rom_path)
    metadata["rom_sha256"] = hashlib.sha256(rom.read_bytes()).hexdigest() if rom.exists() else None
    metadata["observation_access"] = "Jericho room IDs, world-change hashes and valid-action oracle; not text-only play"
    logger.write_metadata(metadata)

    print(f"Trial output directory: {logger.run_directory}")
    world_map = CognitiveMap(args.map_path)
    historian = Historian(
        args.historian_path,
        target_score=args.target_score,
        lessons_per_category=args.historian_lessons_per_category,
    )
    navigator = NavigationController(world_map, risk_weight=args.route_risk_weight if args.nsi else 0)
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

    for episode_number in range(1, args.episodes + 1):
        with logger.episode(episode_number) as episode_logger:
            if agent is None:
                agent = create_agent(args, event_logger=episode_logger.log_chat_event)
                adaptive_system = AdaptiveSystem(args, actor=agent, historian=historian,
                    critic=historian_critic, policy=actor_rl_controller, navigator=navigator,
                    output_dir=logger.run_directory)
            try:
                _, _ = run_episode(
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
                )
                map_initialized = True
            except Exception as exc:
                print(f"Episode {episode_number} failed: {type(exc).__name__}: {exc}")
                raise

    print(f"Trial complete. Logs saved in: {logger.run_directory}")
    return logger.run_directory


if __name__ == "__main__":
    freeze_support()
    main()
