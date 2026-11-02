"""Zork I: LLM Actor + Symbolic Navigator + persistent Historian."""
from __future__ import annotations

import argparse
import os
from multiprocessing import freeze_support

from config import load_project_env

load_project_env()  # Must run before argparse defaults read os.environ.

from historian import Historian
from navigation_controller import NavigationController
from smart_agent import SmartAgent, create_openai_client
from trial_logger import TrialLogger
from world_map import WorldMap

DEFAULT_BASE_URL = "http://cs-steverw.umflint.edu:8080/api"


class SafeDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults for reproducibility without ever printing credentials."""

    SECRET_DESTINATIONS = {"api_key", "historian_api_key"}

    def _get_help_string(self, action):
        if action.dest in self.SECRET_DESTINATIONS:
            return action.help
        return super()._get_help_string(action)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for a reproducible trial run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the neuro-symbolic Zork I agent. Each invocation creates a "
            "unique trial folder, with separate console/chat logs, a Historian "
            "review, and six lessons for every completed episode."
        ),
        formatter_class=SafeDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent-type", choices=("smart",), default="smart",
        help="Actor implementation to use (only 'smart' is currently implemented).",
    )
    parser.add_argument(
        "--model", default=os.getenv("LLM_MODEL", "gemma4:31b"),
        help="LLM model identifier passed to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("OPENAI_API_KEY"),
        help=(
            "API key (prefer OPENAI_API_KEY; command-line values may be visible "
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
        "--max-tokens", type=int, default=300,
        help="Maximum response tokens requested from the LLM.",
    )
    parser.add_argument(
        "--historian-review", action=argparse.BooleanOptionalAction,
        default=True,
        help="Have the Historian review each finished episode and update six trial lessons.",
    )
    parser.add_argument(
        "--historian-model", default=os.getenv("HISTORIAN_MODEL"),
        help="Model used for episode reviews (defaults to --model).",
    )
    parser.add_argument(
        "--historian-base-url", default=os.getenv("HISTORIAN_BASE_URL") or None,
        help=(
            "Optional separate Historian API base URL. Defaults to --base-url "
            "(OPENAI_BASE_URL)."
        ),
    )
    parser.add_argument(
        "--historian-api-key", default=os.getenv("HISTORIAN_API_KEY") or None,
        help=(
            "Optional separate Historian API key. Defaults to --api-key "
            "(OPENAI_API_KEY)."
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
        "--openwebui-parent-id-null",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("OPENWEBUI_PARENT_ID_NULL", "1") == "1",
        help="Send parent_id: null for compatible Open WebUI /api endpoints.",
    )
    return parser


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv)
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
    if args.temperature < 0:
        raise SystemExit("--temperature cannot be negative")
    if args.historian_temperature < 0:
        raise SystemExit("--historian-temperature cannot be negative")
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
        )
    raise ValueError(f"Unsupported agent type: {args.agent_type}")


def create_historian_connection(args, agent):
    """Return the Historian review client and endpoint-specific request options.

    By default the Historian reuses the Actor's OpenAI client. HISTORIAN_API_KEY
    and HISTORIAN_BASE_URL in .env (or their CLI equivalents) can point reviews
    at a separate compatible endpoint.
    """
    base_url = args.historian_base_url or args.base_url
    api_key = args.historian_api_key or args.api_key
    effective_key = api_key
    if not effective_key and base_url.rstrip("/").endswith("/v1"):
        effective_key = "ollama"

    same_url = base_url.rstrip("/") == agent.base_url.rstrip("/")
    same_key = effective_key == agent.api_key
    if same_url and same_key:
        client = agent.client
        extra_body = agent.extra_body
    else:
        client = create_openai_client(api_key=api_key, base_url=base_url)
        extra_body = (
            {"parent_id": None}
            if args.openwebui_parent_id_null
            and base_url.rstrip("/").endswith("/api")
            else None
        )
    return client, extra_body


def run_episode(args, episode_number, world_map, historian, navigator, agent,
                episode_logger, map_initialized=False):
    """Run one fresh Jericho episode while preserving trial-level memory."""
    # Import lazily so CLI help and parser tests work without Jericho installed.
    try:
        from jericho import FrotzEnv
    except ImportError as exc:
        raise RuntimeError(
            "Jericho is required to run the game. Install it with `pip install jericho`."
        ) from exc

    env = FrotzEnv(args.rom_path)
    review_client, review_extra_body = (
        create_historian_connection(args, agent)
        if args.historian_review else (None, None)
    )
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

            action = None
            action_source = "smart_actor"

            if goal.get("kind") == "score_action":
                action = _match_action(goal.get("suggested_action"), valid_actions)
                if action:
                    action_source = "historian_observed_score"

            elif goal.get("kind") in {"navigate_to_home", "navigate_score_goal"}:
                target_node = goal.get("target_node")
                if target_node is not None:
                    if goal.get("kind") == "navigate_to_home":
                        plan = navigator.plan_to_home(target_node)
                    else:
                        plan = navigator.ensure_plan(
                            target_node, require_confirmed=True
                        )
                    if plan is not None:
                        action = navigator.next_action(valid_actions)
                        if action:
                            action_source = "symbolic_navigation"

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
                )
                action = decision["action"]
                historian.record_agent_report(decision.get("report", {}))
            else:
                decision = {
                    "action": action,
                    "report": {},
                    "raw_response": "symbolic navigation or Historian-observed score action",
                }

            print(f"Location: {current_label}")
            print(f"Goal: {goal.get('description')}")
            if home_node is not None:
                print(f"Home route status this episode: {home_route_status}")
            print(f"Action source: {action_source}")
            print(f"Action: {action}")

            observation_before = observation
            score_before = info.get("score")
            before_node = current_node
            before_label = current_label

            observation, reward, done, next_info = env.step(action)
            next_room = env.get_player_location()
            if next_room is None:
                raise RuntimeError("Jericho did not provide the room after the action.")

            navigation_result = navigator.observe_action_result(
                action=action,
                next_location=next_room.name,
                next_location_id=next_room.num,
                observation=observation,
            )
            after_node = world_map.current_location
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
            )

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
            if args.historian_review and finished_episode is not None:
                try:
                    review = historian.review_episode(
                        finished_episode,
                        client=review_client,
                        model=args.historian_model or args.model,
                        temperature=args.historian_temperature,
                        max_tokens=args.historian_max_tokens,
                        event_logger=episode_logger.log_chat_event,
                        extra_body=review_extra_body,
                        trial_episode_number=episode_number,
                    )
                    episode_logger.write_historian_review(review)
                except Exception as review_exc:
                    # A review/logging problem should not erase a completed
                    # game episode or prevent the next episode from running.
                    print(
                        "Historian review failed unexpectedly: "
                        f"{type(review_exc).__name__}: {review_exc}"
                    )
                historian.save()
        if map_initialized or world_map.current_location is not None:
            world_map.save()
        try:
            env.close()
        except Exception as close_exc:
            print(f"Could not close Jericho environment cleanly: {close_exc}")


def main(argv=None):
    args = parse_args(argv)
    logger = TrialLogger(args.output_dir, args.trial_name)
    metadata = vars(args).copy()
    metadata["api_key"] = "<redacted>" if args.api_key else None
    metadata["historian_api_key"] = "<redacted>" if args.historian_api_key else None
    metadata["effective_historian_model"] = args.historian_model or args.model
    metadata["effective_historian_base_url"] = args.historian_base_url or args.base_url
    logger.write_metadata(metadata)

    print(f"Trial output directory: {logger.run_directory}")
    world_map = WorldMap(args.map_path)
    historian = Historian(args.historian_path, target_score=args.target_score)
    navigator = NavigationController(world_map)
    agent = None
    map_initialized = False

    for episode_number in range(1, args.episodes + 1):
        with logger.episode(episode_number) as episode_logger:
            if agent is None:
                agent = create_agent(args, event_logger=episode_logger.log_chat_event)
            try:
                _, _ = run_episode(
                    args=args,
                    episode_number=episode_number,
                    world_map=world_map,
                    historian=historian,
                    navigator=navigator,
                    agent=agent,
                    episode_logger=episode_logger,
                    map_initialized=map_initialized,
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
