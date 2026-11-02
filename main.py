"""Zork I: LLM Actor + Symbolic Navigator + persistent Historian."""
from __future__ import annotations

import os
from multiprocessing import freeze_support

from jericho import FrotzEnv

from historian import Historian
from navigation_controller import NavigationController
from smart_agent import SmartAgent
from world_map import WorldMap

ROM_PATH = os.getenv(
    "ZORK_ROM_PATH",
    "z-machine-games-master/jericho-game-suite/zork1.z5",
)
MAX_STEPS = int(os.getenv("ZORK_MAX_STEPS", "500"))
TARGET_SCORE = int(os.getenv("ZORK_TARGET_SCORE", "76"))
MAP_PATH = os.getenv("ZORK_MAP_PATH", "memory/world_map.json")
HISTORIAN_PATH = os.getenv("ZORK_HISTORIAN_PATH", "memory/historian.json")


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


def main():
    env = FrotzEnv(ROM_PATH)
    historian = Historian(
        memory_path=HISTORIAN_PATH,
        target_score=TARGET_SCORE,
    )
    world_map = WorldMap(MAP_PATH)

    base_url = os.getenv(
        "OPENAI_BASE_URL", "http://cs-steverw.umflint.edu:8080/api"
    )
    extra_body = (
        {"parent_id": None}
        if os.getenv("OPENWEBUI_PARENT_ID_NULL", "1") == "1"
        and base_url.rstrip("/").endswith("/api")
        else None
    )
    navigator = NavigationController(world_map)

    agent = None
    info = {}
    map_initialized = False
    episode_started = False
    termination_reason = "initialization_failed"

    try:
        agent = SmartAgent(base_url=base_url, extra_body=extra_body)
        observation, info = env.reset()
        room = env.get_player_location()
        if room is None:
            raise RuntimeError("Jericho did not provide the starting room.")

        world_map.instantiate()
        map_initialized = True
        current_node = world_map.start_episode(room.name, room.num)
        navigator.start_episode()
        historian.start_episode(
            start_node=current_node,
            start_location=world_map.display_location(current_node),
            initial_score=info.get("score"),
        )
        episode_started = True
        termination_reason = "step_limit"

        print(f"Initial observation: {observation}")
        print(f"Starting location: {world_map.display_location(current_node)}")
        print(f"Target score: {TARGET_SCORE} (must exceed 75)")

        for step in range(MAX_STEPS):
            print(f"\n========== STEP {step + 1}/{MAX_STEPS} ==========")
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
            if isinstance(current_score, (int, float)) and current_score >= TARGET_SCORE:
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

            goal = historian.choose_goal(
                current_node=current_node,
                current_location=current_label,
                valid_actions=valid_actions,
                known_routes=known_routes,
                known_outgoing_actions=known_outgoing,
                current_score=current_score,
                home_node=home_node,
                home_label="Living Room",
            )
            guidance = historian.build_guidance(
                current_node,
                current_label,
                current_score=current_score,
            )
            avoid_actions = historian.get_avoid_actions(current_node)

            action = None
            action_source = "smart_actor"

            # If the Historian has observed a score-positive action at this
            # location, try that exact action when Jericho offers it.
            if goal.get("kind") == "score_action":
                action = _match_action(goal.get("suggested_action"), valid_actions)
                if action:
                    action_source = "historian_observed_score"

            # Symbolic navigation handles known routes to home or known
            # score-positive locations. If no route exists, the Actor gets
            # the strategic goal and must investigate the puzzle/alternate path.
            elif goal.get("kind") in {"navigate_to_home", "navigate_score_goal"}:
                target_node = goal.get("target_node")
                if target_node is not None:
                    plan = navigator.ensure_plan(target_node)
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
                    target_score=TARGET_SCORE,
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
            )

            info = next_info
            print(f"Observation: {observation}")
            print(f"New location: {after_label}")
            print(f"Score: {score_after} (delta: {score_delta}) | reward: {reward}")
            print(f"Navigation status: {navigation_result['plan_status']}")

            world_map.save()
            historian.save()

            if isinstance(score_after, (int, float)) and score_after >= TARGET_SCORE:
                termination_reason = "target_score_reached"
                print(f"Target achieved: score {score_after} >= {TARGET_SCORE}")
                break
            if done:
                termination_reason = "environment_done"
                print("Environment ended.")
                break

        world_map.save()
        world_map.print_graph()

    finally:
        final_score = info.get("score") if isinstance(info, dict) else None
        final_moves = info.get("moves") if isinstance(info, dict) else None
        if episode_started:
            historian.finish_episode(
                final_score=final_score,
                moves=final_moves,
                termination_reason=termination_reason,
            )
        if map_initialized:
            world_map.save()
        env.close()


if __name__ == "__main__":
    freeze_support()
    main()
