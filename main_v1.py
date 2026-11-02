from jericho import FrotzEnv
from multiprocessing import freeze_support
import time

from my_solution.world_map_v1 import WorldMap
from simple_agent import SimpleAgent

ROM_PATH = "z-machine-games-master/jericho-game-suite/zork1.z5"
MAX_STEPS = 100
MODEL = "gemma4:31b"

def walkthrough(env, world_map):
        walkthrough = env.get_walkthrough()
        for action in walkthrough:
        
            # print(f'\n===== STEP {step+1} =====')
            current_room = env.get_player_location()
            if current_room is None:
                raise RuntimeError("Jericho did not provide the current room.")
            # location = current_room.name
            obs, reward, done, info = env.step(action)
            print(f'Observation: {obs}')
            new_room = env.get_player_location()
            if new_room is None:
                raise RuntimeError("Jericho did not provide the room after the action.")
            print(
                        f"{action}: "
                        f"{current_room.name}#{current_room.num} -> "
                        f"{new_room.name}#{new_room.num}"
                    )
            new_location = new_room.name
            world_map.record_step(action, new_location, observation=obs, next_location_id=new_room.num)
            # location = new_location
            print('Total Score', info['score'], 'Moves', info['moves'])
            time.sleep(0.75)  # Optional: slow down for readability

def get_valid_actions(env):
    """Retrieve valid actions without starting Jericho's worker processes."""
    try:
        actions = env.get_valid_actions(use_parallel=False)
        if actions:
            return actions
    except Exception as exc:
        print(f"Jericho action generation failed: {exc}")

    # Last-resort fallback; less capable than Jericho's generated actions.
    return [
        "north", "south", "east", "west",
        "northeast", "northwest", "southeast", "southwest",
        "up", "down", "look", "inventory"
    ]


def main():
    agent = SimpleAgent(model=MODEL)
    env = FrotzEnv(ROM_PATH)

    try:
        observation, info = env.reset()
        print(f"Initial observation: {observation}")

        world_map = WorldMap("memory/world_map.json")
        world_map.instantiate()

        start_room = env.get_player_location()
        if start_room is None:
            raise RuntimeError("Jericho did not provide the starting room.")

        world_map.start_episode(start_room.name, start_room.num)

        for step in range(MAX_STEPS):
            print(f"\n===== STEP {step + 1} =====")

            current_room = env.get_player_location()
            if current_room is None:
                raise RuntimeError("Jericho did not provide the current room.")

            location_name = current_room.name
            current_node = world_map.resolve_location(
                location_name,
                location_id=current_room.num,
            )
            if current_node is None:
                # Defensive recovery if the current room has not yet been added.
                current_node = world_map.add_standard_location(
                    location_name,
                    current_room.num,
                )

            # Give the LLM readable names that disambiguate repeated room names.
            actor_location = world_map.display_location(current_node)
            known_locations = world_map.get_known_locations()
            known_routes = world_map.get_reachable_routes(
                start=current_node,
                max_destinations=10,
                start_id=current_room.num,
            )

            valid_actions = get_valid_actions(env)
            print(f"Current location: {actor_location}")
            print(f"Valid actions: {valid_actions}")

            action = agent.choose_action(
                observation=observation,
                current_location=actor_location,
                valid_actions=valid_actions,
                known_locations=known_locations,
                known_routes=known_routes,
            )
            print(f"Action: {action}")

            observation, reward, done, info = env.step(action)
            print(f"Observation: {observation}")

            new_room = env.get_player_location()
            if new_room is None:
                raise RuntimeError("Jericho did not provide the room after the action.")

            new_location = new_room.name
            agent.add_history(
                observation=observation,
                action=action,
                location=new_location,
            )

            moved_to_new_room = world_map.record_step(
                action=action,
                next_location=new_location,
                observation=observation,
                next_location_id=new_room.num,
            )

            if moved_to_new_room:
                print(
                    "Recorded transition: "
                    f"{actor_location} --[{action}]--> "
                    f"{world_map.display_location(world_map.current_location)}"
                )
            else:
                print("No room-identity change; no navigation edge recorded.")

            print("Total Score", info.get("score"), "Moves", info.get("moves"))

            # Persist after each action so discoveries survive an interrupted run.
            world_map.save()

            if done:
                print("Episode ended.")
                break

        world_map.save()
        world_map.print_graph()

    finally:
        env.close()


if __name__ == "__main__":
    freeze_support()
    main()



# def get_valid_actions(env):
#     """Retrieve valid actions from the environment."""
#     try:
#         valid_actions = env.get_valid_actions(use_parallel=False)
#         if valid_actions:
#             return valid_actions
#     except Exception as e:
#         print(f"Jericho parser failed: {e}")
#     valid_actions = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'up', 'down', 'look', 'inventory']
#     return valid_actions

# MAX_STEPS = 200

# def main():
#     # agent = SimpleAgent()
#     env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")
#     obs, info = env.reset()
#     print(f'Initial observation: {obs}')
#     starting_room = env.get_player_location()
#     if starting_room is None:
#         raise RuntimeError("Jericho did not provide the starting room.")
#     starting_location = starting_room.name
#     world_map = WorldMap("memory/world_map.json")
#     world_map.instantiate()
#     world_map.start_episode(starting_location, starting_room.num)



#     world_map.save()

#     world_map.print_graph()

#     env.close()


# if __name__ == "__main__":
#     freeze_support()
#     main()
