from jericho import *
import time
import json
from multiprocessing import freeze_support

from world_map import WorldMap
from simple_agent import SimpleAgent

def get_valid_actions(env):
    """Retrieve valid actions from the environment."""
    try:
        valid_actions = env.get_valid_actions(use_parallel=False)
        if valid_actions:
            return valid_actions
    except Exception as e:
        print(f"Jericho parser failed: {e}")
    valid_actions = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'up', 'down', 'look', 'inventory']
    return valid_actions

MAX_STEPS = 200

def main():
    agent = SimpleAgent()
    env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")
    obs, info = env.reset()
    print(f'Initial observation: {obs}')
    starting_room = env.get_player_location()
    if starting_room is None:
        raise RuntimeError("Jericho did not provide the starting room.")
    starting_location = starting_room.name
    # walkthrough = env.get_walkthrough()
    world_map = WorldMap("memory/world_map.json")
    world_map.instantiate()
    world_map.start_episode(starting_location, starting_room.num)
    for step in range(MAX_STEPS):
        
        print(f'\n===== STEP {step+1} =====')
        current_room = env.get_player_location()
        if current_room is None:
            raise RuntimeError("Jericho did not provide the current room.")
        location = current_room.name
        valid_actions = get_valid_actions(env)
        print(f'Valid actions: {valid_actions}')
        known_locations = [
            node
            for node, data in world_map.graph.nodes(data=True)
            if data.get("node_type") == "location"
        ]

        action = agent.choose_action(
            observation=obs,
            current_location=location,
            valid_actions=valid_actions,
            known_locations=known_locations
        )
        print(f'Action: {action}')
        
        obs, reward, done, info = env.step(action)
        print(f'Observation: {obs}')
        new_room = env.get_player_location()
        if new_room is None:
            raise RuntimeError("Jericho did not provide the room after the action.")
        new_location = new_room.name
        # update actor history
        agent.add_history(observation=obs, action=action, location=new_location)
        # save obs, action, obs_n, reward to map
        world_map.record_step(action, new_location, observation=obs, next_location_id=new_room.num)
        location = new_location
        print('Total Score', info['score'], 'Moves', info['moves'])


    world_map.save()

    world_map.print_graph()

    env.close()


if __name__ == "__main__":
    freeze_support()
    main()
