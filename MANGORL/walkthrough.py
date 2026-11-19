"""Seed a compatible world map using the ROM's reference walkthrough."""
from .cognitive_map import CognitiveMap


def run_walkthrough(rom_path, map_path="memory/world_model.json", *, seed=0,
                    env_factory=None):
    """Replay all reference commands, preserving interactions as graph metadata.

    The reference walkthrough is not assumed to be a shortest solution. Normal
    episode step/score limits do not truncate this replay.
    """
    if env_factory is None:
        from jericho import FrotzEnv
        env_factory = FrotzEnv
    env = env_factory(str(rom_path), seed=seed)
    world_map = CognitiveMap(map_path)
    started = False
    try:
        observation, info = env.reset()
        commands = list(env.get_walkthrough() or [])
        if not commands:
            raise ValueError("This ROM has no Jericho reference walkthrough.")
        room = env.get_player_location()
        if room is None:
            raise RuntimeError("Jericho did not provide the starting room.")
        world_map.instantiate()
        world_map.start_episode(room.name, room.num)
        replay = {"source": "jericho_reference_walkthrough", "rom_path": str(rom_path),
                  "seed": seed, "commands": commands, "steps": [],
                  "completed": False, "termination_reason": "interrupted"}
        world_map.graph.graph["walkthrough"] = replay
        started = True
        for index, action in enumerate(commands, 1):
            source = world_map.current_location
            observation, reward, done, info = env.step(action)
            room = env.get_player_location()
            if room is None and not done:
                raise RuntimeError(f"Jericho did not provide the room after command {index}.")
            destination = None
            if room is not None:
                destination = world_map.add_standard_location(room.name, room.num)
                # Terminal relocation (e.g. death) is not a navigable edge.
                if not done:
                    world_map.record_step(action, room.name, observation, room.num)
            replay["steps"].append({"action": action, "source": source,
                                    "destination": destination, "observation": observation,
                                    "reward": reward, "score": info.get("score"),
                                    "done": bool(done)})
            print(f"Walkthrough {index}/{len(commands)}: {action} (score: {info.get('score')})")
            if done:
                break
        replay["completed"] = len(replay["steps"]) == len(commands)
        replay["termination_reason"] = "commands_exhausted" if replay["completed"] else "game_over"
        return world_map.save_path
    finally:
        try:
            if started:
                world_map.save()
        finally:
            env.close()
