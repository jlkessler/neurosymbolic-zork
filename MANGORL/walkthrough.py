"""Seed a compatible world map using the ROM's reference walkthrough."""
from .cognitive_map import CognitiveMap


def _state_context(env, observation):
    """Read the same restored-state room and inventory view used during play."""
    if not callable(getattr(env, "get_state", None)) or not callable(getattr(env, "set_state", None)):
        return str(observation or "")
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
        world_map.bind_rom(rom_path)
        world_map.start_episode(room.name, room.num)
        trajectory_id = f"ideal_map:{rom_path}:{seed}"
        world_map.remove_trajectory_experiences(trajectory_id)
        replay = {"source": "jericho_reference_walkthrough", "rom_path": str(rom_path),
                  "seed": seed, "commands": commands, "steps": [],
                  "completed": False, "termination_reason": "interrupted"}
        world_map.graph.graph["walkthrough"] = replay
        started = True
        current_context = _state_context(env, observation)
        score_before = info.get("score")
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
            next_context = observation if done else _state_context(env, observation)
            world_map.record_experience(
                source=source,
                action=action,
                destination=destination,
                before_context=current_context,
                after_context=next_context,
                score_before=score_before,
                score_after=info.get("score"),
                reward=reward,
                done=done,
                trajectory_id=trajectory_id,
                step=index,
                provenance="reference_replay",
            )
            replay["steps"].append({"action": action, "source": source,
                                    "destination": destination, "observation": observation,
                                    "reward": reward, "score": info.get("score"),
                                    "done": bool(done)})
            current_context = next_context
            score_before = info.get("score")
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
