"""Absolute reward schedules; repeated resets never accumulate decrements."""



def reward_schedule(
    env, env_ids, term_name: str, start: float, decrement: float,
    interval: int, minimum: float, parameter: str | None = None,
) -> float:
    """Update a shared reward term from the runner's zero-based iteration."""
    if interval <= 0 or decrement < 0 or minimum > start:
        raise ValueError("Expected interval > 0, decrement >= 0, and minimum <= start")
    value = max(minimum, round(start - decrement * (env.learning_iteration // interval), 12))
    term = env.reward_manager.get_term_cfg(term_name)
    if parameter is None:
        term.weight = value
    else:
        term.params[parameter] = value
    env.reward_manager.set_term_cfg(term_name, term)
    return value
