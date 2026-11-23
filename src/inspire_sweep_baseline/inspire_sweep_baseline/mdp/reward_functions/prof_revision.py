"""Signed progress is a distance increment; formation is a bounded rate."""


def reaching(env):
    return env._stabilize_state.reach_delta / env.step_dt


def contact_formation(env):
    return env._formation
