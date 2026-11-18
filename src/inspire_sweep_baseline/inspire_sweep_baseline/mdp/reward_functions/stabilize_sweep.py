"""Phase-aware rewards; RewardManager applies policy dt once."""

def squeeze(env):
    samples=env._stabilize_contacts
    return samples.squeeze_sum/max(samples.samples,1)


def stage(env):
    return env._stabilize_state.stage_score


def initial_movement(env):
    return env._stabilize_state.move_cost


def initial_rotation(env):
    return env._stabilize_state.rotate_cost


def sweep_progress(env):
    return env._stabilize_state.positive/env.step_dt


def sweep_retreat(env):
    return env._stabilize_state.retreat/env.step_dt
