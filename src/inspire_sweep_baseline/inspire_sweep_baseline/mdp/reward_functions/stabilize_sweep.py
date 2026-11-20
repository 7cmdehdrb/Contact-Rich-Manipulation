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


def action_limit_excess(env, action_group):
    """Mean squared raw-command excess, normalized by each channel's full allowed width."""
    import math
    from ...assets.hand_kinematics import HAND_DRIVE_UPPER_LIMITS

    actions = env.action_manager.action
    if actions.shape[-1] != 12:
        raise ValueError('Sweep Only requires six arm and six hand actions')
    if action_group == 'arm':
        raw = actions[:, :6]
        bounded = raw.clamp(-math.pi / 2, math.pi / 2)
        width = math.pi
    elif action_group == 'hand':
        raw = actions[:, 6:]
        width = raw.new_tensor(HAND_DRIVE_UPPER_LIMITS)
        bounded = raw.clamp_min(0).minimum(width)
    else:
        raise ValueError("action_group must be 'arm' or 'hand'")
    return ((raw - bounded) / width).square().mean(dim=-1)
