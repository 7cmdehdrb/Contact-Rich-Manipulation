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


def incremental_approach_reach(env):
    """Signed 5cm-excess change; the manager applies weight and dt once."""
    from ..common.stabilize_sweep_contact import centroid

    state = env._stabilize_state
    return state.update_reach(centroid(env), env.scene['ee_frame'].data.target_pos_w[:, 0],
                              env.common_step_counter) / env.step_dt


def arm_joint_failure(env):
    """Read the already computed termination event, without a second velocity test."""
    return env.termination_manager.get_term('arm_joint_velocity').float() / env.step_dt


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


def linear_reach(env, near_radius=0.10, decay_distance=0.35, frame_name='ee_frame'):
    """Unit proximity plateau and linear falloff to the live target bounds center.

    No reset-distance or episode-duration scaling; RewardManager applies the
    configured weight and policy dt once, as for the parent's continuous Reach.
    """
    import math
    from ..common.stabilize_sweep_contact import centroid

    if not math.isfinite(near_radius) or near_radius < 0.0:
        raise ValueError('Reach near_radius must be finite and nonnegative')
    if not math.isfinite(decay_distance) or decay_distance <= 0.0:
        raise ValueError('Reach decay_distance must be finite and positive')
    reference = env.scene[frame_name].data.target_pos_w[:, 0]
    gap = ((reference - centroid(env)).norm(dim=-1) - near_radius).clamp_min(0.0)
    return (1.0 - gap / decay_distance).clamp(0.0, 1.0)
