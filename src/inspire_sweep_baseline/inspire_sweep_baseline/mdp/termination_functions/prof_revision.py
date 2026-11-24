"""Timeout success uses the tracked initial visible OBB center, inclusive radius."""


def success_at_timeout(env):
    goal = env.command_manager.get_command('target_goal_pos')
    distance = (env._obb_live_pos_w[:, :2]-goal[:, :2]).norm(dim=-1)
    success = (env.episode_length_buf >= env.max_episode_length) & (distance <= env.cfg.goal_success_radius)
    for name in ('object_fall_or_tilt', 'shelf_failure', 'arm_joint_velocity'):
        success &= ~env.termination_manager.get_term(name)
    return success


def tcp_linear_velocity(env, asset_cfg, threshold=4.0):
    """Inclusive world linear-speed limit at the UR5e last link's origin."""
    import math

    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError('TCP speed threshold must be finite and positive')
    velocity = env.scene[asset_cfg.name].data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    return (velocity.norm(dim=-1) >= threshold).any(dim=-1)


def success_at_timeout_tcp(env):
    """Parent success geometry with TCP failure replacing the joint cutoff."""
    goal = env.command_manager.get_command('target_goal_pos')
    distance = (env._obb_live_pos_w[:, :2]-goal[:, :2]).norm(dim=-1)
    success = (env.episode_length_buf >= env.max_episode_length) & (distance <= env.cfg.goal_success_radius)
    for name in ('object_fall_or_tilt', 'shelf_failure'):
        success &= ~env.termination_manager.get_term(name)
    # Inherited term ordering can evaluate success before the appended TCP term.
    # Evaluate its configured predicate on this step, never a previous-step flag.
    tcp = env.termination_manager.get_term_cfg('tcp_linear_velocity')
    return success & ~tcp.func(env, **tcp.params)
