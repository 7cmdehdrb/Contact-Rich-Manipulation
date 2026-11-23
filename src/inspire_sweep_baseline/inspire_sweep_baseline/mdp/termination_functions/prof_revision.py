"""Timeout success uses the tracked initial visible OBB center, inclusive radius."""


def success_at_timeout(env):
    goal = env.command_manager.get_command('target_goal_pos')
    distance = (env._obb_live_pos_w[:, :2]-goal[:, :2]).norm(dim=-1)
    success = (env.episode_length_buf >= env.max_episode_length) & (distance <= env.cfg.goal_success_radius)
    for name in ('object_fall_or_tilt', 'shelf_failure', 'arm_joint_velocity'):
        success &= ~env.termination_manager.get_term(name)
    return success
