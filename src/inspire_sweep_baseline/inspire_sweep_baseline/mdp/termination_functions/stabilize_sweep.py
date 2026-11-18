"""Timeout success requires Gate, strict current centroid distance, and no failure."""
from ..common.stabilize_sweep_contact import centroid


def success_at_timeout(env):
    goal=env.command_manager.get_command('target_goal_pos')
    distance=(centroid(env)[:,:2]-goal[:,:2]).norm(dim=-1)
    success=(env.episode_length_buf>=env.max_episode_length)&env._sweep_phase&(distance<.03)
    for name in ('object_fall_or_tilt','shelf_failure','arm_joint_velocity'):
        success &= ~env.termination_manager.get_term(name)
    return success
