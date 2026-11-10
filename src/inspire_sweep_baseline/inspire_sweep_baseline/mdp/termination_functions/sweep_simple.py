"""Finger-v2 success classification at the existing timeout."""
from isaaclab.envs.mdp import time_out


def retreat_success_at_timeout(env):
    active = env._finger_retreat.update(env)
    failed = (env.termination_manager.get_term('object_fall_or_tilt')
              | env.termination_manager.get_term('shelf_failure')
              | env.termination_manager.get_term('arm_joint_velocity'))
    return time_out(env) & active & ~failed
