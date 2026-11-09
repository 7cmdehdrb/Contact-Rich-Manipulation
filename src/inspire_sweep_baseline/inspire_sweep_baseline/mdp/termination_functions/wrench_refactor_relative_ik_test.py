"""Collect telemetry before reset without changing timeout semantics."""
from isaaclab.envs.mdp import time_out


def timeout_with_metrics(env):
    env.record_goal_accuracy_metrics()
    return time_out(env)


def success_at_timeout(env):
    """Strict XY < 3 cm at the deadline; simultaneous failures take precedence."""
    import torch
    data = env.scene['object_collection'].data
    rows = torch.arange(env.num_envs, device=env.device)
    target = data.object_pos_w[rows, env.target_id[:, 0].long(), :2]
    goal = env.command_manager.get_command('target_goal_pos')[:, :2]
    distance = (target - goal).norm(dim=-1)
    failed = (env.termination_manager.get_term('object_fall_or_tilt')
              | env.termination_manager.get_term('shelf_failure')
              | env.termination_manager.get_term('arm_joint_velocity'))
    return time_out(env) & torch.isfinite(distance) & (distance < .03) & ~failed
