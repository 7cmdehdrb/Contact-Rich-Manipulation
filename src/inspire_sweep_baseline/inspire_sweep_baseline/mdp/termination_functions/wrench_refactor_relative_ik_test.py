"""Collect telemetry before reset without changing timeout semantics."""
from isaaclab.envs.mdp import time_out


def timeout_with_metrics(env):
    env.record_goal_accuracy_metrics()
    return time_out(env)
