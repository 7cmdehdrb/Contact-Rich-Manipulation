"""Final v1 time-limit accounting without early goal termination."""

from isaaclab.envs.mdp import time_out
from .sweep import shelf_collision
from .goal_terminations import shelf_displacement, _object_position
from ..common.contact_sweep_clutter_fusion_holding import goal_radius


def shelf_termination(env, threshold=0.1, check_fingers=False,
                      wrist_clearance=0.03, displacement_threshold=0.05):
    return (shelf_collision(env, threshold, check_fingers, wrist_clearance)
            | shelf_displacement(env, displacement_threshold))


def timeout_goal_reached(env):
    """Only the live XY radius is tested, with no speed or hold-time gate."""
    goal = env.command_manager.get_command("target_goal_pos")
    return (_object_position(env)[:, :2] - goal[:, :2]).norm(dim=-1) <= goal_radius(env)


def success_at_timeout(env):
    return time_out(env) & timeout_goal_reached(env)


def unsuccessful_timeout(env):
    return time_out(env) & ~timeout_goal_reached(env)


def strict_timeout_goal_reached(env):
    goal = env.command_manager.get_command("target_goal_pos")
    distance = (_object_position(env)[:, :2] - goal[:, :2]).norm(dim=-1)
    return distance < goal_radius(env)


def strict_success_at_timeout(env):
    env.record_goal_accuracy_metrics()
    return time_out(env) & strict_timeout_goal_reached(env)


def strict_unsuccessful_timeout(env):
    return time_out(env) & ~strict_timeout_goal_reached(env)
