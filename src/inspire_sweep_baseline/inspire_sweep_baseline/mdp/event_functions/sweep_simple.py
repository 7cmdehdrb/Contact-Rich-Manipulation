"""Simple Hard startup: cache goal limits before the unchanged Fast reset runs."""
from ..common.sweep_simple import HardCircleResetLayout
from .wrench_refactor_relative_ik_test import initialize_fast_geometry


def initialize_hard_geometry(env, env_ids):
    initialize_fast_geometry(env, env_ids, layout_class=HardCircleResetLayout)
    params = env.cfg.events.object_spawn.params
    env._fast_layout.configure_commands(**{key: params[key] for key in (
        'spawn_radius', 'edge_clearance', 'excluded_x_half_angle',
        'min_goal_distance', 'max_goal_distance')})
