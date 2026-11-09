"""Adapters preserve Holding reward semantics on the independent IK runtime."""
from ..termination_functions.sweep import shelf_collision


def holding_shelf_failure(env):
    """Pay -3 only for Holding's shelf collision predicate, not displacement alone."""
    return shelf_collision(env, threshold=0.1, check_fingers=False, wrist_clearance=0.03).float() / env.step_dt
