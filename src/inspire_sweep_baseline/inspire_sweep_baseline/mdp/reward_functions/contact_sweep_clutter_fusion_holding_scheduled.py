"""One live goal radius for settling, success and contact credit."""

from ..common.contact_sweep_clutter_fusion_contacts import whole_hand_contact_force_w
from ..common.contact_sweep_clutter_fusion_holding import goal_radius
from ..common.contact_sweep_clutter_fusion_holding import target_measurements
import math as math
import torch as torch

def goal_settling(env, position_tolerance=0.02, speed_scale=0.05, command_name="target_goal_pos"):
    """Dense distance/speed reward, restricted to the inclusive goal radius."""
    if not all(math.isfinite(v) and v > 0 for v in (position_tolerance, speed_scale)):
        raise ValueError("Goal radius and speed scale must be finite and positive")
    distance, speed, valid = target_measurements(env, command_name)
    score = torch.exp(-math.log(2) * ((distance / position_tolerance) ** 2 + (speed / speed_scale) ** 2))
    return torch.where(valid & (distance <= position_tolerance), score, 0.0)


def hand_contact_or_goal(env, threshold_n=0.01):
    """Unit rate for fresh whole-hand force OR valid goal occupancy, never both twice.

    Goal occupancy needs neither contact nor the success speed threshold.
    This affects reward only; tactile observations retain measured contacts.
    """
    if not math.isfinite(threshold_n) or threshold_n <= 0:
        raise ValueError("Contact threshold must be finite and positive")
    force = whole_hand_contact_force_w(env)
    contact = (torch.isfinite(force).all(-1) & env.sensor_data_fresh
               & (torch.linalg.vector_norm(force, dim=-1) >= threshold_n))
    goal = env.reward_manager.get_term_cfg("goal_hold")
    distance, _, valid = target_measurements(env, goal.params.get("command_name", "target_goal_pos"))
    in_goal = valid & (distance <= goal_radius(env))
    return (contact | in_goal).to(force.dtype)
