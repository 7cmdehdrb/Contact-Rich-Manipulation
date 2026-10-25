"""One live goal radius for settling, success and contact credit."""

from ..common.contact_sweep_clutter_fusion_contacts import whole_hand_contact_force_w
from ..common.contact_sweep_clutter_fusion_holding import goal_radius
from ..common.contact_sweep_clutter_fusion_holding import target_measurements
from .dex_rewards import DexReachProgress
from .goal_rewards import _target_state
import math as math
import torch as torch
from isaaclab.managers import ManagerTermBase

def goal_settling(env, position_tolerance=0.02, speed_scale=0.05, command_name="target_goal_pos"):
    """Dense distance/speed reward, restricted to the inclusive goal radius."""
    if not all(math.isfinite(v) and v > 0 for v in (position_tolerance, speed_scale)):
        raise ValueError("Goal radius and speed scale must be finite and positive")
    distance, speed, valid = target_measurements(env, command_name)
    score = torch.exp(-math.log(2) * ((distance / position_tolerance) ** 2 + (speed / speed_scale) ** 2))
    return torch.where(valid & (distance <= position_tolerance), score, 0.0)


def goal_settling_centered(env, position_tolerance=0.02, speed_scale=0.05,
                          command_name="target_goal_pos"):
    """Uncapped settling through 2r, with up to twice the rate inside r.

    The multiplier is one at the boundary and two at the center; the inherited
    distance/speed score and all validity checks still apply.
    """
    if not all(math.isfinite(v) and v > 0 for v in (position_tolerance, speed_scale)):
        raise ValueError("Goal radius and speed scale must be finite and positive")
    distance, speed, valid = target_measurements(env, command_name)
    score = torch.exp(-math.log(2) * ((distance / position_tolerance) ** 2 + (speed / speed_scale) ** 2))
    rate = torch.where(valid & (distance <= 2.0 * position_tolerance), score, 0.0)
    center_bonus = (1.0 - distance / position_tolerance).clamp(0.0, 1.0).square()
    return torch.where(distance < position_tolerance, rate * (1.0 + center_bonus), rate)


class ReachingOrGoal(DexReachProgress):
    """New-best approach with a relaxed near radius while the target is in goal.

    Use live goal occupancy, like hand_contact_or_goal, without a speed gate or
    episode latch. Keep a single monotonic best-distance record so toggling the
    trigger cannot repay progress. Entering goal can pay the remaining approach
    progress once, up to the relaxed near radius.
    """

    def __call__(self, env, near_radius=0.10, goal_near_radius=1.0, frame_name="ee_frame"):
        if not (math.isfinite(near_radius) and near_radius > 0
                and math.isfinite(goal_near_radius) and goal_near_radius >= near_radius):
            raise ValueError("Reach radii must be finite, positive and goal radius >= near radius")
        goal = env.reward_manager.get_term_cfg("goal_hold")
        distance, _, valid = target_measurements(env, goal.params.get("command_name", "target_goal_pos"))
        in_goal = valid & (distance <= goal_radius(env))
        radius = torch.where(in_goal, goal_near_radius, near_radius)
        reference = env.scene[frame_name].data.target_pos_w[:, 0]
        gap = (torch.linalg.vector_norm(reference - _target_state(env)[:, :3], dim=-1) - radius).clamp_min(0.0)
        best = torch.minimum(self._best_distance, gap)
        increment = self._best_distance - best
        self._best_distance[:] = best
        return increment / env.step_dt


class CappedGoalSettling(ManagerTermBase):
    """Pay the inherited rate for at most three cumulative eligible seconds.

    Leaving the goal pauses the budget; only episode reset replenishes it.
    Count paid control steps rather than reward magnitude or consecutive success
    time. RewardManager still applies weight and step_dt exactly once.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._paid_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        self._paid_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, position_tolerance=0.02, speed_scale=0.05,
                 max_hold_time=3.0, command_name="target_goal_pos"):
        if not math.isfinite(max_hold_time) or max_hold_time <= 0:
            raise ValueError("Hold reward duration must be finite and positive")
        if not math.isfinite(env.step_dt) or env.step_dt <= 0:
            raise ValueError("Control step duration must be finite and positive")
        rate = goal_settling(env, position_tolerance, speed_scale, command_name)
        # Float64 avoids a missing final step at the exact 3 s boundary.
        remaining = max_hold_time - self._paid_steps.to(torch.float64) * env.step_dt
        fraction = (remaining / env.step_dt).clamp(0.0, 1.0)
        eligible = (rate > 0) & (remaining > 1e-9)
        self._paid_steps += eligible.long()
        return torch.where(eligible, rate * fraction.to(rate.dtype), 0.0)


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
