"""Simple-specific contact maintenance with full credit inside the goal."""
import math
import torch

from .wrench_refactor_relative_ik_test import TargetContactDecayMaintenance
from ..termination_functions.goal_terminations import _object_position


class GoalContactMaintenance(TargetContactDecayMaintenance):
    """XY distance < goal_radius grants full contact credit (default 3 cm).

    Outside the goal, retain the parent's real contact history and decay rule.
    Goal occupancy does not fabricate tactile samples in the 2-of-3 history.
    """

    def __call__(self, env, threshold_n=.01, eef_height_limit=.15, goal_radius=.03):
        if not math.isfinite(goal_radius) or goal_radius <= 0:
            raise ValueError("Contact reward goal_radius must be positive and finite")
        fresh = self._cached_step != env.common_step_counter
        reward = super().__call__(env, threshold_n, eef_height_limit)
        if fresh:
            goal = env.command_manager.get_command('target_goal_pos')[:, :2]
            distance = (_object_position(env)[:, :2] - goal).norm(dim=-1)
            inside = torch.isfinite(distance) & (distance < goal_radius)
            self._count[inside] = 20
            self._misses[inside] = 0
            self._cached_reward[inside] = 20.
            reward = self._cached_reward.clone()
        return reward


def finger_action_rate_l2(env, action_group="arm"):
    """Parent raw-command squared-sum cost, extended to six hand channels."""
    current, previous = env.action_manager.action, env.action_manager.prev_action
    if current.shape[-1] != 12 or previous.shape != current.shape:
        raise ValueError("Finger penalties require six Arm and six Hand actions")
    if action_group == "arm":
        selection = slice(0, 6)
    elif action_group == "hand":
        selection = slice(6, 12)
    else:
        raise ValueError("action_group must be 'arm' or 'hand'")
    return (current[:, selection] - previous[:, selection]).square().sum(dim=-1)


from .dex_rewards import DexPlanarGoalProgress
from .wrench_refactor_relative_ik_test import InverseDistanceReach
from .goal_rewards import _target_state


class AreaMatchedLogGoalProgress(DexPlanarGoalProgress):
    """Log distance potential; match cumulative-curve area for each reset distance.

    Epsilon=1cm, tolerance=0. AUC refers to integral of cumulative reward over
    distance, NOT equal total episode reward (9cm gives 13.92398 vs 9).
    """
    def __call__(self, env, command_name='target_goal_pos', epsilon=.01):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be positive and finite')
        goal = env.command_manager.get_command(command_name)[:, :2]
        initial = (self._initial_position - goal).norm(dim=-1)
        distance = (_target_state(env)[:, :2] - goal).norm(dim=-1)
        # float64 avoids cancellation for very short initial distances. This is
        # tensor-only; the usual reset distribution is 6–12cm.
        d0 = initial.double()
        x = d0 / epsilon
        log = torch.log1p(x)
        denominator = epsilon * (x - log)
        small = x < 1e-3
        series = epsilon * x.square() * (.5 - x / 3 + x.square() / 4)
        denominator = torch.where(small, series, denominator).clamp_min(1e-30)
        coefficient = .5 * d0.square() / denominator
        remaining = torch.minimum(distance.double(), d0)
        progress = (coefficient * (log - torch.log1p(remaining / epsilon))).float().clamp_min(0.)
        best = torch.maximum(self._best_progress, progress)
        increment = best - self._best_progress
        self._best_progress[:] = best
        return increment / env.step_dt


class RetreatReach(InverseDistanceReach):
    def __call__(self, env, near_radius=.10, distance_scale=.10, frame_name='ee_frame', retreat_radius=.02):
        normal = super().__call__(env, near_radius, distance_scale, frame_name)
        state = env._finger_retreat
        active = state.update(env)
        reference = env.scene[frame_name].data.target_pos_w[:, 0]
        gap = ((reference - state.target).norm(dim=-1) - retreat_radius).clamp_min(0.)
        retreat = self._initial_distance / env.cfg.episode_length_s * distance_scale / (distance_scale + gap)
        return torch.where(active, retreat, normal)


class RetreatContactMaintenance(TargetContactDecayMaintenance):
    """No 3cm bypass. Only the latched 2cm/3s gate grants unconditional 20."""
    def __call__(self, env, threshold_n=.01, eef_height_limit=.15):
        reward = super().__call__(env, threshold_n, eef_height_limit)
        active = env._finger_retreat.update(env)
        self._count[active] = 20
        self._misses[active] = 0
        self._cached_reward[active] = 20.
        return torch.where(active, self._cached_reward, reward).clone()


class ContactLogGoalProgress(AreaMatchedLogGoalProgress):
    """Log record improvements gated by 2-of-3 contact and current EEF height.

    Advance the record even on rejected steps: free motion cannot be banked
    for later contact. History contains contact only, never a reward count.
    """

    def __init__(self, cfg, env):
        self._contact_history = torch.zeros((env.num_envs, 3), dtype=torch.bool, device=env.device)
        self._contact_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._contact_value = torch.zeros(env.num_envs, device=env.device)
        super().__init__(cfg, env)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._contact_history[ids] = False
        self._contact_step[ids] = -1
        self._contact_value[ids] = 0.

    def __call__(self, env, command_name='target_goal_pos', epsilon=.01,
                 threshold_n=.01, eef_height_limit=.15):
        from ..common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
        from ..common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT

        fresh = self._contact_step != env.common_step_counter
        # Parent always advances best progress, independently of the contact mask.
        increment = super().__call__(env, command_name, epsilon)
        name = env.scene['object_collection'].object_names[0]
        normal = env.scene[object_hand_sensor_name(name)].data.force_matrix_w
        force = normal.norm(dim=-1).sum(dim=(1, 2))
        contact = env.sensor_data_fresh & torch.isfinite(force) & (force >= threshold_n)
        history = torch.cat((self._contact_history[:, 1:], contact[:, None]), dim=-1)
        self._contact_history[fresh] = history[fresh]
        eef_z = env.scene['ee_frame'].data.target_pos_w[:, 0, 2]
        surface_z = env._revision_initial_shelf_pos[:, 2] + SHELF_SURFACE_HEIGHT
        accepted = ((self._contact_history.sum(-1) >= 2) & env.sensor_data_fresh
                    & torch.isfinite(eef_z) & torch.isfinite(surface_z)
                    & (eef_z < surface_z + eef_height_limit))
        self._contact_value[fresh] = torch.where(accepted, increment, 0.)[fresh]
        self._contact_step[fresh] = env.common_step_counter
        return self._contact_value.clone()


def semi_prof_force_direction(env, min_force=1., near_radius=.01):
    """Compare full 3D force against live planar goal direction (reference Z=0).

    Real contact and >=1N horizontal resultant remain required inside the
    radius. No velocity/height/progress gate and no goal-visit latch.
    RewardManager alone applies weight and dt.
    """
    from isaaclab.utils.math import quat_apply_inverse

    force_w, valid = env._semi_prof_forces.mean(env)
    quat = env._revision_initial_shelf_quat
    force = quat_apply_inverse(quat, force_w)
    goal = env.command_manager.get_command('target_goal_pos')[:, :3]
    direction = quat_apply_inverse(quat, goal - _target_state(env)[:, :3])[:, :2]
    distance = direction.norm(dim=-1)
    horizontal_magnitude = force[:, :2].norm(dim=-1)
    magnitude = force.norm(dim=-1)
    finite = torch.isfinite(direction).all(-1) & torch.isfinite(force).all(-1)
    cosine = (force[:, :2] * direction).sum(-1) / (magnitude * distance).clamp_min(1e-12)
    cosine = torch.nan_to_num(cosine).clamp(-1., 1.)
    score = (1. - torch.acos(cosine) / (math.pi / 2.)).clamp_min(0.).square()
    score = torch.where(cosine > 0., score, 0.)
    score = torch.where(distance < near_radius, 1., score)
    return torch.where(valid & finite & (horizontal_magnitude >= min_force), score, 0.)
