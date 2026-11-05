"""Wrench terminal, contact-maintained goal and action-rate rewards."""

import math

import torch
from isaaclab.managers import ManagerTermBase

from ..common.contact_sweep_clutter_fusion_contacts import target_hand_contact_force_w
from ..termination_functions.goal_terminations import _object_position
from .revision import goal_distance_v2


class WrenchV0_1GoalDistance(ManagerTermBase):
    """Parent XY proximity times signed cosine, gated by consecutive contact.

    Contact is the fresh, finite target/hand resultant >= threshold_n. Keep
    the parent's current EEF height gate. Inside 3 cm, alpha is one to avoid an
    unstable angle near the goal. Outside it, opposite-side overshoot has a
    negative cosine. Repeated calls in one control step never manufacture a
    second contact sample. Reset clears selected rows.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._cached_reward = torch.zeros(env.num_envs, device=env.device)
        self._cached_step = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_contact[env_ids] = False
        self._cached_reward[env_ids] = 0.0

    def __call__(self, env, distance_scale=0.05, threshold_n=0.01, command_name="target_goal_pos"):
        if not math.isfinite(threshold_n) or threshold_n <= 0.0:
            raise ValueError("Contact threshold must be finite and positive")
        step = env.common_step_counter
        if step != self._cached_step:
            force = target_hand_contact_force_w(env)
            contact = (env.sensor_data_fresh & torch.isfinite(force).all(-1)
                       & (torch.linalg.vector_norm(force, dim=-1) >= threshold_n))
            goal = env.command_manager.get_command(command_name)[:, :2]
            rows = torch.arange(env.num_envs, device=env.device)
            initial = env._initial_object_positions_w[rows, env.target_id[:, 0].long(), :2]
            current = _object_position(env)[:, :2]
            alpha = torch.nn.functional.cosine_similarity(initial - goal, current - goal, dim=-1)
            distance = torch.linalg.vector_norm(current - goal, dim=-1)
            alpha = torch.where(distance < 0.03, 1.0, alpha)
            raw = goal_distance_v2(env, distance_scale=distance_scale, command_name=command_name) * alpha
            consecutive = self._cached_step is not None and step == self._cached_step + 1
            valid = contact & self._previous_contact & consecutive & torch.isfinite(raw)
            self._cached_reward[:] = torch.where(valid, raw, 0.0)
            self._previous_contact[:] = contact
            self._cached_step = step
        return self._cached_reward.clone()


def settled_success_bonus(env):
    return env.termination_manager.get_term('target_goal_success').float() / env.step_dt


def desired_wrench_action_rate_penalty(env):
    """Squared change of normalized desired wrench; preserve the raw Hand cost.

    Desired wrench = clip(raw, -limit, limit) * wrench_scale. Dividing by
    the policy target range (limit * wrench_scale) cancels the physical scale.
    Compare targets before feedback, not measured or corrected wrench. Use the
    manager's previous action so selected resets keep its existing zero history.
    """
    from .goal_rewards import action_rate_l2

    current = env.action_manager.action
    previous = env.action_manager.prev_action
    if current.shape[-1] != 8 or previous.shape != current.shape:
        raise ValueError("Wrench penalties require six Arm and two Hand actions")
    limit = env.action_manager.get_term("arm_action").cfg.normalized_wrench_limit
    current_arm = current[:, :6].clamp(-limit, limit) / limit
    previous_arm = previous[:, :6].clamp(-limit, limit) / limit
    arm_cost = (current_arm - previous_arm).square().sum(dim=-1).clamp_max(100.0)
    return 0.03 * arm_cost + 0.03 * action_rate_l2(env, action_group="hand", max_cost=100.0)


def goal_distance_cosine(env, distance_scale=0.05, command_name="target_goal_pos"):
    """v0 proximity times signed initial/current goal-vector cosine, without contact."""
    goal = env.command_manager.get_command(command_name)[:, :2]
    rows = torch.arange(env.num_envs, device=env.device)
    initial = env._initial_object_positions_w[rows, env.target_id[:, 0].long(), :2]
    current = _object_position(env)[:, :2]
    alpha = torch.nn.functional.cosine_similarity(initial - goal, current - goal, dim=-1)
    distance = torch.linalg.vector_norm(current - goal, dim=-1)
    alpha = torch.where(distance < 0.03, 1.0, alpha)
    raw = goal_distance_v2(env, distance_scale=distance_scale, command_name=command_name) * alpha
    return torch.where(torch.isfinite(raw), raw, 0.0)


class WrenchV0_2ContactMaintenance(ManagerTermBase):
    """Accumulate 2-of-3 contact windows up to 10; clear after five rejected windows.

    Rejected windows pay zero but retain the count for four steps. Goal occupancy
    immediately qualifies, even with no force/history. RewardManager applies dt.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._history = torch.zeros((env.num_envs, 3), dtype=torch.bool, device=env.device)
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._misses = torch.zeros_like(self._count)
        self._cached_reward = torch.zeros(env.num_envs, device=env.device)
        self._cached_step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._history[ids] = False
        self._count[ids] = 0
        self._misses[ids] = 0
        self._cached_reward[ids] = 0.0

    def __call__(self, env, threshold_n=0.01, command_name="target_goal_pos"):
        from .revision import target_contact_below_height

        step = env.common_step_counter
        if step != self._cached_step:
            # Missing evaluations cannot establish a consecutive sliding window.
            if self._cached_step is not None and step != self._cached_step + 1:
                self.reset()
            contact = target_contact_below_height(env, threshold_n=threshold_n).bool()
            goal = env.command_manager.get_command(command_name)[:, :2]
            distance = torch.linalg.vector_norm(_object_position(env)[:, :2] - goal, dim=-1)
            inside = torch.isfinite(distance) & (distance < 0.03)
            self._history[:, :2] = self._history[:, 1:].clone()
            self._history[:, 2] = contact | inside
            accepted = inside | (self._history.sum(-1) >= 2)
            self._misses[:] = torch.where(accepted, 0, self._misses + 1)
            self._count += accepted.long()
            self._count.clamp_(max=10)
            self._count[self._misses >= 5] = 0
            self._cached_reward[:] = torch.where(accepted, self._count, 0).float()
            self._cached_step = step
            # TEMP v0_2 diagnostics: inspect env 0 once per control step.
            # Keep this inside the cache update so repeated queries do not print twice.
            # contact_now, goal_inside, window_hits, qualifies, misses, count, reward = torch.stack((
            #     contact[0].long(), inside[0].long(), self._history[0].sum(),
            #     accepted[0].long(), self._misses[0], self._count[0],
            #     self._cached_reward[0].long(),
            # )).tolist()
            # print(
            #     f"[Wrench-v0_2 Contact Count] step={step} env=0 count={count} "
            #     f"contact={contact_now} goal_inside={goal_inside} "
            #     f"window_hits={window_hits}/3 accepted={qualifies} "
            #     f"misses={misses} reward_raw={reward}",
            #     flush=True,
            # )
        return self._cached_reward.clone()
