"""Refactor-specific deltas to Wrench reward functions; previous tasks unchanged."""
import torch
from isaaclab.managers import ManagerTermBase
from ..termination_functions.goal_terminations import _object_position


class RefactorActionRatePenalty(ManagerTermBase):
    """Policy-step change of normalized OSC output torques and clamped Hand openness.

    Sample the final physics substep's applied Arm efforts. Normalize by the
    same per-joint bounds as the OSC action; keep the parent's 0.03 coefficients
    and Arm cap. Reset starts the selected Arm histories at zero.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_arm = torch.zeros((env.num_envs, 6), device=env.device)
        self._cached_cost = torch.zeros(env.num_envs, device=env.device)
        self._cached_step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._previous_arm[ids] = 0.0
        self._cached_cost[ids] = 0.0

    def __call__(self, env):
        if env.common_step_counter != self._cached_step:
            arm = env.action_manager.get_term("arm_action")
            limits = (arm._asset.data.joint_effort_limits[:, arm._joint_ids]
                      * arm.cfg.effort_limit_scale).clamp_min(arm.cfg.minimum_effort_limit)
            current = (arm.joint_efforts / limits).clamp(-1.0, 1.0)
            arm_cost = (current - self._previous_arm).square().sum(-1).clamp_max(100.0)
            hand_delta = env.action_manager.action[:, -2:].clamp(0., 1.) - env.action_manager.prev_action[:, -2:].clamp(0., 1.)
            self._cached_cost[:] = 0.03 * arm_cost + 0.03 * hand_delta.square().sum(-1)
            self._previous_arm[:] = current
            self._cached_step = env.common_step_counter
        return self._cached_cost.clone()


class RefactorContactMaintenance(ManagerTermBase):
    """Accumulate 2-of-3 contact windows up to 10; clear after five rejected windows.

    Rejected windows pay zero but retain the count for four steps. Goal occupancy
    qualifies without force/history only below the current EEF ceiling. RewardManager applies dt.
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
        from .revision import target_contact_below_height, goal_reward_height_valid

        step = env.common_step_counter
        if step != self._cached_step:
            # Missing evaluations cannot establish a consecutive sliding window.
            if self._cached_step is not None and step != self._cached_step + 1:
                self.reset()
            contact = target_contact_below_height(env, threshold_n=threshold_n).bool()
            goal = env.command_manager.get_command(command_name)[:, :2]
            distance = torch.linalg.vector_norm(_object_position(env)[:, :2] - goal, dim=-1)
            height = goal_reward_height_valid(env)
            inside = torch.isfinite(distance) & (distance <= 0.03) & height
            self._history[:, :2] = self._history[:, 1:].clone()
            self._history[:, 2] = contact | inside
            accepted = (inside | (self._history.sum(-1) >= 2)) & height
            self._misses[:] = torch.where(accepted, 0, self._misses + 1)
            self._count += accepted.long()
            self._count.clamp_(max=10)
            self._count[self._misses >= 5] = 0
            self._cached_reward[:] = torch.where(accepted, self._count, 0).float()
            self._cached_step = step
        return self._cached_reward.clone()


class RefactorFirstEntry(ManagerTermBase):
    """Pay exactly once on inclusive goal entry, independent of speed and dt."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._entered = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._event = torch.zeros(env.num_envs, device=env.device)
        self._key = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._entered[ids] = False
        self._event[ids] = 0.

    def __call__(self, env, command_name="target_goal_pos"):
        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._key:
            distance = (_object_position(env)[:, :2] - env.command_manager.get_command(command_name)[:, :2]).norm(dim=-1)
            inside = torch.isfinite(distance) & (distance <= env.cfg.goal_success_radius)
            self._event[:] = (inside & ~self._entered).float()
            self._entered |= inside
            self._key = key
        return self._event.clone() / env.step_dt


def contact_gated_goal_distance(env, distance_scale=0.05, threshold_n=0.01,
                                command_name="target_goal_pos"):
    """Current target-hand contact only: no history/count or goal-inside bypass."""
    from .revision import goal_distance_v2, target_contact_below_height
    contact = target_contact_below_height(env, threshold_n=threshold_n).bool()
    distance_reward = goal_distance_v2(env, distance_scale=distance_scale, command_name=command_name)
    return torch.where(contact, distance_reward, 0.0)


def displacement_cosine_goal_distance(env, distance_scale=0.05,
                                     command_name="target_goal_pos"):
    """v0 distance times signed XY alignment of goal and displacement from reset.

    This measures cumulative displacement, not instantaneous velocity. Retreat
    on the goal side and collinear overshoot retain positive alignment. Vectors
    shorter than 1e-8 m (including coincident start/goal) have undefined direction
    and pay zero. No goal-radius override is applied.
    """
    rows = torch.arange(env.num_envs, device=env.device)
    start = env._initial_object_positions_w[rows, env.target_id[:, 0].long(), :2]
    goal_vector = env.command_manager.get_command(command_name)[:, :2] - start
    displacement = _object_position(env)[:, :2] - start
    goal_norm = torch.linalg.vector_norm(goal_vector, dim=-1)
    move_norm = torch.linalg.vector_norm(displacement, dim=-1)
    valid = (goal_norm > 1e-8) & (move_norm > 1e-8)
    cosine = ((goal_vector / goal_norm.clamp_min(1e-8).unsqueeze(-1)) *
              (displacement / move_norm.clamp_min(1e-8).unsqueeze(-1))).sum(-1).clamp(-1., 1.)
    from .revision import goal_distance_v2
    reward = goal_distance_v2(
        env, distance_scale=distance_scale,
        command_name=command_name,
    ) * cosine
    return torch.where(valid & torch.isfinite(reward), reward, 0.0)
