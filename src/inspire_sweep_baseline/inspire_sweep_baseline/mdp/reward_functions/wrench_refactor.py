"""Refactor-specific deltas to Wrench reward functions; previous tasks unchanged."""
import torch
from isaaclab.managers import ManagerTermBase
from ..termination_functions.goal_terminations import _object_position


def policy_action_rate_penalty(env, arm_max_cost=None, hand_max_cost=None):
    """Raw Arm and clamped Hand changes; optional caps apply to each squared sum."""
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    arm_delta = action[:, :6] - previous[:, :6]
    hand_delta = action[:, -2:].clamp(0., 1.) - previous[:, -2:].clamp(0., 1.)
    arm_cost = arm_delta.square().sum(-1)
    hand_cost = hand_delta.square().sum(-1)
    if arm_max_cost is not None:
        arm_cost = arm_cost.clamp_max(arm_max_cost)
    if hand_max_cost is not None:
        hand_cost = hand_cost.clamp_max(hand_max_cost)
    return 0.03 * arm_cost + 0.03 * hand_cost


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
            # TEMP paired diagnostic only: keep the original returned reward.
            if type(getattr(env, 'cfg', None)).__name__ == 'WrenchRefactorV3TestEnvCfg':
                import json
                action_delta = env.action_manager.action[:, :6] - env.action_manager.prev_action[:, :6]
                proposed_axes = 0.03 * action_delta.square()
                old_axes = 0.03 * (current - self._previous_arm).square()
                hand_cost = 0.03 * hand_delta.square().sum(-1)
                proposed = proposed_axes.sum(-1) + hand_cost
                fields = {
                    'old_cost': self._cached_cost,
                    'proposed_cost': proposed,
                    'proposed_cost_with_arm_cap100': 0.03 * action_delta.square().sum(-1).clamp_max(100.) + hand_cost,
                    'old_arm_axes': old_axes,
                    'proposed_position_cost': proposed_axes[:, :3].sum(-1),
                    'proposed_rotation_cost': proposed_axes[:, 3:].sum(-1),
                    'proposed_arm_axes': proposed_axes,
                    'hand_cost': hand_cost,
                    'policy_arm': env.action_manager.action[:, :6],
                    'policy_arm_delta': action_delta,
                    'normalized_torque': current,
                    'old_step_reward': -env.step_dt * self._cached_cost,
                    'proposed_step_reward': -env.step_dt * proposed,
                }
                batches = {key: value.tolist() for key, value in fields.items()}
                steps = env.episode_length_buf.tolist()
                for env_id in range(env.num_envs):
                    payload = {key: value[env_id] for key, value in batches.items()}
                    payload.update(env=env_id, step=steps[env_id], policy_step=env.common_step_counter)
                    print('[Wrench-Refactor-v3-Test action_penalty] ' + json.dumps(payload), flush=True)
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



def contact_gated_displacement_cosine_goal_distance(env, distance_scale=0.05,
                                                     threshold_n=0.01, command_name="target_goal_pos"):
    """Preserve Relative distance cosine; require contact in this step only."""
    from .revision import target_contact_below_height
    contact = target_contact_below_height(env, threshold_n=threshold_n).bool()
    reward = displacement_cosine_goal_distance(env, distance_scale, command_name)
    return torch.where(contact, reward, 0.0)


class RelativeMotionCosineGoalDistance(ManagerTermBase):
    """Current XY goal direction versus motion from a distance-triggered anchor.

    Evaluate every control step, then advance the anchor only at >= 1 mm.
    Subthreshold motion is still evaluated, not suppressed or normalized by 1 mm.
    Contact/height gates affect payment only; histories continue to advance.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._anchor = _object_position(env)[:, :2].clone()
        self._value = self._anchor.new_zeros(env.num_envs)
        self._step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._anchor[ids] = _object_position(self._env)[ids, :2]
        self._value[ids] = 0.
        self._step[ids] = -1

    def __call__(self, env, distance_scale=0.05, anchor_distance=0.001,
                 require_contact=False, threshold_n=0.01, command_name="target_goal_pos"):
        from .revision import goal_distance_v2, target_contact_below_height

        fresh = self._step != env.common_step_counter
        position = _object_position(env)[:, :2]
        motion = position - self._anchor
        toward_goal = env.command_manager.get_command(command_name)[:, :2] - position
        move_norm = motion.norm(dim=-1)
        goal_norm = toward_goal.norm(dim=-1)
        cosine = ((motion / move_norm.clamp_min(1e-8)[:, None]) *
                  (toward_goal / goal_norm.clamp_min(1e-8)[:, None])).sum(-1).clamp(-1., 1.)
        value = goal_distance_v2(env, distance_scale=distance_scale, command_name=command_name) * cosine
        valid = (move_norm > 1e-8) & (goal_norm > 1e-8) & torch.isfinite(value)
        if require_contact:
            valid &= target_contact_below_height(env, threshold_n=threshold_n).bool()
        self._value[fresh] = torch.where(valid, value, 0.)[fresh]
        advance = fresh & (move_norm >= anchor_distance) & torch.isfinite(position).all(-1)
        self._anchor[advance] = position[advance]
        self._step[fresh] = env.common_step_counter
        return self._value.clone()


class RelativeInverseDistance(ManagerTermBase):
    """Outward XY distance increment; dt cancellation gives weight * delta per step.

    Store the previous control-step position independently of the cosine anchor.
    Reset captures the spawned position, before CommandManager resets its goal;
    evaluate both distances against the episode's current goal on the next call.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = _object_position(env)[:, :2].clone()
        self._value = self._previous.new_zeros(env.num_envs)
        self._step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._previous[ids] = _object_position(self._env)[ids, :2]
        self._value[ids] = 0.
        self._step[ids] = -1

    def __call__(self, env, command_name="target_goal_pos"):
        fresh = self._step != env.common_step_counter
        position = _object_position(env)[:, :2]
        goal = env.command_manager.get_command(command_name)[:, :2]
        delta = (position - goal).norm(dim=-1) - (self._previous - goal).norm(dim=-1)
        self._value[fresh] = delta.clamp_min(0.)[fresh] / env.step_dt
        self._previous[fresh] = position[fresh]
        self._step[fresh] = env.common_step_counter
        return self._value.clone()
