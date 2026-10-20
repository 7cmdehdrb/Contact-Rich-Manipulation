"""World-XY goal occupancy, shared tactile rewards, and bounded scene movement."""

import math

import torch

from ..mdp.tactile_observations import ProjectedTactileContactBits


class SharedTactileContactBits(ProjectedTactileContactBits):
    """Read current contacts once for both reward terms and the observation.

    Rewards run before observations. Never read the previous observation buffer.
    Automatic/partial reset invalidates this cache before reset observations.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._cached_step = None
        self._cached_bits = None

    def reset(self, env_ids=None):
        self._cached_step = None

    def __call__(self, env, sensor_name="hand_tactile", threshold_n=0.01):
        key = (env.common_step_counter, sensor_name, threshold_n)
        if self._cached_step != key:
            self._cached_bits = super().__call__(env, sensor_name, threshold_n)
            self._cached_step = key
        # Observation modifiers must not change the shared physical reading.
        return self._cached_bits.clone()


def goal_hold_xy(env, position_tolerance=0.01, command_name="target_goal_pos"):
    """Unit reward rate while the target is inside the world-horizontal disk.

    Only world X/Y are compared: height, shelf tilt, target velocity, and prior
    occupancy do not alter this condition. RewardManager applies weight * dt.
    """
    if not math.isfinite(position_tolerance) or position_tolerance <= 0.0:
        raise ValueError("position_tolerance must be finite and positive")
    objects = env.scene["object_collection"]
    rows = torch.arange(env.num_envs, device=env.device)
    position = objects.data.object_pos_w[rows, env.target_id[:, 0].long()]
    goal = env.command_manager.get_command(command_name)
    distance = torch.linalg.vector_norm(position[:, :2] - goal[:, :2], dim=-1)
    return (distance <= position_tolerance).to(position.dtype)


def _current_tactile_bits(env):
    # The manager's copied config contains the resolved, shared term instance;
    # env.cfg still contains its class. ObservationManager has no term getter.
    term = env.observation_manager.cfg.policy.tactile
    return term.func(env, **term.params)


def contact_any(env):
    """Unit rate when at least one of the policy's 17 tactile channels is ON."""
    bits = _current_tactile_bits(env)
    return (bits > 0.5).any(dim=-1).to(bits.dtype)


def contact_count(env, max_contacts=5):
    """Linear channel count, normalized to one at five or more ON channels."""
    if not isinstance(max_contacts, int) or max_contacts <= 0:
        raise ValueError("max_contacts must be a positive integer")
    bits = _current_tactile_bits(env)
    count = (bits > 0.5).sum(dim=-1).to(bits.dtype)
    return count.clamp(max=max_contacts) / max_contacts


def obstacle_displacement_l2(env, max_cost=0.04):
    """Sum squared 3D displacement from exact reset poses, excluding target.

    The cap is in m^2, before the negative reward weight and dt. Extreme or
    non-finite obstacle costs saturate; the target never contributes.
    """
    if not math.isfinite(max_cost) or max_cost <= 0.0:
        raise ValueError("max_cost must be finite and positive")
    position = env.scene["object_collection"].data.object_pos_w
    squared = (position - env._initial_object_positions_w).square().sum(dim=-1)
    is_obstacle = (
        torch.arange(position.shape[1], device=env.device)[None, :]
        != env.target_id.long()
    )
    cost = torch.where(is_obstacle, squared, 0.0).sum(dim=-1)
    return torch.nan_to_num(
        cost, nan=max_cost, posinf=max_cost, neginf=max_cost,
    ).clamp(max=max_cost)
