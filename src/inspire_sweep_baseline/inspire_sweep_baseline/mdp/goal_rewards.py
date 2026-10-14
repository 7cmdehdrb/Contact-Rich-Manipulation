"""Goal-task rewards, with event rewards independent of the control timestep.

Isaac Lab's RewardManager multiplies every configured term by ``step_dt``.
Only reaching progress and terminal events divide by that interval here; the
remaining functions are costs per second (the plan uses T_ref = 1 second).
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat


def _target_state(env):
    rows = torch.arange(env.num_envs, device=env.device)
    targets = env.target_id[:, 0].long()
    return env.scene["object_collection"].data.object_link_state_w[rows, targets]


def _scale_tensor(scale, values):
    if isinstance(scale, Real):
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Normalization scales must be finite and positive")
    else:
        scale_cpu = torch.as_tensor(scale)
        if not bool(torch.isfinite(scale_cpu).all() & (scale_cpu > 0.0).all()):
            raise ValueError("Normalization scales must be finite and positive")
    result = torch.as_tensor(scale, device=values.device, dtype=values.dtype)
    if result.ndim > 1 or (result.ndim == 1 and result.numel() != values.shape[-1]):
        raise ValueError("Normalization scales must be scalar or match the group size")
    return result


def planar_goal_cost(env, command_name="target_goal_pos", sigma=0.10):
    """Negative isotropic world-XY distance cost; zero at the goal."""
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    position = _target_state(env)[:, :2]
    goal = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.linalg.vector_norm(position - goal, dim=-1)
    return torch.expm1(-distance / sigma)


def normalized_action_rate(env, action_group="arm", normalization_scale=1.0):
    """Mean squared changes of normalized commands for the baseline 6+2 actions.

    Arm inputs already use the action term's normalized coordinates: their
    physical joint-target change is 0.5 times this value. Hand inputs are
    absolute openness commands, so consecutive inputs are clamped to the same
    [0, 1] interval used by HandOpennessAction before their changes are measured.
    """
    current = env.action_manager.action
    previous = env.action_manager.prev_action
    if current.shape[-1] != 8 or previous.shape != current.shape:
        raise ValueError("Goal rewards require the baseline six Arm and two Hand actions")
    if action_group == "arm":
        current, previous = current[:, :6], previous[:, :6]
    elif action_group == "hand":
        current, previous = current[:, 6:].clamp(0.0, 1.0), previous[:, 6:].clamp(0.0, 1.0)
    else:
        raise ValueError("action_group must be 'arm' or 'hand'")
    scale = _scale_tensor(normalization_scale, current)
    return ((current - previous) / scale).square().mean(dim=-1)


def normalized_joint_velocity(env, asset_cfg=SceneEntityCfg("robot"), reference_velocity=1.0):
    """Mean squared actual joint speeds, using the configured named joint group."""
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    scale = _scale_tensor(reference_velocity, velocity)
    return (velocity / scale).square().mean(dim=-1)


class ReachProgress(ManagerTermBase):
    """Reward only new episode-best progress toward the initial object bottom.

    ``reset`` runs after scene reset events and before command reset. It reads
    scene state directly so a previous episode's command snapshot cannot leak
    into the new episode. The fixed EEF point is attached to the wrist assembly
    and cannot move when only finger joints bend.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initial_object_position = torch.zeros((env.num_envs, 3), device=env.device)
        self._initial_distance = torch.zeros(env.num_envs, device=env.device)
        self._best_distance = torch.zeros_like(self._initial_distance)
        self.reset()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        params = self.cfg.params
        near_radius = params.get("near_radius", 0.10)
        frame_name = params.get("frame_name", "ee_frame")
        if not math.isfinite(near_radius) or near_radius < 0.0:
            raise ValueError("near_radius must be finite and nonnegative")
        # Articulation's public pose accessor updates forward kinematics after
        # joint writes. FrameTransformer then reads the refreshed rigid poses.
        # Minimal CPU scene doubles need no kinematic refresh.
        getattr(self._env.scene["robot"].data, "body_link_pos_w", None)
        initial = _target_state(self._env)[ids, :3]
        reference = self._env.scene[frame_name].data.target_pos_w[ids, 0]
        distance = (torch.linalg.vector_norm(reference - initial, dim=-1) - near_radius).clamp_min(0.0)
        self._initial_object_position[ids] = initial
        self._initial_distance[ids] = distance
        self._best_distance[ids] = distance

    def __call__(self, env, near_radius=0.10, frame_name="ee_frame", epsilon=1.0e-6):
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        reference = env.scene[frame_name].data.target_pos_w[:, 0]
        distance = (
            torch.linalg.vector_norm(reference - self._initial_object_position, dim=-1) - near_radius
        ).clamp_min(0.0)
        best = torch.minimum(self._best_distance, distance)
        progress = (self._best_distance - best) / self._initial_distance.clamp_min(epsilon)
        self._best_distance[:] = best
        return progress / env.step_dt


def object_tilt_angle(env):
    """Angle between the cylinder's local up axis and the current shelf normal."""
    object_up = matrix_from_quat(_target_state(env)[:, 3:7])[..., :, 2]
    shelf_up = matrix_from_quat(env.scene["shelf"].data.root_quat_w)[..., :, 2]
    cosine = (object_up * shelf_up).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def object_tilt_cost(env, free_angle=math.radians(5.0), fail_angle=0.9):
    """Squared, bounded tilt cost with a dead zone; angle parameters are radians."""
    if not (0.0 <= free_angle < fail_angle <= math.pi):
        raise ValueError("Tilt thresholds must satisfy 0 <= free_angle < fail_angle <= pi")
    return ((object_tilt_angle(env) - free_angle) / (fail_angle - free_angle)).clamp(0.0, 1.0).square()


def time_cost(env):
    """One unit per second before applying the configured negative weight."""
    return torch.ones(env.num_envs, device=env.device)


def success_event(env, termination_name="success"):
    """Terminal success indicator with RewardManager's dt multiplication undone.

    Terminations are evaluated before rewards and successful environments reset
    immediately afterward, so the indicator is paid on exactly one transition.
    """
    return env.termination_manager.get_term(termination_name).float() / env.step_dt


def failure_event(env, failure_terms):
    """One failure penalty per terminal transition, even if several failures fire."""
    failed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in failure_terms:
        if name in ("success", "time_out"):
            raise ValueError("failure_terms must exclude success and time_out")
        failed |= env.termination_manager.get_term(name)
    return failed.float() / env.step_dt
