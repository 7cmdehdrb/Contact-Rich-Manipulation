"""Per-environment episode buffers shared by reset and MDP terms."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from .commands import direction_from_angle, goal_from_command, is_allowed_sweep_angle, wrap_s_angle


Tensor = torch.Tensor

PALM_MODE = 0
DORSAL_MODE = 1
ACTION_DIM = 8

OBSERVATION_LAYOUT: tuple[tuple[str, int], ...] = (
    ("arm_joint_position", 6),
    ("arm_joint_velocity", 6),
    ("c_relative_position", 3),
    ("c_relative_orientation", 3),
    ("hand_state", 2),
    ("surface_header_and_tactile", 18),
    ("wrist_wrench_c", 6),
    ("initial_object_relative_position", 3),
    ("command_direction_y_s", 1),
    ("command_distance", 1),
    ("last_action", 8),
)


def _build_observation_slices() -> Mapping[str, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    for name, width in OBSERVATION_LAYOUT:
        slices[name] = slice(offset, offset + width)
        offset += width
    return MappingProxyType(slices)


OBSERVATION_SLICES = _build_observation_slices()
OBSERVATION_DIM = sum(width for _, width in OBSERVATION_LAYOUT)


@dataclass
class EpisodeState:
    """Tensor buffers for command identity, reset references, and latches."""

    mode: Tensor
    command_angle_s: Tensor
    command_distance: Tensor
    command_direction_s: Tensor
    initial_object_position_s: Tensor
    goal_position_s: Tensor
    c0_position_w: Tensor
    c0_rotation_w: Tensor
    initial_object_upright_s: Tensor
    initial_object_height_s: Tensor
    initial_hand_state: Tensor
    previous_action: Tensor
    previous_action_valid: Tensor
    sensor_valid: Tensor
    board_force_peak: Tensor
    topple_latched: Tensor
    board_contact_latched: Tensor
    height_latched: Tensor
    termination_reason: Tensor

    def __post_init__(self) -> None:
        count = self.mode.shape[0]
        for name, value in vars(self).items():
            if value.ndim == 0 or value.shape[0] != count:
                raise ValueError(f"{name} must have leading environment dimension {count}")
        expected_shapes = {
            "command_direction_s": (count, 3),
            "initial_object_position_s": (count, 3),
            "goal_position_s": (count, 3),
            "c0_position_w": (count, 3),
            "c0_rotation_w": (count, 3, 3),
            "initial_object_upright_s": (count, 3),
            "initial_hand_state": (count, 2),
            "previous_action": (count, ACTION_DIM),
        }
        for name, expected in expected_shapes.items():
            actual = tuple(getattr(self, name).shape)
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}; got {actual}")

    @property
    def num_envs(self) -> int:
        return self.mode.shape[0]

    @classmethod
    def allocate(
        cls,
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "EpisodeState":
        """Allocate deterministic buffers without sampling an episode spec."""

        if num_envs < 0:
            raise ValueError("num_envs must be non-negative")
        zeros = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)
        false = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)
        identity = torch.eye(3, dtype=dtype, device=device).expand(num_envs, 3, 3).clone()
        upright = zeros(num_envs, 3)
        if num_envs:
            upright[:, 2] = 1.0
        return cls(
            mode=torch.full((num_envs,), -1, dtype=torch.int64, device=device),
            command_angle_s=zeros(num_envs),
            command_distance=zeros(num_envs),
            command_direction_s=zeros(num_envs, 3),
            initial_object_position_s=zeros(num_envs, 3),
            goal_position_s=zeros(num_envs, 3),
            c0_position_w=zeros(num_envs, 3),
            c0_rotation_w=identity,
            initial_object_upright_s=upright,
            initial_object_height_s=zeros(num_envs),
            initial_hand_state=zeros(num_envs, 2),
            previous_action=zeros(num_envs, ACTION_DIM),
            previous_action_valid=false(),
            sensor_valid=false(),
            board_force_peak=zeros(num_envs),
            topple_latched=false(),
            board_contact_latched=false(),
            height_latched=false(),
            termination_reason=torch.zeros(num_envs, dtype=torch.int64, device=device),
        )

    def _index(self, env_ids: Tensor | list[int] | tuple[int, ...] | None) -> slice | Tensor:
        if env_ids is None:
            return slice(None)
        return torch.as_tensor(env_ids, dtype=torch.int64, device=self.mode.device)

    @staticmethod
    def _as_target(value: Tensor | float | int, target: Tensor) -> Tensor:
        tensor = torch.as_tensor(value, dtype=target.dtype, device=target.device)
        try:
            return torch.broadcast_to(tensor, target.shape)
        except RuntimeError as error:
            raise ValueError(
                f"value with shape {tuple(tensor.shape)} cannot fill target {tuple(target.shape)}"
            ) from error

    def set_episode_spec(
        self,
        env_ids: Tensor | list[int] | tuple[int, ...] | None,
        *,
        mode: Tensor | int,
        angle_s: Tensor | float,
        distance: Tensor | float,
        initial_object_position_s: Tensor,
    ) -> None:
        """Store one authoritative command spec for the selected environments."""

        index = self._index(env_ids)
        mode_value = self._as_target(mode, self.mode[index])
        if torch.any((mode_value != PALM_MODE) & (mode_value != DORSAL_MODE)):
            raise ValueError("mode must be PALM_MODE (0) or DORSAL_MODE (1)")

        angle_value = self._as_target(angle_s, self.command_angle_s[index])
        angle_value = wrap_s_angle(angle_value)
        if torch.any(~is_allowed_sweep_angle(angle_value)):
            raise ValueError("angle_s must select shelf-left or shelf-right (+/-pi/2)")

        distance_value = self._as_target(distance, self.command_distance[index])
        if torch.any(distance_value < 0):
            raise ValueError("distance must be non-negative")
        position_value = self._as_target(
            initial_object_position_s, self.initial_object_position_s[index]
        )
        direction_value = direction_from_angle(angle_value)
        goal_value = goal_from_command(position_value, angle_value, distance_value)

        self.mode[index] = mode_value
        self.command_angle_s[index] = angle_value
        self.command_distance[index] = distance_value
        self.command_direction_s[index] = direction_value
        self.initial_object_position_s[index] = position_value
        self.goal_position_s[index] = goal_value

    def reset_transient(
        self, env_ids: Tensor | list[int] | tuple[int, ...] | None = None
    ) -> None:
        """Clear action/sensor history and substep failure latches."""

        index = self._index(env_ids)
        self.previous_action[index] = 0
        self.previous_action_valid[index] = False
        self.sensor_valid[index] = False
        self.board_force_peak[index] = 0
        self.topple_latched[index] = False
        self.board_contact_latched[index] = False
        self.height_latched[index] = False
        self.termination_reason[index] = 0

    def record_action(
        self,
        action: Tensor,
        env_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Record an executed normalized policy action."""

        index = self._index(env_ids)
        self.previous_action[index] = self._as_target(action, self.previous_action[index])
        self.previous_action_valid[index] = True

    def update_board_force_peak(
        self,
        board_force: Tensor,
        env_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Retain the maximum robot-board load seen in a policy interval."""

        index = self._index(env_ids)
        force = self._as_target(board_force, self.board_force_peak[index])
        self.board_force_peak[index] = torch.maximum(self.board_force_peak[index], force)

    def latch_failures(
        self,
        *,
        topple: Tensor | bool,
        board_contact: Tensor | bool,
        height: Tensor | bool,
        env_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """OR physics-substep hard-threshold results into episode latches."""

        index = self._index(env_ids)
        self.topple_latched[index] |= self._as_target(topple, self.topple_latched[index])
        self.board_contact_latched[index] |= self._as_target(
            board_contact, self.board_contact_latched[index]
        )
        self.height_latched[index] |= self._as_target(height, self.height_latched[index])
