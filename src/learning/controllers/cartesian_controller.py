"""Cartesian action conversion for reaching."""

from __future__ import annotations

from collections.abc import Callable

import torch


ForwardKinematicsFn = Callable[[torch.Tensor], torch.Tensor]


class CartesianController:
    """Maps 3D EEF displacement actions to joint deltas with differential IK."""

    def __init__(self, action_scale: float, damping: float, max_delta: float, jacobian_eps: float = 1.0e-3) -> None:
        self.action_scale = action_scale
        self.damping = damping
        self.max_delta = max_delta
        self.jacobian_eps = jacobian_eps

    def to_joint_delta(
        self,
        actions: torch.Tensor,
        joint_pos: torch.Tensor,
        forward_kinematics: ForwardKinematicsFn,
    ) -> torch.Tensor:
        if actions.shape[-1] != 3:
            raise ValueError(f"Cartesian actions must have 3 dimensions, got {actions.shape[-1]}.")

        delta_x = torch.clamp(actions, -1.0, 1.0) * self.action_scale
        jacobian = self._finite_difference_jacobian(joint_pos, forward_kinematics)
        jjt = jacobian @ jacobian.transpose(1, 2)
        eye = torch.eye(3, device=joint_pos.device, dtype=joint_pos.dtype).expand(joint_pos.shape[0], 3, 3)
        lhs = jjt + (self.damping**2) * eye
        solved = torch.linalg.solve(lhs, delta_x.unsqueeze(-1))
        joint_delta = jacobian.transpose(1, 2) @ solved
        return torch.clamp(joint_delta.squeeze(-1), -self.max_delta, self.max_delta)

    def _finite_difference_jacobian(
        self,
        joint_pos: torch.Tensor,
        forward_kinematics: ForwardKinematicsFn,
    ) -> torch.Tensor:
        columns = []
        for joint_id in range(joint_pos.shape[-1]):
            offset = torch.zeros_like(joint_pos)
            offset[:, joint_id] = self.jacobian_eps
            plus = forward_kinematics(joint_pos + offset)
            minus = forward_kinematics(joint_pos - offset)
            columns.append((plus - minus) / (2.0 * self.jacobian_eps))
        return torch.stack(columns, dim=-1)
