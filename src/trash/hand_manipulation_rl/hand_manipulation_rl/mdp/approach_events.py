"""Reset specialization for the inherited approach-shaped sweep task."""

from __future__ import annotations

import torch

from .events import StableOffsetPoseReset


class OrientationAwareStableOffsetPoseReset(StableOffsetPoseReset):
    """Choose a safe lower C height from the hand's vertical orientation.

    The source hand collision envelope is asymmetric along Hand +X.  The
    parent reset maps +X to either world-up or world-down depending on surface
    mode and sweep direction.  A single lower height would therefore make one
    branch start inside the board.  This subclass keeps the parent's sampling,
    IK, and two collision certificates, but substitutes the certified height
    assigned to the resulting +X orientation.
    """

    def _sample_stable_pose(
        self,
        env,
        env_ids: torch.Tensor,
        *,
        position_offset_task: tuple[float, float, float],
        position_jitter_task: tuple[float, float, float],
        orientation_offset_rpy: tuple[float, float, float],
        orientation_jitter_rpy: tuple[float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        desired_pos, desired_quat = super()._sample_stable_pose(
            env,
            env_ids,
            position_offset_task=position_offset_task,
            position_jitter_task=position_jitter_task,
            orientation_offset_rpy=orientation_offset_rpy,
            orientation_jitter_rpy=orientation_jitter_rpy,
        )
        safe_height = env.safe_approach_height_offset_m(env_ids)
        desired_pos[:, 2] += safe_height - position_offset_task[2]
        return desired_pos, desired_quat


__all__ = ["OrientationAwareStableOffsetPoseReset"]
