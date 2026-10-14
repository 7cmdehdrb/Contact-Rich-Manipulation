"""Fixed initial Cube target for the inherited contact-reaching task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from ..geometry import control_point_pose_w
from .commands import EpisodeTargetPositionCommand, EpisodeTargetPositionCommandCfg


class CubeInitialPositionCommand(EpisodeTargetPositionCommand):
    """Snapshot the reset Cube centre once, alongside the reset EEF frame."""

    cfg: "CubeInitialPositionCommandCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.target_pos_w[:] = env.scene[cfg.object_name].data.root_pos_w

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        if isinstance(env_ids, slice):
            index = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        position, quaternion = control_point_pose_w(self._env)
        self.initial_eef_pos_w[index] = position[index]
        self.initial_eef_quat_w[index] = quaternion[index]
        # Reset events have already written the Cube's world pose. It already
        # includes env_origins, so adding the origins here would offset twice.
        self.target_pos_w[index] = self._env.scene[self.cfg.object_name].data.root_pos_w[index]


@configclass
class CubeInitialPositionCommandCfg(EpisodeTargetPositionCommandCfg):
    class_type: type = CubeInitialPositionCommand
    object_name: str = "target_object"
    # The physical Cube already marks the task target in the viewport.
    debug_vis: bool = False


__all__ = ["CubeInitialPositionCommand", "CubeInitialPositionCommandCfg"]
