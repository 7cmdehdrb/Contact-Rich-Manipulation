"""Fixed world-+Y object goal command for the single Cube."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass


class FixedObjectGoalCommand(CommandTerm):
    """Set the goal from each reset pose without time-based state transitions."""

    cfg: "FixedObjectGoalCommandCfg"

    def __init__(self, cfg: "FixedObjectGoalCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_name]
        self.target: RigidObject = env.scene[cfg.object_name]
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_b = torch.zeros_like(self.goal_pos_w)
        self.initial_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.initial_quat_w[:, 0] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.goal_pos_w, self.initial_quat_w), dim=-1)

    def _env_ids(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        ids = self._env_ids(env_ids)
        if ids.numel() == 0:
            return
        goal_offset = torch.tensor(
            self.cfg.goal_offset, dtype=torch.float32, device=self.device
        )
        self.goal_pos_w[ids] = self.target.data.root_pos_w[ids] + goal_offset
        self.initial_quat_w[ids] = self.target.data.root_quat_w[ids]
        self.goal_pos_b[ids], _ = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w[ids],
            self.robot.data.root_quat_w[ids],
            self.goal_pos_w[ids],
        )

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_marker_cfg)
                self.target_visualizer = VisualizationMarkers(self.cfg.target_marker_cfg)
            self.goal_visualizer.set_visibility(True)
            self.target_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self.target.is_initialized:
            return
        self.goal_visualizer.visualize(self.goal_pos_w, self.initial_quat_w)
        self.target_visualizer.visualize(
            self.target.data.root_pos_w, self.target.data.root_quat_w
        )


@configclass
class FixedObjectGoalCommandCfg(CommandTermCfg):
    class_type: type = FixedObjectGoalCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    robot_name: str = MISSING
    object_name: str = MISSING
    goal_offset: tuple[float, float, float] = (0.0, 0.18, 0.0)
    debug_vis: bool = True
    goal_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/SweepPolicyCube/Goal"
    )
    target_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/SweepPolicyCube/Target"
    )

    goal_marker_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
    target_marker_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
