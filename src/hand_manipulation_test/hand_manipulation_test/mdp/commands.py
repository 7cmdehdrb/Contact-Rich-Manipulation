"""Episode-fixed, massless reaching targets and EEF reset references."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from ..geometry import control_point_pose_w


class EpisodeTargetPositionCommand(CommandTerm):
    """Sample one target per reset without spawning any physical target body."""

    cfg: "EpisodeTargetPositionCommandCfg"

    def __init__(self, cfg, env) -> None:
        low = torch.as_tensor(cfg.target_position_range_low, device=env.device, dtype=torch.float32)
        high = torch.as_tensor(cfg.target_position_range_high, device=env.device, dtype=torch.float32)
        if low.shape != (3,) or high.shape != (3,) or not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("Target bounds must contain three finite positions")
        if torch.any(high < low):
            raise ValueError("Each target upper bound must be at least its lower bound")
        self._range_low = low
        self._range_high = high
        self.target_pos_w = (0.5 * (low + high)).expand(env.num_envs, -1).clone() + env.scene.env_origins
        position, quaternion = control_point_pose_w(env)
        self.initial_eef_pos_w = position.clone()
        self.initial_eef_quat_w = quaternion.clone()
        self._target_marker = None
        super().__init__(cfg, env)
        self.metrics["distance_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reached"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos_w

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        index = torch.arange(self.num_envs, device=self.device)[env_ids]
        position, quaternion = control_point_pose_w(self._env)
        self.initial_eef_pos_w[index] = position[index]
        self.initial_eef_quat_w[index] = quaternion[index]
        sampled = self._range_low + torch.rand((index.numel(), 3), device=self.device) * (
            self._range_high - self._range_low
        )
        self.target_pos_w[index] = sampled + self._env.scene.env_origins[index]

    def _update_command(self) -> None:
        # Targets and reset references remain fixed between episode resets.
        pass

    def _update_metrics(self) -> None:
        position, _ = control_point_pose_w(self._env)
        distance = torch.linalg.vector_norm(position - self.target_pos_w, dim=-1)
        self.metrics["distance_m"][:] = distance
        self.metrics["reached"][:] = (distance <= self._env.cfg.task.success_distance_m).float()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis and self._target_marker is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._target_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/HandManipulationTest/Target",
                    markers={
                        "target": sim_utils.SphereCfg(
                            radius=0.025,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.35, 0.02), opacity=0.65
                            ),
                        ),
                    },
                )
            )
        if self._target_marker is not None:
            self._target_marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        del event
        if self._target_marker is None:
            return
        robot = self._env.scene[self.cfg.asset_name]
        if not robot.is_initialized:
            return
        self._target_marker.visualize(translations=self.target_pos_w)


@configclass
class EpisodeTargetPositionCommandCfg(CommandTermCfg):
    class_type: type = EpisodeTargetPositionCommand
    asset_name: str = "robot"
    debug_vis: bool = True
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    target_position_range_low: tuple[float, float, float] = (-0.77, -0.22, 1.08)
    target_position_range_high: tuple[float, float, float] = (-0.58, 0.22, 1.08)


__all__ = ["EpisodeTargetPositionCommand", "EpisodeTargetPositionCommandCfg"]
