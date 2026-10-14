"""Fixed world-+Y command for the single-stage sweep task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils import configclass

from .task_math import contact_force_mask


class FixedSweepCommand(CommandTerm):
    """Create one fixed world-+Y sweep goal without internal task phases."""

    cfg: "FixedSweepCommandCfg"

    def __init__(self, cfg: "FixedSweepCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_name]
        self.target: RigidObject = env.scene[cfg.object_name]
        self.shelf: RigidObject = env.scene[cfg.shelf_name]
        self.frames: FrameTransformer = env.scene[cfg.frame_name]
        self.target_contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]

        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.direction_w[:, 1] = 1.0
        self.initial_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

        for name in (
            "precontact_distance",
            "push_pose_distance",
            "upright_alignment",
            "goal_distance",
            "object_speed",
            "progress_ratio",
            "target_contact",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _as_env_ids(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def contact_point_pos_w(self) -> torch.Tensor:
        """Return the virtual point at the physical center of the open gripper."""
        return self.frames.data.target_pos_w[:, self.cfg.contact_frame_index]

    def wrist_pos_w(self) -> torch.Tensor:
        """Return the tracked wrist position used by the pushing gate."""
        return self.frames.data.target_pos_w[:, self.cfg.wrist_frame_index]

    def target_contact(self) -> torch.Tensor:
        force_matrix_w = self.target_contact_sensor.data.force_matrix_w
        if force_matrix_w is None:
            raise RuntimeError(
                f"Contact sensor '{self.cfg.contact_sensor_name}' has no filtered force matrix."
            )
        return contact_force_mask(
            force_matrix_w,
            force_threshold=self.cfg.contact_force_threshold,
            fail_on_nonfinite=False,
        )

    def precontact_point_w(self) -> torch.Tensor:
        return (
            self.target.data.root_pos_w
            - self.cfg.precontact_standoff * self.direction_w
            + self.cfg.external_finger_offset * self.shelf_up_axis_w()
        )

    def push_point_w(self) -> torch.Tensor:
        return (
            self.target.data.root_pos_w
            - self.cfg.push_standoff * self.direction_w
            + self.cfg.external_finger_offset * self.shelf_up_axis_w()
        )

    def upright_alignment(self) -> torch.Tensor:
        eef_rotation = math_utils.matrix_from_quat(
            self.frames.data.target_quat_w[:, self.cfg.eef_frame_index]
        )
        eef_y_axis = eef_rotation[..., 1]
        shelf_z_axis = math_utils.matrix_from_quat(self.shelf.data.root_quat_w)[..., 2]
        return torch.sum(eef_y_axis * shelf_z_axis, dim=-1)

    def shelf_up_axis_w(self) -> torch.Tensor:
        """Return the shelf-local up direction in world coordinates."""
        return math_utils.matrix_from_quat(self.shelf.data.root_quat_w)[..., 2]

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        ids = self._as_env_ids(env_ids)
        if ids.numel() == 0:
            return
        self.initial_pos_w[ids] = self.target.data.root_pos_w[ids]
        self.goal_pos_w[ids] = self.initial_pos_w[ids] + (
            self.cfg.sweep_distance * self.direction_w[ids]
        )
        goal_pos_b, _ = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w[ids],
            self.robot.data.root_quat_w[ids],
            self.goal_pos_w[ids],
        )
        self.goal_pos_b[ids] = goal_pos_b
        self._command[ids] = goal_pos_b

    def _state_values(self) -> tuple[torch.Tensor, ...]:
        contact_pos_w = self.contact_point_pos_w()
        precontact_distance = torch.linalg.norm(
            contact_pos_w - self.precontact_point_w(),
            dim=-1,
        )
        push_pose_distance = torch.linalg.norm(
            contact_pos_w - self.push_point_w(), dim=-1
        )
        upright_alignment = self.upright_alignment()
        goal_distance = torch.linalg.norm(
            self.target.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        object_speed = torch.linalg.norm(self.target.data.root_lin_vel_w, dim=-1)
        progress = torch.sum(
            (self.target.data.root_pos_w - self.initial_pos_w) * self.direction_w,
            dim=-1,
        )
        target_contact = self.target_contact()
        return (
            precontact_distance,
            push_pose_distance,
            upright_alignment,
            goal_distance,
            object_speed,
            progress,
            target_contact,
        )

    def _update_metrics(self) -> None:
        (
            precontact_distance,
            push_pose_distance,
            upright_alignment,
            goal_distance,
            object_speed,
            progress,
            target_contact,
        ) = self._state_values()
        self.metrics["precontact_distance"][:] = precontact_distance
        self.metrics["push_pose_distance"][:] = push_pose_distance
        self.metrics["upright_alignment"][:] = upright_alignment
        self.metrics["goal_distance"][:] = goal_distance
        self.metrics["object_speed"][:] = object_speed
        self.metrics["progress_ratio"][:] = progress / self.cfg.sweep_distance
        self.metrics["target_contact"][:] = target_contact.float()

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "object_visualizer"):
                self.object_visualizer = VisualizationMarkers(self.cfg.object_marker_cfg)
                self.precontact_visualizer = VisualizationMarkers(self.cfg.precontact_marker_cfg)
                self.push_visualizer = VisualizationMarkers(self.cfg.push_marker_cfg)
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_marker_cfg)
                self.contact_visualizer = VisualizationMarkers(
                    self.cfg.contact_marker_cfg
                )
            for visualizer in (
                self.object_visualizer,
                self.precontact_visualizer,
                self.push_visualizer,
                self.goal_visualizer,
                self.contact_visualizer,
            ):
                visualizer.set_visibility(True)
        elif hasattr(self, "object_visualizer"):
            for visualizer in (
                self.object_visualizer,
                self.precontact_visualizer,
                self.push_visualizer,
                self.goal_visualizer,
                self.contact_visualizer,
            ):
                visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self.target.is_initialized:
            return
        self.object_visualizer.visualize(translations=self.target.data.root_pos_w)
        self.precontact_visualizer.visualize(translations=self.precontact_point_w())
        self.push_visualizer.visualize(translations=self.push_point_w())
        self.goal_visualizer.visualize(translations=self.goal_pos_w)
        self.contact_visualizer.visualize(translations=self.contact_point_pos_w())


@configclass
class FixedSweepCommandCfg(CommandTermCfg):
    class_type: type = FixedSweepCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    robot_name: str = MISSING
    object_name: str = MISSING
    shelf_name: str = MISSING
    frame_name: str = MISSING
    eef_frame_index: int = 0
    contact_frame_index: int = 0
    wrist_frame_index: int = 0
    contact_sensor_name: str = MISSING
    contact_force_threshold: float = 0.25
    precontact_standoff: float = 0.06
    push_standoff: float = 0.04
    external_finger_offset: float = 0.055
    sweep_distance: float = 0.40
    target_speed: float = 0.10
    debug_vis: bool = False
    object_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SweepBasic/ObjectRoot",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.009,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
            )
        },
    )
    precontact_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SweepBasic/Precontact",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.009,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
            )
        },
    )
    push_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SweepBasic/PushPoint",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.009,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.1)),
            )
        },
    )
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SweepBasic/Goal",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
            )
        },
    )
    contact_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SweepBasic/EEFContactCenter",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.007,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.9)),
            )
        },
    )
