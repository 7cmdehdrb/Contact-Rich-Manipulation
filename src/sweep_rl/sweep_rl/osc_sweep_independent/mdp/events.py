"""Object geometry and reset randomization for the standalone task."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.events import randomize_rigid_body_scale
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.utils.stage import get_current_stage

TARGET_SIZE_BUFFER = "_independent_sweep_target_sizes"
INITIAL_VISUALIZER = "_independent_sweep_initial_position_visualizer"
GOAL_VISUALIZER = "_independent_sweep_goal_position_visualizer"


def create_sweep_position_visualizers(env, env_ids: torch.Tensor | None) -> None:
    """Create single-environment GUI markers before simulation startup.

    Training does not need these markers.  Avoiding PointInstancers in headless or
    vectorized runs also keeps Fabric from trying to reconcile debug prototypes
    across cloned environments.
    """
    del env_ids
    if not env.sim.has_gui() or env.scene.num_envs != 1:
        return
    if hasattr(env, INITIAL_VISUALIZER):
        return
    initial_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/sweep_target_positions/initial",
        markers={
            "initial": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.10, 0.35, 1.00),
                    emissive_color=(0.02, 0.08, 0.30),
                    opacity=0.35,
                ),
            )
        },
    )
    goal_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/sweep_target_positions/goal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.050,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.00, 0.05, 0.75),
                    emissive_color=(0.40, 0.01, 0.15),
                    opacity=0.90,
                ),
            )
        },
    )
    setattr(env, INITIAL_VISUALIZER, VisualizationMarkers(initial_cfg))
    setattr(env, GOAL_VISUALIZER, VisualizationMarkers(goal_cfg))


def randomize_target_cube_size(
    env,
    env_ids: torch.Tensor | None,
    size_range: tuple[float, float],
    base_size: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> None:
    """Sample one cube size per parallel environment before simulation starts."""
    if base_size <= 0.0 or size_range[0] <= 0.0 or size_range[1] < size_range[0]:
        raise ValueError("Cube size range must be positive and ordered.")
    randomize_rigid_body_scale(
        env,
        env_ids,
        scale_range=(size_range[0] / base_size, size_range[1] / base_size),
        asset_cfg=asset_cfg,
    )
    selected_ids = (
        torch.arange(env.scene.num_envs, device="cpu")
        if env_ids is None
        else env_ids.to(device="cpu", dtype=torch.long)
    )
    prim_paths = sim_utils.find_matching_prim_paths(env.scene[asset_cfg.name].cfg.prim_path)
    stage = get_current_stage()
    sampled_sizes: list[float] = []
    for env_id in selected_ids.tolist():
        scale = stage.GetPrimAtPath(prim_paths[env_id]).GetAttribute("xformOp:scale").Get()
        if scale is None:
            raise RuntimeError(f"Missing scale attribute on '{prim_paths[env_id]}'.")
        sampled_sizes.append(base_size * float(scale[0]))

    if not hasattr(env, TARGET_SIZE_BUFFER):
        setattr(
            env,
            TARGET_SIZE_BUFFER,
            torch.full((env.scene.num_envs,), base_size, device=env.device),
        )
    buffer = getattr(env, TARGET_SIZE_BUFFER)
    buffer[selected_ids.to(env.device)] = torch.tensor(sampled_sizes, device=env.device)


def reset_variable_size_target(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    table_top_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> None:
    """Reset every cube upright with its bottom face on the table."""
    if not hasattr(env, TARGET_SIZE_BUFFER):
        raise RuntimeError("Target sizes were not initialized by the prestartup event.")
    if len(env_ids) == 0:
        return
    target: RigidObject = env.scene[asset_cfg.name]
    count = len(env_ids)
    state = target.data.default_root_state[env_ids].clone()
    position = state[:, :3] + env.scene.env_origins[env_ids]
    position[:, 0] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("x", (0.0, 0.0))
    )
    position[:, 1] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("y", (0.0, 0.0))
    )
    position[:, 2] = table_top_height + 0.5 * getattr(env, TARGET_SIZE_BUFFER)[env_ids]
    yaw = torch.empty(count, device=env.device).uniform_(
        *pose_range.get("yaw", (0.0, 0.0))
    )
    zero = torch.zeros_like(yaw)
    orientation = math_utils.quat_from_euler_xyz(zero, zero, yaw)
    target.write_root_pose_to_sim(torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    target.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]), env_ids=env_ids)
