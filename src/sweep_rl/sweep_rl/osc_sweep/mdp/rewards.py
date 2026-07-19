"""Direction, distance, and force-centered reward formulation."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .common import (
    active_gripper_side_direction_b,
    desired_direction_b,
    filtered_contact_mask,
    object_displacement_b,
    side_pad_contact_quality,
    target_contact_data_w,
)


def reaching_precontact_pose(
    env,
    command_name: str,
    std: float,
    stand_off: float,
    eef_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward approaching behind the cube along the requested push direction."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    initial_pos_b = command.initial_pose_b[:, :3]
    direction_b = desired_direction_b(env, command_name)
    precontact_b = initial_pos_b - stand_off * direction_b
    precontact_w, _ = math_utils.combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        precontact_b,
    )
    distance = torch.linalg.norm(eef_pos_w - precontact_w, dim=-1)
    return torch.exp(-torch.square(distance / std))


def current_precontact_pose_error(
    env,
    command_name: str,
    distance_scale: float,
    stand_off: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Normalized EEF error from the moving object's desired push pose.

    This is intended as a penalty.  Unlike ``reaching_precontact_pose``, the
    reference follows the object, so pushing it away from its initial pose does
    not sacrifice an accumulated positive reward.
    """
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive.")
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    precontact_w = target.data.root_pos_w - stand_off * command.direction_w
    error = torch.linalg.norm(eef_pos_w - precontact_w, dim=-1)
    return torch.clamp(error / distance_scale, max=3.0)


def external_pad_precontact_pose_error(
    env,
    command_name: str,
    distance_scale: float,
    stand_off: float,
    pad_center_offset: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Guide the object toward either pad center, outside the gripper gap.

    In the EEF frame, local X is the pad-face normal and the two pads are
    centered at local Y = +/- ``pad_center_offset``.  The nearer pad is used as
    the target, with positive Y breaking an exact tie.  This leaves the world
    orientation free while making the dense approach target consistent with
    the gripper-interior termination.
    """
    if distance_scale <= 0.0 or stand_off <= 0.0 or pad_center_offset <= 0.0:
        raise ValueError("Pre-contact distances must be positive.")

    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_body_id = eef_cfg.body_ids[0]
    eef_pos_w = robot.data.body_pos_w[:, eef_body_id]
    eef_quat_w = robot.data.body_quat_w[:, eef_body_id]
    object_pos_eef, _ = math_utils.subtract_frame_transforms(
        eef_pos_w,
        eef_quat_w,
        target.data.root_pos_w,
    )
    direction_eef = math_utils.quat_apply_inverse(
        eef_quat_w, command.direction_w
    )

    face_sign = torch.where(direction_eef[:, 0] >= 0.0, 1.0, -1.0)
    pad_sign = torch.where(object_pos_eef[:, 1] >= 0.0, 1.0, -1.0)
    desired_object_pos_eef = torch.zeros_like(object_pos_eef)
    desired_object_pos_eef[:, 0] = face_sign * stand_off
    desired_object_pos_eef[:, 1] = pad_sign * pad_center_offset

    error = torch.linalg.norm(
        object_pos_eef - desired_object_pos_eef, dim=-1
    )
    return torch.clamp(error / distance_scale, max=3.0)


def variable_size_precontact_pose_error(
    env,
    command_name: str,
    distance_scale: float,
    surface_clearance: float,
    table_side_pad_offset: float,
    size_buffer_name: str,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Pre-contact error for pushing with the table-side pad.

    The EEF center is raised above the cube by ``table_side_pad_offset`` so
    that the lower pad, rather than the open gripper gap, is centered on the
    object.  The horizontal stand-off still follows the randomized cube size.
    """
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive.")
    if surface_clearance < 0.0:
        raise ValueError("surface_clearance must be non-negative.")
    if table_side_pad_offset < 0.0:
        raise ValueError("table_side_pad_offset must be non-negative.")
    if not hasattr(env, size_buffer_name):
        raise RuntimeError(f"Environment has no cube-size buffer '{size_buffer_name}'.")

    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    side_lengths = getattr(env, size_buffer_name)
    stand_off = 0.5 * side_lengths + surface_clearance
    precontact_w = target.data.root_pos_w - stand_off.unsqueeze(-1) * command.direction_w
    precontact_w[:, 2] += table_side_pad_offset
    error = torch.linalg.norm(eef_pos_w - precontact_w, dim=-1)
    return torch.clamp(error / distance_scale, max=3.0)


def object_velocity_along_direction(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Signed object velocity along the desired world-frame direction."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    return torch.sum(target.data.root_lin_vel_w * command.direction_w, dim=-1)


def normalized_progress(
    env,
    command_name: str,
) -> torch.Tensor:
    """Current longitudinal displacement divided by desired sweep length."""
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    length = env.command_manager.get_command(command_name)[:, 2]
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    return torch.clamp(progress / torch.clamp(length, min=1.0e-6), -1.0, 1.2)


def direction_alignment(
    env,
    command_name: str,
    minimum_displacement: float = 0.005,
) -> torch.Tensor:
    """Cosine alignment between actual XY displacement and desired direction."""
    displacement_b = object_displacement_b(env, command_name)
    displacement_xy = displacement_b[:, :2]
    norm = torch.linalg.norm(displacement_xy, dim=-1)
    direction_xy = env.command_manager.get_command(command_name)[:, :2]
    cosine = torch.sum(displacement_xy * direction_xy, dim=-1) / torch.clamp(
        norm, min=1.0e-6
    )
    return torch.where(norm > minimum_displacement, torch.clamp(cosine, -1.0, 1.0), 0.0)


def lateral_displacement(
    env,
    command_name: str,
) -> torch.Tensor:
    """Normalized displacement perpendicular to the desired direction."""
    displacement_b = object_displacement_b(env, command_name)
    direction = env.command_manager.get_command(command_name)[:, :2]
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    lateral = torch.abs(torch.sum(displacement_b[:, :2] * perpendicular, dim=-1))
    length = env.command_manager.get_command(command_name)[:, 2]
    return lateral / torch.clamp(length, min=1.0e-6)


def endpoint_tracking(
    env,
    command_name: str,
    std: float,
    coarse_std: float | None = None,
    coarse_weight: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Gaussian reward for stopping at the requested sweep endpoint.

    An optional wider Gaussian supplies a useful gradient while the object is
    still far from the endpoint.  The narrow component retains the precision
    incentive close to the goal.
    """
    if std <= 0.0:
        raise ValueError("std must be positive.")
    if not 0.0 <= coarse_weight <= 1.0:
        raise ValueError("coarse_weight must be in [0, 1].")
    if coarse_weight > 0.0 and (coarse_std is None or coarse_std <= 0.0):
        raise ValueError("coarse_std must be positive when coarse_weight is used.")

    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    error = torch.linalg.norm(target.data.root_pos_w - command.goal_pos_w, dim=-1)
    fine_tracking = torch.exp(-torch.square(error / std))
    if coarse_weight == 0.0:
        return fine_tracking

    coarse_tracking = torch.exp(-torch.square(error / coarse_std))
    return (1.0 - coarse_weight) * fine_tracking + coarse_weight * coarse_tracking


def normalized_endpoint_error(
    env,
    command_name: str,
    maximum_error: float = 2.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Endpoint distance normalized by commanded travel length.

    Used with a negative weight, this makes every stationary step costly and
    continuously reduces that cost as the object approaches the goal.
    """
    if maximum_error <= 0.0:
        raise ValueError("maximum_error must be positive.")
    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    error = torch.linalg.norm(
        target.data.root_pos_w - command_term.goal_pos_w, dim=-1
    )
    normalized = error / torch.clamp(command[:, 2], min=1.0e-6)
    return torch.clamp(normalized, max=maximum_error)


def overshoot_penalty(env, command_name: str) -> torch.Tensor:
    """Penalty for longitudinal motion beyond the desired length."""
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    command = env.command_manager.get_command(command_name)
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    return torch.relu(progress - command[:, 2]) / torch.clamp(command[:, 2], min=1.0e-6)


def target_force_tracking(
    env,
    command_name: str,
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
) -> torch.Tensor:
    """Track requested target-contact force magnitude within its sampled band."""
    _, force_w, contact_mask = target_contact_data_w(env, sensor_names)
    measured_force = torch.linalg.norm(force_w, dim=-1)
    command = env.command_manager.get_command(command_name)
    desired_force = command[:, 3]
    tolerance = torch.clamp(command[:, 4], min=1.0e-3)
    tracking = torch.exp(-torch.square((measured_force - desired_force) / tolerance))
    # Force is meaningful only while the gripper is touching the target.
    return tracking * contact_mask.float()


def gripper_side_direction_alignment(
    env,
    command_name: str,
    side_axis_local: tuple[float, float, float],
    proximity_std: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Align the active broad-side normal with the requested sweep direction."""
    if proximity_std <= 0.0:
        raise ValueError("proximity_std must be positive.")

    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_body_id = eef_cfg.body_ids[0]
    active_side_b = active_gripper_side_direction_b(
        env,
        eef_cfg=eef_cfg,
        object_cfg=object_cfg,
        side_axis_local=side_axis_local,
    )
    desired_b = desired_direction_b(env, command_name)
    side_xy = active_side_b[:, :2]
    side_xy = side_xy / torch.clamp(
        torch.linalg.norm(side_xy, dim=-1, keepdim=True), min=1.0e-6
    )
    alignment = torch.sum(side_xy * desired_b[:, :2], dim=-1)

    eef_pos_w = robot.data.body_pos_w[:, eef_body_id]
    distance = torch.linalg.norm(target.data.root_pos_w - eef_pos_w, dim=-1)
    proximity = torch.exp(-torch.square(distance / proximity_std))
    return torch.clamp(alignment, -1.0, 1.0) * proximity


def gripper_side_direction_error(
    env,
    command_name: str,
    side_axis_local: tuple[float, float, float],
    proximity_std: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize pad-direction error only while the EEF is near the object."""
    if proximity_std <= 0.0:
        raise ValueError("proximity_std must be positive.")

    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    active_side_b = active_gripper_side_direction_b(
        env,
        eef_cfg=eef_cfg,
        object_cfg=object_cfg,
        side_axis_local=side_axis_local,
    )
    desired_b = desired_direction_b(env, command_name)
    side_xy = active_side_b[:, :2]
    side_xy = side_xy / torch.clamp(
        torch.linalg.norm(side_xy, dim=-1, keepdim=True), min=1.0e-6
    )
    alignment = torch.clamp(
        torch.sum(side_xy * desired_b[:, :2], dim=-1), 0.0, 1.0
    )

    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    distance = torch.linalg.norm(target.data.root_pos_w - eef_pos_w, dim=-1)
    proximity = torch.exp(-torch.square(distance / proximity_std))
    return proximity * (1.0 - alignment)


def eef_axis_upright_alignment(
    env,
    local_up_axis: tuple[float, float, float],
    allowed_deviation_deg: float,
    zero_reward_deviation_deg: float,
    eef_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward an upright EEF axis with an angular tolerance cone.

    Orientations inside ``allowed_deviation_deg`` receive the same full
    reward.  Outside that cone the reward decreases smoothly and reaches zero
    at ``zero_reward_deviation_deg``.  This makes upright a preference rather
    than an exact orientation constraint.
    """
    if not 0.0 <= allowed_deviation_deg < zero_reward_deviation_deg <= 180.0:
        raise ValueError(
            "Expected 0 <= allowed_deviation_deg < "
            "zero_reward_deviation_deg <= 180."
        )
    axis = torch.tensor(local_up_axis, dtype=torch.float32, device=env.device)
    axis_norm = torch.linalg.norm(axis)
    if axis_norm < 1.0e-6:
        raise ValueError("local_up_axis must be non-zero.")
    axis = axis / axis_norm

    robot: Articulation = env.scene[eef_cfg.name]
    eef_quat_w = robot.data.body_quat_w[:, eef_cfg.body_ids[0]]
    axis_w = math_utils.quat_apply(eef_quat_w, axis.expand(env.num_envs, -1))
    cosine = torch.clamp(axis_w[:, 2], -1.0, 1.0)
    full_reward_cosine = math.cos(math.radians(allowed_deviation_deg))
    zero_reward_cosine = math.cos(math.radians(zero_reward_deviation_deg))
    score = torch.clamp(
        (cosine - zero_reward_cosine)
        / max(full_reward_cosine - zero_reward_cosine, 1.0e-6),
        0.0,
        1.0,
    )
    return score * score * (3.0 - 2.0 * score)


def object_in_gripper_gap(
    env,
    gap_axis: int,
    gap_half_width: float,
    transition_width: float,
    proximity_scale: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize placing the object center inside the open gripper gap.

    The gap score is active only while the object is near the EEF in the two
    axes perpendicular to the opening axis.  This avoids charging the policy
    merely for approaching from a distant, aligned pose.
    """
    if gap_axis not in (0, 1, 2):
        raise ValueError("gap_axis must be 0, 1, or 2.")
    if gap_half_width <= 0.0 or transition_width <= 0.0 or proximity_scale <= 0.0:
        raise ValueError("Gap dimensions and proximity_scale must be positive.")

    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_body_id = eef_cfg.body_ids[0]
    object_pos_eef, _ = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, eef_body_id],
        robot.data.body_quat_w[:, eef_body_id],
        target.data.root_pos_w,
    )
    gap_distance = torch.abs(object_pos_eef[:, gap_axis])
    inside_score = torch.clamp(
        (gap_half_width - gap_distance) / transition_width, 0.0, 1.0
    )

    perpendicular_mask = torch.ones(3, dtype=object_pos_eef.dtype, device=env.device)
    perpendicular_mask[gap_axis] = 0.0
    perpendicular_distance = torch.linalg.norm(
        object_pos_eef * perpendicular_mask, dim=-1
    )
    proximity = torch.exp(-torch.square(perpendicular_distance / proximity_scale))
    return inside_score * proximity


def dual_pad_target_contact(
    env,
    sensor_names: tuple[str, str] = ("left_contact", "right_contact"),
    force_threshold: float = 0.25,
) -> torch.Tensor:
    """Return one when both pads contact the target at the same time."""
    if len(sensor_names) != 2:
        raise ValueError("dual_pad_target_contact requires exactly two sensors.")
    first = filtered_contact_mask(env, sensor_names[0], force_threshold)
    second = filtered_contact_mask(env, sensor_names[1], force_threshold)
    return (first & second).float()


def side_pad_center_contact(
    env,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
) -> torch.Tensor:
    """Reward target contact at the center of either broad gripper side pad."""
    quality, _ = side_pad_contact_quality(
        env,
        sensor_names=sensor_names,
        pad_size=pad_size,
        face_normal_axis=face_normal_axis,
        center_sigma=center_sigma,
        face_sigma=face_sigma,
    )
    return quality


def off_center_target_contact(
    env,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
) -> torch.Tensor:
    """Penalize target contact on pad edges, narrow faces, or inner faces."""
    quality, contact_mask = side_pad_contact_quality(
        env,
        sensor_names=sensor_names,
        pad_size=pad_size,
        face_normal_axis=face_normal_axis,
        center_sigma=center_sigma,
        face_sigma=face_sigma,
    )
    return contact_mask.float() * (1.0 - quality)


def side_target_force_tracking(
    env,
    command_name: str,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
) -> torch.Tensor:
    """Track desired force only through broad-side central pad contact."""
    _, force_w, _ = target_contact_data_w(env, sensor_names)
    measured_force = torch.linalg.norm(force_w, dim=-1)
    command = env.command_manager.get_command(command_name)
    desired_force = command[:, 3]
    tolerance = torch.clamp(command[:, 4], min=1.0e-3)
    tracking = torch.exp(-torch.square((measured_force - desired_force) / tolerance))
    quality, _ = side_pad_contact_quality(
        env,
        sensor_names=sensor_names,
        pad_size=pad_size,
        face_normal_axis=face_normal_axis,
        center_sigma=center_sigma,
        face_sigma=face_sigma,
    )
    return tracking * quality


def target_contact_bonus(
    env,
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
) -> torch.Tensor:
    """Binary reward for making target-specific contact."""
    _, _, contact_mask = target_contact_data_w(env, sensor_names)
    return contact_mask.float()


def contact_forward_progress(
    env,
    command_name: str,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    endpoint_threshold: float,
    maximum_normalized_speed: float = 1.25,
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward forward object motion caused while target contact is present.

    Contact alone receives only a small bridge reward elsewhere.  This term is
    larger only when contact produces motion in the commanded direction, and
    it fades with the velocity profile as the object approaches the endpoint.
    """
    if endpoint_threshold <= 0.0:
        raise ValueError("endpoint_threshold must be positive.")
    if maximum_normalized_speed <= 0.0:
        raise ValueError("maximum_normalized_speed must be positive.")

    _, _, contact_mask = target_contact_data_w(env, sensor_names)
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    desired_velocity_w, remaining, _ = desired_velocity_profile_w(
        env,
        command_name,
        acceleration_distance=acceleration_distance,
        stopping_distance=stopping_distance,
        initial_speed_fraction=initial_speed_fraction,
    )
    target_speed = torch.clamp(command[:, 3], min=1.0e-3)
    forward_speed = torch.sum(
        target.data.root_lin_vel_w * command_term.direction_w, dim=-1
    )
    normalized_forward_speed = torch.clamp(
        forward_speed / target_speed, 0.0, maximum_normalized_speed
    )
    desired_speed_fraction = torch.clamp(
        torch.linalg.norm(desired_velocity_w, dim=-1) / target_speed, 0.0, 1.0
    )
    transit_mask = remaining > endpoint_threshold
    return (
        contact_mask.float()
        * normalized_forward_speed
        * desired_speed_fraction
        * transit_mask.float()
    )


def remaining_horizon_failure_penalty(
    env,
    term_names: tuple[str, ...],
    minimum_penalty_time: float = 1.0,
) -> torch.Tensor:
    """Charge early safety failures for the remaining episode horizon.

    Isaac Lab scales reward terms by ``step_dt``.  Returning a number of steps
    here makes the configured weight act as a cost per avoided second.  This
    prevents the agent from deliberately ending an episode to escape running
    endpoint and stall penalties.
    """
    if minimum_penalty_time < 0.0:
        raise ValueError("minimum_penalty_time must be non-negative.")
    if not term_names:
        raise ValueError("term_names must contain at least one failure term.")

    failure = torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    for term_name in term_names:
        failure |= env.termination_manager.get_term(term_name).bool()

    remaining_steps = torch.clamp(
        env.max_episode_length - env.episode_length_buf, min=0
    ).float()
    minimum_steps = minimum_penalty_time / env.step_dt
    charged_steps = torch.clamp(remaining_steps, min=minimum_steps)
    return failure.float() * charged_steps


class PhaseGatedReward(ManagerTermBase):
    """Evaluate another reward only during one task phase.

    Scene entity configurations inside ``reward_params`` are nested one level
    deeper than Isaac Lab's manager normally scans.  Resolve them explicitly
    once here rather than carrying unresolved body/joint IDs into the wrapped
    reward.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for value in cfg.params["reward_params"].values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)

    def __call__(
        self,
        env,
        phase_command_name: str,
        active_phase: int,
        reward_func: Callable,
        reward_params: dict,
    ) -> torch.Tensor:
        command = env.command_manager.get_term(phase_command_name)
        if not hasattr(command, "task_phase"):
            raise RuntimeError(
                f"Command '{phase_command_name}' does not expose task_phase."
            )
        value = reward_func(env, **reward_params)
        return value * (command.task_phase == active_phase).float()


def filtered_contact_indicator(
    env,
    sensor_name: str,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    """Binary reward value for any contact reported by a filtered sensor."""
    return filtered_contact_mask(env, sensor_name, force_threshold).float()


def home_joint_pose_reward(
    env,
    command_name: str,
    joint_std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward arm joints approaching their default Home positions."""
    if joint_std <= 0.0:
        raise ValueError("joint_std must be positive.")
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    current = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = math_utils.wrap_to_pi(current - home)
    tracking = torch.exp(-torch.mean(torch.square(error / joint_std), dim=-1))
    return tracking * (command.task_phase == 1).float()


def home_joint_error(
    env,
    command_name: str,
    error_scale: float,
    maximum_normalized_error: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense normalized Home-joint error during the return phase."""
    if error_scale <= 0.0 or maximum_normalized_error <= 0.0:
        raise ValueError("Home joint-error scales must be positive.")
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    current = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.mean(torch.abs(math_utils.wrap_to_pi(current - home)), dim=-1)
    normalized = torch.clamp(
        error / error_scale, max=maximum_normalized_error
    )
    return normalized * (command.task_phase == 1).float()


def home_eef_object_clearance(
    env,
    command_name: str,
    safe_distance: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward opening clearance between the EEF and object after sweeping."""
    if safe_distance <= 0.0:
        raise ValueError("safe_distance must be positive.")
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    distance = torch.linalg.norm(eef_pos_w - target.data.root_pos_w, dim=-1)
    normalized = torch.clamp(distance / safe_distance, 0.0, 1.0)
    clearance = normalized * normalized * (3.0 - 2.0 * normalized)
    return clearance * (command.task_phase == 1).float()


def home_object_speed_penalty(
    env,
    command_name: str,
    speed_scale: float,
    maximum_normalized_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize disturbing the parked object during Home return."""
    if speed_scale <= 0.0 or maximum_normalized_speed <= 0.0:
        raise ValueError("Object speed-penalty scales must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    normalized = torch.clamp(
        speed / speed_scale, max=maximum_normalized_speed
    )
    return normalized * (command.task_phase == 1).float()


def home_object_displacement_penalty(
    env,
    command_name: str,
    displacement_scale: float,
    maximum_normalized_displacement: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize moving the object away from its phase-transition pose."""
    if displacement_scale <= 0.0 or maximum_normalized_displacement <= 0.0:
        raise ValueError("Object displacement-penalty scales must be positive.")
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "parked_object_pos_w"):
        raise RuntimeError(f"Command '{command_name}' has no parked object pose.")
    target: RigidObject = env.scene[object_cfg.name]
    displacement = torch.linalg.norm(
        target.data.root_pos_w - command.parked_object_pos_w, dim=-1
    )
    normalized = torch.clamp(
        displacement / displacement_scale,
        max=maximum_normalized_displacement,
    )
    return normalized * (command.task_phase == 1).float()


def home_phase_time(
    env,
    command_name: str,
) -> torch.Tensor:
    """Unit running cost used to discourage lingering after the sweep."""
    command = env.command_manager.get_term(command_name)
    return (command.task_phase == 1).float()


def home_success_bonus(
    env,
    command_name: str,
    joint_position_threshold: float,
    joint_speed_threshold: float,
    endpoint_threshold: float,
    object_speed_threshold: float,
    object_displacement_threshold: float,
    contact_sensor_name: str,
    contact_force_threshold: float,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Sparse pre-dwell signal for a parked object and contact-free Home pose."""
    if min(
        joint_position_threshold,
        joint_speed_threshold,
        endpoint_threshold,
        object_speed_threshold,
        object_displacement_threshold,
    ) <= 0.0:
        raise ValueError("Home success thresholds must be positive.")
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    joint_error = torch.abs(
        math_utils.wrap_to_pi(
            robot.data.joint_pos[:, asset_cfg.joint_ids]
            - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
        )
    )
    joint_speed = torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids])
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    object_speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    object_displacement = torch.linalg.norm(
        target.data.root_pos_w - command.parked_object_pos_w, dim=-1
    )
    contact_mask = filtered_contact_mask(
        env, contact_sensor_name, contact_force_threshold
    )
    return (
        (command.task_phase == 1)
        & torch.all(joint_error < joint_position_threshold, dim=-1)
        & torch.all(joint_speed < joint_speed_threshold, dim=-1)
        & (endpoint_error < endpoint_threshold)
        & (object_speed < object_speed_threshold)
        & (object_displacement < object_displacement_threshold)
        & (~contact_mask)
    ).float()


def ft_torque_excess(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize F/T torque above a safe free-space/load threshold."""
    robot: Articulation = env.scene[asset_cfg.name]
    wrench = -robot.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids[0], :]
    torque_norm = torch.linalg.norm(wrench[:, 3:6], dim=-1)
    return torch.relu(torque_norm - threshold)


def commanded_joint_effort_l2(
    env,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Normalized OSC torque penalty."""
    action = env.action_manager.get_term(action_name)
    effort_limits = action._asset.data.joint_effort_limits[:, action._joint_ids]
    normalized = action.joint_efforts / torch.clamp(effort_limits, min=1.0e-6)
    return torch.sum(torch.square(normalized), dim=-1)


def torque_saturation(
    env,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Penalty indicator for invalid/clipped actions or saturated OSC torque."""
    return env.action_manager.get_term(action_name).torque_saturated.float()


def sweep_success_bonus(
    env,
    command_name: str,
    endpoint_threshold: float,
    lateral_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Sparse success bonus for correct direction and requested length."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    lateral_error = lateral_displacement(env, command_name)
    return (
        (endpoint_error < endpoint_threshold) & (lateral_error < lateral_threshold)
    ).float()


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def desired_velocity_profile_w(
    env,
    command_name: str,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return desired world velocity, remaining distance, and progress.

    Speed ramps up over the initial travel distance, stays constant through
    transit, and smoothly falls to zero over the final stopping distance.  A
    non-zero initial fraction prevents zero motion from matching the profile.
    """
    if acceleration_distance <= 0.0:
        raise ValueError("acceleration_distance must be positive.")
    if stopping_distance <= 0.0:
        raise ValueError("stopping_distance must be positive.")
    if not 0.0 < initial_speed_fraction <= 1.0:
        raise ValueError("initial_speed_fraction must be in (0, 1].")

    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    remaining = command[:, 2] - progress

    acceleration_ramp = _smoothstep01(progress / acceleration_distance)
    acceleration_factor = (
        initial_speed_fraction + (1.0 - initial_speed_fraction) * acceleration_ramp
    )
    stopping_factor = _smoothstep01(remaining / stopping_distance)
    desired_speed = command[:, 3] * acceleration_factor * stopping_factor
    desired_velocity_w = command_term.direction_w * desired_speed.unsqueeze(-1)
    return desired_velocity_w, remaining, progress


def object_velocity_profile_tracking(
    env,
    command_name: str,
    std: float,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    endpoint_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Track desired velocity, with zero reward for stationary transit states."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    if endpoint_threshold <= 0.0:
        raise ValueError("endpoint_threshold must be positive.")
    command_term = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    desired_velocity_w, remaining, _ = desired_velocity_profile_w(
        env,
        command_name,
        acceleration_distance=acceleration_distance,
        stopping_distance=stopping_distance,
        initial_speed_fraction=initial_speed_fraction,
    )
    velocity_error = torch.linalg.norm(
        target.data.root_lin_vel_w - desired_velocity_w, dim=-1
    )
    tracking = torch.exp(-torch.square(velocity_error / std))

    desired_speed = torch.linalg.norm(desired_velocity_w, dim=-1)
    forward_speed = torch.sum(
        target.data.root_lin_vel_w * command_term.direction_w, dim=-1
    )
    movement_gate = torch.clamp(
        forward_speed / torch.clamp(desired_speed, min=1.0e-3), 0.0, 1.0
    )
    transit_mask = remaining > endpoint_threshold
    return tracking * movement_gate * transit_mask.float()


def side_contact_velocity_profile_tracking(
    env,
    command_name: str,
    std: float,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    endpoint_threshold: float,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Track object velocity only through a central broad-side pad contact."""
    tracking = object_velocity_profile_tracking(
        env,
        command_name=command_name,
        std=std,
        acceleration_distance=acceleration_distance,
        stopping_distance=stopping_distance,
        initial_speed_fraction=initial_speed_fraction,
        endpoint_threshold=endpoint_threshold,
        object_cfg=object_cfg,
    )
    quality, _ = side_pad_contact_quality(
        env,
        sensor_names=sensor_names,
        pad_size=pad_size,
        face_normal_axis=face_normal_axis,
        center_sigma=center_sigma,
        face_sigma=face_sigma,
    )
    return tracking * quality


def object_stall_penalty(
    env,
    command_name: str,
    startup_grace_time: float,
    endpoint_threshold: float,
    minimum_speed_fraction: float,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize failing to move after startup while the goal remains ahead."""
    if startup_grace_time < 0.0:
        raise ValueError("startup_grace_time must be non-negative.")
    if endpoint_threshold <= 0.0:
        raise ValueError("endpoint_threshold must be positive.")
    if not 0.0 <= minimum_speed_fraction <= 1.0:
        raise ValueError("minimum_speed_fraction must be in [0, 1].")

    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    desired_velocity_w, remaining, _ = desired_velocity_profile_w(
        env,
        command_name,
        acceleration_distance=acceleration_distance,
        stopping_distance=stopping_distance,
        initial_speed_fraction=initial_speed_fraction,
    )
    desired_speed = torch.linalg.norm(desired_velocity_w, dim=-1)
    forward_speed = torch.sum(
        target.data.root_lin_vel_w * command_term.direction_w, dim=-1
    )
    speed_shortfall = torch.relu(
        minimum_speed_fraction * desired_speed - forward_speed
    ) / torch.clamp(command[:, 3], min=1.0e-6)
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    transit_mask = (elapsed >= startup_grace_time) & (remaining > endpoint_threshold)
    return speed_shortfall * transit_mask.float()


def object_linear_acceleration_penalty(
    env,
    acceleration_scale: float,
    maximum_normalized_penalty: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize abrupt object-speed changes using center-of-mass acceleration."""
    if acceleration_scale <= 0.0:
        raise ValueError("acceleration_scale must be positive.")
    if maximum_normalized_penalty <= 0.0:
        raise ValueError("maximum_normalized_penalty must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    acceleration = target.data.body_com_lin_acc_w[:, 0, :]
    normalized = torch.linalg.norm(acceleration, dim=-1) / acceleration_scale
    return torch.clamp(torch.square(normalized), max=maximum_normalized_penalty)


def stopped_at_goal_reward(
    env,
    command_name: str,
    position_std: float,
    speed_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward accurate endpoint placement only when the object is nearly still."""
    if position_std <= 0.0 or speed_std <= 0.0:
        raise ValueError("position_std and speed_std must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    position_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return torch.exp(
        -torch.square(position_error / position_std) - torch.square(speed / speed_std)
    )


def stopped_sweep_success_bonus(
    env,
    command_name: str,
    endpoint_threshold: float,
    lateral_threshold: float,
    speed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Sparse success signal requiring both endpoint accuracy and stopping."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return (
        (endpoint_error < endpoint_threshold)
        & (lateral_displacement(env, command_name) < lateral_threshold)
        & (speed < speed_threshold)
    ).float()
