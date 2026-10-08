"""Hand state, physical palmar tactile channels, and uncancelled wrist F/T."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from hand_manipulation_rl.sensors import (
    FixedJointWrenchReader,
    PALM_CHANNEL_NAMES,
    contact_magnitudes_and_bits,
)


def _sensor_fresh_mask(env) -> torch.Tensor | None:
    """A reset row has no live contact/reaction sample until physics advances."""

    mask = getattr(env, "sensor_data_fresh", None)
    if mask is not None and (mask.shape != (env.num_envs,) or mask.dtype != torch.bool):
        raise RuntimeError("sensor_data_fresh must be one boolean per environment")
    return mask


def actual_hand_synergy(env, action_name: str = "hand_action") -> torch.Tensor:
    """Read measured [common flexion openness, thumb1 openness]."""

    synergy = env.action_manager.get_term(action_name).actual_synergy
    if synergy.shape != (env.num_envs, 2):
        raise RuntimeError(f"Unexpected actual hand synergy shape: {tuple(synergy.shape)}")
    return synergy


def palm_tactile_bits(env, sensor_name: str = "palm_tactile", threshold_n: float = 0.05) -> torch.Tensor:
    """Return 17 physical pad contact bits in the source hand's stable order."""

    fresh = _sensor_fresh_mask(env)
    if fresh is not None and not bool(fresh.any()):
        return torch.zeros((env.num_envs, 17), dtype=torch.float32, device=env.device)
    sensor = env.scene[sensor_name]
    cache = getattr(env, "_sweep_inspire_palm_indices", None)
    if cache is None:
        cache = env._sweep_inspire_palm_indices = {}
    if sensor_name not in cache:
        names = tuple(sensor.body_names)
        if len(names) != 17 or len(set(names)) != 17 or set(names) != set(PALM_CHANNEL_NAMES):
            raise RuntimeError(f"Palm tactile sensor must resolve the 17 physical pads, got {names}")
        cache[sensor_name] = torch.tensor(
            [names.index(name) for name in PALM_CHANNEL_NAMES], dtype=torch.long, device=env.device
        )
    ordered_forces = torch.index_select(sensor.data.net_forces_w, 1, cache[sensor_name])
    _, bits = contact_magnitudes_and_bits(ordered_forces, threshold_n)
    result = bits.to(dtype=torch.float32)
    return result if fresh is None else torch.where(fresh.unsqueeze(-1), result, torch.zeros_like(result))


def wrist_wrench_c(
    env,
    asset_name: str = "robot",
    c_offset_h: tuple[float, float, float] = (0.0, 0.0, 0.10),
) -> torch.Tensor:
    """Return [Fx,Fy,Fz,Mx,My,Mz] expressed in H and referenced at C.

    The incoming fixed-joint reaction is indexed by its child hand body and
    expressed in its Axia80 parent frame. Rotation and moment-arm transport
    are the only operations; hand self-weight remains in the observation.
    """

    fresh = _sensor_fresh_mask(env)
    if fresh is not None and not bool(fresh.any()):
        return torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)
    cache = getattr(env, "_sweep_inspire_wrench_readers", None)
    if cache is None:
        cache = env._sweep_inspire_wrench_readers = {}
    robot = env.scene[asset_name]
    if asset_name not in cache:
        names = tuple(robot.body_names)
        if names.count("inspire_base_link") != 1 or names.count("axia80_link") != 1:
            raise RuntimeError("Robot must contain one Inspire base and one Axia80 body")
        cache[asset_name] = (
            FixedJointWrenchReader(robot, "inspire_base_link"),
            names.index("inspire_base_link"),
            names.index("axia80_link"),
        )
    reader, h_body_id, f_body_id = cache[asset_name]
    h_pos_w = robot.data.body_pos_w[:, h_body_id]
    h_quat_w = robot.data.body_quat_w[:, h_body_id]
    f_pos_w = robot.data.body_pos_w[:, f_body_id]
    f_quat_w = robot.data.body_quat_w[:, f_body_id]
    offset_h = torch.tensor(c_offset_h, dtype=h_pos_w.dtype, device=env.device).expand(env.num_envs, -1)
    c_pos_w = h_pos_w + math_utils.quat_apply(h_quat_w, offset_h)
    rotation_h_from_f = (
        math_utils.matrix_from_quat(h_quat_w).transpose(-1, -2)
        @ math_utils.matrix_from_quat(f_quat_w)
    )
    c_to_f_h = math_utils.quat_apply_inverse(h_quat_w, f_pos_w - c_pos_w)
    result = reader.read(rotation_h_from_f, c_to_f_h).measured_c
    return result if fresh is None else torch.where(fresh.unsqueeze(-1), result, torch.zeros_like(result))


__all__ = ["actual_hand_synergy", "palm_tactile_bits", "wrist_wrench_c"]
