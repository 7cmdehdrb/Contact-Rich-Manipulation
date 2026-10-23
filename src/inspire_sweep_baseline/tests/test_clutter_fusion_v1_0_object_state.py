"""Parked capacity slots do not affect reward or episode failure."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from test_contact_sweep_clutter_mdp import multiobject_env
from test_clutter_fusion_v1_0_rewards import rewards


@pytest.fixture
def object_state(rewards):
    from _source_units import import_unit
    return import_unit("contact_sweep_clutter_fusion_v1_0.object_state")


def masked_env(make_env, count=4):
    env = multiobject_env(make_env, count=count, objects=7)
    env._initial_object_positions_w = env.scene["object_collection"].data.object_pos_w.clone()
    env._active_object_mask = torch.zeros(count, 7, dtype=torch.bool)
    env._active_object_mask[:, 0] = True
    return env


def test_active_cost_preserves_target_tilt_obstacle_displacement_and_aggregate_cap(
    object_state, make_env, tensor_math,
):
    env = masked_env(make_env)
    env._active_object_mask[2, 1] = True
    env._active_object_mask[3, :3] = True
    env.target_id[3] = 2
    position = env.scene["object_collection"].data.object_pos_w
    quaternion = env.scene["object_collection"].data.object_quat_w
    position[~env._active_object_mask] = math.nan
    quaternion[~env._active_object_mask] = math.nan
    position[0, 0, 0] += 100.0  # Target translation is free.
    quaternion[1, 0] = tensor_math.quat_from_euler_xyz(
        torch.tensor([.9]), torch.tensor([0.]), torch.tensor([0.]),
    )[0]
    position[2, 1, 0] += .1
    position[3, :2, 0] += .2
    position[3, 2, 0] += 100.0
    assert object_state.combined_active_object_cost(env).tolist() == pytest.approx([0., 1., .025, .1])


def test_parked_fallen_tilted_slots_are_ignored_but_active_targets_and_obstacles_fail(
    object_state, make_env, tensor_math,
):
    env = masked_env(make_env)
    env._active_object_mask[2, 6] = True
    env._active_object_mask[3, 2] = True
    env.target_id[3] = 2
    position = env.scene["object_collection"].data.object_pos_w
    quaternion = env.scene["object_collection"].data.object_quat_w
    position[~env._active_object_mask] = -1000.
    quaternion[~env._active_object_mask] = tensor_math.quat_from_euler_xyz(
        torch.tensor([1.5]), torch.tensor([0.]), torch.tensor([0.]),
    )[0]
    position[1, 0, 2] = 1.03  # Active target falls.
    position[2, 6, 2] = 1.03  # Last active obstacle falls.
    quaternion[3, 2] = tensor_math.quat_from_euler_xyz(
        torch.tensor([1.0]), torch.tensor([0.]), torch.tensor([0.]),
    )[0]  # A dynamically selected target tilts.
    assert object_state.any_active_object_fall_or_tilt(env).tolist() == [False, True, True, True]
    env.scene.env_origins[:, 2] = 10.0
    position[..., 2] += 10.0
    assert object_state.any_active_object_fall_or_tilt(env).tolist() == [False, True, True, True]


def test_contact_resultant_ignores_inactive_force_and_dynamic_target(rewards):
    contacts = sys.modules["_baseline_cpu_pkg.mdp.common.contact_sweep_clutter_fusion_contacts"]
    names = ["target", *(f"obstacle_{i}" for i in range(6))]
    env = SimpleNamespace(
        num_envs=2, target_id=torch.tensor([[0], [6]]),
        _active_object_mask=torch.tensor([
            [True, True, False, False, False, False, False],
            [True, False, False, False, False, False, True],
        ]),
        scene={"object_collection": SimpleNamespace(object_names=names)},
    )
    for index, name in enumerate(names):
        forces = torch.full((2, 1, 1, 3), math.nan)
        forces[env._active_object_mask[:, index]] = torch.tensor([index + 1., 0., 0.])
        env.scene[contacts.object_hand_sensor_name(name)] = SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=forces),
        )
    assert contacts.hand_obstacle_contact_force_w(env).tolist() == [[-2., 0., 0.], [-1., 0., 0.]]


def test_mask_fallback_preserves_all_active_and_rejects_wrong_shape(object_state, make_env):
    env = make_env(2)
    values = torch.zeros(2, 7, 3)
    assert object_state.active_object_mask(env, values).all()
    env._active_object_mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="boolean"):
        object_state.active_object_mask(env, values)
