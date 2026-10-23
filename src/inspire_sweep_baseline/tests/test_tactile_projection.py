"""Classify actual per-point parent contacts without changing collision shapes."""

from _source_units import source_text

import json
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT


REGIONS = json.loads(source_text(MODULE_ROOT / "assets/data/tactile_regions.json"))["regions"]


def _raw(points, normals, counts, starts, magnitudes=None):
    # Spare capacity and invalid unused storage verify we gather active slices.
    capacity = len(points) + 2
    positions = torch.full((capacity, 3), torch.nan)
    directions = torch.full((capacity, 3), torch.nan)
    scalars = torch.full((capacity, 1), torch.nan)
    positions[:len(points)] = torch.as_tensor(points).reshape(-1, 3)
    directions[:len(points)] = torch.as_tensor(normals).reshape(-1, 3)
    scalars[:len(points), 0] = 0.1 if magnitudes is None else torch.as_tensor(magnitudes)
    return (scalars, positions, directions, torch.zeros(capacity, 1),
            torch.tensor(counts, dtype=torch.uint32), torch.tensor(starts, dtype=torch.uint32))


def _poses(num_envs=1):
    return torch.zeros(num_envs, 17, 3), torch.tensor([1., 0., 0., 0.]).expand(num_envs, 17, 4).clone()


@pytest.mark.parametrize("channel", range(17))
def test_existing_parent_surface_can_activate_each_pad_independently(load_mdp, channel):
    tactile = load_mdp("tactile_observations")
    parent = REGIONS[channel]["parent_name"]
    channels = [index for index, region in enumerate(REGIONS) if region["parent_name"] == parent]
    selected_regions = [REGIONS[index] for index in channels]
    proof = REGIONS[channel]["verification"]
    point = torch.tensor([proof["parent_surface_point_local_m"]])
    normal = -torch.tensor([proof["parent_surface_outward_normal_local"]])
    positions, quaternions = _poses()
    # Each neighboring pad has its own link frame. Place its region far away
    # to isolate the channel proof and detect unintended force broadcasting.
    positions[:, :, 0] = 1.0
    positions[:, channel] = 0.0
    output = torch.zeros(1, 17, 3)
    mapper = tactile.ParentContactRegions(selected_regions, channels, "cpu")
    mapper.accumulate(output, point, normal * 0.1, normal, torch.tensor([0]), positions, quaternions)
    bits = tactile.contact_magnitudes_and_bits(output)[1]
    expected = torch.zeros(1, 17, dtype=torch.uint8)
    expected[0, channel] = 1
    assert torch.equal(bits, expected)


def test_two_pad_contacts_on_same_parent_stay_separate_without_averaging(load_mdp, tensor_math):
    tactile = load_mdp("tactile_observations")
    channels = [6, 7]  # index pad2 and pad3 belong to the same distal link.
    assert REGIONS[channels[0]]["parent_name"] == REGIONS[channels[1]]["parent_name"]
    positions, quaternions = _poses(2)
    angles = torch.tensor([0.0, 0.7])
    q = tensor_math.quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), angles)
    positions[:, channels[0]] = torch.tensor([0.2, 0.0, 0.3])
    positions[:, channels[1]] = torch.tensor([-0.2, 0.0, 0.3])
    quaternions[:, channels] = q[:, None]
    points, normals, ids = [], [], []
    for env_id in range(2):
        for channel in channels:
            proof = REGIONS[channel]["verification"]
            local = torch.tensor(proof["parent_surface_point_local_m"])
            inward = -torch.tensor(proof["parent_surface_outward_normal_local"])
            points.append(tensor_math.quat_apply(q[env_id:env_id+1], local[None])[0] + positions[env_id, channel])
            normals.append(tensor_math.quat_apply(q[env_id:env_id+1], inward[None])[0])
            ids.append(env_id)
    points, normals = torch.stack(points), torch.stack(normals)
    output = torch.zeros(2, 17, 3)
    tactile.ParentContactRegions([REGIONS[i] for i in channels], channels, "cpu").accumulate(
        output, points, normals * 0.1, normals, torch.tensor(ids), positions, quaternions,
    )
    bits = tactile.contact_magnitudes_and_bits(output)[1]
    expected = torch.zeros(2, 17, dtype=torch.uint8)
    expected[:, channels] = 1
    assert torch.equal(bits, expected)


@pytest.mark.parametrize("kind", ["outside_xy", "backside", "tangential"])
def test_off_region_parent_contacts_do_not_activate_palm(load_mdp, kind):
    tactile = load_mdp("tactile_observations")
    region = REGIONS[0]
    point = torch.tensor([region["verification"]["parent_surface_point_local_m"]])
    normal = torch.tensor([[0.0, 0.0, -1.0]])
    if kind == "outside_xy":
        point[0, 0] += 0.1
    elif kind == "backside":
        point[0, 2] = region["outward_z_min_m"] - 0.01
    else:
        normal[0] = torch.tensor([1.0, 0.0, 0.0])
    output = torch.zeros(1, 17, 3)
    positions, quaternions = _poses()
    tactile.ParentContactRegions([region], [0], "cpu").accumulate(
        output, point, normal * 0.1, normal, torch.tensor([0]), positions, quaternions,
    )
    assert output.count_nonzero() == 0


def test_raw_contact_slices_respect_filter_indices_and_stale_reset_rows(load_mdp):
    tactile = load_mdp("tactile_observations")
    raw = _raw([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [torch.nan] * 3],
               [[0., 0., -1.]] * 4,
               [[0, 2, 1], [1, 0, 0]], [[0, 1, 0], [3, 0, 0]])
    points, forces, normals, env_ids = tactile.unpack_contact_points(raw, torch.tensor([True, False]), 2)
    assert points.tolist() == [[4., 5., 6.], [7., 8., 9.], [1., 2., 3.]]
    assert env_ids.tolist() == [0, 0, 0]
    assert torch.allclose(forces, normals * 0.1)


@pytest.mark.parametrize("kind", ["non_finite", "truncated", "full"])
def test_corrupt_or_saturated_contact_buffers_fail_explicitly(load_mdp, kind):
    tactile = load_mdp("tactile_observations")
    raw = list(_raw([[1., 2., 3.]], [[0., 0., -1.]], [[1]], [[0]]))
    if kind == "non_finite":
        raw[1][0, 0] = torch.nan
    elif kind == "truncated":
        raw[5] = torch.tensor([[3]], dtype=torch.uint32)
    else:
        raw[4] = torch.tensor([[3]], dtype=torch.uint32)
    with pytest.raises(RuntimeError, match="non-finite|truncated|buffer full"):
        tactile.unpack_contact_points(tuple(raw), torch.tensor([True]), 1)


def test_configured_projected_reader_combines_real_pad_and_parent_contacts(load_mdp, make_env):
    tactile = load_mdp("tactile_observations")
    from _baseline_cpu_pkg.assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, parent_sensor_name

    env = make_env(2)
    names = [region["name"] for region in REGIONS]
    env.physics_dt = 0.01
    env.scene["robot"].body_names = names[::-1]
    # Reader must resolve link poses independently of ContactSensor/body order.
    positions, quaternions = _poses(2)
    env.scene["robot"].data.body_link_pos_w = positions.flip(1)
    env.scene["robot"].data.body_link_quat_w = quaternions.flip(1)
    env.scene["hand_tactile"] = SimpleNamespace(
        body_names=names,
        data=SimpleNamespace(net_forces_w=torch.zeros(2, 17, 3)),
    )
    env.scene["hand_tactile"].data.net_forces_w[0, 4, 0] = 0.1
    # Opposite projected palm force must never turn off an already active
    # direct pad. Binary sources combine by OR rather than force cancellation.
    env.scene["hand_tactile"].data.net_forces_w[0, 0, 2] = 0.1
    env.scene["hand_tactile"].data.net_forces_w[1] = torch.nan
    env.sensor_data_fresh[1] = False
    proof = REGIONS[0]["verification"]
    for parent in TACTILE_PARENT_BODY_NAMES:
        if parent == REGIONS[0]["parent_name"]:
            raw = _raw([proof["parent_surface_point_local_m"], [torch.nan] * 3],
                       [[0., 0., -1.], [torch.nan] * 3], [[1, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0]])
        else:
            raw = _raw([], [], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]])

        class View:
            def __init__(self, value):
                self.value = value

            def get_contact_data(self, dt):
                assert dt == 0.01
                return self.value

        env.scene[parent_sensor_name(parent)] = SimpleNamespace(body_names=[parent], data=None, contact_physx_view=View(raw))
    cfg = SimpleNamespace(params={"sensor_name": "hand_tactile", "threshold_n": 0.05})
    term = tactile.ProjectedTactileContactBits(cfg, env)
    result = term(env)
    assert result.shape == (2, 17)
    assert result[0].nonzero().flatten().tolist() == [0, 4]
    assert result[1].count_nonzero() == 0
