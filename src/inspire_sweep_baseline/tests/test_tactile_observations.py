"""Independent pad bits, deterministic body order, and reset-safe readings."""

from _source_units import source_text

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT


@pytest.fixture
def tactile_sensor(load_mdp, make_env):
    module = load_mdp("tactile_observations")
    expected_names = tuple(module.PALM_SENSOR_BODY_NAMES)

    def make(count=3, names=None, forces=None):
        env = make_env(count)
        names = list(expected_names if names is None else names)
        if forces is None:
            forces = torch.zeros(count, len(names), 3)
        env.scene["hand_tactile"] = SimpleNamespace(
            body_names=names, data=SimpleNamespace(net_forces_w=forces),
        )
        cfg = SimpleNamespace(params={"sensor_name": "hand_tactile", "threshold_n": 0.05})
        return env, module.TactileContactBits(cfg, env)

    return SimpleNamespace(make=make, names=expected_names, module=module)


def test_all_seventeen_pads_have_independent_bits_in_stable_name_order(tactile_sensor):
    names = tactile_sensor.names
    assert len(names) == len(set(names)) == 17
    # PhysX is free to enumerate bodies differently. A shuffled view must still
    # produce each physical pad's bit in the documented policy input position.
    actual_names = list(names[7:] + names[:7])[::-1]
    forces = torch.zeros(17, 17, 3, dtype=torch.float64)
    for row, pad_name in enumerate(names):
        forces[row, actual_names.index(pad_name), row % 3] = -0.08
    env, reader = tactile_sensor.make(17, actual_names, forces)
    bits = reader(env)
    assert bits.shape == (17, 17)
    assert bits.dtype == torch.float32 and bits.device.type == "cpu"
    assert torch.equal(bits, torch.eye(17))
    # A subsequent live sample is read rather than a cached force snapshot.
    env.scene["hand_tactile"].data.net_forces_w.zero_()
    assert reader(env).count_nonzero() == 0


def test_simultaneous_contacts_use_each_pad_magnitude_without_cross_pad_cancellation(tactile_sensor):
    env, reader = tactile_sensor.make(2)
    forces = env.scene["hand_tactile"].data.net_forces_w
    forces[0, 0, 0], forces[0, 1, 0] = 0.06, -0.06
    forces[1, 2] = torch.tensor([0.03, 0.04, 0.0])
    forces[1, 3, 2] = 0.049
    bits = reader(env)
    expected = torch.zeros(2, 17)
    expected[0, :2], expected[1, 2] = 1.0, 1.0
    assert torch.equal(bits, expected)


def test_threshold_is_inclusive_and_can_be_adjusted_without_changing_channels(tactile_sensor):
    env, reader = tactile_sensor.make(3)
    env.scene["hand_tactile"].data.net_forces_w[:, 4, 0] = torch.tensor([0.049, 0.05, 0.051])
    assert reader(env)[:, 4].tolist() == [0.0, 1.0, 1.0]
    assert reader(env, threshold_n=0.051)[:, 4].tolist() == [0.0, 0.0, 1.0]


def test_partial_reset_masks_stale_nan_rows_and_preserves_other_live_pads(tactile_sensor):
    env, reader = tactile_sensor.make(4, names=list(reversed(tactile_sensor.names)))
    forces = env.scene["hand_tactile"].data.net_forces_w
    forces[0, -1, 0], forces[2, 0, 2] = 0.1, -0.1
    forces[1].fill_(torch.nan)
    forces[3].fill_(torch.inf)
    env.sensor_data_fresh[:] = torch.tensor([True, False, True, False])
    bits = reader(env)
    expected = torch.zeros(4, 17)
    expected[0, 0], expected[2, -1] = 1.0, 1.0
    assert torch.equal(bits, expected)
    assert torch.isfinite(bits).all()
    # Once that environment has a fresh sample, its invalid data must surface.
    env.sensor_data_fresh[1] = True
    with pytest.raises(RuntimeError, match="non-finite"):
        reader(env)


def test_all_reset_rows_return_zero_without_accessing_stale_sim_data(tactile_sensor):
    env, reader = tactile_sensor.make(2)
    env.sensor_data_fresh.zero_()

    class StaleData:
        @property
        def net_forces_w(self):
            raise AssertionError("Reset observations must not fetch stale PhysX force data")

    env.scene["hand_tactile"].data = StaleData()
    assert torch.equal(reader(env), torch.zeros(2, 17))


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_live_nonfinite_forces_are_rejected(tactile_sensor, invalid):
    env, reader = tactile_sensor.make(1)
    env.scene["hand_tactile"].data.net_forces_w[0, 16, 1] = invalid
    with pytest.raises(RuntimeError, match="non-finite"):
        reader(env)


@pytest.mark.parametrize("threshold", [0.0, -0.05, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(tactile_sensor, threshold):
    env, reader = tactile_sensor.make(1)
    with pytest.raises(ValueError, match="positive and finite"):
        reader(env, threshold_n=threshold)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "substituted"])
def test_incorrect_sensor_body_set_fails_at_term_construction(load_mdp, make_env, change):
    module = load_mdp("tactile_observations")
    names = list(module.PALM_SENSOR_BODY_NAMES)
    if change == "missing":
        names.pop()
    elif change == "extra":
        names.append("inspire_left_index_1")
    elif change == "duplicate":
        names[-1] = names[0]
    else:
        names[-1] = "inspire_left_index_1"
    env = make_env(1)
    # Constructor validation uses body metadata and must fail before force reads.
    env.scene["hand_tactile"] = SimpleNamespace(body_names=names)
    with pytest.raises(RuntimeError):
        module.TactileContactBits(SimpleNamespace(params={"sensor_name": "hand_tactile"}), env)


@pytest.mark.parametrize("shape", [(2, 16, 3), (2, 18, 3), (2, 17, 2), (1, 17, 3), (2, 51)])
def test_force_tensor_must_match_exact_environment_and_pad_dimensions(tactile_sensor, shape):
    env, reader = tactile_sensor.make(2)
    env.scene["hand_tactile"].data.net_forces_w = torch.zeros(*shape)
    with pytest.raises((ValueError, RuntimeError), match="shape"):
        reader(env)


def test_virtual_region_calibration_matches_the_original_collision_assets(load_mdp):
    module = load_mdp("tactile_observations")
    data_root = MODULE_ROOT / "assets/data"
    metadata = json.loads(source_text(data_root / "tactile_regions.json"))
    assert tuple(region["name"] for region in metadata["regions"]) == module.PALM_SENSOR_BODY_NAMES
    # Changed source collision geometry requires recalibration rather than
    # silently applying region coordinates from a different Hand asset.
    layers = metadata["source"]["layer_sha256"]
    assert metadata["source"]["usd_path"] in layers
    for relative_path, expected in layers.items():
        assert hashlib.sha256((data_root / relative_path).read_bytes()).hexdigest() == expected, relative_path
    assert metadata["virtual_margin_m"] <= 0.0005
    for region in metadata["regions"]:
        planes = torch.tensor(region["footprint_halfspaces"], dtype=torch.float64)
        assert planes.ndim == 2 and planes.shape[1] == 3
        assert torch.isfinite(planes).all()
        assert torch.allclose(torch.linalg.vector_norm(planes[:, :2], dim=1), torch.ones(len(planes), dtype=torch.float64))
        assert region["outward_z_min_m"] < region["outward_z_max_m"]
