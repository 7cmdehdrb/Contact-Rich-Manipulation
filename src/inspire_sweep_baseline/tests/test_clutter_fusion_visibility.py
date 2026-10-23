"""Exact reset-time visibility, independent misclassification, and fixed caches."""

import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, _module_from_file


@pytest.fixture
def fusion_events(load_mdp, monkeypatch):
    load_mdp("goal_events")
    for package_name in ("contact_sweep_clutter", "contact_sweep_clutter_fusion"):
        name = f"_baseline_cpu_pkg.{package_name}"
        package = types.ModuleType(name)
        package.__path__ = [str(MODULE_ROOT / package_name)]
        monkeypatch.setitem(sys.modules, name, package)
    name = "_baseline_cpu_pkg.mdp.event_functions.contact_sweep_clutter"
    monkeypatch.setitem(sys.modules, name, _module_from_file(name, MODULE_ROOT / "contact_sweep_clutter/events.py"))
    name = "_baseline_cpu_pkg.mdp.event_functions.contact_sweep_clutter_fusion"
    module = _module_from_file(name, MODULE_ROOT / "contact_sweep_clutter_fusion/events.py")
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def visibility_env(tensor_math):
    def make(count=1, shelf_angle=0.0):
        shelf_quat = torch.tensor([
            math.cos(shelf_angle / 2), 0.0, 0.0, math.sin(shelf_angle / 2),
        ]).expand(count, -1).clone()
        shelf_pos = torch.tensor([-0.7, 0.0, 0.0]).expand(count, -1).clone()
        shelf_pos[:, 1] += torch.arange(count) * 2.0
        # Target occludes obstacle 0; obstacle 1 occludes obstacle 2.
        local = torch.tensor([
            [0.10, 0.00, 1.05], [-0.10, 0.00, 1.05],
            [-0.10, 0.20, 1.05], [-0.20, 0.20, 1.05],
        ]).expand(count, -1, -1).clone()
        states = torch.zeros(count, 4, 13)
        states[..., :3] = shelf_pos[:, None] + tensor_math.quat_apply(shelf_quat[:, None].expand(-1, 4, -1), local)
        states[..., 3] = 1.0

        class Collection:
            num_objects = 4
            object_names = ["mass_cylinder", "obstacle_0", "obstacle_1", "obstacle_2"]

            def __init__(self):
                self.data = SimpleNamespace(object_pos_w=states[..., :3], object_link_state_w=states)

            def write_object_link_state_to_sim(self, values, env_ids):
                states[env_ids] = values

        slots = torch.zeros(count, 10, 4)
        slots[:, :3, 0] = 1.0
        slots[:, :3, 1:] = states[:, 1:, :3] + 0.01
        return SimpleNamespace(
            num_envs=count, device="cpu", cfg=SimpleNamespace(seed=123),
            scene={
                "object_collection": Collection(),
                "shelf": SimpleNamespace(data=SimpleNamespace(root_pos_w=shelf_pos, root_quat_w=shelf_quat)),
                "robot": SimpleNamespace(data=SimpleNamespace(
                    root_pos_w=torch.zeros(count, 3),
                    root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, -1).clone(),
                )),
            },
            target_id=torch.zeros(count, 1, dtype=torch.long),
            target_width=torch.full((count, 1), 0.04), sweep_dir=torch.zeros(count, 3),
            _initial_obstacle_observation=slots.flatten(1),
        )

    return make


@pytest.mark.parametrize("shelf_angle", [0.0, math.pi / 2.0, -0.7])
def test_front_projection_uses_exact_shelf_geometry_and_all_objects(fusion_events, visibility_env, shelf_angle):
    env = visibility_env(count=2, shelf_angle=shelf_angle)
    before_positions = env.scene["object_collection"].data.object_pos_w.clone()
    original = env._initial_obstacle_observation.reshape(2, 10, 4).clone()
    fusion_events.apply_initial_obstacle_visibility(env, None, visibility_flip_probability=0.0)
    slots = env._initial_obstacle_observation.reshape(2, 10, 4)
    assert torch.equal(slots[:, :3, 0], torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1))
    assert slots[:, [0, 2]].count_nonzero() == 0
    assert torch.equal(slots[:, 1], original[:, 1])  # Preserve existing observation noise.
    assert slots[:, 3:].count_nonzero() == 0
    assert torch.equal(env.scene["object_collection"].data.object_pos_w, before_positions)


def test_threshold_is_projected_visible_fraction(fusion_events, visibility_env):
    env = visibility_env()
    # Move hidden obstacle sideways to reveal half its projected diameter.
    env.scene["object_collection"].data.object_pos_w[0, 1, 1] += 0.04
    original = env._initial_obstacle_observation.clone()
    fusion_events.apply_initial_obstacle_visibility(env, None, visibility_threshold=0.49, visibility_flip_probability=0.0)
    assert env._initial_obstacle_observation[0, 0] == 1.0
    env._initial_obstacle_observation[:] = original
    fusion_events.apply_initial_obstacle_visibility(env, None, visibility_threshold=0.51, visibility_flip_probability=0.0)
    assert env._initial_obstacle_observation[0, :4].count_nonzero() == 0


def test_flips_both_directions_only_on_real_obstacles(fusion_events, visibility_env, monkeypatch):
    env = visibility_env(count=2)
    original = env._initial_obstacle_observation.reshape(2, 10, 4).clone()
    # Deliberately poison padding: reset masking must always zero every slot.
    env._initial_obstacle_observation.reshape(2, 10, 4)[:, 3:] = 99.0
    calls = []

    def fake_rand(shape, device):
        calls.append((shape, device))
        return torch.tensor([[0.009, 0.009, 0.5], [0.5, 0.5, 0.009]])

    monkeypatch.setattr(fusion_events.torch, "rand", fake_rand)
    fusion_events.apply_initial_obstacle_visibility(env, None)
    slots = env._initial_obstacle_observation.reshape(2, 10, 4)
    assert calls == [(torch.Size([2, 3]), "cpu")]
    assert torch.equal(slots[:, :3, 0], torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
    assert torch.equal(slots[0, 0], original[0, 0])
    assert torch.equal(slots[1, 2], original[1, 2])
    assert slots[:, 3:].count_nonzero() == 0


def test_seed_reproduces_independent_one_percent_flips(fusion_events, visibility_env):
    first, second = visibility_env(10000), visibility_env(10000)
    torch.manual_seed(675)
    fusion_events.apply_initial_obstacle_visibility(first, None)
    torch.manual_seed(675)
    fusion_events.apply_initial_obstacle_visibility(second, None)
    assert torch.equal(first._initial_obstacle_observation, second._initial_obstacle_observation)
    observed = first._initial_obstacle_observation.reshape(10000, 10, 4)[:, :3, 0].bool()
    flips = observed != torch.tensor([False, True, False])
    assert 0.007 < flips.float().mean().item() < 0.013
    assert flips[:, 0].any() and flips[:, 1].any() and flips[:, 2].any()


def test_partial_reset_and_reads_leave_other_cache_entries_fixed(fusion_events, visibility_env):
    env = visibility_env(3)
    original = env._initial_obstacle_observation.clone()
    fusion_events.apply_initial_obstacle_visibility(env, [1], visibility_flip_probability=0.0)
    cached = env._initial_obstacle_observation.clone()
    assert torch.equal(cached[[0, 2]], original[[0, 2]])
    assert not torch.equal(cached[1], original[1])
    source = _module_from_file("_baseline_cpu_pkg.mdp.reward_functions.contact_sweep_clutter", MODULE_ROOT / "contact_sweep_clutter/mdp.py")
    env.scene["object_collection"].data.object_pos_w += 5.0
    assert torch.equal(source.initial_obstacle_positions(env), cached)
    fusion_events.apply_initial_obstacle_visibility(env, [])
    assert torch.equal(env._initial_obstacle_observation, cached)


def test_wrapper_preserves_goals_and_layout_and_masks_only_initial_inputs(fusion_events, visibility_env):
    plain, filtered = visibility_env(16), visibility_env(16)
    torch.manual_seed(49)
    fusion_events.randomize_clutter_scene(plain, None, excluded_x_half_angle=math.pi / 3)
    torch.manual_seed(49)
    fusion_events.randomize_visible_clutter_scene(filtered, None, visibility_flip_probability=0.0)
    assert torch.equal(plain.scene["object_collection"].data.object_link_state_w,
                       filtered.scene["object_collection"].data.object_link_state_w)
    assert torch.equal(plain.sweep_dir, filtered.sweep_dir)
    assert torch.equal(plain.target_id, filtered.target_id)
    assert torch.equal(plain.target_width, filtered.target_width)
    before = plain._initial_obstacle_observation.reshape(16, 10, 4)
    after = filtered._initial_obstacle_observation.reshape(16, 10, 4)
    visible = after[:, :3, 0].bool()
    assert torch.equal(after[:, :3][visible], before[:, :3][visible])
    assert after[:, :3][~visible].count_nonzero() == 0
    assert after[:, 3:].count_nonzero() == 0
    # Initial 짹60째 cones exclude the forwards/backwards shelf X directions.
    directions = filtered.sweep_dir[:, :2] / filtered.sweep_dir[:, :2].norm(dim=-1, keepdim=True)
    assert (directions[:, 0].abs() <= 0.5 + 1e-6).all()


@pytest.mark.parametrize("params", [
    {"object_radius": 0.0}, {"visibility_threshold": 0.0},
    {"visibility_threshold": 1.1}, {"visibility_flip_probability": -0.01},
    {"visibility_flip_probability": 1.01},
])
def test_invalid_visibility_settings_are_rejected(fusion_events, visibility_env, params):
    with pytest.raises(ValueError):
        fusion_events.apply_initial_obstacle_visibility(visibility_env(), None, **params)
