"""Behavioral checks for clutter failures, cached observations, and shared physics."""

import importlib.util
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT


@pytest.fixture
def clutter_modules(load_mdp, monkeypatch):
    parent_physics = load_mdp("realistic_physics")
    parent_rewards = load_mdp("goal_rewards")
    parent_terminations = load_mdp("goal_terminations")
    package_name = "_baseline_cpu_pkg.contact_sweep_clutter"
    package = ModuleType(package_name)
    package.__path__ = [str(MODULE_ROOT / "contact_sweep_clutter")]
    monkeypatch.setitem(sys.modules, package_name, package)
    loaded = {}
    for name in ("mdp", "physics"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name, MODULE_ROOT / "contact_sweep_clutter" / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded, parent_physics=parent_physics, parent_rewards=parent_rewards,
                           parent_terminations=parent_terminations)


def multiobject_env(make_env, count=3, objects=4):
    env = make_env(count)
    state = env.scene["object_collection"].data.object_link_state_w.repeat(1, objects, 1)
    env.scene["object_collection"].num_objects = objects
    env.scene["object_collection"].data = SimpleNamespace(
        object_link_state_w=state,
        object_pos_w=state[..., :3],
        object_quat_w=state[..., 3:7],
    )
    env.scene["shelf"].data.root_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, -1).clone()
    return env


def test_all_object_tilt_cost_is_sum_of_parent_costs_and_uses_current_shelf_frame(
    clutter_modules, make_env, tensor_math,
):
    env = multiobject_env(make_env)
    roll = torch.tensor([[0.0, 0.3, 0.5, 1.5], [0.6, 0.6, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0]])
    zeros = torch.zeros_like(roll)
    env.scene["object_collection"].data.object_quat_w[:] = tensor_math.quat_from_euler_xyz(roll, zeros, zeros)
    shelf_roll = torch.tensor([0.0, 0.6, 0.0])
    env.scene["shelf"].data.root_quat_w[:] = tensor_math.quat_from_euler_xyz(
        shelf_roll, torch.zeros(3), torch.zeros(3),
    )
    expected = torch.zeros(3)
    for object_index in range(4):
        env.target_id[:] = object_index
        expected += clutter_modules.parent_rewards.object_tilt_cost(env)
    actual = clutter_modules.mdp.all_object_tilt_cost(env)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual[0] > 1.0  # Bound each term, not the combined scene cost.
    assert torch.allclose(actual[1:], torch.zeros(2), atol=1e-6)
    # Target selection must not omit a cylinder from the all-object sum.
    env.target_id[:] = 0
    assert torch.equal(actual, clutter_modules.mdp.all_object_tilt_cost(env))


def test_any_object_fall_or_tilt_matches_parent_union_and_environment_origins(
    clutter_modules, make_env, tensor_math,
):
    env = multiobject_env(make_env, count=4, objects=3)
    origins_z = torch.tensor([0.0, 10.0, -10.0, 20.0])
    env.scene.env_origins[:, 2] = origins_z
    env.scene["object_collection"].data.object_pos_w[..., 2] = origins_z[:, None] + 1.05
    # Only an obstacle falls in env 0, another obstacle tips in env 1.
    env.scene["object_collection"].data.object_pos_w[0, 2, 2] = 1.03
    roll = torch.zeros(4, 3)
    roll[1, 1] = 0.91
    roll[2, :] = 0.95
    env.scene["object_collection"].data.object_quat_w[:] = tensor_math.quat_from_euler_xyz(
        roll, torch.zeros_like(roll), torch.zeros_like(roll),
    )
    # A tilted shelf and equally tilted cylinders remain upright relative to it.
    env.scene["shelf"].data.root_quat_w[2] = env.scene["object_collection"].data.object_quat_w[2, 0]
    expected = torch.zeros(4, dtype=torch.bool)
    for object_index in range(3):
        env.target_id[:] = object_index
        expected |= clutter_modules.parent_terminations.object_fall_or_tilt(env)
    actual = clutter_modules.mdp.any_object_fall_or_tilt(env)
    assert torch.equal(actual, expected)
    assert actual.tolist() == [True, True, False, False]
    env.scene["object_collection"].data.object_pos_w[3, 0, 2] = origins_z[3] + 1.03
    assert clutter_modules.mdp.any_object_fall_or_tilt(env).tolist() == [True, True, False, True]


def test_initial_obstacle_observation_keeps_snapshot_and_zero_dummy_slots(clutter_modules, make_env):
    env = multiobject_env(make_env, count=2, objects=3)
    slots = torch.zeros(2, 10, 4)
    slots[:, :2, 0] = 1.0
    slots[:, :2, 1:] = torch.tensor([[-0.6, 0.1, 1.05], [-0.7, -0.1, 1.05]])
    env._initial_obstacle_observation = slots.flatten(1)
    expected = env._initial_obstacle_observation.clone()
    env.scene["object_collection"].data.object_pos_w.add_(50.0)
    observation = clutter_modules.mdp.initial_obstacle_positions(env)
    assert observation.shape == (2, 40)
    assert torch.equal(observation, expected)
    assert torch.count_nonzero(observation.reshape(2, 10, 4)[:, 2:]) == 0


@pytest.mark.parametrize("free_angle,fail_angle", [(-0.1, 0.9), (0.9, 0.9), (0.1, math.pi + 0.1)])
def test_tilt_threshold_validation_matches_parent(clutter_modules, make_env, free_angle, fail_angle):
    with pytest.raises(ValueError, match="Tilt thresholds"):
        clutter_modules.mdp.all_object_tilt_cost(
            multiobject_env(make_env), free_angle=free_angle, fail_angle=fail_angle,
        )


class PhysicsView:
    def __init__(self, rows, shapes):
        self.max_shapes = shapes
        self.mass = torch.full((rows, 1), 0.5475)
        unit_inertia = torch.tensor([0.000933333, 0, 0, 0, 0.000933333, 0, 0, 0, 0.0008])
        self.inertia = unit_inertia[None, :].repeat(rows, 1) * self.mass
        self.properties = torch.ones(rows, shapes, 3)
        self.properties[..., 2] = torch.arange(shapes) / 10.0
        self.reads = {"mass": 0, "inertia": 0, "material": 0}
        self.writes = {"mass": [], "inertia": [], "material": []}

    def get_masses(self):
        self.reads["mass"] += 1
        return self.mass

    def get_inertias(self):
        self.reads["inertia"] += 1
        return self.inertia

    def get_material_properties(self):
        self.reads["material"] += 1
        return self.properties

    def set_masses(self, values, ids):
        self.writes["mass"].append(ids.clone())
        self.mass[ids] = values[ids]

    def set_inertias(self, values, ids):
        self.writes["inertia"].append(ids.clone())
        self.inertia[ids] = values[ids]

    def set_material_properties(self, values, ids):
        self.writes["material"].append(ids.clone())
        self.properties[ids] = values[ids]


def physics_env(clutter_modules, monkeypatch, count=4, objects=3):
    shelf, cylinders, robot = PhysicsView(count, 3), PhysicsView(count * objects, 2), PhysicsView(count, 6)
    env = SimpleNamespace(num_envs=count, scene={
        "shelf": SimpleNamespace(root_physx_view=shelf),
        "object_collection": SimpleNamespace(num_objects=objects, root_physx_view=cylinders),
        "robot": SimpleNamespace(root_physx_view=robot),
    })
    for module in (clutter_modules.physics, clutter_modules.parent_physics):
        monkeypatch.setattr(module, "hand_material_shape_ids", lambda robot: torch.tensor([1, 3, 4]))
    return env, (shelf, cylinders, robot)


def test_physics_reset_uses_object_major_rows_and_preserves_other_environments(
    clutter_modules, monkeypatch,
):
    physics = clutter_modules.physics
    env, views = physics_env(clutter_modules, monkeypatch)
    original_mass = views[1].mass.clone()
    original_inertia = views[1].inertia.clone()
    original_materials = [view.properties.clone() for view in views]
    physics.initialize_clutter_materials(env)
    initial_reads = [dict(view.reads) for view in views]
    ids = torch.tensor([3, 1])
    flat_ids = torch.tensor([3, 1, 7, 5, 11, 9])
    unaffected = torch.tensor([0, 2, 4, 6, 8, 10])
    for seed in (7, 13):
        torch.manual_seed(seed)
        physics.randomize_clutter_episode_physics(env, ids)
        state = env._clutter_physics
        assert torch.equal(views[1].writes["mass"][-1], flat_ids)
        assert torch.equal(views[1].writes["inertia"][-1], flat_ids)
        assert torch.equal(views[1].writes["material"][-1], flat_ids)
        expected_mass = state.mass[ids].repeat(3)
        assert torch.equal(views[1].mass[flat_ids, 0], expected_mass)
        expected_inertia = original_inertia[ids] * (state.mass[ids] / original_mass[ids, 0])[:, None]
        assert torch.allclose(views[1].inertia[flat_ids], expected_inertia.repeat(3, 1))
        assert torch.equal(views[1].mass[unaffected], original_mass[unaffected])
        assert torch.equal(views[1].inertia[unaffected], original_inertia[unaffected])
        assert torch.equal(views[1].properties[unaffected], original_materials[1][unaffected])
        assert torch.equal(views[1].properties[flat_ids, :, 0], state.static_friction[ids, 1].repeat(3)[:, None].expand(-1, 2))
        assert torch.equal(views[1].properties[flat_ids, :, 1], state.dynamic_friction[ids, 1].repeat(3)[:, None].expand(-1, 2))
        for role in (0, 2):
            assert torch.equal(views[role].properties[[0, 2]], original_materials[role][[0, 2]])
            assert torch.equal(views[role].writes["material"][-1], ids)
        assert torch.equal(views[2].properties[:, [0, 2, 5]], original_materials[2][:, [0, 2, 5]])
        for view, original, reads in zip(views, original_materials, initial_reads):
            assert torch.equal(view.properties[..., 2], original[..., 2])
            assert view.reads == reads


def test_clutter_physics_preserves_parent_sampling_and_exact_object_properties(
    clutter_modules, monkeypatch,
):
    parent, parent_views = physics_env(clutter_modules, monkeypatch, objects=1)
    child, child_views = physics_env(clutter_modules, monkeypatch, objects=4)
    clutter_modules.parent_physics.initialize_realistic_materials(parent)
    clutter_modules.physics.initialize_clutter_materials(child)
    child._clutter_physics.friction_buckets = parent._realistic_physics.friction_buckets.clone()
    for seed in (11, 42):
        torch.manual_seed(seed)
        clutter_modules.parent_physics.randomize_scaled_episode_physics(parent, None, mass_scale=1.5)
        torch.manual_seed(seed)
        clutter_modules.physics.randomize_clutter_episode_physics(child, None)
        assert torch.equal(child_views[1].mass, parent_views[1].mass.repeat(4, 1))
        assert torch.equal(child_views[1].inertia, parent_views[1].inertia.repeat(4, 1))
        assert torch.equal(child_views[1].properties, parent_views[1].properties.repeat(4, 1, 1))
        for role in (0, 2):
            assert torch.equal(child_views[role].properties, parent_views[role].properties)
        assert child._clutter_physics.mass.min() >= 0.345
        assert child._clutter_physics.mass.max() <= 0.750


@pytest.mark.parametrize("mass_scale", [0.0, -1.0, math.inf, math.nan])
def test_invalid_mass_scale_rejected_before_writes(clutter_modules, monkeypatch, mass_scale):
    env, views = physics_env(clutter_modules, monkeypatch)
    with pytest.raises(ValueError, match="mass_scale"):
        clutter_modules.physics.randomize_clutter_episode_physics(env, None, mass_scale=mass_scale)
    assert all(not any(view.writes.values()) for view in views)


def test_empty_reset_does_not_read_or_write_physics(clutter_modules, monkeypatch):
    env, views = physics_env(clutter_modules, monkeypatch)
    clutter_modules.physics.randomize_clutter_episode_physics(env, [])
    assert all(not any(view.writes.values()) and not any(view.reads.values()) for view in views)
