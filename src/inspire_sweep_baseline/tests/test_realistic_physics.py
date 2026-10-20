"""Mass/friction distribution and reset isolation of the inherited experiment."""

import ast
from types import SimpleNamespace
import sys

import pytest
import torch

from conftest import MODULE_ROOT


def test_gaussian_properties_and_average_contact_pair_statistics(load_mdp):
    physics = load_mdp("realistic_physics")
    torch.manual_seed(20261013)
    mass, static, dynamic = physics.sample_episode_properties(200_000)
    assert mass.min().item() == pytest.approx(0.230)
    assert mass.max().item() == pytest.approx(0.500)
    assert mass.mean().item() == pytest.approx(0.365, abs=0.0005)
    # Clipping, rather than rejection sampling, retains a probability mass at
    # both limits (about 2.3% each for the unmodified two-sigma Gaussian tails).
    assert 0.018 < (mass == mass.min()).float().mean().item() < 0.028
    assert 0.018 < (mass == mass.max()).float().mean().item() < 0.028
    assert torch.allclose(static.mean(0), torch.tensor([0.315, 0.315, 0.585]), atol=0.001)
    assert torch.allclose(static.std(0), torch.tensor([0.051, 0.040, 0.145]), atol=0.001)
    assert torch.allclose(dynamic.mean(0), torch.tensor([0.252, 0.252, 0.468]), atol=0.001)
    assert torch.allclose(dynamic.std(0), torch.tensor([0.041, 0.032, 0.116]), atol=0.001)
    assert not torch.allclose(dynamic, static * 0.8)
    assert torch.allclose(dynamic.mean(0) / static.mean(0), torch.full((3,), 0.8), atol=0.003)
    assert static.min() >= 0.0
    floor_object = (static[:, 0] + static[:, 1]) / 2
    object_tool = (static[:, 1] + static[:, 2]) / 2
    assert floor_object.mean().item() == pytest.approx(0.315, abs=0.001)
    assert object_tool.mean().item() == pytest.approx(0.450, abs=0.001)
    assert floor_object.std().item() == pytest.approx((0.051**2 + 0.040**2)**0.5 / 2, abs=0.001)
    assert object_tool.std().item() == pytest.approx((0.040**2 + 0.145**2)**0.5 / 2, abs=0.001)


class FakeView:
    def __init__(self, count, shapes):
        self.max_shapes = shapes
        self.mass = torch.full((count, 1), 0.365)
        self.inertia = torch.tensor([0.000933333, 0, 0, 0, 0.000933333, 0, 0, 0, 0.0008]).expand(count, -1).clone() * 0.365
        self.properties = torch.ones(count, shapes, 3)
        self.properties[..., 2] = torch.arange(shapes) / 10
        self.reads = {"mass": 0, "inertia": 0, "material": 0}
        self.writes = {"mass": 0, "inertia": 0, "material": 0}

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
        self.writes["mass"] += 1
        self.mass[ids] = values[ids]

    def set_inertias(self, values, ids):
        self.writes["inertia"] += 1
        self.inertia[ids] = values[ids]

    def set_material_properties(self, values, ids):
        self.writes["material"] += 1
        self.properties[ids] = values[ids]


def make_physics_env(physics, monkeypatch, count=3):
    shelf, cylinder, robot = FakeView(count, 3), FakeView(count, 1), FakeView(count, 6)
    env = SimpleNamespace(num_envs=count, scene={
        "shelf": SimpleNamespace(root_physx_view=shelf),
        "object_collection": SimpleNamespace(num_objects=1, root_physx_view=cylinder),
        "robot": SimpleNamespace(root_physx_view=robot),
    })
    monkeypatch.setattr(physics, "hand_material_shape_ids", lambda robot: torch.tensor([1, 3, 4]))
    return env, (shelf, cylinder, robot)


def test_batched_reset_preserves_other_rows_arm_housing_and_restitution_without_readbacks(load_mdp, monkeypatch):
    physics = load_mdp("realistic_physics")
    # A runtime dependency on USD editing would fail this regression test.
    monkeypatch.setitem(sys.modules, "pxr", None)
    env, views = make_physics_env(physics, monkeypatch)
    original_materials = [view.properties.clone() for view in views]
    original_inertia = views[1].inertia.clone()
    physics.initialize_realistic_materials(env)
    initial_reads = [dict(view.reads) for view in views]
    state = env._realistic_physics
    buffers = [material.properties for material in state.materials]
    physics.randomize_episode_physics(env, torch.tensor([1]))
    first_mass = views[1].mass[1].clone()
    physics.randomize_episode_physics(env, torch.tensor([1]))
    assert not torch.equal(views[1].mass[1], first_mass)
    assert torch.equal(views[1].mass[[0, 2]], torch.full((2, 1), 0.365))
    assert torch.allclose(views[1].inertia[1], original_inertia[1] * (views[1].mass[1] / 0.365))
    assert torch.equal(views[1].inertia[[0, 2]], original_inertia[[0, 2]])
    for role, (view, original) in enumerate(zip(views, original_materials)):
        assert torch.equal(view.properties[[0, 2]], original[[0, 2]])
        assert torch.equal(view.properties[..., 2], original[..., 2])
        shapes = state.materials[role].shape_ids
        assert torch.all(view.properties[1, shapes, 0] == state.static_friction[1, role])
        assert torch.all(view.properties[1, shapes, 1] == state.dynamic_friction[1, role])
        assert state.materials[role].properties is buffers[role]
        assert view.reads == initial_reads[role]
        assert view.writes["material"] == 2
    assert torch.equal(views[2].properties[:, [0, 2, 5]], original_materials[2][:, [0, 2, 5]])


@pytest.mark.parametrize("count", [1, 2048])
def test_reset_has_constant_number_of_physx_calls(load_mdp, monkeypatch, count):
    physics = load_mdp("realistic_physics")
    env, views = make_physics_env(physics, monkeypatch, count)
    physics.initialize_realistic_materials(env)
    physics.randomize_episode_physics(env, torch.arange(count))
    assert [view.writes["material"] for view in views] == [1, 1, 1]
    assert views[1].writes["mass"] == views[1].writes["inertia"] == 1
    assert views[1].reads["mass"] == views[1].reads["inertia"] == 1


def test_mass_scale_changes_mass_and_inertia_without_compounding_or_changing_friction(load_mdp, monkeypatch):
    physics = load_mdp("realistic_physics")
    parent, parent_views = make_physics_env(physics, monkeypatch)
    child, child_views = make_physics_env(physics, monkeypatch)
    physics.initialize_realistic_materials(parent)
    physics.initialize_realistic_materials(child)
    # Use the same friction pool and random draws to isolate the mass change.
    child._realistic_physics.friction_buckets = parent._realistic_physics.friction_buckets.clone()
    original_inertia = child_views[1].inertia.clone()
    ids = torch.tensor([1])
    for seed in (7, 11):
        torch.manual_seed(seed)
        physics.randomize_episode_physics(parent, ids)
        torch.manual_seed(seed)
        physics.randomize_scaled_episode_physics(child, ids, mass_scale=1.5)
        assert torch.allclose(child_views[1].mass[ids], parent_views[1].mass[ids] * 1.5)
        assert torch.allclose(child_views[1].inertia[ids], parent_views[1].inertia[ids] * 1.5)
        assert 0.345 <= child._realistic_physics.mass[1] <= 0.750
        assert torch.equal(child_views[1].mass[[0, 2]], torch.full((2, 1), 0.365))
        assert torch.equal(child_views[1].inertia[[0, 2]], original_inertia[[0, 2]])
        for a, b in zip(parent_views, child_views):
            assert torch.equal(a.properties, b.properties)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_mass_scale_rejected_before_physics_writes(load_mdp, monkeypatch, scale):
    physics = load_mdp("realistic_physics")
    env, views = make_physics_env(physics, monkeypatch)
    with pytest.raises(ValueError, match="mass_scale"):
        physics.randomize_scaled_episode_physics(env, None, mass_scale=scale)
    assert all(not any(view.writes.values()) for view in views)


def test_gaussian_bucket_pool_retains_marginals_and_bounds_material_count(load_mdp):
    physics = load_mdp("realistic_physics")
    torch.manual_seed(42)
    pools = physics.make_friction_buckets()
    assert pools.shape == (3, 4096, 2)
    expected_means = torch.tensor([physics.STATIC_FRICTION_MEANS, physics.DYNAMIC_FRICTION_MEANS]).T
    expected_stds = torch.tensor([physics.STATIC_FRICTION_STDS, physics.DYNAMIC_FRICTION_STDS]).T
    assert torch.allclose(pools.mean(1), expected_means, atol=1e-6)
    assert torch.allclose(pools.std(1), expected_stds, atol=0.0001)
    for role in range(3):
        assert abs(torch.corrcoef(pools[role].T)[0, 1]) < 0.08
    mass, static, dynamic = physics.sample_episode_properties(20_000, pools)
    assert mass.min() >= 0.23 and mass.max() <= 0.50
    pairs = torch.stack((static, dynamic), dim=-1)
    for role in range(3):
        assert torch.unique(pairs[:, role], dim=0).shape[0] <= 4096
    assert not torch.equal(dynamic, static * 0.8)


def test_hand_shape_mapping_uses_body_names_and_excludes_arm_and_axia(load_mdp):
    physics = load_mdp("realistic_physics")
    from _baseline_cpu_pkg.assets.robot import HAND_CONTACT_BODY_NAMES
    names = ["axia80_link", *reversed(HAND_CONTACT_BODY_NAMES), "wrist_3_link"]
    sizes = [1, *([1] * 30), 2]
    paths = [f"/Robot/{name}" for name in names]
    size_by_path = dict(zip(paths, sizes))
    robot = SimpleNamespace(
        body_names=names,
        root_physx_view=SimpleNamespace(link_paths=[paths], max_shapes=33),
        _physics_sim_view=SimpleNamespace(create_rigid_body_view=lambda path: SimpleNamespace(max_shapes=size_by_path[path])),
    )
    assert physics.hand_material_shape_ids(robot).tolist() == list(range(1, 31))


def test_variant_uses_replication_and_spawn_time_bindings_with_no_prestartup_edits():
    tree = ast.parse((MODULE_ROOT / "realistic_physics_env_cfg.py").read_text())
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InspireSweepRealisticPhysicsEnvCfg")
    assert [node.id for node in cfg.bases] == ["InspireSweepEnvCfg"]
    assert [node.target.id for node in cfg.body if isinstance(node, ast.AnnAssign)] == ["events"]
    assert "replicate_physics = False" not in ast.unparse(tree)
    assert "apply_realistic_physics(self)" in ast.unparse(cfg)
    physics_setup = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == "apply_realistic_physics")
    assert "RealisticRobotSpawnerCfg" in ast.unparse(physics_setup)
    assert "RealisticShelfSpawnerCfg" in ast.unparse(physics_setup)
    assert "RealisticCylinderSpawnerCfg" in ast.unparse(physics_setup)
    events = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RealisticPhysicsEventsCfg")
    assert [node.id for node in events.bases] == ["EventsCfg"]
    event_terms = {node.targets[0].id: node.value for node in events.body if isinstance(node, ast.Assign)}
    assert list(event_terms) == ["physics_materials", "physics_randomization"]
    assert {name: ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == "mode"))
            for name, call in event_terms.items()} == {"physics_materials": "startup", "physics_randomization": "reset"}
    runner = ast.parse((MODULE_ROOT / "agents/realistic_physics_ppo_cfg.py").read_text())
    runner = next(node for node in runner.body if isinstance(node, ast.ClassDef))
    assert runner.bases[0].id == "InspireSweepPPORunnerCfg"
    assert [node.targets[0].id for node in runner.body if isinstance(node, ast.Assign)] == ["experiment_name"]
