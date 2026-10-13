"""Mass/friction distribution and reset isolation of the inherited experiment."""

import ast
from contextlib import nullcontext
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


def test_episode_resampling_updates_only_selected_envs_without_inertia_drift(load_mdp, monkeypatch):
    physics = load_mdp("realistic_physics")
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Sdf=SimpleNamespace(ChangeBlock=nullcontext)))

    class Attribute:
        def __init__(self, value):
            self.value = value

        def Set(self, value):
            self.value = value

    class View:
        def __init__(self):
            self.mass = torch.full((3, 1), 0.365)
            self.inertia = torch.tensor([0.000933333, 0, 0, 0, 0.000933333, 0, 0, 0, 0.0008]).expand(3, -1).clone() * 0.365

        def get_masses(self):
            return self.mass

        def get_inertias(self):
            return self.inertia

        def set_masses(self, values, ids):
            self.mass[ids] = values[ids]

        def set_inertias(self, values, ids):
            self.inertia[ids] = values[ids]

    view = View()
    original_inertia = view.inertia.clone()
    bindings = [[[(Attribute(mean), Attribute(mean * 0.8))] for mean in physics.STATIC_FRICTION_MEANS] for _ in range(3)]
    state = physics.EpisodePhysicsState(
        bindings, view.mass.clone(), original_inertia.clone(), torch.full((3,), 0.365),
        torch.tensor(physics.STATIC_FRICTION_MEANS).expand(3, 3).clone(),
        torch.tensor(physics.DYNAMIC_FRICTION_MEANS).expand(3, 3).clone(),
    )
    env = SimpleNamespace(num_envs=3, scene={"object_collection": SimpleNamespace(root_physx_view=view)}, _realistic_physics=state)
    torch.manual_seed(1)
    physics.randomize_episode_physics(env, torch.tensor([1]))
    first_mass = view.mass[1].clone()
    first_friction = state.static_friction[1].clone()
    assert torch.equal(view.mass[[0, 2]], torch.full((2, 1), 0.365))
    assert torch.equal(view.inertia[[0, 2]], original_inertia[[0, 2]])
    assert torch.allclose(view.inertia[1], original_inertia[1] * (view.mass[1] / 0.365))
    physics.randomize_episode_physics(env, torch.tensor([1]))
    assert not torch.equal(view.mass[1], first_mass)
    assert not torch.equal(state.static_friction[1], first_friction)
    assert torch.allclose(view.inertia[1], original_inertia[1] * (view.mass[1] / 0.365))
    for role, pairs in enumerate(bindings[1]):
        assert pairs[0][0].value == pytest.approx(state.static_friction[1, role].item())
        assert pairs[0][1].value == pytest.approx(state.dynamic_friction[1, role].item())
    # No new material is allocated on repeated resets.
    assert env._realistic_physics.materials is bindings


def test_variant_changes_only_physics_events_object_mass_and_required_replication_mode():
    tree = ast.parse((MODULE_ROOT / "realistic_physics_env_cfg.py").read_text())
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InspireSweepRealisticPhysicsEnvCfg")
    assert [node.id for node in cfg.bases] == ["InspireSweepEnvCfg"]
    assignments = [node for node in cfg.body if isinstance(node, ast.AnnAssign)]
    assert [node.target.id for node in assignments] == ["events"]
    post = next(node for node in cfg.body if isinstance(node, ast.FunctionDef))
    writes = [node.targets[0] for node in post.body if isinstance(node, ast.Assign)]
    assert len(writes) == 3
    assert writes[0].attr == "replicate_physics"
    replication = next(node for node in post.body if isinstance(node, ast.Assign) and node.targets[0] is writes[0])
    assert ast.literal_eval(replication.value) is False
    assert writes[2].attr == "mass_props"
    events = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RealisticPhysicsEventsCfg")
    assert [node.id for node in events.bases] == ["EventsCfg"]
    event_terms = {node.targets[0].id: node.value for node in events.body if isinstance(node, ast.Assign)}
    assert list(event_terms) == ["physics_material_bindings", "physics_materials", "physics_randomization"]
    assert {name: ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == "mode"))
            for name, call in event_terms.items()} == {
        "physics_material_bindings": "prestartup", "physics_materials": "startup", "physics_randomization": "reset",
    }
    runner = ast.parse((MODULE_ROOT / "agents/realistic_physics_ppo_cfg.py").read_text())
    runner = next(node for node in runner.body if isinstance(node, ast.ClassDef))
    assert runner.bases[0].id == "InspireSweepPPORunnerCfg"
    assert [node.targets[0].id for node in runner.body if isinstance(node, ast.Assign)] == ["experiment_name"]


def test_material_binding_is_rejected_after_physics_starts(load_mdp):
    physics = load_mdp("realistic_physics")
    env = SimpleNamespace(sim=SimpleNamespace(is_playing=lambda: True))
    with pytest.raises(RuntimeError, match="before simulation starts"):
        physics.prepare_realistic_materials(env)


def test_startup_without_prepared_materials_never_falls_back_to_usd_binding(load_mdp):
    physics = load_mdp("realistic_physics")
    env = SimpleNamespace()
    with pytest.raises(RuntimeError, match="Missing prestartup material bindings"):
        physics.initialize_realistic_materials(env)


def test_prestartup_needs_no_physics_views_and_startup_only_caches_defaults(load_mdp, monkeypatch):
    physics = load_mdp("realistic_physics")
    stage = object()
    binds = []
    materials_module = SimpleNamespace(
        material_defaults=lambda cfg: {},
        bind_role_materials=lambda *args: binds.append(args) or [(object(), object())],
    )
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", SimpleNamespace(get_current_stage=lambda: stage))
    monkeypatch.setitem(sys.modules, "_baseline_cpu_pkg.assets.realistic_materials", materials_module)

    class Scene(dict):
        env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]

    # No asset/view is available before simulation starts.
    env = SimpleNamespace(
        sim=SimpleNamespace(is_playing=lambda: False),
        scene=Scene(), num_envs=2,
        cfg=SimpleNamespace(sim=SimpleNamespace(physics_material=object())),
    )
    physics.prepare_realistic_materials(env)
    assert len(binds) == 6  # Two environments, three contact roles.
    assert all(args[0] is stage for args in binds)
    assert len(binds[2][1]) == 30
    assert len(env._realistic_material_bindings) == 2

    # When physics views become available, startup must not call binding again.
    env.sim.is_playing = lambda: True
    masses = torch.full((2, 1), 0.365)
    inertias = torch.ones(2, 9)
    env.scene["object_collection"] = SimpleNamespace(
        num_objects=1,
        root_physx_view=SimpleNamespace(get_masses=lambda: masses, get_inertias=lambda: inertias),
    )
    physics.initialize_realistic_materials(env)
    assert len(binds) == 6
    assert env._realistic_physics.materials is env._realistic_material_bindings
    assert torch.equal(env._realistic_physics.default_mass, masses)
    masses.fill_(0.5)
    assert torch.equal(env._realistic_physics.default_mass, torch.full((2, 1), 0.365))
