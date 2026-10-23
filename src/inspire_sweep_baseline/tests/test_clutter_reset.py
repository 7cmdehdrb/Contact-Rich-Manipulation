"""Reset integration: inherited targets, goal-first placement, fixed noisy snapshots."""

import importlib.util
import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_contact_sweep import contact_config
from test_omni_configuration import _execute_definitions, omni_config
from test_tactile_configuration import tactile_config
from test_tactile_renewal_configuration import renewal_tactile, renewal_tactile_v1


@pytest.fixture
def clutter_modules(load_mdp, monkeypatch):
    from _source_units import import_unit
    modules = {name: import_unit("contact_sweep_clutter." + name)
               for name in ("layout", "events", "mdp", "physics")}
    return SimpleNamespace(**modules, parent_events=load_mdp("goal_events"))


def _env(make_env, tensor_math, count=4):
    env = make_env(count)
    env.cfg = SimpleNamespace(seed=42)
    env._initial_obstacle_observation = torch.zeros(count, 40)
    shelf = env.scene["shelf"].data
    shelf.root_pos_w = torch.tensor([-.7, 0., 0.]).expand(count, -1).clone()
    shelf.root_quat_w = torch.tensor([1., 0., 0., 0.]).expand(count, -1).clone()
    # A translated, rotated robot root detects accidental world-frame caching.
    env.scene["robot"].data.root_pos_w[:] = torch.tensor([.2, -.3, .1])
    env.scene["robot"].data.root_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    objects = env.scene["object_collection"]
    objects.num_objects = 4
    objects.object_names = ["mass_cylinder"] + [f"obstacle_{i}" for i in range(3)]
    states = torch.zeros(count, 4, 13)
    states[..., 3] = 1
    objects.data = SimpleNamespace(
        object_link_state_w=states, object_pos_w=states[..., :3],
        object_quat_w=states[..., 3:7],
    )
    objects.write_object_link_state_to_sim = lambda values, env_ids: states.__setitem__(env_ids, values)
    return env


def test_reset_preserves_parent_target_distribution_and_goal_before_obstacles(clutter_modules, make_env, tensor_math):
    env = _env(make_env, tensor_math)
    torch.manual_seed(91)
    target = clutter_modules.parent_events._sample_uniform_disk(4, .02, "cpu")
    direction = clutter_modules.parent_events._sample_planar_unit_vectors(4, "cpu", math.radians(15))
    length = clutter_modules.parent_events._goal_ray_distance(target, direction, .15, (.14, .44))
    torch.manual_seed(91)
    clutter_modules.events.randomize_clutter_scene(env, None, observation_noise=0)
    state = env.scene["object_collection"].data.object_link_state_w
    assert torch.allclose(state[:, 0, :2], target + torch.tensor([-.7, 0.]))
    assert torch.allclose(env.sweep_dir[:, :2], direction * length[:, None])
    slots = env._initial_obstacle_observation.reshape(4, 10, 4)
    assert torch.equal(slots[:, :3, 0], torch.ones(4, 3))
    assert torch.count_nonzero(slots[:, 3:]) == 0
    robot = env.scene["robot"].data
    expected, _ = tensor_math.subtract_frame_transforms(
        robot.root_pos_w[:, None].expand(-1, 3, -1),
        robot.root_quat_w[:, None].expand(-1, 3, -1), state[:, 1:, :3],
    )
    assert torch.allclose(slots[:, :3, 1:], expected)


def test_initial_noise_is_bounded_and_cache_survives_motion_and_partial_resets(clutter_modules, make_env, tensor_math):
    env = _env(make_env, tensor_math)
    clutter_modules.events.randomize_clutter_scene(env, None)
    initial = env._initial_obstacle_observation.clone()
    robot, objects = env.scene["robot"].data, env.scene["object_collection"].data
    truth, _ = tensor_math.subtract_frame_transforms(
        robot.root_pos_w[:, None].expand(-1, 3, -1), robot.root_quat_w[:, None].expand(-1, 3, -1),
        objects.object_pos_w[:, 1:],
    )
    noise = initial.reshape(4, 10, 4)[:, :3, 1:] - truth
    assert noise.abs().max() <= .010001
    assert noise.abs().sum() > 0
    objects.object_pos_w[:, 1:] += 3
    assert torch.equal(clutter_modules.mdp.initial_obstacle_positions(env), initial)
    clutter_modules.events.randomize_clutter_scene(env, torch.tensor([2]))
    assert torch.equal(env._initial_obstacle_observation[[0, 1, 3]], initial[[0, 1, 3]])
    assert not torch.equal(env._initial_obstacle_observation[2], initial[2])


def test_clutter_changes_only_requested_manager_terms(contact_config, clutter_modules, monkeypatch):
    def as_dict(value):
        return value.to_dict() if hasattr(value, "to_dict") else vars(value).copy()

    namespace = contact_config.new.copy()
    namespace["DEFAULT_LAYOUT_CONFIG"] = clutter_modules.layout.DEFAULT_LAYOUT_CONFIG
    namespace.update(vars(clutter_modules.mdp))
    namespace.update(vars(clutter_modules.physics))
    namespace["randomize_clutter_scene"] = clutter_modules.events.randomize_clutter_scene
    namespace["ObservationGroupCfg"] = namespace["ObsGroup"]
    namespace["ObservationTermCfg"] = namespace["ObsTerm"]
    for name in ("scene_cfg", "observations_cfg", "events_cfg", "rewards_cfg", "terminations_cfg", "env_cfg"):
        module = types.ModuleType(f"_clutter_cfg_{name}")
        module.__dict__.update(namespace, __name__=module.__name__)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        result = _execute_definitions(MODULE_ROOT / "contact_sweep_clutter" / f"{name}.py", module.__dict__)
        namespace.update(vars(result))
    parent = namespace["ContactSweepEnvCfg"]()
    child = namespace["ContactSweepClutterEnvCfg"]()
    assert isinstance(child, type(parent))
    for name in ("actions", "commands", "sim", "viewer"):
        assert as_dict(getattr(child, name)) == as_dict(getattr(parent, name))
    assert as_dict(child.observations.policy) == as_dict(parent.observations.policy)
    for name in vars(parent.rewards):
        actual = as_dict(getattr(child.rewards, name))
        expected = as_dict(getattr(parent.rewards, name))
        if name == "object_tilt":
            actual["func"] = expected["func"]
        assert actual == expected, name
    for name in vars(parent.terminations):
        actual, expected = getattr(child.terminations, name), getattr(parent.terminations, name)
        if actual is None:
            assert expected is None
            continue
        a, b = as_dict(actual), as_dict(expected)
        if name == "object_drop":
            a["func"] = b["func"]
        assert a == b, name
    assert len(parent.scene.object_collection.rigid_objects) == 1
    assert len(child.scene.object_collection.rigid_objects) == 4
    assert list(vars(child.events)) == list(vars(parent.events))
    for parent_name in namespace["TACTILE_PARENT_BODY_NAMES"]:
        sensor = getattr(child.scene, namespace["parent_sensor_name"](parent_name))
        assert len(sensor.filter_prim_paths_expr) == 4
