"""Whole-hand coverage and target-specific contact force aggregation."""

import importlib.util
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_omni_configuration import _AssetNode, _execute_definitions, omni_config


@pytest.fixture
def contacts(monkeypatch):
    name = "_fusion_v1_0_contacts_test"
    spec = importlib.util.spec_from_file_location(
        name, MODULE_ROOT / "contact_sweep_clutter_fusion_v1_0/contacts.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scene_cfg(omni_config, contacts, monkeypatch):
    import importlib

    geometry = importlib.import_module("_baseline_cpu_pkg.assets.tactile_regions")
    robot = sys.modules["_baseline_cpu_pkg.assets.robot"]
    namespace = dict(vars(omni_config.baseline))
    namespace.update(
        __name__="_fusion_v1_0_parent_scene_test",
        RealisticShelfSpawnerCfg=_AssetNode, RealisticCylinderSpawnerCfg=_AssetNode,
        PALM_SENSOR_BODY_NAMES=robot.PALM_SENSOR_BODY_NAMES,
        TACTILE_PARENT_BODY_NAMES=geometry.TACTILE_PARENT_BODY_NAMES,
        parent_sensor_name=geometry.parent_sensor_name,
        DEFAULT_LAYOUT_CONFIG=SimpleNamespace(obstacles=3),
        OBJECT_MASS=1.0, TOOL_ROTATION_H=(1.0, 0.0, 0.0, 0.0),
    )
    parent = ModuleType(namespace["__name__"])
    parent.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion/scene_cfg.py", parent.__dict__)
    child = ModuleType("_fusion_v1_0_scene_test")
    child.__dict__.update(
        configclass=omni_config.baseline.configclass,
        sim_utils=namespace["sim_utils"],
        ContactSensorCfg=_AssetNode, HAND_CONTACT_BODY_NAMES=robot.HAND_CONTACT_BODY_NAMES,
        ContactSweepClutterFusionSceneCfg=parent.ContactSweepClutterFusionSceneCfg,
        object_hand_sensor_name=contacts.object_hand_sensor_name,
        TACTILE_PARENT_BODY_NAMES=geometry.TACTILE_PARENT_BODY_NAMES,
        parent_sensor_name=geometry.parent_sensor_name,
        validate_capacity=importlib.import_module("_baseline_cpu_pkg.contact_sweep_clutter_fusion_v1_0.population").validate_capacity,
        PARKING_HEIGHT=100.0,
    )
    monkeypatch.setitem(sys.modules, child.__name__, child)
    _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion_v1_0/scene_cfg.py", child.__dict__)
    return SimpleNamespace(
        child=child.ContactSweepClutterFusionV1_0SceneCfg,
        parent=parent.ContactSweepClutterFusionSceneCfg,
        robot=robot, geometry=geometry,
    )


def test_scene_preserves_tactile_and_covers_all_hand_bodies(scene_cfg, contacts):
    parent = scene_cfg.parent()
    snapshot = parent.to_dict()
    child = scene_cfg.child()
    hand_names = scene_cfg.robot.HAND_CONTACT_BODY_NAMES
    assert len(hand_names) == 30
    pattern = child.whole_hand_contact.prim_path.rsplit("/", 1)[-1]
    assert all(re.fullmatch(pattern, name) for name in hand_names)
    assert not any(re.fullmatch(pattern, name) for name in (*scene_cfg.robot.ARM_BODY_NAMES, "axia80_link"))
    assert child.whole_hand_contact.force_threshold == 0.01
    assert child.hand_tactile == parent.hand_tactile
    assert len(child.object_collection.rigid_objects) == 7
    for body in scene_cfg.geometry.TACTILE_PARENT_BODY_NAMES:
        name = scene_cfg.geometry.parent_sensor_name(body)
        actual, prior = getattr(child, name), getattr(parent, name)
        assert actual.prim_path == prior.prim_path
        assert actual.filter_prim_paths_expr == [obj.prim_path for obj in child.object_collection.rigid_objects.values()]
        assert actual.max_contact_data_count_per_prim == 16 * 7
    for name, obj in child.object_collection.rigid_objects.items():
        sensor = getattr(child, contacts.object_hand_sensor_name(name))
        assert sensor.prim_path == obj.prim_path
        assert obj.spawn.activate_contact_sensors is True
        assert sensor.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Robot/" + body for body in hand_names]
        assert not any("axia80_link" in path for path in sensor.filter_prim_paths_expr)
        assert sensor.update_period == 0.0
        assert sensor.history_length == 0
    assert parent.to_dict() == snapshot
    assert scene_cfg.parent().to_dict() == snapshot


def test_capacity_rebuild_removes_stale_sensors_and_keeps_instances_isolated(scene_cfg, contacts):
    child, other = scene_cfg.child(), scene_cfg.child()
    child.configure_population_capacity(max_gray_obstacles=1, max_white_obstacles=2)
    assert len(child.object_collection.rigid_objects) == 4
    assert not hasattr(child, contacts.object_hand_sensor_name("obstacle_5"))
    assert not hasattr(child, contacts.object_hand_sensor_name("obstacle_3"))
    assert hasattr(child, contacts.object_hand_sensor_name("obstacle_2"))
    assert len(other.object_collection.rigid_objects) == 7
    for body in scene_cfg.geometry.TACTILE_PARENT_BODY_NAMES:
        name = scene_cfg.geometry.parent_sensor_name(body)
        assert len(getattr(child, name).filter_prim_paths_expr) == 4
        assert len(getattr(other, name).filter_prim_paths_expr) == 7
    with pytest.raises(ValueError, match="at most 10"):
        child.configure_population_capacity(6, 5)


def test_whole_hand_sums_vectors_before_reward_norm(contacts):
    forces = torch.zeros(2, 30, 3)
    forces[0, 0] = torch.tensor([3.0, 4.0, 0.0])
    forces[0, 1] = torch.tensor([-3.0, -4.0, 0.0])
    forces[1, -1] = torch.tensor([0.0, 0.0, 0.01])
    env = SimpleNamespace(num_envs=2, scene={
        contacts.WHOLE_HAND_SENSOR_NAME: SimpleNamespace(data=SimpleNamespace(net_forces_w=forces)),
    })
    result = contacts.whole_hand_contact_force_w(env)
    torch.testing.assert_close(result, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]))


def test_obstacle_filter_excludes_each_env_target_and_preserves_vector_sign(contacts):
    names = ["mass_cylinder", "obstacle_0", "obstacle_1", "obstacle_2"]
    env = SimpleNamespace(
        num_envs=3, target_id=torch.tensor([[0], [2], [3]]),
        scene={"object_collection": SimpleNamespace(object_names=names)},
    )
    vectors = [
        torch.tensor([100.0, 0.0, 0.0]), torch.tensor([3.0, 4.0, 0.0]),
        torch.tensor([-3.0, -4.0, 0.0]), torch.tensor([0.0, 0.0, 6.0]),
    ]
    for name, force in zip(names, vectors):
        # Two matching filter columns exercise vector addition before norm.
        matrix = (force / 2).expand(3, 1, 2, 3).clone()
        env.scene[contacts.object_hand_sensor_name(name)] = SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=matrix),
        )
    actual = contacts.hand_obstacle_contact_force_w(env)
    torch.testing.assert_close(actual, torch.tensor([
        [0.0, 0.0, -6.0], [-103.0, -4.0, -6.0], [-100.0, 0.0, 0.0],
    ]))


def test_missing_filter_matrix_fails_instead_of_rewarding_unfiltered_forces(contacts):
    env = SimpleNamespace(
        num_envs=1, target_id=torch.tensor([[0]]),
        scene={
            "object_collection": SimpleNamespace(object_names=["mass_cylinder"]),
            contacts.object_hand_sensor_name("mass_cylinder"): SimpleNamespace(
                data=SimpleNamespace(force_matrix_w=None),
            ),
        },
    )
    with pytest.raises(RuntimeError, match="Object/hand contact sensor"):
        contacts.hand_obstacle_contact_force_w(env)
