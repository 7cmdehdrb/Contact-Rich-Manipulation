"""Exercise real configclass inheritance and task-relative scene placement on CPU."""

import ast
from dataclasses import fields
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "hand_manipulation_test"


def _class_node(relative, name):
    tree = ast.parse((PACKAGE_ROOT / relative).read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


@pytest.fixture(scope="module")
def task_classes():
    # Load the installed workspace's actual decorator without importing Kit.
    # Serialization is outside this test: its two unused dependency functions
    # deliberately raise if construction/copy starts calling them.
    decorator_path = PROJECT_ROOT.parents[1] / "IsaacLab/source/isaaclab/isaaclab/utils/configclass.py"
    if not decorator_path.is_file():
        pytest.skip("Isaac Lab configclass source is unavailable")
    package = ModuleType("_table_config_fixture")
    package.__path__ = []
    serialization = ModuleType(package.__name__ + ".dict")

    def unexpected_serialization(*args, **kwargs):
        raise AssertionError("Serialization is not part of this fixture")

    serialization.class_to_dict = serialization.update_class_from_dict = unexpected_serialization
    spec = importlib.util.spec_from_file_location(package.__name__ + ".configclass", decorator_path)
    decorator = importlib.util.module_from_spec(spec)
    namespace = ModuleType(package.__name__ + ".task_classes")
    with patch.dict(sys.modules, {package.__name__: package, serialization.__name__: serialization,
                                  decorator.__name__: decorator, namespace.__name__: namespace}):
        spec.loader.exec_module(decorator)
        namespace.__dict__.update(configclass=decorator.configclass, fields=fields, math=math,
                                  cube_circumsphere_radius=lambda size: math.sqrt(3) * size / 2)
        for relative, name in (("env_cfg.py", "ReachTaskCfg"),
                               ("config/ur5e/contact_env_cfg.py", "ContactTaskCfg"),
                               ("config/ur5e/push_env_cfg.py", "PushTaskCfg")):
            node = _class_node(relative, name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace.__dict__)
        yield namespace.ContactTaskCfg, namespace.PushTaskCfg


def test_contact_and_push_share_one_table_center_and_preserve_separate_cube_boxes(task_classes):
    contact, push = (cls() for cls in task_classes)
    assert contact.table_center == push.table_center == (-.75, 0., 1.03)
    assert contact.object_xy_range_low == pytest.approx((-.82, -.22))
    assert contact.object_xy_range_high == pytest.approx((-.63, .22))
    assert push.object_xy_range_low == pytest.approx((-.71, -.06))
    assert push.object_xy_range_high == pytest.approx((-.69, .06))
    assert contact.reset_position_offset_task == push.reset_position_offset_task == (.14, 0., .10)
    # No derived class literal may silently retain an older center.
    push_node = _class_node("config/ur5e/push_env_cfg.py", "PushTaskCfg")
    assigned = {item.target.id for item in push_node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
    assert not {"table_center", "reset_position_offset_task"} & assigned


@pytest.mark.parametrize("kind", (0, 1), ids=("contact", "push"))
def test_cube_box_and_height_follow_instance_center_after_configclass_copy(task_classes, kind):
    original = task_classes[kind]()
    copied = original.copy()
    copied.table_center = (-.90, .15, 1.13)
    assert copied.object_xy_range_low == pytest.approx(tuple(copied.table_center[i] + copied.object_xy_offset_low[i] for i in range(2)))
    assert copied.object_xy_range_high == pytest.approx(tuple(copied.table_center[i] + copied.object_xy_offset_high[i] for i in range(2)))
    assert copied.cube_center_height_m == pytest.approx(1.18)
    assert original.table_center == (-.75, 0., 1.03)
    # Derived values remain descriptors, avoiding configclass's stale copied
    # instance attributes when inherited properties are not redeclared.
    assert not {"object_xy_range_low", "object_xy_range_high", "cube_center_height_m"} & vars(copied).keys()


@pytest.mark.parametrize("kind", (0, 1), ids=("contact", "push"))
def test_inherited_scene_post_init_uses_current_task_table_and_spawn_midpoint(task_classes, kind):
    task = task_classes[kind]()
    task.table_center = (-.90, .15, 1.13)
    node = _class_node("config/ur5e/contact_env_cfg.py", "UR5eInspireContactEnvCfg")
    node.decorator_list = []
    node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__post_init__"]

    class Parent:
        def __post_init__(self):
            self.parent_calls += 1

    namespace = {"UR5eInspireReachEnvCfg": Parent}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "contact_env_cfg.py", "exec"), namespace)
    cfg = namespace[node.name]()
    cfg.task, cfg.parent_calls, cfg.decimation = task, 0, 2
    cfg.scene = SimpleNamespace(
        table=SimpleNamespace(spawn=SimpleNamespace(), init_state=SimpleNamespace()),
        target_object=SimpleNamespace(spawn=SimpleNamespace(mass_props=SimpleNamespace()), init_state=SimpleNamespace()),
        table_contacts=SimpleNamespace(), lazy_sensor_update=True,
    )
    cfg.viewer = SimpleNamespace()
    cfg.__post_init__()
    assert cfg.parent_calls == 1
    assert cfg.scene.table.init_state.pos == task.table_center
    assert cfg.scene.table.spawn.size == task.table_size
    expected = tuple((task.object_xy_range_low[i] + task.object_xy_range_high[i]) / 2 for i in range(2)) + (task.cube_center_height_m,)
    assert cfg.scene.target_object.init_state.pos == pytest.approx(expected)
    assert cfg.viewer.lookat == pytest.approx((*task.table_center[:2], task.cube_center_height_m))
    assert cfg.scene.table_contacts.history_length == 2
    assert not cfg.scene.lazy_sensor_update
