"""Check Push v1 registration and inherited public entry points without Kit."""

import argparse
import ast
from copy import deepcopy
from dataclasses import fields
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_shared_table_config import task_classes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "hand_manipulation_test"
TASK_ID = "Isaac-Hand-Manipulation-Push-v1"


def _class(relative, name):
    path = PACKAGE_ROOT / relative
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_v1_registration_is_idempotent_and_preserves_v0_registration():
    gym = pytest.importorskip("gymnasium")
    package = importlib.import_module("hand_manipulation_test")
    original = gym.spec(package.PUSH_TASK_ID)
    package._register_task()
    spec = gym.spec(TASK_ID)
    assert package.PUSH_V1_TASK_ID == TASK_ID
    assert spec.entry_point == "hand_manipulation_test.push_v1_env:HandManipulationPushV1Env"
    assert spec.disable_env_checker
    assert spec.kwargs == {
        "env_cfg_entry_point": "hand_manipulation_test.config.ur5e.push_v1_env_cfg:UR5eInspirePushV1EnvCfg",
        "rsl_rl_cfg_entry_point": "hand_manipulation_test.agents.rsl_rl_push_v1_ppo_cfg:PushV1PPORunnerCfg",
    }
    package._register_task()
    assert gym.spec(TASK_ID) is spec
    assert gym.spec(package.PUSH_TASK_ID) is original


def test_v1_environment_inherits_standard_manager_flow_without_overriding_step_or_reset():
    cls = _class("push_v1_env.py", "HandManipulationPushV1Env")
    assert [ast.unparse(base) for base in cls.bases] == ["HandManipulationPushEnv"]
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in cls.body)


def test_v1_runner_inherits_all_ppo_settings_and_only_changes_log_name():
    cls = _class("agents/rsl_rl_push_v1_ppo_cfg.py", "PushV1PPORunnerCfg")
    assert [ast.unparse(base) for base in cls.bases] == ["PushPPORunnerCfg"]
    overrides = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in cls.body if isinstance(node, ast.Assign)}
    assert overrides == {"experiment_name": "hand_manipulation_push_v1"}
    assert not any(isinstance(node, (ast.AnnAssign, ast.FunctionDef)) for node in cls.body)


def test_v1_cfg_inherits_push_scene_and_appends_only_three_controller_observations():
    cfg = _class("config/ur5e/push_v1_env_cfg.py", "UR5eInspirePushV1EnvCfg")
    assert [ast.unparse(base) for base in cfg.bases] == ["UR5eInspirePushEnvCfg"]
    declared = {node.target.id for node in cfg.body if isinstance(node, ast.AnnAssign)}
    assert not {"scene", "terminations"} & declared
    observations = _class("config/ur5e/push_v1_env_cfg.py", "PushV1ObservationsCfg")
    assert [ast.unparse(base) for base in observations.bases] == ["PushObservationsCfg"]
    policy = next(node for node in observations.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
    assert [ast.unparse(base) for base in policy.bases] == ["PushObservationsCfg.PolicyCfg"]
    terms = {node.targets[0].id: node.value for node in policy.body if isinstance(node, ast.Assign)}
    assert tuple(terms) == ("accumulated_translation_error",)
    assert {keyword.arg for keyword in terms["accumulated_translation_error"].keywords} == {"func"}
    from hand_manipulation_test.constants import (
        ACTION_DIM, PUSH_OBSERVATION_LAYOUT, PUSH_OBSERVATION_SLICES,
        PUSH_V1_OBSERVATION_DIM, PUSH_V1_OBSERVATION_LAYOUT, PUSH_V1_OBSERVATION_SLICES,
    )
    assert ACTION_DIM == 8 and PUSH_V1_OBSERVATION_DIM == 64
    assert PUSH_V1_OBSERVATION_LAYOUT[:-1] == PUSH_OBSERVATION_LAYOUT
    assert PUSH_V1_OBSERVATION_SLICES["accumulated_translation_error"] == slice(61, 64)
    assert {name: PUSH_V1_OBSERVATION_SLICES[name] for name in PUSH_OBSERVATION_SLICES} == dict(PUSH_OBSERVATION_SLICES)


def test_v1_task_retains_parent_workspace_and_live_copied_geometry_properties(task_classes):
    _, push_class = task_classes
    namespace = sys.modules[push_class.__module__].__dict__
    node = _class("config/ur5e/push_v1_env_cfg.py", "PushV1TaskCfg")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "push_v1_env_cfg.py", "exec"), namespace)
    parent, v1 = push_class(), namespace["PushV1TaskCfg"]()
    overrides = {"object_xy_offset_low", "object_xy_offset_high"}
    for field in fields(push_class):
        if field.name in overrides:
            continue
        assert getattr(v1, field.name) == getattr(parent, field.name), field.name
    assert parent.table_size == (.36, 1., .04)
    assert v1.table_size == parent.table_size == (.36, 1., .04)
    assert v1.table_center[0] + v1.table_size[0]/2 == pytest.approx(-.57)
    declared = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
    assert not {"table_center", "table_size", "reset_position_offset_task"} & declared
    assert v1.object_xy_range_low == pytest.approx((-.73, -.03))
    assert v1.object_xy_range_high == pytest.approx((-.67, .03))
    assert v1.reset_position_offset_task == (.14, 0., .10)
    assert v1.reset_position_jitter_task == (.004, .004, .003)
    assert v1.command_centered_path
    assert v1.command_midpoint_x_offset_m == .05
    assert v1.command_initial_x_jitter_m == .002
    assert v1.contact_palm_height_m == .025
    assert v1.left_contact_reference_body == "inspire_thumb_force_sensor_4"
    # Execute the constructor's selector with the real config, a legacy task
    # lacking the new field, and an explicit neighboring-pad override. This
    # keeps startup's fallback consistent with the configured approach point
    # without importing Kit or claiming a synthetic contact is physical.
    reset = _class("mdp/push_v1_events.py", "PushV1SafePoseReset")
    constructor = next(item for item in reset.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
    selector = next(item.value for item in constructor.body if isinstance(item, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "name" for target in item.targets))
    expression = compile(ast.Expression(selector), "push_v1_events.py", "eval")
    for task, expected in ((v1, "inspire_thumb_force_sensor_4"),
                           (SimpleNamespace(), "inspire_thumb_force_sensor_4"),
                           (SimpleNamespace(left_contact_reference_body="inspire_thumb_force_sensor_3"),
                            "inspire_thumb_force_sensor_3")):
        assert eval(expression, {"env": SimpleNamespace(cfg=SimpleNamespace(task=task))}) == expected
    assert v1.enforce_outward_fingers and v1.minimum_finger_outward_cos == .25
    assert v1.wrist_2_branch_sin_margin == .15
    assert v1.reset_joint_seed_offsets == parent.reset_joint_seed_offsets
    assert len(v1.reset_joint_seed_offsets) == 3 and v1.reset_max_iterations == 80
    for seed in (v1.right_reset_arm_seed, v1.left_reset_arm_seed):
        assert len(seed) == 6 and all(math.isfinite(value) for value in seed)
        assert math.sin(seed[4]) > v1.wrist_2_branch_sin_margin
    assert not hasattr(v1, "left_finger_down_rad")
    copied = v1.copy()
    copied.table_center = (-.90, .15, 1.13)
    assert copied.object_xy_range_low == pytest.approx((-.88, .12))
    assert copied.object_xy_range_high == pytest.approx((-.82, .18))
    assert copied.cube_center_height_m == pytest.approx(1.18)
    assert v1.table_center == parent.table_center == (-.75, 0., 1.03)
    assert not {"object_xy_range_low", "object_xy_range_high", "cube_center_height_m"} & vars(copied).keys()


def test_v1_post_init_preserves_parent_osc_fields_and_hand_when_replacing_arm_term():
    cfg_node = _class("config/ur5e/push_v1_env_cfg.py", "UR5eInspirePushV1EnvCfg")
    cfg_node.decorator_list = []
    cfg_node.body = [node for node in cfg_node.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"]

    class Parent:
        def __post_init__(self):
            self.parent_calls += 1

    def accumulating_cfg(**values):
        return SimpleNamespace(**values, class_type="accumulated controller")

    def v1_spawner(*args, **kwargs):
        raise AssertionError("Configuration must assign the spawner without executing it")

    namespace = {"UR5eInspirePushEnvCfg": Parent, "deepcopy": deepcopy,
                 "AccumulatedTranslationOscActionCfg": accumulating_cfg,
                 "spawn_push_v1_robot": v1_spawner}
    exec(compile(ast.Module(body=[cfg_node], type_ignores=[]), "push_v1_env_cfg.py", "exec"), namespace)
    cfg = namespace[cfg_node.name]()
    inherited = SimpleNamespace(class_type="original controller", motion_stiffness=(200.,)*6,
        translation_scale=(.012, .012, .012), rotation_scale=(.05,)*3,
        body_offset_pos=(0., .05, .10), gravity_compensation=True, mutable_setting=[1., 2.],
        contact_aware_impedance=False)
    hand = SimpleNamespace(synergy_range=(.8, 1.), enforce_synergy_joint_limits=True)
    cfg.actions = SimpleNamespace(arm_action=inherited, hand_action=hand)
    spawn = SimpleNamespace(func="inherited spawner", usd_path="original robot asset", activate_contact_sensors=True)
    cfg.scene = SimpleNamespace(robot=SimpleNamespace(spawn=spawn))
    cfg.task = SimpleNamespace(accumulated_position_error_limit_m=.06)
    cfg.parent_calls = 0
    cfg.__post_init__()
    assert cfg.parent_calls == 1
    assert cfg.actions.hand_action is hand
    assert cfg.actions.arm_action.class_type == "accumulated controller"
    assert cfg.actions.arm_action.position_error_limit_m == .06
    assert cfg.actions.arm_action.contact_aware_impedance is True
    assert inherited.contact_aware_impedance is False
    assert cfg.scene.robot.spawn is spawn and spawn.func is v1_spawner
    assert spawn.usd_path == "original robot asset" and spawn.activate_contact_sensors
    for name, value in vars(inherited).items():
        if name not in {"class_type", "contact_aware_impedance"}:
            assert getattr(cfg.actions.arm_action, name) == value, name
    assert cfg.actions.arm_action.mutable_setting is not inherited.mutable_setting
    cfg.__post_init__()
    assert cfg.parent_calls == 2
    assert cfg.actions.arm_action.contact_aware_impedance is True
    assert cfg.scene.robot.spawn is spawn and spawn.func is v1_spawner
    assert cfg.actions.hand_action is hand


def test_real_configclass_replace_reinitializes_existing_v1_controller_without_duplicate_fields(task_classes):
    """replace() runs the production post-init with an already-converted arm."""

    _, push_class = task_classes
    decorator = sys.modules[push_class.__module__].configclass

    def accumulating_cfg(**values):
        return SimpleNamespace(**values, class_type="accumulated controller")

    def v1_spawner(*args, **kwargs):
        raise AssertionError("Config copy must not spawn simulator assets")

    namespace = {"__name__": push_class.__module__, "configclass": decorator, "SimpleNamespace": SimpleNamespace,
                 "deepcopy": deepcopy, "AccumulatedTranslationOscActionCfg": accumulating_cfg,
                 "spawn_push_v1_robot": v1_spawner}
    exec("""
@configclass
class UR5eInspirePushEnvCfg:
    parent_calls: int = 0
    task: SimpleNamespace = SimpleNamespace(accumulated_position_error_limit_m=.06)
    actions: SimpleNamespace = SimpleNamespace(
        arm_action=SimpleNamespace(class_type='original controller', motion_stiffness=(200.,)*6,
                                   gravity_compensation=True, mutable_setting=[1., 2.]),
        hand_action=SimpleNamespace(synergy_range=(.8, 1.), enforce_synergy_joint_limits=True))
    scene: SimpleNamespace = SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(
        func='inherited spawner', usd_path='original robot asset', activate_contact_sensors=True)))
    def __post_init__(self):
        self.parent_calls += 1
""", namespace)
    node = _class("config/ur5e/push_v1_env_cfg.py", "UR5eInspirePushV1EnvCfg")
    node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__post_init__"]
    exec(compile(ast.Module(body=[node], type_ignores=[]), "push_v1_env_cfg.py", "exec"), namespace)
    original = namespace[node.name]()
    assert original.parent_calls == 1
    assert original.actions.arm_action.position_error_limit_m == .06
    assert original.actions.arm_action.contact_aware_impedance is True
    assert original.scene.robot.spawn.func is v1_spawner
    replaced = original.replace(task=SimpleNamespace(accumulated_position_error_limit_m=.04))
    assert replaced.parent_calls == 2
    assert replaced.actions.arm_action.position_error_limit_m == .04
    assert original.actions.arm_action.position_error_limit_m == .06
    assert replaced.actions.arm_action.motion_stiffness == (200.,)*6
    assert replaced.actions.arm_action.gravity_compensation is True
    assert replaced.actions.arm_action.contact_aware_impedance is True
    assert replaced.scene.robot.spawn.func is v1_spawner
    assert replaced.scene.robot.spawn.usd_path == "original robot asset"
    assert replaced.scene.robot.spawn.activate_contact_sensors
    assert replaced.actions.hand_action.synergy_range == (.8, 1.)
    replaced.actions.arm_action.mutable_setting.append(3.)
    assert original.actions.arm_action.mutable_setting == [1., 2.]
    # Explicit post-init repetition is also supported for already-converted configs.
    replaced.__post_init__()
    assert replaced.parent_calls == 3
    assert replaced.actions.arm_action.position_error_limit_m == .04
    assert replaced.actions.arm_action.contact_aware_impedance is True
    assert replaced.scene.robot.spawn.func is v1_spawner


@pytest.mark.parametrize("name", ("train.py", "play.py"))
def test_runtime_cli_accepts_v1_and_resolves_the_selected_registry_configuration(name):
    path = PROJECT_ROOT / "scripts" / name
    tree = ast.parse(path.read_text())
    bindings = {node.targets[0].id: ast.literal_eval(node.value)
                for node in tree.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id.endswith("TASK_ID")}
    task_option = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                       and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--task")
    parser = argparse.ArgumentParser()
    namespace = {**bindings, "parser": parser}
    option_tree = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=task_option)], type_ignores=[]))
    exec(compile(option_tree, str(path), "exec"), namespace)
    assert parser.parse_args(["--task", TASK_ID]).task == TASK_ID
    assert parser.parse_args([]).task == "Isaac-Hand-Manipulation-Test-v0"
    config_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "load_cfg_from_registry"]
    assert {ast.literal_eval(node.args[1]) for node in config_calls} == {"env_cfg_entry_point", "rsl_rl_cfg_entry_point"}
    assert all(ast.unparse(node.args[0]) == "args.task" for node in config_calls)
