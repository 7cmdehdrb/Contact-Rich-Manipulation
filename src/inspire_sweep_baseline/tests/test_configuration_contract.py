"""Wiring, independence, and numerical PPO parity without a GPU launcher."""

import ast
from pathlib import Path

from conftest import MODULE_ROOT, PACKAGE_ROOT, PROJECT_ROOT


def _tree(path):
    return ast.parse(Path(path).read_text())


def _class(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def _assignments(class_node):
    result = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            result[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign):
            result[node.target.id] = node.value
    return result


def _keywords(call):
    return {keyword.arg: keyword.value for keyword in call.keywords}


def test_production_imports_do_not_depend_on_existing_project_packages():
    forbidden = {"sweeping_policy", "sweep_inspire_rl", "hand_manipulation_rl", "hand_manipulation_test"}
    for path in [*MODULE_ROOT.rglob("*.py"), *(PACKAGE_ROOT / "scripts").glob("*.py")]:
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                assert not roots & forbidden, path
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in forbidden, path
        source = path.read_text()
        assert "/home/min/7cmdehdrb/grad" not in source, path
        assert "example/Sweep-Policy" not in source, path


def test_ppo_numeric_configuration_matches_source_v2_mass():
    baseline = _class(_tree(PROJECT_ROOT / "example/Sweep-Policy/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py"), "UR5eSweepPPORunnerCfg")
    actual = _class(_tree(MODULE_ROOT / "agents/rsl_rl_ppo_cfg.py"), "InspireSweepPPORunnerCfg")
    source_values, actual_values = _assignments(baseline), _assignments(actual)
    for name in ("num_steps_per_env", "max_iterations", "save_interval"):
        assert ast.literal_eval(actual_values[name]) == ast.literal_eval(source_values[name])
    for name in ("policy", "algorithm"):
        source_kwargs, actual_kwargs = _keywords(source_values[name]), _keywords(actual_values[name])
        assert {key: ast.literal_eval(value) for key, value in actual_kwargs.items()} == {
            key: ast.literal_eval(value) for key, value in source_kwargs.items()
        }
    assert ast.literal_eval(actual_values["obs_groups"]) == {"actor": ["policy"], "critic": ["policy"]}


def test_reward_weights_match_original_sweep_and_sensor_channels_are_observation_only():
    baseline = _class(_tree(PROJECT_ROOT / "example/Sweep-Policy/sweeping_policy/shelf_sweep_random_env_cfg.py"), "RewardsCfg")
    actual = _class(_tree(MODULE_ROOT / "env_cfg.py"), "RewardsCfg")
    source_weights = {
        name: ast.literal_eval(_keywords(value)["weight"])
        for name, value in _assignments(baseline).items() if isinstance(value, ast.Call)
    }
    actual_weights = {
        name: ast.literal_eval(_keywords(value)["weight"])
        for name, value in _assignments(actual).items() if isinstance(value, ast.Call)
    }
    assert actual_weights == source_weights
    reward_source = (MODULE_ROOT / "mdp/rewards.py").read_text()
    assert "palm_tactile" not in reward_source
    assert "body_incoming_joint_wrench" not in reward_source


def test_configuration_connects_nine_policy_terms_and_both_height_terms():
    tree = _tree(MODULE_ROOT / "env_cfg.py")
    policy = _assignments(_class(tree, "PolicyCfg"))
    assert [_keywords(value)["func"].attr for value in policy.values()] == [
        "arm_joint_pos_rel", "arm_joint_vel_rel", "last_action", "object_position_in_root",
        "object_width", "eef_pose_in_root", "goal_position_in_root", "palm_contact_bit", "wrist_wrench_eef",
    ]
    cfg = _class(tree, "ObservationsCfg")
    assert set(_assignments(cfg)) == {"policy"}
    rewards = _assignments(_class(tree, "RewardsCfg"))
    for name in ("reaching", "sweeping_object"):
        params = _keywords(rewards[name])["params"]
        param_values = dict(zip((ast.literal_eval(key) for key in params.keys), params.values))
        assert isinstance(param_values["z_offset"], ast.Name)
        assert param_values["z_offset"].id == "TARGET_Z_OFFSET"
    constants = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)}
    assert ast.literal_eval(constants["TARGET_Z_OFFSET"]) == 0.09
    arm = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_arm")
    arm_keywords = _keywords(arm.body[0].value)
    assert ast.literal_eval(arm_keywords["preserve_order"]) is True
    assert arm_keywords["joint_names"].args[0].id == "ARM_JOINT_NAMES"
    velocity = _assignments(_class(tree, "TerminationsCfg"))["hand_velocity"]
    params_node = _keywords(velocity)["params"]
    params = {ast.literal_eval(key): value for key, value in zip(params_node.keys, params_node.values)}
    assert ast.literal_eval(params["threshold"]) == 1.0
    assert params["asset_cfg"].func.id == "_arm"
    actions = _assignments(_class(tree, "ActionsCfg"))
    assert set(actions) == {"arm_action", "gripper_action"}
    assert actions["arm_action"].func.attr == "JointPositionActionCfg"
    assert _keywords(actions["arm_action"])["joint_names"].args[0].id == "ARM_JOINT_NAMES"
    assert actions["gripper_action"].func.attr == "HandOpennessActionCfg"
