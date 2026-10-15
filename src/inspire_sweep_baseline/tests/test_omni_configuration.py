"""Execute Omni wiring with real configclass semantics, without starting Sim.

Asset configuration nodes stand in for launcher-dependent Isaac Sim types.
The checked-in config classes, manager term cfgs, MDP callables and CLI helper
execute unchanged, including configclass copying and inherited field order.
"""

import ast
from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING
import importlib.util
import inspect
import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PACKAGE_ROOT, PROJECT_ROOT


class _AssetNode(SimpleNamespace):
    def copy(self):
        return deepcopy(self)

    def replace(self, **kwargs):
        result = self.copy()
        result.__dict__.update(kwargs)
        return result


_AssetNode.InitialStateCfg = _AssetNode
_AssetNode.FrameCfg = _AssetNode


def _execute_definitions(path, namespace, selected_classes=None):
    """Retain production bodies while supplying only their import dependencies."""
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    if selected_classes is not None:
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in selected_classes]
        tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def omni_config(load_mdp, monkeypatch):
    # Import the official decorator and its CPU-only dependencies by file path.
    utils_path = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/utils"
    package = types.ModuleType("_omni_official_utils")
    package.__path__ = [str(utils_path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.configclass", utils_path / "configclass.py")
    official = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, official)
    spec.loader.exec_module(official)
    configclass = official.configclass
    monkeypatch.setattr(sys.modules["isaaclab.utils"], "configclass", configclass)

    scene_path = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/managers/scene_entity_cfg.py"
    scene_spec = importlib.util.spec_from_file_location("_omni_official_scene_entity", scene_path)
    scene_module = importlib.util.module_from_spec(scene_spec)
    monkeypatch.setitem(sys.modules, scene_spec.name, scene_module)
    scene_spec.loader.exec_module(scene_module)
    monkeypatch.setattr(sys.modules["isaaclab.managers"], "SceneEntityCfg", scene_module.SceneEntityCfg)

    module = types.ModuleType("_omni_official_manager_terms")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(configclass=configclass, SceneEntityCfg=scene_module.SceneEntityCfg,
                           torch=torch, Callable=Callable, MISSING=MISSING)
    terms = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py", module.__dict__,
        {"ManagerTermBaseCfg", "EventTermCfg", "RewardTermCfg", "TerminationTermCfg", "ObservationTermCfg", "ObservationGroupCfg"},
    )

    mdp = SimpleNamespace(JointPositionActionCfg=_AssetNode)
    for name in ("actions", "commands", "events", "observations", "rewards", "terminations"):
        mdp.__dict__.update(vars(load_mdp(name)))

    def reset_scene_to_default(env, env_ids):
        pass

    def time_out(env):
        pass

    mdp.reset_scene_to_default = reset_scene_to_default
    mdp.time_out = time_out
    robot_constants = vars(sys.modules["_baseline_cpu_pkg.assets.robot"])
    robot_namespace = dict(robot_constants, ArticulationCfg=_AssetNode, ImplicitActuatorCfg=_AssetNode,
                           Ur5eAxia80InspireSpawnerCfg=lambda: _AssetNode(sensor_height=0.02))
    robot_tree = ast.parse((MODULE_ROOT / "assets/robot.py").read_text())
    robot_tree.body = [node for node in robot_tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_robot_cfg"]
    exec(compile(robot_tree, str(MODULE_ROOT / "assets/robot.py"), "exec"), robot_namespace)

    @configclass
    class SceneBase:
        num_envs: int = 1
        env_spacing: float = 1.0
        replicate_physics: bool = True

    @configclass
    class EnvBase:
        sim: object = _AssetNode(dt=0.01, render_interval=2, physx=_AssetNode(), physics_material=_AssetNode())

    def mass_cylinder_cfg(**kwargs):
        return _AssetNode(**({"radius": 0.04, "height": 0.10, "mass_props": _AssetNode(mass=1.0)} | kwargs))

    sim = SimpleNamespace(**{name: _AssetNode for name in (
        "GroundPlaneCfg", "DomeLightCfg", "UsdFileCfg", "MassPropertiesCfg", "RigidBodyPropertiesCfg",
        "CollisionPropertiesCfg", "PreviewSurfaceCfg",
    )})
    baseline_namespace = dict(
        __name__="_omni_cpu_baseline_cfg", configclass=configclass, mdp=mdp, sim_utils=sim,
        SceneEntityCfg=scene_module.SceneEntityCfg, EventTerm=terms.EventTermCfg,
        RewTerm=terms.RewardTermCfg, DoneTerm=terms.TerminationTermCfg,
        ObsTerm=terms.ObservationTermCfg, ObsGroup=terms.ObservationGroupCfg,
        InteractiveSceneCfg=SceneBase, ManagerBasedRLEnvCfg=EnvBase,
        AssetBaseCfg=_AssetNode, RigidObjectCfg=_AssetNode, RigidObjectCollectionCfg=_AssetNode,
        ViewerCfg=_AssetNode, FrameTransformerCfg=_AssetNode, ContactSensorCfg=_AssetNode,
        OffsetCfg=_AssetNode, Unoise=_AssetNode, MassCylinderCfg=mass_cylinder_cfg,
        FRAME_MARKER_CFG=_AssetNode(markers={"frame": _AssetNode(scale=(1.0, 1.0, 1.0))}),
        SHELF_USD_PATH="test-shelf.usd", make_robot_cfg=robot_namespace["make_robot_cfg"],
        ARM_JOINT_NAMES=robot_constants["ARM_JOINT_NAMES"],
    )
    baseline_module = types.ModuleType(baseline_namespace["__name__"])
    baseline_module.__dict__.update(baseline_namespace)
    monkeypatch.setitem(sys.modules, baseline_module.__name__, baseline_module)
    baseline = _execute_definitions(MODULE_ROOT / "env_cfg.py", baseline_module.__dict__)

    class RealisticRobotSpawnerCfg(_AssetNode):
        pass

    class RealisticShelfSpawnerCfg(_AssetNode):
        pass

    class RealisticCylinderSpawnerCfg(_AssetNode):
        pass

    def replace_friction_spawner(original, spawner_type, defaults):
        return spawner_type(**deepcopy(vars(original)), friction_defaults=defaults.copy())

    physics = load_mdp("realistic_physics")
    physics_module = types.ModuleType("_omni_cpu_realistic_cfg")
    physics_module.__dict__.update(
        configclass=configclass, sim_utils=sim, EventTerm=terms.EventTermCfg,
        EventsCfg=baseline.EventsCfg, InspireSweepEnvCfg=baseline.InspireSweepEnvCfg,
        MASS_MEAN=physics.MASS_MEAN, initialize_realistic_materials=physics.initialize_realistic_materials,
        randomize_episode_physics=physics.randomize_episode_physics,
        RealisticRobotSpawnerCfg=RealisticRobotSpawnerCfg, RealisticShelfSpawnerCfg=RealisticShelfSpawnerCfg,
        RealisticCylinderSpawnerCfg=RealisticCylinderSpawnerCfg, replace_friction_spawner=replace_friction_spawner,
    )
    monkeypatch.setitem(sys.modules, physics_module.__name__, physics_module)
    realistic = _execute_definitions(MODULE_ROOT / "realistic_physics_env_cfg.py", physics_module.__dict__)
    omni_module = types.ModuleType("_omni_cpu_omni_cfg")
    omni_module.__dict__.update(
        configclass=configclass, math=math, mdp=mdp, EventTerm=terms.EventTermCfg,
        RewTerm=terms.RewardTermCfg, DoneTerm=terms.TerminationTermCfg,
        SceneEntityCfg=scene_module.SceneEntityCfg, HAND_JOINT_NAMES=robot_constants["HAND_JOINT_NAMES"],
        InspireSweepEnvCfg=baseline.InspireSweepEnvCfg, TerminationsCfg=baseline.TerminationsCfg, _arm=baseline._arm,
        goal_rewards=load_mdp("goal_rewards"), goal_terminations=load_mdp("goal_terminations"),
        randomize_goal_object_scene=load_mdp("goal_events").randomize_goal_object_scene,
        RealisticPhysicsEventsCfg=realistic.RealisticPhysicsEventsCfg,
        apply_realistic_physics=realistic.apply_realistic_physics,
    )
    monkeypatch.setitem(sys.modules, omni_module.__name__, omni_module)
    omni = _execute_definitions(MODULE_ROOT / "omni_env_cfg.py", omni_module.__dict__)
    return SimpleNamespace(baseline=baseline, realistic=realistic, omni=omni, terms=terms, mdp=mdp)


def test_new_task_registration_uses_baseline_environment_and_exact_baseline_ppo():
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    namespace = {"gym": gym}
    _execute_definitions(MODULE_ROOT / "__init__.py", namespace)
    baseline = registrations[namespace["TASK_ID"]]
    omni = registrations[namespace["OMNI_TASK_ID"]]
    assert namespace["OMNI_TASK_ID"] == "Isaac-Inspire-Sweep-Omni-v0"
    assert omni["entry_point"] == baseline["entry_point"]
    assert omni["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.omni_env_cfg:InspireSweepOmniEnvCfg"
    assert omni["kwargs"]["rsl_rl_cfg_entry_point"] == baseline["kwargs"]["rsl_rl_cfg_entry_point"]
    assert omni["kwargs"]["rsl_rl_cfg_entry_point"] == "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg"
    v1 = registrations[namespace["OMNI_V1_TASK_ID"]]
    assert namespace["OMNI_V1_TASK_ID"] == "Isaac-Inspire-Sweep-Omni-v1"
    assert v1["entry_point"] == omni["entry_point"]
    assert v1["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.omni_env_cfg:InspireSweepOmniV1EnvCfg"
    assert v1["kwargs"]["rsl_rl_cfg_entry_point"] == omni["kwargs"]["rsl_rl_cfg_entry_point"]


def test_v1_velocity_boundary_and_config_isolation(omni_config, make_env):
    v0 = omni_config.omni.InspireSweepOmniEnvCfg()
    v1 = omni_config.omni.InspireSweepOmniV1EnvCfg()
    another_v0 = omni_config.omni.InspireSweepOmniEnvCfg()
    assert isinstance(v1, omni_config.omni.InspireSweepOmniEnvCfg)
    assert v0.terminations.hand_velocity.params["threshold"] == 1.0
    assert another_v0.terminations.hand_velocity.params["threshold"] == 1.0
    threshold = v1.terminations.hand_velocity.params["threshold"]
    assert threshold > v0.terminations.hand_velocity.params["threshold"]
    assert v1.terminations.hand_velocity.params is not v0.terminations.hand_velocity.params
    # Evaluate the configured production callable with resolved, non-contiguous
    # arm indices. Fast fingers must not trigger either version's arm cutoff.
    env = make_env(4)
    arm_ids = [1, 4, 7, 10, 13, 16]
    env.scene["robot"].data.joint_vel[:, 0] = threshold * 10.0
    env.scene["robot"].data.joint_vel[:, 7] = torch.tensor([
        (1.0 + threshold) / 2.0, threshold, threshold + 0.001, -threshold - 0.001,
    ])
    for cfg, expected in ((v0, [True] * 4), (v1, [False, False, True, True])):
        term = cfg.terminations.hand_velocity
        params = deepcopy(term.params)
        params["asset_cfg"].joint_ids = arm_ids
        assert term.func(env, **params).tolist() == expected


def test_omni_arm_costs_match_baseline_and_exclude_hand(omni_config, make_env):
    baseline = omni_config.baseline.InspireSweepEnvCfg()
    env = make_env(2)
    arm_ids = [1, 4, 7, 10, 13, 16]
    env.scene["robot"].data.joint_vel[:, arm_ids] = torch.tensor([
        [2.0, -1.0, 0.0, 0.5, -3.0, 1.5], [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
    ])
    env.action_manager.action[:, :6] = torch.tensor([
        [1.0, -2.0, 0.5, 0.0, 3.0, -1.0], [0.2, 0.3, -0.5, 0.1, -0.4, 0.6],
    ])
    env.action_manager.prev_action[:, :6] = 0.25
    velocity_params = deepcopy(baseline.rewards.joint_vel.params)
    velocity_params["asset_cfg"].joint_ids = arm_ids
    expected_velocity = baseline.rewards.joint_vel.func(env, **velocity_params) * baseline.rewards.joint_vel.weight
    # Baseline sums all action channels: with Hand held fixed it is exactly
    # the requested Arm component, independent of how Hand costs are defined.
    expected_action = baseline.rewards.action_rate.func(env) * baseline.rewards.action_rate.weight
    env.scene["robot"].data.joint_vel[:, 0] = 1000.0
    env.action_manager.action[:, 6:] = 100.0
    for cfg_class in (omni_config.omni.InspireSweepOmniEnvCfg, omni_config.omni.InspireSweepOmniV1EnvCfg):
        cfg = cfg_class()
        term = cfg.rewards.arm_joint_velocity
        params = deepcopy(term.params)
        params["asset_cfg"].joint_ids = arm_ids
        assert torch.allclose(term.func(env, **params) * term.weight, expected_velocity)
        term = cfg.rewards.arm_action_rate
        assert torch.allclose(term.func(env, **term.params) * term.weight, expected_action)


def test_omni_reward_set_has_complete_plan_and_no_old_shaping(omni_config):
    cfg = omni_config.omni.InspireSweepOmniEnvCfg()
    rewards = vars(cfg.rewards)
    assert set(rewards) == {
        "goal_distance", "reaching", "arm_action_rate", "hand_action_rate", "arm_joint_velocity",
        "hand_joint_velocity", "object_tilt", "shelf_collision", "time", "success", "failure",
    }
    old_functions = {omni_config.mdp.pushing_target, omni_config.mdp.homing_reward,
                     omni_config.mdp.reaching_target, omni_config.mdp.align_eef_y_with_shelf_z}
    assert not {term.func for term in rewards.values()} & old_functions
    assert rewards["shelf_collision"].func is omni_config.mdp.ShelfCollisionPenalty
    assert rewards["arm_action_rate"].params["action_group"] == "arm"
    assert rewards["hand_action_rate"].params["action_group"] == "hand"
    arm = rewards["arm_joint_velocity"].params["asset_cfg"].joint_names
    hand = rewards["hand_joint_velocity"].params["asset_cfg"].joint_names
    assert len(arm) == 6 and len(hand) == len(set(hand)) == 12
    assert not set(arm) & set(hand)


def test_every_configured_manager_parameter_binds_to_its_production_callable(omni_config):
    cfg = omni_config.omni.InspireSweepOmniEnvCfg()
    for group_name in ("events", "rewards", "terminations"):
        for name, term in vars(getattr(cfg, group_name)).items():
            if inspect.isclass(term.func):
                signature, positional = inspect.signature(term.func.__call__), (None, None)
            else:
                signature = inspect.signature(term.func)
                positional = (None, None) if group_name == "events" else (None,)
            try:
                signature.bind(*positional, **term.params)
            except TypeError as error:
                pytest.fail(f"{group_name}.{name}: {error}")
    spawn = cfg.events.object_spawn
    assert spawn.mode == "reset"
    assert spawn.params["spawn_radius"] == 0.02 and spawn.params["goal_distance"] == 0.15
    assert "sweep_direction" not in spawn.params
    assert cfg.events.physics_materials.mode == "startup"
    assert cfg.events.physics_randomization.mode == "reset"


def test_instantiated_success_follows_all_failures_and_reward_uses_the_same_failure_union(omni_config):
    cfg = omni_config.omni.InspireSweepOmniEnvCfg()
    names = list(vars(cfg.terminations))
    success = cfg.terminations.success
    failure_names = success.params["failure_terms"]
    assert set(failure_names) == {name for name in names if name != "success" and not getattr(cfg.terminations, name).time_out}
    assert all(names.index(name) < names.index("success") for name in failure_names)
    assert cfg.rewards.failure.params["failure_terms"] == failure_names
    for name in ("push_fast", "shelf_collision", "hand_velocity"):
        assert getattr(cfg.terminations, name) == getattr(omni_config.baseline.TerminationsCfg(), name)


def test_omni_inherits_scene_control_live_observations_and_timing(omni_config, make_env):
    baseline = omni_config.baseline.InspireSweepEnvCfg()
    cfg = omni_config.omni.InspireSweepOmniEnvCfg()
    assert isinstance(cfg, type(baseline))
    assert type(cfg.scene) is type(baseline.scene)
    assert cfg.actions.to_dict() == baseline.actions.to_dict()
    assert cfg.observations.to_dict() == baseline.observations.to_dict()
    assert cfg.commands.to_dict() == baseline.commands.to_dict()
    assert cfg.scene.robot.init_state == baseline.scene.robot.init_state
    assert cfg.scene.robot.actuators == baseline.scene.robot.actuators
    assert cfg.scene.shelf.init_state == baseline.scene.shelf.init_state
    assert cfg.scene.replicate_physics == baseline.scene.replicate_physics
    assert cfg.decimation == baseline.decimation == 2
    assert cfg.sim.dt == baseline.sim.dt == 0.01
    assert cfg.episode_length_s == baseline.episode_length_s == 10.0

    env = make_env(2)
    env.sensor_data_fresh.zero_()  # Exercise sensor output shapes without physical handles.
    columns = []
    for term in vars(cfg.observations.policy).values():
        if not isinstance(term, omni_config.terms.ObservationTermCfg):
            continue
        params = deepcopy(term.params)
        if "asset_cfg" in params:
            params["asset_cfg"].joint_ids = list(range(6))
        columns.append(term.func(env, **params))
    assert torch.cat(columns, dim=-1).shape == (2, 41)
    target = cfg.observations.policy.target_obs_state
    before = target.func(env, **target.params).clone()
    env.scene["object_collection"].data.object_pos_w[1, 0, 1] += 0.03
    after = target.func(env, **target.params)
    assert after[1, 1] - before[1, 1] == pytest.approx(0.03)


def test_instances_isolate_physics_reward_and_event_mutations_from_baseline(omni_config):
    baseline_before = omni_config.baseline.InspireSweepEnvCfg()
    first = omni_config.omni.InspireSweepOmniEnvCfg()
    second = omni_config.omni.InspireSweepOmniEnvCfg()
    assert first.scene.object_collection.rigid_objects["mass_cylinder"].spawn.mass_props.mass == 0.365
    first.events.object_spawn.params["spawn_radius"] = 0.09
    first.rewards.success.weight = 99.0
    first.scene.robot.init_state.joint_pos["elbow_joint"] = -99.0
    first.scene.object_collection.rigid_objects["mass_cylinder"].spawn.mass_props.mass = 3.0
    assert second.events.object_spawn.params["spawn_radius"] == 0.02
    assert second.rewards.success.weight == 30.0
    assert second.scene.robot.init_state.joint_pos["elbow_joint"] == 2.2
    assert second.scene.object_collection.rigid_objects["mass_cylinder"].spawn.mass_props.mass == 0.365
    baseline_after = omni_config.baseline.InspireSweepEnvCfg()
    assert baseline_after.to_dict() == baseline_before.to_dict()
    assert baseline_after.scene.object_collection.rigid_objects["mass_cylinder"].spawn.mass_props.mass == 1.0
    assert baseline_after.events.object_spawn.params["sweep_direction"] == "both"


def test_direction_cli_preserves_legacy_and_rejects_omni_without_mutation(omni_config):
    spec = importlib.util.spec_from_file_location("_omni_cli_args", PACKAGE_ROOT / "scripts/cli_args.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    baseline = omni_config.baseline.InspireSweepEnvCfg()
    cli.apply_sweep_direction(baseline, "left")
    assert baseline.events.object_spawn.params["sweep_direction"] == "left"
    cli.apply_sweep_direction(baseline, None)
    assert baseline.events.object_spawn.params["sweep_direction"] == "left"
    omni = omni_config.omni.InspireSweepOmniEnvCfg()
    before = deepcopy(omni.events.object_spawn.params)
    cli.apply_sweep_direction(omni, None)
    with pytest.raises(ValueError, match="omit --sweep-direction"):
        cli.apply_sweep_direction(omni, "right")
    assert omni.events.object_spawn.params == before
