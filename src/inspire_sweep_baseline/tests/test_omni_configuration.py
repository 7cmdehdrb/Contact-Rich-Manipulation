"""Execute Omni wiring with real configclass semantics, without starting Sim.

Asset configuration nodes stand in for launcher-dependent Isaac Sim types.
The checked-in config classes, manager term cfgs, MDP callables and CLI helper
execute unchanged, including configclass copying and inherited field order.
"""

from _source_units import source_text

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
    """Execute the selected live production definitions with CPU dependencies."""
    from _source_units import execute_definitions
    return execute_definitions(path, namespace, selected_classes)


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
    robot_tree = ast.parse(source_text(MODULE_ROOT / "assets/robot.py"))
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
    dex_module = types.ModuleType("_omni_cpu_dex_cfg")
    dex_module.__dict__.update(
        configclass=configclass, RewTerm=terms.RewardTermCfg,
        dex_rewards=load_mdp("dex_rewards"),
        OmniV2RewardsCfg=omni.OmniV2RewardsCfg,
        InspireSweepOmniV2EnvCfg=omni.InspireSweepOmniV2EnvCfg,
    )
    monkeypatch.setitem(sys.modules, dex_module.__name__, dex_module)
    dex = _execute_definitions(MODULE_ROOT / "dex_omni_env_cfg.py", dex_module.__dict__)
    renewal_module = types.ModuleType("_omni_cpu_renewal_cfg")
    renewal_module.__dict__.update(
        configclass=configclass, math=math, mdp=mdp,
        EventTerm=terms.EventTermCfg, RewTerm=terms.RewardTermCfg,
        DoneTerm=terms.TerminationTermCfg, SceneEntityCfg=scene_module.SceneEntityCfg,
        HAND_JOINT_NAMES=robot_constants["HAND_JOINT_NAMES"],
        InspireSweepEnvCfg=baseline.InspireSweepEnvCfg, TerminationsCfg=baseline.TerminationsCfg, _arm=baseline._arm,
        goal_rewards=omni.goal_rewards, goal_terminations=omni.goal_terminations,
        randomize_goal_object_scene=omni.randomize_goal_object_scene,
        OBJECT_RADIUS=omni.OBJECT_RADIUS, SHELF_HALF_EXTENTS=omni.SHELF_HALF_EXTENTS,
        SHELF_SURFACE_HEIGHT=omni.SHELF_SURFACE_HEIGHT,
        RealisticPhysicsEventsCfg=realistic.RealisticPhysicsEventsCfg,
        apply_realistic_physics=realistic.apply_realistic_physics,
    )
    monkeypatch.setitem(sys.modules, renewal_module.__name__, renewal_module)
    renewal = _execute_definitions(MODULE_ROOT / "omni_renewal_env_cfg.py", renewal_module.__dict__)
    dex_renewal_module = types.ModuleType("_omni_cpu_dex_renewal_cfg")
    dex_renewal_module.__dict__.update(
        configclass=configclass, RewTerm=terms.RewardTermCfg, dex_rewards=dex.dex_rewards,
        InspireSweepOmniRenewalEnvCfg=renewal.InspireSweepOmniRenewalEnvCfg,
        OmniRenewalRewardsCfg=renewal.OmniRenewalRewardsCfg,
    )
    monkeypatch.setitem(sys.modules, dex_renewal_module.__name__, dex_renewal_module)
    dex_renewal = _execute_definitions(MODULE_ROOT / "dex_omni_renewal_env_cfg.py", dex_renewal_module.__dict__)
    return SimpleNamespace(baseline=baseline, realistic=realistic, omni=omni, dex=dex,
                           renewal=renewal, dex_renewal=dex_renewal, terms=terms, mdp=mdp)


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
    assert omni["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniEnvCfg"
    assert omni["kwargs"]["rsl_rl_cfg_entry_point"] == baseline["kwargs"]["rsl_rl_cfg_entry_point"]
    assert omni["kwargs"]["rsl_rl_cfg_entry_point"] == "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg"
    v1 = registrations[namespace["OMNI_V1_TASK_ID"]]
    assert namespace["OMNI_V1_TASK_ID"] == "Isaac-Inspire-Sweep-Omni-v1"
    assert v1["entry_point"] == omni["entry_point"]
    assert v1["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV1EnvCfg"
    assert v1["kwargs"]["rsl_rl_cfg_entry_point"] == omni["kwargs"]["rsl_rl_cfg_entry_point"]
    v2 = registrations[namespace["OMNI_V2_TASK_ID"]]
    assert namespace["OMNI_V2_TASK_ID"] == "Isaac-Inspire-Sweep-Omni-v2"
    assert v2["entry_point"] == v1["entry_point"]
    assert v2["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV2EnvCfg"
    assert v2["kwargs"]["rsl_rl_cfg_entry_point"] == v1["kwargs"]["rsl_rl_cfg_entry_point"]
    v3 = registrations[namespace["OMNI_V3_TASK_ID"]]
    assert namespace["OMNI_V3_TASK_ID"] == "Isaac-Inspire-Sweep-Omni-v3"
    assert v3["entry_point"] == v2["entry_point"]
    assert v3["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV3EnvCfg"
    assert v3["kwargs"]["rsl_rl_cfg_entry_point"] == v2["kwargs"]["rsl_rl_cfg_entry_point"]
    v4 = registrations[namespace["OMNI_V4_TASK_ID"]]
    assert namespace["OMNI_V4_TASK_ID"] == "Isaac-Inspire-Sweep-Omni-v4"
    assert "OMNI_V4_TASK_ID" in namespace["__all__"]
    assert v4["entry_point"] == v3["entry_point"]
    assert v4["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV4EnvCfg"
    assert v4["kwargs"]["rsl_rl_cfg_entry_point"] == v3["kwargs"]["rsl_rl_cfg_entry_point"]
    dex = registrations[namespace["DEX_OMNI_TASK_ID"]]
    assert namespace["DEX_OMNI_TASK_ID"] == "Isaac-Inspire-Dex-Omni-v0"
    assert "DEX_OMNI_TASK_ID" in namespace["__all__"]
    assert dex["entry_point"] == baseline["entry_point"]
    assert dex["disable_env_checker"] == baseline["disable_env_checker"]
    assert dex["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.envs.sweep_omni_envs:InspireDexOmniEnvCfg"
    assert dex["kwargs"]["rsl_rl_cfg_entry_point"] == baseline["kwargs"]["rsl_rl_cfg_entry_point"]
    for constant, task_id, entry in (
        ("OMNI_RENEWAL_TASK_ID", "Isaac-Inspire-Sweep-Omni-Renewal-v0",
         "envs.sweep_omni_renewal_envs:InspireSweepOmniRenewalEnvCfg"),
        ("DEX_OMNI_RENEWAL_TASK_ID", "Isaac-Inspire-Dex-Sweep-Omni-Renewal-v0",
         "envs.sweep_omni_renewal_envs:InspireDexSweepOmniRenewalEnvCfg"),
    ):
        assert namespace[constant] == task_id
        assert constant in namespace["__all__"]
        registered = registrations[task_id]
        assert registered["entry_point"] == baseline["entry_point"]
        assert registered["disable_env_checker"] == baseline["disable_env_checker"]
        assert registered["kwargs"]["env_cfg_entry_point"] == f"inspire_sweep_baseline.{entry}"
        assert registered["kwargs"]["rsl_rl_cfg_entry_point"] == baseline["kwargs"]["rsl_rl_cfg_entry_point"]


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


def test_v2_disables_success_preserves_other_v1_terminations_and_isolates_mutations(omni_config):
    before = omni_config.omni.InspireSweepOmniV1EnvCfg()
    v2 = omni_config.omni.InspireSweepOmniV2EnvCfg()
    assert isinstance(v2, type(before))
    expected_terminations = before.terminations.to_dict()
    expected_terminations["success"] = None
    assert v2.terminations.to_dict() == expected_terminations
    assert v2.terminations.success is None
    assert before.terminations.success is not None
    for name in ("scene", "actions", "commands", "observations", "events"):
        assert getattr(v2, name).to_dict() == getattr(before, name).to_dict()
    assert v2.episode_length_s == before.episode_length_s
    assert v2.sim.dt == before.sim.dt and v2.decimation == before.decimation
    assert all(getattr(v2.rewards, name) is None for name in ("time", "success", "failure"))
    assert v2.rewards.reaching.func.__name__ == "CurrentObjectReachProgress"
    assert v2.rewards.goal_distance.func.__name__ == "PlanarGoalProgress"
    assert v2.rewards.reaching.weight > 0.0
    assert v2.rewards.goal_distance.weight > 0.0
    for name in ("arm_action_rate", "hand_action_rate", "arm_joint_velocity", "hand_joint_velocity"):
        assert getattr(v2.rewards, name) == getattr(before.rewards, name)
    assert v2.rewards.object_tilt == before.rewards.object_tilt
    assert v2.rewards.shelf_collision == before.rewards.shelf_collision
    v2.rewards.reaching.params["near_radius"] = 9.0
    v2.terminations.hand_velocity.params["threshold"] = 99.0
    assert omni_config.omni.InspireSweepOmniV1EnvCfg().to_dict() == before.to_dict()
    assert omni_config.omni.InspireSweepOmniV2EnvCfg().rewards.reaching.params["near_radius"] == 0.10


def test_v2_staged_approach_then_sweep_pays_separate_budgets(omni_config, make_env):
    cfg = omni_config.omni.InspireSweepOmniV2EnvCfg()
    env = make_env(1)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    initial = position.clone()
    reference = env.scene["ee_frame"].data.target_pos_w[:, 0]
    reference[:] = position + torch.tensor([0.6, 0.0, 0.0])
    goal = initial + torch.tensor([0.15, 0.0, 0.0])
    env.command_manager.get_command = lambda name: goal
    configs = [cfg.rewards.reaching, cfg.rewards.goal_distance]
    terms = [term.func(term, env) for term in configs]

    def reward():
        return [(term(env, **config.params) * config.weight * env.step_dt).item()
                for term, config in zip(terms, configs)]

    assert reward() == [0.0, 0.0]
    reference[:] = position
    assert reward() == pytest.approx([cfg.rewards.reaching.weight, 0.0])
    assert reward() == [0.0, 0.0]  # Waiting beside the object does not pay.
    position[:, 0] += 0.07
    reference[:] = position
    assert reward() == pytest.approx([0.0, cfg.rewards.goal_distance.weight / 2], abs=2e-4)
    position[:, 0] += 0.07
    reference[:] = position
    assert reward() == pytest.approx([0.0, cfg.rewards.goal_distance.weight / 2], abs=2e-4)
    assert reward() == [0.0, 0.0]


def test_dex_changes_only_shaping_and_preserves_complete_v2_environment_contract(omni_config):
    v2 = omni_config.omni.InspireSweepOmniV2EnvCfg()
    dex = omni_config.dex.InspireDexOmniEnvCfg()
    assert isinstance(dex, type(v2))
    assert isinstance(dex.rewards, type(v2.rewards))
    expected = v2.to_dict()
    expected["rewards"] = dex.rewards.to_dict()
    assert dex.to_dict() == expected
    # These equalities include object type/geometry, spawn position/ranges,
    # realistic physics, both shelf termination rules, and all four costs.
    for name in ("scene", "events", "actions", "observations", "commands", "terminations"):
        assert getattr(dex, name).to_dict() == getattr(v2, name).to_dict()
    assert dex.sim.dt == v2.sim.dt == 0.01
    assert dex.decimation == v2.decimation == 2
    assert dex.episode_length_s == v2.episode_length_s == 10.0
    assert dex.terminations.success is None
    assert dex.terminations.hand_velocity.params["threshold"] == v2.terminations.hand_velocity.params["threshold"]
    assert dex.terminations.shelf_collision.params["threshold"] == 0.1
    assert dex.terminations.shelf_displacement.params["threshold"] == 0.05
    assert set(vars(dex.rewards)) == set(vars(v2.rewards))
    for name, term in vars(v2.rewards).items():
        if name not in ("reaching", "goal_distance"):
            assert getattr(dex.rewards, name) == term
    assert dex.rewards.reaching.func is omni_config.dex.dex_rewards.DexReachProgress
    assert dex.rewards.reaching.weight == 30.0
    assert dex.rewards.reaching.params == {"near_radius": 0.10, "frame_name": "ee_frame"}
    assert dex.rewards.goal_distance.func is omni_config.dex.dex_rewards.DexPlanarGoalProgress
    assert dex.rewards.goal_distance.weight == 300.0
    assert dex.rewards.goal_distance.params == {"position_tolerance": 0.01}


def test_dex_instances_isolate_reward_spawn_shelf_and_control_mutations(omni_config):
    v2_before = omni_config.omni.InspireSweepOmniV2EnvCfg().to_dict()
    first = omni_config.dex.InspireDexOmniEnvCfg()
    second = omni_config.dex.InspireDexOmniEnvCfg()
    dex_before = second.to_dict()
    first.rewards.reaching.params["near_radius"] = 9.0
    first.rewards.goal_distance.weight = 999.0
    first.rewards.arm_action_rate.weight = -99.0
    first.events.object_spawn.params["spawn_radius"] = 0.2
    first.scene.object_collection.rigid_objects["mass_cylinder"].spawn.radius = 0.2
    first.scene.robot.init_state.joint_pos["elbow_joint"] = -99.0
    first.actions.arm_action.scale = 0.9
    first.terminations.shelf_collision.params["threshold"] = 99.0
    first.terminations.shelf_displacement.params["threshold"] = 99.0
    assert second.to_dict() == dex_before
    assert omni_config.dex.InspireDexOmniEnvCfg().to_dict() == dex_before
    assert omni_config.omni.InspireSweepOmniV2EnvCfg().to_dict() == v2_before


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
    for cfg_class in (
        omni_config.omni.InspireSweepOmniEnvCfg, omni_config.omni.InspireSweepOmniV1EnvCfg,
        omni_config.omni.InspireSweepOmniV2EnvCfg, omni_config.omni.InspireSweepOmniV3EnvCfg,
        omni_config.omni.InspireSweepOmniV4EnvCfg,
        omni_config.renewal.InspireSweepOmniRenewalEnvCfg,
        omni_config.dex_renewal.InspireDexSweepOmniRenewalEnvCfg,
        omni_config.dex.InspireDexOmniEnvCfg,
    ):
        cfg = cfg_class()
        for name, group in (("arm_action_rate", "arm"), ("hand_action_rate", "hand")):
            term = getattr(cfg.rewards, name)
            assert term.func is omni_config.omni.goal_rewards.action_rate_l2
            assert term.params == {"action_group": group}
            assert term.weight == -0.03
        for name, weight in (("arm_joint_velocity", -0.03), ("hand_joint_velocity", -0.01)):
            term = getattr(cfg.rewards, name)
            assert term.func is omni_config.omni.goal_rewards.joint_vel_l2
            assert set(term.params) == {"asset_cfg"}
            assert term.weight == weight
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


@pytest.mark.parametrize("namespace,config_name", [
    ("omni", "InspireSweepOmniEnvCfg"),
    ("omni", "InspireSweepOmniV2EnvCfg"),
    ("omni", "InspireSweepOmniV3EnvCfg"),
    ("omni", "InspireSweepOmniV4EnvCfg"),
    ("dex", "InspireDexOmniEnvCfg"),
    ("renewal", "InspireSweepOmniRenewalEnvCfg"),
    ("dex_renewal", "InspireDexSweepOmniRenewalEnvCfg"),
])
def test_every_configured_manager_parameter_binds_to_its_production_callable(omni_config, namespace, config_name):
    cfg = getattr(getattr(omni_config, namespace), config_name)()
    for group_name in ("events", "rewards", "terminations"):
        for name, term in vars(getattr(cfg, group_name)).items():
            if term is None:
                continue
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


def test_v3_only_changes_reaching_geometry_and_preserves_v2(omni_config):
    v2 = omni_config.omni.InspireSweepOmniV2EnvCfg()
    before = v2.to_dict()
    v3 = omni_config.omni.InspireSweepOmniV3EnvCfg()
    assert isinstance(v3, type(v2))
    reaching = v3.rewards.reaching
    assert reaching.func.__name__ == "HeightGatedPlanarReachProgress"
    assert reaching.weight == v2.rewards.reaching.weight
    assert reaching.params == dict(v2.rewards.reaching.params, z_offset=0.09, z_tolerance=0.02)
    # Replacing just the new Reaching term must recover the entire v2 config.
    v3.rewards.reaching = deepcopy(v2.rewards.reaching)
    assert v3.to_dict() == before
    another_v3 = omni_config.omni.InspireSweepOmniV3EnvCfg()
    another_v3.rewards.reaching.params["near_radius"] = 9.0
    assert v2.to_dict() == before
    assert omni_config.omni.InspireSweepOmniV2EnvCfg().to_dict() == before


def test_v4_preserves_v3_except_speed_cost_and_requested_terminations(omni_config, make_env):
    v3 = omni_config.omni.InspireSweepOmniV3EnvCfg()
    before = v3.to_dict()
    v4 = omni_config.omni.InspireSweepOmniV4EnvCfg()
    assert isinstance(v4, type(v3))
    assert v4.terminations.object_outside_shelf is None
    assert v4.terminations.success is None
    assert v4.terminations.hand_velocity.params["threshold"] == 5.0
    assert v4.terminations.push_fast.params["speed_condition"] == 0.5
    cost = v4.rewards.pushing_speed
    assert cost.func is omni_config.omni.goal_rewards.pushing_speed_cost
    assert cost.weight == -0.05
    env = make_env(3)
    env.scene["object_collection"].data.object_lin_vel_w[:, 0, 0] = torch.tensor([0.3, 0.5, 0.501])
    assert torch.allclose(cost.func(env, **cost.params) * cost.weight,
                          torch.tensor([0.0, -0.05, -0.05050125]))
    term = v4.terminations.push_fast
    assert term.func(env, **term.params).tolist() == [False, False, True]
    # Match the saved run boundary; fast fingers remain outside this Arm term.
    arm_ids = [1, 4, 7, 10, 13, 16]
    env.scene["robot"].data.joint_vel[:, 0] = 100.0
    env.scene["robot"].data.joint_vel[:, 7] = torch.tensor([4.0, 5.0, 5.001])
    term = v4.terminations.hand_velocity
    params = deepcopy(term.params)
    params["asset_cfg"].joint_ids = arm_ids
    assert term.func(env, **params).tolist() == [False, False, True]
    excess = v4.rewards.arm_speed_excess
    assert excess.func is omni_config.omni.goal_rewards.joint_speed_cost
    assert excess.weight == -0.03
    assert excess.params["asset_cfg"].joint_names == term.params["asset_cfg"].joint_names
    params = deepcopy(excess.params)
    params["asset_cfg"].joint_ids = arm_ids
    env.scene["robot"].data.joint_vel[:, 7] = torch.tensor([1.0, 1.5, -2.0])
    assert torch.allclose(excess.func(env, **params) * excess.weight, torch.tensor([0.0, -0.0075, -0.03]))
    # Restoring just the requested differences must recover the complete V3.
    del v4.rewards.pushing_speed
    del v4.rewards.arm_speed_excess
    v4.terminations = deepcopy(v3.terminations)
    assert v4.to_dict() == before
    assert v3.to_dict() == before
    assert omni_config.omni.InspireSweepOmniV3EnvCfg().to_dict() == before
    another_v4 = omni_config.omni.InspireSweepOmniV4EnvCfg()
    another_v4.rewards.pushing_speed.params["free_speed"] = 0.1
    assert omni_config.omni.InspireSweepOmniV4EnvCfg().rewards.pushing_speed.params["free_speed"] == 0.3


def test_renewal_composes_v4_with_exponential_costs_and_current_overrides(omni_config):
    v4 = omni_config.omni.InspireSweepOmniV4EnvCfg()
    renewal = omni_config.renewal.InspireSweepOmniRenewalEnvCfg()
    assert renewal.__class__.__bases__ == (omni_config.baseline.InspireSweepEnvCfg,)
    expected = v4.to_dict()
    shared = omni_config.renewal.OmniRenewalRewardsCfg().to_dict()
    for name in ("pushing_speed", "arm_speed_excess"):
        expected["rewards"][name] = shared[name]
    expected["commands"]["target_goal_pos"]["debug_vis"] = True
    expected["terminations"]["shelf_collision"]["params"]["check_fingers"] = False
    expected["terminations"]["push_fast"]["params"]["speed_condition"] = 0.8
    # Renewal omits these disabled fields rather than declaring None terms.
    for name in ("time", "success", "failure"):
        assert expected["rewards"].pop(name) is None
    assert expected["terminations"].pop("success") is None
    assert renewal.to_dict() == expected
    for group in ("events", "rewards", "terminations"):
        assert [name for name, value in vars(getattr(renewal, group)).items() if value is not None] == [
            name for name, value in vars(getattr(v4, group)).items() if value is not None
        ]
    renewal.rewards.reaching.params["z_offset"] = 99.0
    renewal.terminations.hand_velocity.params["threshold"] = 99.0
    renewal.events.object_spawn.params["spawn_radius"] = 99.0
    assert omni_config.renewal.InspireSweepOmniRenewalEnvCfg().to_dict() == expected


def test_dex_renewal_changes_distance_rewards_and_inherits_exponential_speed_costs(omni_config):
    renewal = omni_config.renewal.InspireSweepOmniRenewalEnvCfg()
    before = renewal.to_dict()
    dex_renewal = omni_config.dex_renewal.InspireDexSweepOmniRenewalEnvCfg()
    assert isinstance(dex_renewal, type(renewal))
    old_dex = omni_config.dex.InspireDexOmniEnvCfg()
    for name in ("reaching", "goal_distance"):
        assert getattr(dex_renewal.rewards, name).to_dict() == getattr(old_dex.rewards, name).to_dict()
        setattr(dex_renewal.rewards, name, deepcopy(getattr(renewal.rewards, name)))
    shared = omni_config.renewal.OmniRenewalRewardsCfg()
    for name in ("pushing_speed", "arm_speed_excess"):
        assert getattr(dex_renewal.rewards, name).to_dict() == getattr(shared, name).to_dict()
        setattr(dex_renewal.rewards, name, deepcopy(getattr(renewal.rewards, name)))
    assert dex_renewal.to_dict() == before
    assert renewal.to_dict() == before
    another = omni_config.dex_renewal.InspireDexSweepOmniRenewalEnvCfg()
    another.rewards.arm_speed_excess.weight = -99.0
    another.rewards.reaching.params["near_radius"] = 99.0
    another.terminations.hand_velocity.params["threshold"] = 99.0
    fresh = omni_config.dex_renewal.InspireDexSweepOmniRenewalEnvCfg()
    assert fresh.rewards.arm_speed_excess.weight == -1.0
    assert fresh.rewards.reaching.params["near_radius"] == 0.10
    assert fresh.terminations.hand_velocity.params["threshold"] == 5.0
    assert omni_config.renewal.InspireSweepOmniRenewalEnvCfg().to_dict() == before


def test_renewal_osc_changes_arm_control_and_speed_costs_and_matches_eef(omni_config):
    configclass = omni_config.baseline.configclass
    controllers = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/controllers/operational_space_cfg.py",
        dict(configclass=configclass, MISSING=MISSING, OperationalSpaceController=object),
        {"OperationalSpaceControllerCfg"},
    )

    @configclass
    class ActionTermCfg:
        asset_name: str = "robot"

    actions = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py",
        dict(configclass=configclass, MISSING=MISSING, ActionTermCfg=ActionTermCfg,
             task_space_actions=SimpleNamespace(OperationalSpaceControllerAction=object)),
        {"OperationalSpaceControllerActionCfg"},
    )
    module = _execute_definitions(
        MODULE_ROOT / "omni_renewal_osc_env_cfg.py",
        dict(configclass=configclass,
             OperationalSpaceControllerCfg=controllers.OperationalSpaceControllerCfg,
             OperationalSpaceControllerActionCfg=actions.OperationalSpaceControllerActionCfg,
             ARM_JOINT_NAMES=omni_config.baseline.ARM_JOINT_NAMES,
             OmniRenewalRewardsCfg=omni_config.renewal.OmniRenewalRewardsCfg,
             InspireSweepOmniRenewalEnvCfg=omni_config.renewal.InspireSweepOmniRenewalEnvCfg),
    )
    parent = omni_config.renewal.InspireSweepOmniRenewalEnvCfg()
    child = module.InspireSweepOmniRenewalOSCEnvCfg()
    action = child.actions.arm_action
    eef = child.scene.ee_frame.target_frames[0]
    assert action.body_name == "axia80_link"
    assert action.body_offset.pos == eef.offset.pos
    assert action.body_offset.rot == eef.offset.rot
    assert action.joint_names == list(omni_config.baseline.ARM_JOINT_NAMES)
    assert action.controller_cfg.target_types == ["pose_rel"]
    assert action.controller_cfg.impedance_mode == "fixed"
    assert action.controller_cfg.motion_stiffness_task == 200.0
    assert action.controller_cfg.motion_damping_ratio_task == 1.0
    assert action.controller_cfg.gravity_compensation
    assert action.controller_cfg.inertial_dynamics_decoupling
    assert action.position_scale == 1.0
    assert action.orientation_scale == 1.0
    assert child.scene.robot.actuators["arm"].stiffness == 0.0
    assert child.scene.robot.actuators["arm"].damping == 0.0
    assert child.terminations.shelf_collision.params["check_fingers"] is False
    assert child.commands.target_goal_pos.debug_vis is True
    assert parent.terminations.shelf_collision.params["check_fingers"] is False
    assert parent.commands.target_goal_pos.debug_vis is True
    shared = omni_config.renewal.OmniRenewalRewardsCfg()
    for name in ("pushing_speed", "arm_speed_excess"):
        assert getattr(child.rewards, name).to_dict() == getattr(shared, name).to_dict()
        setattr(child.rewards, name, deepcopy(getattr(parent.rewards, name)))
    # Undo the requested overrides: every other task setting must match.
    child.actions.arm_action = deepcopy(parent.actions.arm_action)
    child.scene.robot.actuators["arm"] = deepcopy(parent.scene.robot.actuators["arm"])
    assert child.to_dict() == parent.to_dict()
    assert omni_config.renewal.InspireSweepOmniRenewalEnvCfg().to_dict() == parent.to_dict()

    @configclass
    class OSCDescendant(module.InspireSweepOmniRenewalOSCEnvCfg):
        pass

    descendant = OSCDescendant()
    for name in ("pushing_speed", "arm_speed_excess"):
        assert getattr(descendant.rewards, name).to_dict() == getattr(shared, name).to_dict()
    descendant.rewards.pushing_speed.weight = -99.0
    assert module.InspireSweepOmniRenewalOSCEnvCfg().rewards.pushing_speed.weight == -1.0


def test_renewal_osc_registration():
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    child = registrations["Isaac-Inspire-Sweep-Omni-Renewal-OSC-v0"]
    parent = registrations["Isaac-Inspire-Sweep-Omni-Renewal-v0"]
    assert child["entry_point"] == parent["entry_point"]
    assert child["kwargs"]["rsl_rl_cfg_entry_point"] == parent["kwargs"]["rsl_rl_cfg_entry_point"]
    assert child["kwargs"]["env_cfg_entry_point"] == (
        "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireSweepOmniRenewalOSCEnvCfg"
    )


def test_renewal_family_inherits_wrist_only_shelf_check_and_visible_goal(omni_config):
    for cls in (omni_config.renewal.InspireSweepOmniRenewalEnvCfg,
                omni_config.dex_renewal.InspireDexSweepOmniRenewalEnvCfg):
        cfg = cls()
        assert cfg.terminations.shelf_collision.params == {"threshold": 0.1, "check_fingers": False}
        assert cfg.commands.target_goal_pos.debug_vis is True
    baseline = omni_config.baseline.InspireSweepEnvCfg()
    assert baseline.terminations.shelf_collision.params == {"threshold": 0.1}
    assert baseline.commands.target_goal_pos.debug_vis is False
