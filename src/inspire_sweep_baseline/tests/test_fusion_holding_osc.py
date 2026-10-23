"""Holding OSC base-frame targets, direct inheritance and model isolation."""

import ast
from copy import deepcopy
from dataclasses import MISSING
import math
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_omni_configuration import _execute_definitions, omni_config
from test_fusion_holding_v2 import cfg, curricula


@pytest.fixture
def osc(omni_config, tensor_math):
    configclass = omni_config.baseline.configclass
    controller = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/controllers/operational_space.py",
        dict(vars(tensor_math), torch=torch), {"OperationalSpaceController"},
    ).OperationalSpaceController
    controller_cfg = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/controllers/operational_space_cfg.py",
        dict(configclass=configclass, MISSING=MISSING, OperationalSpaceController=controller),
        {"OperationalSpaceControllerCfg"},
    ).OperationalSpaceControllerCfg
    # Retain upstream processing and frame selection; PhysX construction is
    # covered by the separate simulation smoke check.
    path = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py"
    tree = ast.parse(path.read_text())
    action = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "OperationalSpaceControllerAction")
    action.bases = []
    action.body = [node for node in action.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("process_actions", "_preprocess_actions", "_compute_task_frame_pose")]
    ns = dict(torch=torch, math_utils=tensor_math)
    exec(compile(ast.Module(body=[action], type_ignores=[]), str(path), "exec"), ns)
    action = ns["OperationalSpaceControllerAction"]

    @configclass
    class ActionTermCfg:
        asset_name: str = "robot"

    action_cfg = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py",
        dict(configclass=configclass, MISSING=MISSING, ActionTermCfg=ActionTermCfg,
             task_space_actions=SimpleNamespace(OperationalSpaceControllerAction=action)),
        {"OperationalSpaceControllerActionCfg"},
    ).OperationalSpaceControllerActionCfg
    parent = omni_config.baseline.InspireSweepEnvCfg
    v0 = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_osc_v0/env_cfg.py",
        dict(configclass=configclass, FusionHoldingV0EnvCfg=parent,
             ARM_JOINT_NAMES=omni_config.baseline.ARM_JOINT_NAMES,
             OperationalSpaceControllerActionCfg=action_cfg,
             OperationalSpaceControllerCfg=controller_cfg),
    )
    v1_parent = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_v1/env_cfg.py",
        dict(configclass=configclass, FusionHoldingV0EnvCfg=parent),
    ).FusionHoldingV1EnvCfg
    v1 = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_osc_v1/env_cfg.py",
        dict(configclass=configclass, FusionHoldingV1EnvCfg=v1_parent,
             configure_robot_base_osc=v0.configure_robot_base_osc),
    )
    return SimpleNamespace(
        action=action, controller=controller, parent=parent, v1_parent=v1_parent,
        v0=v0.FusionHoldingOSCV0EnvCfg, v1=v1.FusionHoldingOSCV1EnvCfg,
    )


@pytest.mark.parametrize("version", ["v0", "v1"])
def test_osc_preserves_parent_managers_hand_and_body_offset(osc, version):
    cls = getattr(osc, version)
    parent_cls = osc.parent if version == "v0" else osc.v1_parent
    assert cls.__bases__ == (parent_cls,)
    child, parent = cls(), parent_cls()
    snapshot = parent.to_dict()
    arm = child.actions.arm_action
    assert arm.class_type is osc.action
    assert arm.task_frame_rel_path is None
    assert arm.position_scale == arm.orientation_scale == 1.0
    assert arm.controller_cfg.motion_stiffness_task == 100.0
    assert arm.controller_cfg.motion_damping_ratio_task == 1.0
    assert arm.controller_cfg.target_types == ["pose_rel"]
    assert arm.controller_cfg.impedance_mode == "fixed"
    assert arm.controller_cfg.gravity_compensation
    assert arm.controller_cfg.inertial_dynamics_decoupling
    eef = child.scene.ee_frame.target_frames[0]
    assert arm.body_name == eef.prim_path.rsplit("/", 1)[-1]
    assert arm.body_offset.pos == eef.offset.pos
    assert arm.body_offset.rot == eef.offset.rot
    assert child.scene.robot.actuators["arm"].stiffness == 0.0
    assert child.scene.robot.actuators["arm"].damping == 0.0
    child.actions.arm_action = deepcopy(parent.actions.arm_action)
    child.scene.robot.actuators["arm"] = deepcopy(parent.scene.robot.actuators["arm"])
    assert child.to_dict() == snapshot
    assert parent_cls().to_dict() == snapshot


def test_native_osc_translation_and_rotation_stay_in_base_axes(osc, tensor_math):
    cfg = osc.v0().actions.arm_action
    controller = osc.controller(cfg.controller_cfg, 2, "cpu")
    term = cfg.class_type()
    term._osc = controller
    term._ee_pose_b = torch.tensor([
        [.3, -.2, .5, 1., 0., 0., 0.],
        [.3, -.2, .5, math.sqrt(.5), 0., 0., math.sqrt(.5)],
    ])
    initial_pose = term._ee_pose_b.clone()
    term._compute_ee_pose = lambda: None
    term._task_frame_transformer = term._task_frame_pose_b = None
    term._raw_actions = torch.zeros(2, 6)
    term._processed_actions = torch.zeros(2, 6)
    term._position_scale, term._orientation_scale = cfg.position_scale, cfg.orientation_scale
    term._pose_rel_idx = 0
    term._pose_abs_idx = term._wrench_abs_idx = term._stiffness_idx = term._damping_ratio_idx = None
    # A +X command is +X in base coordinates for both identity and yawed EEFs.
    raw = torch.tensor([[.02, 0., 0., .1, 0., 0.]]).repeat(2, 1)
    term.process_actions(raw)
    desired = controller.desired_ee_pose_b
    assert torch.allclose(desired[:, :3], initial_pose[:, :3] + raw[:, :3])
    # The incremental rotation is left-multiplied, i.e. about base X; a body
    # rotation would instead right-multiply and rotate around base Y for row 1.
    delta = tensor_math.quat_from_angle_axis(torch.full((2,), .1), torch.tensor([[1., 0., 0.]]).repeat(2, 1))
    expected = tensor_math.quat_mul(delta, initial_pose[:, 3:])
    assert torch.allclose(desired[:, 3:], expected, atol=1e-6)
    body_axis_rotation = tensor_math.quat_mul(initial_pose[:, 3:], delta)
    assert not torch.allclose(desired[1, 3:], body_axis_rotation[1], atol=1e-6)
    assert torch.equal(term._raw_actions, raw)
    assert torch.equal(term._processed_actions, raw)
    # Each subsequent relative command starts at the newly sampled EEF pose.
    term._ee_pose_b[:, :3] += .3
    term.process_actions(torch.zeros_like(raw))
    assert torch.allclose(controller.desired_ee_pose_b, term._ee_pose_b, atol=1e-6)


def test_v1_runner_descendants_preserve_16_slot_actor_and_critic(omni_config):
    configclass = omni_config.baseline.configclass

    @configclass
    class Parent:
        actor = SimpleNamespace(obstacle_observation_slots=10, encoder_hidden_dims=[32], embedding_dim=16)
        critic = SimpleNamespace(obstacle_observation_slots=10, encoder_hidden_dims=[32], embedding_dim=16)
        experiment_name: str = "holding_v0"

    holding = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_v1/ppo_cfg.py",
        dict(configclass=configclass, FusionHoldingV0PPORunnerCfg=Parent),
    ).FusionHoldingV1PPORunnerCfg
    holding_v2 = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_v2/ppo_cfg.py",
        dict(configclass=configclass, FusionHoldingV1PPORunnerCfg=holding),
    ).FusionHoldingV2PPORunnerCfg
    osc = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_osc_v2/ppo_cfg.py",
        dict(configclass=configclass, FusionHoldingV2PPORunnerCfg=holding_v2),
    ).FusionHoldingOSCV2PPORunnerCfg
    random = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_osc_object_random_v1/ppo_cfg.py",
        dict(configclass=configclass, FusionHoldingOSCV2PPORunnerCfg=osc),
    ).FusionHoldingOSCObjectRandomV1PPORunnerCfg
    assert osc.__bases__ == (holding_v2,)
    assert random.__bases__ == (osc,)
    for cls in (holding, osc, random):
        cfg = cls()
        assert cfg.actor.obstacle_observation_slots == cfg.critic.obstacle_observation_slots == 16
        assert cfg.actor.encoder_hidden_dims == cfg.critic.encoder_hidden_dims == [32]
        assert cfg.actor.embedding_dim == cfg.critic.embedding_dim == 16
    assert len({cls().experiment_name for cls in (holding, osc, random)}) == 3
    assert Parent().actor.obstacle_observation_slots == 10


def test_object_random_v1_preserves_prop_setup_with_16_slot_capacity(omni_config, cfg):
    configclass = omni_config.baseline.configclass

    @configclass
    class Scene:
        num_envs: int = 1
        env_spacing: float = 2.5

        def configure_population_capacity(self, gray, white, observation_slots=14):
            self.capacity = (gray, white, observation_slots)

    @configclass
    class Parent:
        obstacle_observation_slots: int = 16

        def __post_init__(self):
            self.scene.configure_population_capacity(
                self.max_gray_obstacles, self.max_white_obstacles, self.obstacle_observation_slots,
            )

    class Events:
        def __init__(self):
            self.object_spawn = SimpleNamespace(params={"object_radius": .04})

    cfgs = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_osc_object_random_v1/env_cfg.py",
        dict(configclass=configclass, FusionHoldingOSCV2EnvCfg=Parent,
             ObjectRandomSceneCfg=Scene, ObjectRandomEventsCfg=Events,
             FusionHoldingV2CurriculumCfg=type(cfg)),
    )
    cls = cfgs.FusionHoldingOSCObjectRandomV1EnvCfg
    assert cls.__bases__ == (Parent,)
    cfg = cls()
    assert cfg.scene.capacity == (7, 7, 16)
    assert cfg.obstacle_observation_slots == 16
    assert cfg.events.object_spawn.params == dict(object_radius=.04, gray_obstacles=7, white_obstacles=7)
    assert cfg.curriculum.obstacle_population is None
    assert cfg.curriculum.goal_position_scale is None
    assert cfg.curriculum.goal_position_tolerance.params["points"][-1] == (20000, .005)
    cfg.scene.configure_population_capacity(8, 8)
    assert cfg.scene.capacity == (8, 8, 16)


def test_holding_v2_and_osc_v2_preserve_managers_and_change_only_arm(omni_config, cfg):
    from test_omni_configuration import _AssetNode

    cc = omni_config.baseline.configclass
    base = omni_config.baseline.InspireSweepEnvCfg
    holding1 = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_v1/env_cfg.py",
        dict(configclass=cc, FusionHoldingV0EnvCfg=base),
    ).FusionHoldingV1EnvCfg
    holding2 = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_v2/env_cfg.py",
        dict(configclass=cc, FusionHoldingV1EnvCfg=holding1,
             FusionHoldingV2RewardsCfg=lambda: SimpleNamespace(marker="v2_reward"),
             FusionHoldingV2TerminationsCfg=lambda: SimpleNamespace(marker="v2_success"),
             FusionHoldingV2CurriculumCfg=type(cfg)),
    ).FusionHoldingV2EnvCfg

    class ArmCfg(_AssetNode):
        OffsetCfg = _AssetNode

    osc2 = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_osc_v2/env_cfg.py",
        dict(configclass=cc, FusionHoldingV2EnvCfg=holding2,
             ARM_JOINT_NAMES=omni_config.baseline.ARM_JOINT_NAMES,
             OperationalSpaceControllerCfg=_AssetNode,
             UR10OperationalSpaceControllerActionCfg=ArmCfg),
    ).FusionHoldingOSCV2EnvCfg
    assert holding2.__bases__ == (holding1,)
    assert osc2.__bases__ == (holding2,)
    parent, child = holding2(), osc2()
    snapshot = parent.to_dict()
    arm = child.actions.arm_action
    assert arm.controller_cfg.motion_stiffness_task == (80., 80., 80., 25., 25., 25.)
    assert not arm.controller_cfg.gravity_compensation
    assert not arm.controller_cfg.inertial_dynamics_decoupling
    assert (arm.position_scale, arm.orientation_scale) == (.03, .15)
    child.actions.arm_action = deepcopy(parent.actions.arm_action)
    child.scene.robot.actuators["arm"] = deepcopy(parent.scene.robot.actuators["arm"])
    assert child.to_dict() == snapshot
    assert holding2().to_dict() == snapshot
    assert parent.obstacle_observation_slots == 16
