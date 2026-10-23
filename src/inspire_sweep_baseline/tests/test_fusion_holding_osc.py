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


@pytest.mark.parametrize("joint_position", [False, True])
def test_object_random_v1_preserves_prop_setup_with_16_slot_capacity(omni_config, cfg, joint_position):
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
            self.events.object_spawn.params.update(gray_obstacles=1, white_obstacles=1)

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
    if joint_position:
        cls = _execute_definitions(
            MODULE_ROOT / "contact_sweep_clutter_fusion_holding_object_random_test_v0/env_cfg.py",
            dict(configclass=configclass, FusionHoldingV2EnvCfg=Parent,
                 ObjectRandomV1SceneCfg=cfgs.ObjectRandomV1SceneCfg,
                 ObjectRandomEventsCfg=Events,
                 ObjectRandomV1CurriculumCfg=cfgs.ObjectRandomV1CurriculumCfg),
        ).FusionHoldingObjectRandomTestV0EnvCfg
    assert cls.__bases__ == (Parent,)
    cfg = cls()
    assert cfg.scene.capacity == (8, 8, 16)
    assert cfg.obstacle_observation_slots == 16
    assert cfg.events.object_spawn.params == dict(object_radius=.04, gray_obstacles=1, white_obstacles=1)
    assert cfg.curriculum.obstacle_population.params == dict(
        initial_gray=1, initial_white=1, final_gray=8, final_white=8,
        gray_start_iteration=4500, white_start_iteration=3000, interval=3000,
    )
    assert cfg.curriculum.goal_position_scale is None
    assert cfg.curriculum.goal_position_tolerance.params["points"][-1] == (20000, .005)
    cfg.scene.configure_population_capacity(8, 8)
    assert cfg.scene.capacity == (8, 8, 16)

    # Exercise the production schedule against the configured scene capacity,
    # including every transition, the final cap and checkpoint rewind.
    import runpy
    schedule = runpy.run_path(str(MODULE_ROOT / "contact_sweep_clutter_fusion_v1_0/population.py"))
    event = deepcopy(cfg.events.object_spawn)
    env = SimpleNamespace(cfg=cfg, event_manager=SimpleNamespace(
        get_term_cfg=lambda _: event, set_term_cfg=lambda *_: None))
    checks = [(0, 1, 1), (2999, 1, 1)]
    for increment in range(1, 8):
        white_step = 3000 + (increment - 1) * 3000
        gray_step = white_step + 1500
        checks += [(white_step, increment, increment + 1),
                   (gray_step - 1, increment, increment + 1),
                   (gray_step, increment + 1, increment + 1),
                   (gray_step + 1499, increment + 1, increment + 1)]
    for iteration, gray, white in checks + [(90000, 8, 8)] + list(reversed(checks)):
        env.learning_iteration = iteration
        assert schedule["obstacle_population_schedule"](
            env, None, **cfg.curriculum.obstacle_population.params,
        ) == dict(gray_obstacles=gray, white_obstacles=white)
        mask = schedule["active_population_mask"](1, 8, 8, gray, white, "cpu", observation_slots=16)
        assert mask.sum().item() == 1 + gray + white
        assert mask[0, 1:9].sum().item() == gray
        assert mask[0, 9:].sum().item() == white


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
    assert arm.controller_cfg.motion_stiffness_task == (100., 100., 100., 50., 50., 50.)
    assert arm.controller_cfg.gravity_compensation
    assert arm.controller_cfg.inertial_dynamics_decoupling
    assert (arm.position_scale, arm.orientation_scale) == (.03, .15)
    child.actions.arm_action = deepcopy(parent.actions.arm_action)
    child.scene.robot.actuators["arm"] = deepcopy(parent.scene.robot.actuators["arm"])
    assert child.to_dict() == snapshot
    assert holding2().to_dict() == snapshot
    assert parent.obstacle_observation_slots == 16

    joint_random = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_object_random_test_v0/env_cfg.py",
        dict(configclass=cc, FusionHoldingV2EnvCfg=holding2,
             ObjectRandomV1SceneCfg=type(parent.scene),
             ObjectRandomEventsCfg=type(parent.events),
             ObjectRandomV1CurriculumCfg=type(cfg)),
    ).FusionHoldingObjectRandomTestV0EnvCfg
    assert joint_random.__bases__ == (holding2,)
    randomized = joint_random()
    assert randomized.actions.to_dict() == parent.actions.to_dict()
    assert randomized.scene.robot.actuators["arm"] == parent.scene.robot.actuators["arm"]
    assert randomized.scene.robot.actuators["arm"].stiffness != 0.
    assert randomized.scene.robot.actuators["arm"].damping != 0.
    assert randomized.rewards == parent.rewards
    assert randomized.terminations == parent.terminations
    assert randomized.obstacle_observation_slots == 16


def test_joint_random_runner_and_registration_preserve_holding_model(omni_config):
    cc = omni_config.baseline.configclass

    @cc
    class Parent:
        experiment_name: str = "inspire_contact_sweep_clutter_fusion_holding_v2"
        actor = SimpleNamespace(obstacle_observation_slots=16, encoder_hidden_dims=[32], embedding_dim=16)
        critic = SimpleNamespace(obstacle_observation_slots=16, encoder_hidden_dims=[32], embedding_dim=16)

    cls = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_holding_object_random_test_v0/ppo_cfg.py",
        dict(configclass=cc, FusionHoldingV2PPORunnerCfg=Parent),
    ).FusionHoldingObjectRandomTestV0PPORunnerCfg
    assert cls.__bases__ == (Parent,)
    before, after = Parent().to_dict(), cls().to_dict()
    assert after.pop("experiment_name") == "inspire_contact_sweep_clutter_fusion_holding_object_random_test_v0"
    before.pop("experiment_name")
    assert before == after
    registry = {}
    _execute_definitions(MODULE_ROOT / "__init__.py", dict(gym=SimpleNamespace(
        registry=registry, register=lambda **kw: registry.update({kw["id"]: kw}))))
    spec = registry["Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-ObjectRandom-Test-v0"]
    assert spec["entry_point"].endswith("fusion_v1_0.env:ContactSweepClutterFusionV1_0Env")
    assert spec["kwargs"]["env_cfg_entry_point"].endswith("holding_object_random_test_v0.env_cfg:FusionHoldingObjectRandomTestV0EnvCfg")
    assert spec["kwargs"]["rsl_rl_cfg_entry_point"].endswith("holding_object_random_test_v0.ppo_cfg:FusionHoldingObjectRandomTestV0PPORunnerCfg")
