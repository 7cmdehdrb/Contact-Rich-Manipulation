"""Exercise the real upstream OSC preprocessor and the local command bounds."""

import ast
from copy import deepcopy
from dataclasses import MISSING
import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_omni_configuration import _execute_definitions, omni_config
from test_tactile_configuration import tactile_config
from test_tactile_renewal_configuration import renewal_tactile, renewal_tactile_v2


@pytest.fixture
def bounded_osc(omni_config, monkeypatch):
    configclass = omni_config.baseline.configclass
    # Keep the production upstream command-processing methods. Construction
    # of PhysX views is covered by the manual GPU probe, rather than mocked.
    path = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py"
    source = ast.parse(path.read_text())
    upstream = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "OperationalSpaceControllerAction")
    upstream.bases = []
    upstream.body = [
        node for node in upstream.body
        if isinstance(node, ast.FunctionDef) and node.name in ("process_actions", "_preprocess_actions")
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[upstream], type_ignores=[]), str(path), "exec"), namespace)

    @configclass
    class ActionTermCfg:
        asset_name: str = "robot"

    actions = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py",
        dict(configclass=configclass, MISSING=MISSING, ActionTermCfg=ActionTermCfg,
             task_space_actions=SimpleNamespace(OperationalSpaceControllerAction=namespace["OperationalSpaceControllerAction"])),
        {"OperationalSpaceControllerActionCfg"},
    )
    module = types.ModuleType("_bounded_osc_cpu")
    module.__dict__.update(
        math=math, torch=torch, configclass=configclass,
        OperationalSpaceControllerAction=namespace["OperationalSpaceControllerAction"],
        OperationalSpaceControllerActionCfg=actions.OperationalSpaceControllerActionCfg,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    local = _execute_definitions(MODULE_ROOT / "mdp/osc_actions.py", module.__dict__)
    controllers = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/controllers/operational_space_cfg.py",
        dict(configclass=configclass, MISSING=MISSING, OperationalSpaceController=object),
        {"OperationalSpaceControllerCfg"},
    )
    return SimpleNamespace(local=local, controllers=controllers)


def test_bounds_reach_controller_after_scaling_without_modifying_raw_actions(bounded_osc):
    cls = bounded_osc.local.BoundedOperationalSpaceControllerAction
    term = cls.__new__(cls)
    term.cfg = SimpleNamespace(position_command_limit=0.01, orientation_command_limit=0.05)
    raw = torch.tensor([
        [10., -20., 30., -40., 50., -60.],
        [0.1, -0.2, 0.3, -0.4, 0.2, -0.1],
        [0., 0., 0., 0., 0., 0.],
    ])
    snapshot = raw.clone()
    term._raw_actions = torch.zeros_like(raw)
    term._processed_actions = torch.zeros_like(raw)
    term._position_scale, term._orientation_scale = 0.01, 0.05
    term._pose_rel_idx = 0
    term._pose_abs_idx = term._wrench_abs_idx = term._stiffness_idx = term._damping_ratio_idx = None
    term._compute_ee_pose = term._compute_task_frame_pose = lambda: None
    term._ee_pose_b = term._task_frame_pose_b = None
    received = []
    term._osc = SimpleNamespace(set_command=lambda **kwargs: received.append(kwargs["command"].clone()))
    term.process_actions(raw)
    assert torch.equal(raw, snapshot)
    assert torch.equal(term._raw_actions, snapshot)
    command = received[0]
    assert command[0, :3].norm().item() == pytest.approx(0.01)
    assert command[0, 3:].norm().item() == pytest.approx(0.05)
    # Below-limit commands pass linearly, and saturation preserves direction.
    assert torch.allclose(command[1, :3], raw[1, :3] * 0.01)
    assert torch.allclose(command[1, 3:], raw[1, 3:] * 0.05)
    assert torch.allclose(command[0, :3] / 0.01, raw[0, :3] / raw[0, :3].norm())
    assert torch.allclose(command[0, 3:] / 0.05, raw[0, 3:] / raw[0, 3:].norm())
    assert command[2].count_nonzero() == 0
    # Translation and rotation saturate independently, with no stale target.
    raw[0, :3] = 0
    term.process_actions(raw)
    assert received[1][0, :3].count_nonzero() == 0
    assert received[1][0, 3:].norm().item() == pytest.approx(0.05)


@pytest.mark.parametrize("position,rotation", [(0., .05), (-.01, .05), (math.inf, .05), (.01, math.nan)])
def test_invalid_physical_limits_fail_before_constructing_sim_views(bounded_osc, position, rotation):
    cfg = SimpleNamespace(
        controller_cfg=SimpleNamespace(target_types=["pose_rel"], impedance_mode="fixed"),
        position_command_limit=position, orientation_command_limit=rotation,
    )
    with pytest.raises(ValueError, match="positive and finite"):
        bounded_osc.local.BoundedOperationalSpaceControllerAction(cfg, None)


def test_v2_osc_changes_only_arm_control_and_keeps_v2_rewards_and_terminations(
    bounded_osc, renewal_tactile_v2, renewal_tactile, omni_config, monkeypatch,
):
    parent_type = renewal_tactile_v2.InspireDexTactileSweepOmniRenewalV2EnvCfg
    module = types.ModuleType("_dex_tactile_v2_osc_cpu_cfg")
    module.__dict__.update(
        configclass=omni_config.baseline.configclass,
        OperationalSpaceControllerCfg=bounded_osc.controllers.OperationalSpaceControllerCfg,
        BoundedOperationalSpaceControllerActionCfg=bounded_osc.local.BoundedOperationalSpaceControllerActionCfg,
        ARM_JOINT_NAMES=omni_config.baseline.ARM_JOINT_NAMES,
        InspireDexTactileSweepOmniRenewalV2EnvCfg=parent_type,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = _execute_definitions(MODULE_ROOT / "dex_tactile_omni_renewal_v2_osc_env_cfg.py", module.__dict__)
    parent = parent_type()
    snapshot = parent.to_dict()
    child = cfg.InspireDexTactileSweepOmniRenewalV2OSCEnvCfg()
    assert isinstance(child, parent_type)
    action = child.actions.arm_action
    assert action.class_type is bounded_osc.local.BoundedOperationalSpaceControllerAction
    assert action.position_scale == action.position_command_limit == 0.005
    assert action.orientation_scale == action.orientation_command_limit == 0.025
    assert action.controller_cfg.motion_stiffness_task == 100.0
    assert action.controller_cfg.motion_damping_ratio_task == 1.0
    assert action.controller_cfg.target_types == ["pose_rel"]
    assert action.controller_cfg.impedance_mode == "fixed"
    assert action.controller_cfg.inertial_dynamics_decoupling
    assert action.controller_cfg.gravity_compensation
    eef = child.scene.ee_frame.target_frames[0]
    assert action.body_name == "axia80_link"
    assert action.body_offset.pos == eef.offset.pos
    assert action.body_offset.rot == eef.offset.rot
    assert child.scene.robot.actuators["arm"].stiffness == 0.0
    assert child.scene.robot.actuators["arm"].damping == 0.0
    child.actions.arm_action = deepcopy(parent.actions.arm_action)
    child.scene.robot.actuators["arm"] = deepcopy(parent.scene.robot.actuators["arm"])
    assert child.to_dict() == snapshot  # All rewards, terminations, Hand and sensors.
    assert parent_type().to_dict() == snapshot

    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    registration = registrations["Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2"]
    assert registration["kwargs"]["env_cfg_entry_point"] == (
        "inspire_sweep_baseline.dex_tactile_omni_renewal_v2_osc_env_cfg:InspireDexTactileSweepOmniRenewalV2OSCEnvCfg"
    )
    assert registration["kwargs"]["rsl_rl_cfg_entry_point"] == (
        "inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV2OSCPPORunnerCfg"
    )
    runner = renewal_tactile.runners.InspireDexTactileSweepOmniRenewalV2OSCPPORunnerCfg()
    expected = renewal_tactile.runners.InspireDexTactileSweepOmniRenewalV2PPORunnerCfg().to_dict()
    expected["experiment_name"] = "inspire_dex_tactile_sweep_omni_renewal_v2_osc"
    assert runner.to_dict() == expected
