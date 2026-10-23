"""Run real Holding family configuration, reset, encoder and OSC smoke checks."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--full-capacity", action="store_true",
                    help="Build 8 Gray + 8 White slots, retaining the configured active population schedule.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from tensordict import TensorDict
from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_mul
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import inspire_sweep_baseline  # noqa: F401
from inspire_sweep_baseline.contact_sweep_clutter_fusion.models import ContactSweepClutterFusionModel
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_v0.env_cfg import FusionHoldingV0EnvCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_v1.env_cfg import FusionHoldingV1EnvCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_v2.env_cfg import FusionHoldingV2EnvCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_osc_v2.env_cfg import FusionHoldingOSCV2EnvCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_osc_v2.actions import UR10OperationalSpaceControllerActionCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_osc_v0.env_cfg import FusionHoldingOSCV0EnvCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_osc_v1.env_cfg import FusionHoldingOSCV1EnvCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_osc_object_random_v0.env_cfg import FusionHoldingOSCObjectRandomV0EnvCfg
from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_osc_object_random_v1.env_cfg import FusionHoldingOSCObjectRandomV1EnvCfg


def check_configs():
    relationships = (
        (FusionHoldingV1EnvCfg, FusionHoldingV0EnvCfg),
        (FusionHoldingOSCV0EnvCfg, FusionHoldingV0EnvCfg),
        (FusionHoldingOSCV1EnvCfg, FusionHoldingV1EnvCfg),
        (FusionHoldingOSCObjectRandomV0EnvCfg, FusionHoldingOSCV0EnvCfg),
        (FusionHoldingV2EnvCfg, FusionHoldingV1EnvCfg),
        (FusionHoldingOSCV2EnvCfg, FusionHoldingV2EnvCfg),
        (FusionHoldingOSCObjectRandomV1EnvCfg, FusionHoldingOSCV2EnvCfg),
    )
    for child, parent in relationships:
        assert child.__bases__ == (parent,)
    base, upgraded = FusionHoldingV0EnvCfg(), FusionHoldingV1EnvCfg()
    before, after = base.to_dict(), upgraded.to_dict()
    assert before.pop("obstacle_observation_slots") == 10
    assert after.pop("obstacle_observation_slots") == 16
    assert before == after, "Holding v1 changed more than obstacle capacity"
    for cls in (FusionHoldingOSCV0EnvCfg, FusionHoldingOSCV1EnvCfg,
                FusionHoldingOSCObjectRandomV0EnvCfg):
        cfg = cls()
        arm = cfg.actions.arm_action
        assert type(arm) is OperationalSpaceControllerActionCfg
        assert arm.task_frame_rel_path is None
        assert arm.controller_cfg.motion_stiffness_task == 100.0
        assert arm.controller_cfg.target_types == ["pose_rel"]
        assert cfg.scene.robot.actuators["arm"].stiffness == 0.0
        assert cfg.scene.robot.actuators["arm"].damping == 0.0
    parent = FusionHoldingOSCV2EnvCfg()
    randomized = FusionHoldingOSCObjectRandomV1EnvCfg()
    assert randomized.actions.to_dict() == parent.actions.to_dict()
    assert randomized.rewards.to_dict() == parent.rewards.to_dict()
    assert randomized.terminations.to_dict() == parent.terminations.to_dict()
    current, expected = randomized.curriculum.to_dict(), parent.curriculum.to_dict()
    assert current.pop("obstacle_population") is None
    expected.pop("obstacle_population")
    assert current == expected
    assert randomized.obstacle_observation_slots == 16
    assert randomized.max_gray_obstacles == randomized.max_white_obstacles == 7



def check_base_frame(raw):
    term = raw.action_manager.get_term("arm_action")
    original = term._compute_ee_pose
    poses = torch.tensor([[0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 2**-.5, 0., 0., 2**-.5]], device=raw.device)
    try:
        def synthetic_pose():
            term._ee_pose_b[:] = poses
        term._compute_ee_pose = synthetic_pose
        command = torch.zeros(2, 6, device=raw.device)
        command[:, 0], command[:, 3] = .01, .1
        term.process_actions(command)
        desired = term._osc.desired_ee_pose_b
        torch.testing.assert_close(desired[:, :3], command[:, :3] * term.cfg.position_scale)
        delta = quat_from_angle_axis(torch.full((2,), .1, device=raw.device),
                                    torch.tensor([[1., 0., 0.]] * 2, device=raw.device))
        if isinstance(term.cfg, UR10OperationalSpaceControllerActionCfg):
            delta = quat_from_euler_xyz(*(command[:, 3:] * term.cfg.orientation_scale).unbind(-1))
        torch.testing.assert_close(desired[:, 3:], quat_mul(delta, poses[:, 3:]))
        assert term._task_frame_pose_b is None
        torch.testing.assert_close(term.raw_actions, command)
    finally:
        term._compute_ee_pose = original


env = None
try:
    check_configs()
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=2)
    cfg.seed = 42
    cfg.log_dir = str(args.output.parent)
    if args.full_capacity:
        assert cfg.obstacle_observation_slots == 16
        cfg.max_gray_obstacles = cfg.max_white_obstacles = 8
    slots = cfg.obstacle_observation_slots
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=42)
    assert {key: list(value.shape) for key, value in obs.items()} == {
        "policy": [2, 55], "obstacles": [2, 4 * slots], "privileged": [2, 3],
    }
    assert raw.action_manager.total_action_dim == 8
    assert raw.scene["object_collection"].num_objects == 1 + cfg.max_gray_obstacles + cfg.max_white_obstacles
    if "-OSC-" in args.task:
        check_base_frame(raw)
        obs, _ = env.reset(seed=42)
    initial_active_counts = raw._active_object_mask.sum(-1).tolist()
    runner = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = handle_deprecated_rsl_rl_cfg(runner, importlib.metadata.version("rsl-rl-lib"))
    for name, dim, latent in (("actor", 8, 71), ("critic", 1, 74)):
        model_cfg = getattr(runner, name).to_dict()
        model_cfg.pop("class_name")
        assert model_cfg["obstacle_observation_slots"] == slots
        model = ContactSweepClutterFusionModel(
            TensorDict(obs, batch_size=[2]), runner.obs_groups, name, dim, **model_cfg,
        ).to(raw.device)
        assert model.get_latent(obs).shape == (2, latent)
        assert [(layer.in_features, layer.out_features) for layer in model.obstacle_encoder.mlp
                if isinstance(layer, torch.nn.Linear)] == [(4 * slots, 32), (32, 16)]
        assert torch.isfinite(model(obs)).all()
    done_count = 0
    with torch.inference_mode():
        for step in range(32):
            action = torch.zeros(2, 8, device=raw.device)
            action[:, 6:] = 1.
            if "-OSC-" in args.task and step >= 16:
                action[:, 0] = .001
            obs, reward, terminated, truncated, _ = env.step(action)
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(reward).all()
            assert torch.isfinite(raw.scene["robot"].data.joint_pos).all()
            assert torch.isfinite(raw.scene["robot"].data.applied_torque).all()
            done_count += int((terminated | truncated).sum())
        before = raw._initial_obstacle_observation[1].clone()
        raw.reset(seed=43, env_ids=torch.tensor([0], device=raw.device))
        torch.testing.assert_close(raw._initial_obstacle_observation[1], before)
    report = dict(task=args.task, seed=42, num_envs=2, steps=32, slots=slots,
                  objects=raw.scene["object_collection"].num_objects,
                  observation_shapes={key: list(value.shape) for key, value in obs.items()},
                  done_count=done_count, full_capacity=args.full_capacity,
                  initial_active_object_counts=initial_active_counts,
                  base_frame_checked="-OSC-" in args.task,
                  config_inheritance_checked=True, partial_reset_checked=True,
                  actor_critic_encoder_checked=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("HOLDING FAMILY SMOKE PASSED: " + args.task, flush=True)
except BaseException:
    # SimulationApp.close may terminate the process before Python reports a
    # pending exception; surface the failure and preserve a nonzero exit code.
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
finally:
    if env is not None:
        env.close()
    launcher.app.close()
