"""Real Isaac Sim reset/rollout check, writing only outputs/wrench_refactor_validation."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['Inspire-Wrench-Refactor-v0','Inspire-Wrench-Refactor-v1','Inspire-Wrench-Refactor-v2','Inspire-Wrench-Refactor-v3'], default='Inspire-Wrench-Refactor-v0')
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--config-only', action='store_true')
parser.add_argument('--output', type=Path, default=Path('outputs/wrench_refactor_validation'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
env = None
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.wrench_refactor_envs import (
        WrenchRefactorEnvCfg, WrenchRefactorV1EnvCfg, WrenchRefactorV2EnvCfg, WrenchRefactorV3EnvCfg,
    )
    from inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg import (
        WrenchRefactorPPORunnerCfg, WrenchRefactorV1PPORunnerCfg,
        WrenchRefactorV2PPORunnerCfg, WrenchRefactorV3PPORunnerCfg,
    )
    from inspire_sweep_baseline.mdp.action_functions.wrench_refactor import WrenchRefactorOSCAction
    classes = [WrenchRefactorEnvCfg, WrenchRefactorV1EnvCfg, WrenchRefactorV2EnvCfg, WrenchRefactorV3EnvCfg]
    configs = [cls() for cls in classes]
    for item in configs:
        item.scene.num_envs, item.sim.device, item.seed = args.num_envs, args.device, 42
        item.log_dir = str(args.output)
    snapshots = [item.to_dict() for item in configs]
    import copy
    for version, parent_version in [(1,0),(2,0),(3,2)]:
        expected = copy.deepcopy(snapshots[parent_version])
        if version == 1:
            expected['rewards']['target_contact'] = None
            expected['rewards']['target_goal_distance'] = configs[version].rewards.target_goal_distance.to_dict()
        elif version == 2:
            expected['rewards']['target_goal_distance']['func'] = configs[version].rewards.target_goal_distance.to_dict()['func']
        else:
            expected['actions']['arm_action']['class_type'] = configs[version].actions.arm_action.to_dict()['class_type']
        assert snapshots[version] == expected, f'Unexpected v{version} environment delta'
        assert classes[version].__bases__ == (classes[parent_version],)
    assert [item.to_dict() for item in configs] == snapshots
    runner_classes = [WrenchRefactorPPORunnerCfg, WrenchRefactorV1PPORunnerCfg,
                      WrenchRefactorV2PPORunnerCfg, WrenchRefactorV3PPORunnerCfg]
    runners = [cls() for cls in runner_classes]
    for version in (1,2,3):
        expected_runner = runners[0].to_dict()
        expected_runner['experiment_name'] = f'inspire_wrench_refactor_v{version}'
        assert runners[version].to_dict() == expected_runner
    version = int(args.task[-1])
    cfg, runner = configs[version], runners[version]
    print('ALL_CONFIG_DIFF_OK: v1<-v0 contact/distance; v2<-v0 cosine only; v3<-v2 OSC class only', flush=True)
    if args.config_only:
        def symbolic(value):
            if callable(value):
                return f'{value.__module__}:{value.__qualname__}'
            return str(value)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'resolved_config.json').write_text(json.dumps(
            {'env':cfg.to_dict(), 'runner':runner.to_dict()}, default=symbolic, indent=2))
        print(json.dumps({'config_import':True,'objects':len(cfg.scene.object_collection.rigid_objects),
                          'runner_groups':runner.obs_groups,'goal_radius':cfg.commands.target_goal_pos.goal_radius}),flush=True)
        # No simulation has been created. Kit's no-device shutdown itself can
        # segfault on a driverless host; the config-only subprocess ends here.
        import os
        os._exit(0)
    started = time.perf_counter()
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    from inspire_sweep_baseline.mdp.action_functions.contact_sweep_clutter_fusion_holding_osc import UR10OperationalSpaceControllerAction
    expected_type = WrenchRefactorOSCAction if version == 3 else UR10OperationalSpaceControllerAction
    assert type(raw.action_manager.get_term('arm_action')) is expected_type
    obs, _ = env.reset(seed=42)
    assert {k: v.shape[-1] for k, v in obs.items()} == {'policy':74, 'critic':74, 'obstacles':64, 'critic_obstacles':64}
    assert raw.scene['object_collection'].num_objects == 17
    shelf_mass = raw.scene['shelf'].root_physx_view.get_masses()
    torch.testing.assert_close(shelf_mass, torch.full_like(shelf_mass, 30.))
    from inspire_sweep_baseline.assets.robot import HAND_CONTACT_BODY_NAMES
    robot = raw.scene['robot']
    assert cfg.scene.robot.spawn.usd_path.endswith('wrench_right_hand/ur5e_inspire_right.usda')
    arm = raw.action_manager.get_term('arm_action')
    assert arm.cfg.body_name == 'axia80_link'
    eef_cfg = cfg.scene.ee_frame.target_frames[0].offset
    assert tuple(arm.cfg.body_offset.pos) == tuple(eef_cfg.pos)
    assert tuple(arm.cfg.body_offset.rot) == tuple(eef_cfg.rot)
    if version != 3:
        assert arm._task_frame_pose_b is None
    # Test actual right-hand articulation: root-axis delta, regardless of wrist orientation.
    for axis in range(3):
        command = torch.zeros(raw.num_envs, 6, device=raw.device)
        command[:, axis] = 1.
        arm.process_actions(command)
        delta = arm._osc.desired_ee_pose_b[:, :3] - arm._ee_pose_b[:, :3]
        expected_delta = command[:, :3] * .03
        if version == 3:
            from isaaclab.utils.math import quat_apply
            expected_delta = quat_apply(arm._ee_pose_b[:, 3:], expected_delta)
            torch.testing.assert_close(arm._task_frame_pose_b, arm._ee_pose_b)
        torch.testing.assert_close(delta, expected_delta, atol=1e-6, rtol=0.)
    arm.process_actions(torch.zeros(raw.num_envs, 6, device=raw.device))
    hand_ids = [robot.body_names.index(name) for name in HAND_CONTACT_BODY_NAMES]
    hand_mass = robot.root_physx_view.get_masses()[:, hand_ids].sum(-1)
    torch.testing.assert_close(hand_mass, torch.full_like(hand_mass, .790))
    expected = np.array([0, *[i for i in range(8) for _ in range(2)]])
    assert np.array_equal(raw._revision_object_kinds, np.tile(expected, (args.num_envs,1)))
    mass = raw._clutter_physics.mass_buffer.clone()
    materials = [m.properties.clone() for m in raw._clutter_physics.materials]
    bounds = raw._revision_bounds_t.clone()
    ids = torch.arange(0, args.num_envs, 2, device=raw.device)
    other = torch.arange(1, args.num_envs, 2, device=raw.device)
    objects = raw.scene['object_collection']
    before = objects.data.object_link_state_w[other].clone()
    raw.reset(env_ids=ids)
    torch.testing.assert_close(objects.data.object_link_state_w[other], before)
    actions = torch.zeros(args.num_envs, 8, device=raw.device); actions[:, -2:] = 1.
    timings = []
    for iteration in (0,9999,10000,13000,16000,19000,22000,25000):
        raw.set_learning_iteration(iteration)
        t = time.perf_counter(); env.reset()
        if raw.device != 'cpu': torch.cuda.synchronize()
        timings.append(time.perf_counter()-t)
        goal_term = raw.command_manager.get_term('target_goal_pos')
        initial_distance = (goal_term.command[:, :2] - goal_term.target_init_state_w[:, :2]).norm(dim=-1)
        assert (initial_distance >= .05 - 1e-6).all(), initial_distance.min()
        for _ in range(36):
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
        assert torch.equal(raw._clutter_physics.mass_buffer, mass)
        assert torch.equal(raw._revision_bounds_t, bounds)
        assert all(torch.equal(m.properties, old) for m, old in zip(raw._clutter_physics.materials, materials))
    report = {'task':args.task, 'num_envs':args.num_envs, 'elapsed_s':time.perf_counter()-started,
              'reset_seconds':timings, 'fixed_physics':True, 'partial_reset':True, 'finite_rollout':True,
              'right_hand_control_axes':('eef' if version == 3 else 'root'), 'minimum_goal_distance_m':.05}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output / f'probe_{args.num_envs}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
except Exception:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None: env.close()
    app.close()
