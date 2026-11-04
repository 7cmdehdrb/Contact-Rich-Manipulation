"""Real Isaac Lab full-config comparison, reset and 36-step collection timing.

Run on an Isaac Sim GPU host. Does not train or write logs/rsl_rl.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser=argparse.ArgumentParser()
parser.add_argument('--version', choices=('v0', 'v1'), default='v0')
parser.add_argument('--num_envs',type=int,default=64)
parser.add_argument('--collections',type=int,default=5)
parser.add_argument('--output',type=Path,default=Path('outputs/wrench_validation'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
apply_renderer_compatibility(args)
app=AppLauncher(args).app
env=None
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.revision_envs import RevisionV2TestEnvCfg
    from inspire_sweep_baseline.envs.wrench_envs import WrenchEnvCfg, WrenchV1EnvCfg
    from inspire_sweep_baseline.agent.revision_ppo_cfg import RevisionV2TestPPORunnerCfg
    from inspire_sweep_baseline.agent.wrench_ppo_cfg import WrenchPPORunnerCfg, WrenchV1PPORunnerCfg

    def differences(a,b,prefix=''):
        if isinstance(a,dict) and isinstance(b,dict):
            out=[]
            for key in sorted(a.keys()|b.keys()):
                path=f'{prefix}.{key}' if prefix else key
                out += [path] if key not in a or key not in b else differences(a[key],b[key],path)
            return out
        return [] if a==b else [prefix]

    parent=RevisionV2TestEnvCfg() if args.version == 'v0' else WrenchEnvCfg()
    baseline=parent.to_dict()
    cfg=WrenchEnvCfg() if args.version == 'v0' else WrenchV1EnvCfg()
    config_diff=differences(baseline,cfg.to_dict())
    allowed=(
        'goal_reward_height_limit','contact_height_limit','layout_bank_size','layout_build_batch_size',
        'red_margin','rear_offset','rewards.target_goal_timeout_bonus','rewards.target_goal_success_bonus',
        'terminations.target_goal_success_at_timeout','terminations.target_goal_success',
        'terminations.target_goal_miss_at_timeout.func','events.layout_bank',
        'events.object_spawn.func','events.object_physics_randomization.mode',
        'curriculum.target_goal_direction_deadzone.',
    )
    if args.version == 'v1':
        allowed = ('actions.arm_action.', 'rewards.robot_action_rate_penalty.func')
    assert all(path.startswith(allowed) for path in config_diff),config_diff
    assert type(parent)().to_dict()==baseline
    assert differences(RevisionV2TestPPORunnerCfg().to_dict(),WrenchPPORunnerCfg().to_dict())==['experiment_name']
    assert differences(WrenchPPORunnerCfg().to_dict(),WrenchV1PPORunnerCfg().to_dict())==['experiment_name']
    cfg.scene.num_envs=args.num_envs
    cfg.sim.device=args.device
    cfg.seed=42
    cfg.log_dir=str(args.output)
    started=time.perf_counter()
    task=f'Inspire-Wrench-{args.version}'
    env=gym.make(task,cfg=cfg)
    raw=env.unwrapped
    initialized=time.perf_counter()-started
    obs,_=env.reset(seed=42)
    assert {k:v.shape[-1] for k,v in obs.items()}=={'policy':57,'obstacles':64,'privileged':3,'critic_obstacles':64}
    assert raw.scene['object_collection'].num_objects==10
    assert np.all(np.asarray(raw._revision_object_kinds)[:,0]==0)
    assert len(raw.reward_manager.active_terms)==15
    arm = raw.action_manager.get_term('arm_action')
    if args.version == 'v1':
        assert arm.cfg.body_offset is None and arm.cfg.body_name == 'axia80_link'
        assert arm.action_dim == 6
    mass=raw._clutter_physics.mass_buffer.clone()
    friction=raw._clutter_physics.static_friction.clone()
    ids=torch.arange(0,args.num_envs,2,device=raw.device)
    other=torch.arange(1,args.num_envs,2,device=raw.device)
    objects=raw.scene['object_collection']
    before=objects.data.object_link_state_w[other].clone()
    success=raw.termination_manager.get_term_cfg('target_goal_success').func
    success._steps[:]=100
    raw.reset(env_ids=ids)
    assert not success._steps[ids].any() and (success._steps[other]==100).all()
    torch.testing.assert_close(objects.data.object_link_state_w[other],before)
    assert torch.equal(raw._clutter_physics.mass_buffer,mass)
    assert torch.equal(raw._clutter_physics.static_friction,friction)
    actions=torch.zeros(args.num_envs,raw.action_manager.total_action_dim,device=raw.device)
    actions[:,-2:]=1
    measurements=[]
    for iteration in (0,10000,12000,13000,14000,16000,19000,22000,25000):
        raw.set_learning_iteration(iteration)
        env.reset()
        for _ in range(args.collections):
            torch.cuda.synchronize()
            started=time.perf_counter()
            for step in range(36):
                obs,reward,terminated,truncated,_=env.step(actions)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in obs.values())
                if args.version == 'v1':
                    assert torch.isfinite(arm.current_wrench_s).all()
                    assert torch.isfinite(arm.joint_efforts).all()
                    limits = raw.scene['robot'].data.joint_effort_limits[:, arm._joint_ids]
                    assert (arm.joint_efforts.abs() <= limits + 1e-6).all()
            torch.cuda.synchronize()
            measurements.append({'iteration':iteration,'seconds':time.perf_counter()-started})
    report=dict(task=task,num_envs=args.num_envs,initialize_s=initialized,
                config_diff=config_diff,partial_reset=True,physics_fixed=True,
                collection_36_steps=measurements,
                scope='Zero actions with finite checks; excludes policy inference and PPO updates')
    args.output.mkdir(parents=True,exist_ok=True)
    suffix = '' if args.version == 'v0' else '_v1'
    (args.output/f'gpu_probe{suffix}_{args.num_envs}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
finally:
    if env is not None: env.close()
    app.close()
