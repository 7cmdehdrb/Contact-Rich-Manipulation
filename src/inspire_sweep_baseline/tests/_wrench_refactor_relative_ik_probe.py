"""GPU smoke for Relative-IK registrations, observations, models and termination."""
import argparse, copy, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src/inspire_sweep_baseline'))
sys.path.insert(0,str(Path.cwd()/'src/inspire_sweep_baseline/scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility
parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,choices=(0,1),required=True)
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();apply_renderer_compatibility(args)
app=AppLauncher(args).app;env=None
try:
 import torch,gymnasium as gym
 from tensordict import TensorDict
 import inspire_sweep_baseline
 from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorRelativeV2_7EnvCfg,WrenchRefactorRelativeIKV0EnvCfg,WrenchRefactorRelativeIKV1EnvCfg
 from inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg import WrenchRefactorRelativePPORunnerCfg,WrenchRefactorRelativeIKV0PPORunnerCfg,WrenchRefactorRelativeIKV1PPORunnerCfg
 from inspire_sweep_baseline.agent.wrench_refactor_models import WrenchRefactorRelativeModel
 parent=WrenchRefactorRelativeV2_7EnvCfg();before=copy.deepcopy(parent.to_dict())
 cfg=(WrenchRefactorRelativeIKV0EnvCfg,WrenchRefactorRelativeIKV1EnvCfg)[args.version]()
 v0=WrenchRefactorRelativeIKV0EnvCfg();v0_before=copy.deepcopy(v0.to_dict())
 expected=copy.deepcopy(before);expected['observations']=v0.observations.to_dict()
 expected['debug_initial_ik_bounds']=True
 expected['events']['robot_initial_pose']['params'].update(
  front_clearance=(.08,.10),lateral_offset=(.10,.18),height_jitter=.02,
  height_offset=.01)
 assert v0_before==expected
 if args.version==1:
  expected=copy.deepcopy(v0_before)
  expected['rewards']['target_goal_distance']=cfg.rewards.target_goal_distance.to_dict()
  for name in ('target_contact','target_goal_hold','target_goal_first_entry_bonus','target_goal_success_bonus'):
   expected['rewards'][name]=None
  expected['terminations']['target_goal_success']=None
 assert cfg.to_dict()==expected
 assert parent.to_dict()==before==WrenchRefactorRelativeV2_7EnvCfg().to_dict()
 assert v0.to_dict()==v0_before==WrenchRefactorRelativeIKV0EnvCfg().to_dict()
 runner=(WrenchRefactorRelativeIKV0PPORunnerCfg,WrenchRefactorRelativeIKV1PPORunnerCfg)[args.version]()
 expected_runner=WrenchRefactorRelativePPORunnerCfg().to_dict()
 expected_runner['experiment_name']=f'inspire_wrench_refactor_relative_ik_v{args.version}'
 assert runner.to_dict()==expected_runner
 assert runner.obs_groups=={'actor':['policy','obstacles'],'critic':['policy','obstacles']}
 assert not getattr(cfg,'debug_reaching_point',False)
 assert cfg.rewards.target_inverse_distance.weight==-12.97
 print('CONFIG_PARENT_SHARED_PPO_OK',args.version,flush=True)
 cfg.scene.num_envs=4;cfg.sim.device=args.device;cfg.seed=42
 cfg.log_dir=f'outputs/relative_ik_validation/v{args.version}'
 cfg.observations.policy.enable_corruption=False
 env=gym.make(f'Inspire-Wrench-Refactor-Relative-IK-v{args.version}',cfg=cfg);e=env.unwrapped
 obs,_=env.reset(seed=42)
 assert {k:v.shape[-1] for k,v in obs.items()}=={'policy':58,'obstacles':64}
 assert not hasattr(e,'_reaching_point_marker')
 from pxr import Usd, UsdPhysics
 from isaaclab.sim.utils import get_current_stage
 stage=get_current_stage()
 root=stage.GetPrimAtPath('/Visuals/InspireWrenchRefactor/initial_ik_bounds')
 assert root.IsValid() and e._ik_bounds_active.all()
 for prim in Usd.PrimRange(root):
  assert not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
 edges=e._ik_bounds_positions.reshape(4,2,12,3)
 low,high=edges.amin(2),edges.amax(2)
 assert torch.allclose(high-low,torch.tensor([.02,.08,.04],device=e.device).expand_as(low),atol=1e-6)
 desired=e._goal_reset_desired_pos_w[:,None,:]
 assert (((desired>=low-1e-6)&(desired<=high+1e-6)).all(-1).any(-1)).all()
 previous=e._ik_bounds_positions.clone()
 e.reset(env_ids=torch.tensor([1],device=e.device))
 assert torch.equal(previous[[0,2,3]],e._ik_bounds_positions[[0,2,3]])
 print('TWO_IK_BOUNDING_BOXES_NONPHYSICAL_BOUNDS_PARTIAL_RESET_OK',flush=True)
 # Confirm initial-only target observation even after live target movement.
 term=e.observation_manager.cfg.policy.target_obs_state
 initial=term.func(e,**term.params).clone()
 objects=e.scene['object_collection'];saved=objects.data.object_link_state_w.clone();moved=saved.clone();moved[:,0,0]+=.01
 objects.write_object_link_state_to_sim(moved)
 assert torch.allclose(term.func(e,**term.params),initial,atol=1e-6)
 objects.write_object_link_state_to_sim(saved)
 td=TensorDict(obs,batch_size=[4])
 from importlib.metadata import version
 from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
 runner=handle_deprecated_rsl_rl_cfg(runner,version('rsl-rl-lib'))
 for name,size in [('actor',8),('critic',1)]:
  params=copy.deepcopy(getattr(runner,name).to_dict());params.pop('class_name')
  model=WrenchRefactorRelativeModel(td,runner.obs_groups,name,size,**params).to(e.device)
  output=model(td);assert output.shape==(4,size) and torch.isfinite(output).all()
  output.square().mean().backward()
  assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 print('INITIAL_ONLY_OBS_AND_ACTOR_CRITIC_FORWARD_BACKWARD_OK',flush=True)
 assert ('target_goal_success' in e.termination_manager.active_terms)==(args.version==0)
 for name in ('target_contact','target_goal_hold','target_goal_first_entry_bonus','target_goal_success_bonus'):
  assert (name in e.reward_manager.active_terms)==(args.version==0)
 action=torch.zeros(4,8,device=e.device);action[:,-2:]=1
 for seed in range(42,74):
  env.reset(seed=seed)
  from inspire_sweep_baseline.mdp.common.wrench_refactor_ik import initial_pose_quality
  robot=e.scene['robot'];arm=e._goal_reset_arm_ids
  quality,_,_=initial_pose_quality(e._goal_reset_ik,robot.data.joint_pos[:,arm],robot.data.joint_pos_limits[:,arm])
  assert quality.all()
  assert not hasattr(e,'_goal_reset_condition')  # No runtime SVD/rejection sampling.
  for _ in range(4):
   _,reward,terminated,truncated,_=env.step(action)
   assert not terminated.any() and not truncated.any() and torch.isfinite(reward).all(), (seed,terminated)
 print('EXPANDED_IK_128_RESETS_QUALITY_AND_4_STEPS_OK',flush=True)
 for _ in range(32):
  obs,reward,terminated,truncated,_=env.step(action)
  assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
  assert not terminated.any() and not truncated.any()
 # Existing 10s time-limit remains enabled; force its boundary, not its config.
 e.episode_length_buf[:]=e.max_episode_length-1
 _,_,_,truncated,_=env.step(action)
 assert truncated.all()
 print('GPU_IK_SHARED_OBS_32_STEPS_TIMEOUT_OK',args.version,flush=True)
except BaseException:
 import traceback;traceback.print_exc();print('PROBE_FAILED',flush=True);raise
finally:
 if env is not None:env.close()
 app.close()
