"""GPU v2_7 reset IK regression: run from the workspace root with --headless."""
import argparse, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'src/inspire_sweep_baseline'))
sys.path.insert(0, str(Path.cwd()/'src/inspire_sweep_baseline/scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility
parser=argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();apply_renderer_compatibility(args)
app=AppLauncher(args).app
env=None
try:
 import torch, gymnasium as gym
 import inspire_sweep_baseline
 from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_mul,quat_conjugate,axis_angle_from_quat
 from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorRelativeV2_1EnvCfg,WrenchRefactorRelativeV2_7EnvCfg
 from inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg import WrenchRefactorRelativeV2_1PPORunnerCfg,WrenchRefactorRelativeV2_7PPORunnerCfg
 parent=WrenchRefactorRelativeV2_1EnvCfg();before=copy.deepcopy(parent.to_dict());cfg=WrenchRefactorRelativeV2_7EnvCfg()
 expected=copy.deepcopy(before);expected['events']['robot_initial_pose']=cfg.events.robot_initial_pose.to_dict()
 assert cfg.to_dict()==expected and parent.to_dict()==before==WrenchRefactorRelativeV2_1EnvCfg().to_dict()
 runner=WrenchRefactorRelativeV2_7PPORunnerCfg().to_dict();rp=WrenchRefactorRelativeV2_1PPORunnerCfg().to_dict();rp['experiment_name']='inspire_wrench_refactor_relative_v2_7';assert runner==rp
 print('CONFIG_PARENT_PPO_OK',flush=True)
 cfg.scene.num_envs=8;cfg.sim.device=args.device;cfg.seed=42;cfg.log_dir='outputs/relative_v2_7_validation'
 env=gym.make('Inspire-Wrench-Refactor-Relative-v2_7',cfg=cfg);e=env.unwrapped
 obs,_=env.reset(seed=123)
 first=e._goal_reset_desired_pos_w.clone()
 env.reset(seed=123)
 assert torch.equal(first,e._goal_reset_desired_pos_w)
 robot=e.scene['robot'];shelf=e.scene['shelf'].data;arm=e._goal_reset_arm_ids
 # Validate the independently reconstructed FK against the real frame sensor.
 q=robot.data.default_joint_pos[:,arm]
 robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=arm)
 _=robot.data.body_link_pos_w
 e.scene['ee_frame'].reset()
 actual=e.scene['ee_frame'].data
 bp=robot.data.body_link_pos_w[:,e._goal_reset_base_id];bq=robot.data.body_link_quat_w[:,e._goal_reset_base_id]
 p,r=e._goal_reset_ik.forward(q)
 fkerr=(bp+quat_apply(bq,p)-actual.target_pos_w[:,0]).norm(dim=-1)
 qr=axis_angle_from_quat(quat_mul(quat_mul(bq,r),quat_conjugate(actual.target_quat_w[:,0]))).norm(dim=-1)
 print('DEFAULT_FK_ERROR',fkerr.max().item(),qr.max().item(),flush=True)
 assert fkerr.max()<.0001 and qr.max()<.001
 from inspire_sweep_baseline.mdp.common.wrench_refactor_ik import sample_initial_eef
 n=4096
 target=torch.zeros(n,3,device=e.device)
 direction=torch.zeros_like(target);direction[:n//2,1]=-1;direction[n//2:,1]=1
 default_local=quat_apply_inverse(shelf.root_quat_w[:1],bp[:1]+quat_apply(bq[:1],p[:1])-shelf.root_pos_w[:1]).expand(n,-1)
 desired_local=sample_initial_eef(target,direction,default_local,torch.Generator(device=e.device).manual_seed(819))
 world=shelf.root_pos_w[:1]+quat_apply(shelf.root_quat_w[:1].expand(n,-1),desired_local)
 desired_base=quat_apply_inverse(bq[:1].expand(n,-1),world-bp[:1])
 import time
 torch.cuda.synchronize();started=time.perf_counter()
 _,ok,pe,re=e._goal_reset_ik.solve(desired_base,r[:1].expand(n,-1),q[:1].expand(n,-1),robot.data.joint_pos_limits[:1,arm].expand(n,-1,-1))
 torch.cuda.synchronize();elapsed=time.perf_counter()-started
 assert ok.all()
 print('IK_BATCH_4096_OK',elapsed,pe.max().item(),re.max().item(),flush=True)
 records=[]
 for i in range(16):
  obs,_=env.reset(seed=42+i)
  actual=e.scene['ee_frame'].data
  pos=actual.target_pos_w[:,0];quat=actual.target_quat_w[:,0]
  local=quat_apply_inverse(shelf.root_quat_w,pos-shelf.root_pos_w)
  target=quat_apply_inverse(shelf.root_quat_w,e._revision_spawn_positions[:,0]-shelf.root_pos_w)
  sweep=quat_apply_inverse(shelf.root_quat_w,e.sweep_dir)
  side=torch.where(sweep[:,1]<=0,1.,-1.)
  pe=(pos-e._goal_reset_desired_pos_w).norm(dim=-1)
  re=axis_angle_from_quat(quat_mul(quat,quat_conjugate(e._goal_reset_desired_quat_w))).norm(dim=-1)
  print('RESET',i,'max_error',pe.max().item(),re.max().item(),flush=True)
  assert pe.max()<.001 and re.max()<.004
  assert ((local[:,0]>.299)&(local[:,0]<.331)).all()
  assert ((local[:,1]-target[:,1])*side>.099).all()
  assert ((local[:,1]-target[:,1])*side<.141).all()
  assert robot.data.joint_vel[:,arm].abs().max()<1e-5
  assert all(torch.isfinite(v).all() for v in obs.values())
  records.append({'local':local.tolist(),'sweep':sweep.tolist(),'position_error':pe.tolist(),'rotation_error':re.tolist()})
  reach=e.reward_manager.get_term_cfg('target_reaching').func
  gap=(pos-e.scene['object_collection'].data.object_pos_w[:,0]).norm(dim=-1).sub(.1).clamp_min(0)
  torch.testing.assert_close(reach._best_distance,gap,atol=1e-5,rtol=0.)
  action=torch.zeros(8,8,device=e.device);action[:,-2:]=1.
  for _ in range(4):
   _,reward,terminated,truncated,_=env.step(action)
   if terminated.any() or truncated.any():
    print('EARLY_TERMINATION',i,{name:e.termination_manager.get_term(name).tolist() for name in e.termination_manager.active_terms},flush=True)
   assert not terminated.any() and not truncated.any() and torch.isfinite(reward).all()
 # Seed repeatability and partial reset isolation, including manager snapshots.
 env.reset(seed=901);a=e._goal_reset_desired_pos_w.clone();qa=robot.data.joint_pos.clone();env.reset(seed=901)
 assert torch.equal(a,e._goal_reset_desired_pos_w) and torch.allclose(qa,robot.data.joint_pos)
 before_q=robot.data.joint_pos.clone();before_goal=e.command_manager.get_command('target_goal_pos').clone()
 before_point=e._goal_reset_desired_pos_w.clone()
 e.reset(env_ids=torch.tensor([1,5],device=e.device))
 other=torch.tensor([0,2,3,4,6,7],device=e.device)
 assert torch.equal(robot.data.joint_pos[other],before_q[other])
 assert torch.equal(e.command_manager.get_command('target_goal_pos')[other],before_goal[other])
 assert torch.equal(e._goal_reset_desired_pos_w[other],before_point[other])
 action=torch.zeros(8,8,device=e.device);action[:,-2:]=1.
 terms={}
 for i in range(32):
  obs,reward,terminated,truncated,info=env.step(action)
  assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
  for name in e.termination_manager.active_terms:
   terms[name]=terms.get(name,0)+int(e.termination_manager.get_term(name).sum())
 print('ROLLOUT_TERMINATIONS',terms,flush=True)
 assert sum(terms.values())==0
 Path('outputs/relative_v2_7_validation').mkdir(parents=True,exist_ok=True)
 Path('outputs/relative_v2_7_validation/results.json').write_text(json.dumps({'resets':records,'terminations':terms},indent=2))
 print('GPU_RESET_IK_PARTIAL_RESET_FINITE_OK',flush=True)
except BaseException:
 import traceback
 traceback.print_exc()
 print('PROBE_FAILED',flush=True)
 raise
finally:
 if env is not None: env.close()
 app.close()
