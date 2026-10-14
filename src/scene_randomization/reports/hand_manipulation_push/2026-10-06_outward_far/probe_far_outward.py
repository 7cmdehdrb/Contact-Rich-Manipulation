"""Finite actual FK/full robot OBB grid for outward fingers at a far Table."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/hand_manipulation_test'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--table-width',type=float,default=.36)
parser.add_argument('--cube-x',type=float,nargs='+',default=[-.70])
parser.add_argument('--left-up-deg',type=float,nargs='+',default=[0,30,60,70,80])
parser.add_argument('--palm-height',type=float,nargs='+',default=[.10,.025,.04])
parser.add_argument('--backoff',type=float,nargs='+',default=[.14,.038])
parser.add_argument('--output',type=Path,default=Path(__file__).with_name('far_outward.json'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();launcher=AppLauncher(args,fast_shutdown=True);app=launcher.app

def save(r):
 tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(args.output)

def main():
 import gymnasium as gym
 import torch
 import hand_manipulation_test
 import isaaclab.utils.math as m
 from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
 from hand_manipulation_test.mdp.push_events import PushSafePoseReset
 from hand_manipulation_test.contact_math import palm_facing_rotation
 cfg=load_cfg_from_registry('Isaac-Hand-Manipulation-Push-v1','env_cfg_entry_point')
 cfg.scene.num_envs=4;cfg.scene.env_spacing=.5;cfg.sim.device=args.device
 cfg.task.table_size=(args.table_width,1.,.04);cfg.scene.table.spawn.size=cfg.task.table_size
 cfg.task.object_xy_offset_low=(min(args.cube_x)-cfg.task.table_center[0]-.005,-.03)
 cfg.task.object_xy_offset_high=(max(args.cube_x)-cfg.task.table_center[0]+.005,.03)
 cfg.task.command_centered_path=False
 cfg.task.initial_hand_open_range=(.94,.94)
 cfg.events.safe_hand.func=PushSafePoseReset
 cfg.commands.target_position.debug_vis=False;cfg.scene.ee_frame.debug_vis=False
 cfg.seed=42
 report={'table_size':cfg.task.table_size,'table_center':cfg.task.table_center,'robot_default_preserved':True,
         'reset_seeds':cfg.task.reset_joint_seed_offsets,'max_iterations':cfg.task.reset_max_iterations,
         'clearance_m':cfg.task.reset_clearance_margin_m,'hand_synergy':(.94,.94),'cases':[]}
 env=gym.make('Isaac-Hand-Manipulation-Push-v1',cfg=cfg).unwrapped
 try:
  safe=env.event_manager.get_term_cfg('safe_hand').func;command=env.command_manager.get_term('target_position')
  ids=torch.arange(4,device=env.device);angles=torch.tensor((0.,0.,math.pi,math.pi),device=env.device)
  offsets=torch.zeros(4,3,device=env.device);tilts=torch.zeros(4,device=env.device)
  histories=[[] for _ in range(4)]
  original_solve=safe._solve_seed
  def solve(index,hand):
   old=safe.ik_iterations[index].clone();result=original_solve(index,hand)
   pos=safe.robot.data.body_link_pos_w[index];quat=safe.robot.data.body_link_quat_w[index]
   centres=pos+m.quat_apply(quat,safe.collision_bounds.centers.expand_as(pos))
   radii=(m.matrix_from_quat(quat).abs()@safe.collision_bounds.half_extents.expand_as(pos).unsqueeze(-1)).squeeze(-1)
   minima=centres[:,:,2]-radii[:,:,2]
   for row,i in enumerate(index.cpu().tolist()):
    histories[i].append({'seed':int(safe.attempts[i])-1,'converged':bool(result[row]),
      'iterations':int(safe.ik_iterations[i]-old[row]),'arm_q':safe.robot.data.joint_pos[i,safe.arm_joint_ids].cpu().tolist(),
      'body_zmin_local':dict(zip(safe.robot.body_names,minima[row].cpu().tolist()))})
   return result
  safe._solve_seed=solve
  def sample(index):
   cube=env.scene['target_object'].data.root_pos_w[index];normal=command.direction_w[index]
   rotation=palm_facing_rotation(normal);up=torch.zeros_like(normal);up[:,2]=1
   finger=tilts[index].cos()[:,None]*rotation[:,:,2]+tilts[index].sin()[:,None]*up
   thumb=torch.linalg.cross(normal,finger,dim=-1)
   rotation=torch.stack((thumb,normal,finger),dim=-1)
   palm=cube-offsets[index,:1]*normal+offsets[index,2:]*up
   safe.desired_palm_position_w[index]=palm
   safe.desired_c_pos_w[index]=palm+(rotation@(safe.c_offset_h-safe.palm_reference_h).unsqueeze(-1)).squeeze(-1)
   safe.desired_c_quat_w[index]=m.quat_unique(m.quat_mul(m.quat_from_matrix(rotation),safe.c_quat_h.expand(len(index),-1)))
   safe.direction_w[index]=normal
   if not bool(((safe.desired_c_pos_w[index,1]-env.scene.env_origins[index,1]).abs()<=.25).all()):
    raise RuntimeError('C-Y outside fixed production range')
  safe._sample_pose=sample
  for cube_x,tilt,backoff,height in itertools.product(args.cube_x,args.left_up_deg,args.backoff,args.palm_height):
   cfg.task.object_xy_offset_low=(cube_x-cfg.task.table_center[0]-.005,-.03)
   cfg.task.object_xy_offset_high=(cube_x-cfg.task.table_center[0]+.005,.03)
   positions=torch.tensor([(cube_x,-.03,cfg.task.cube_center_height_m),(cube_x,.03,cfg.task.cube_center_height_m)]*2,device=env.device)
   command.set_reset_specs(ids,positions,angles,.30)
   offsets[:,0]=backoff;offsets[:,2]=height;tilts[:]=0;tilts[2:]=math.radians(tilt)
   histories=[[] for _ in range(4)];start=time.perf_counter();failure=None
   try:env.reset(seed=42)
   except RuntimeError as exc:failure=str(exc)
   accepted=safe.accepted_seed_index>=0;certified=torch.zeros(4,dtype=torch.bool,device=env.device)
   if bool(accepted.any()):certified[accepted]=safe.check_final(ids[accepted])
   diagnostics=[safe.seed_diagnostics(i) for i in range(4)]
   for i,d in enumerate(diagnostics):
    for entry,hist in zip(d,histories[i]):entry.update(hist)
   palm,handquat=safe.palm_reference_pose_w()
   finger=m.quat_apply(handquat,palm.new_tensor((0.,0.,1.)).expand(4,-1))
   thumb=m.quat_apply(handquat,palm.new_tensor((1.,0.,0.)).expand(4,-1))
   case={'cube_x':cube_x,'left_finger_up_deg':tilt,'backoff':backoff,'palm_height':height,
    'certified':certified.cpu().tolist(),'accepted_seed':safe.accepted_seed_index.cpu().tolist(),
    'actual_arm_q':safe.robot.data.joint_pos[:,safe.arm_joint_ids].cpu().tolist(),
    'finger_w':finger.cpu().tolist(),'thumb_w':thumb.cpu().tolist(),'failure':failure,
    'desired_c_local':(safe.desired_c_pos_w-env.scene.env_origins).cpu().tolist(),
    'desired_palm_local':(safe.desired_palm_position_w-env.scene.env_origins).cpu().tolist(),
    'seed_diagnostics':diagnostics,'wall_s':time.perf_counter()-start}
   report['cases'].append(case);save(report)
   print('FAR_OUTWARD',json.dumps({k:case[k] for k in ('cube_x','left_finger_up_deg','backoff','palm_height','certified','accepted_seed','wall_s')}),flush=True)
  report['summary']={'cases':len(report['cases']),'all4_pass':sum(all(c['certified']) for c in report['cases']),
                     'left2_pass':sum(all(c['certified'][2:]) for c in report['cases'])}
  save(report);print('FAR_OUTWARD_SUMMARY',json.dumps(report['summary']),flush=True)
 finally:env.close()
status=0
try:main()
except Exception:traceback.print_exc();status=1
finally:
 import omni.kit.app
 sys.stdout.flush();omni.kit.app.get_app().post_quit(status);app.close()
raise SystemExit(status)
