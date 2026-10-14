"""Far outward-flat reset and real palmar-thumb OSC contact, original physics."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/hand_manipulation_test'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pad',type=int,nargs='+',default=[4,3,2])
parser.add_argument('--contact-height',type=float,nargs='+',default=[.025])
parser.add_argument('--direct-impedance',action='store_true')
parser.add_argument('--implicit-impedance',action='store_true')
parser.add_argument('--measured-external-wrench',action='store_true')
parser.add_argument('--live-pad-anchor',action='store_true')
parser.add_argument('--settle-steps',type=int,default=80)
parser.add_argument('--advance-step',type=float,default=.0008)
parser.add_argument('--pad-exposure',type=float,default=0.)
parser.add_argument('--steps',type=int,default=350)
parser.add_argument('--output',type=Path,default=Path(__file__).with_name('far_thumb_contact.json'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();launcher=AppLauncher(args,fast_shutdown=True);app=launcher.app

def save(r):
 tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(args.output)

def main():
 import gymnasium as gym
 import torch
 import hand_manipulation_test
 import isaaclab.utils.math as m
 from isaaclab.sim.utils import get_current_stage
 from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
 from hand_manipulation_test.mdp.push_events import PushSafePoseReset
 from hand_manipulation_test.contact_math import palm_facing_rotation
 from hand_manipulation_test.collision_geometry import _points_in_frame
 from hand_manipulation_test.geometry import control_point_pose_w
 cfg=load_cfg_from_registry('Isaac-Hand-Manipulation-Push-v1','env_cfg_entry_point')
 cfg.scene.num_envs=4;cfg.scene.env_spacing=.5;cfg.sim.device=args.device;cfg.seed=42
 if args.direct_impedance:cfg.actions.arm_action.inertial_dynamics_decoupling=False
 cfg.task.table_size=(.36,1.,.04);cfg.scene.table.spawn.size=cfg.task.table_size
 cfg.task.object_xy_offset_low=(.045,-.03);cfg.task.object_xy_offset_high=(.055,.03)
 cfg.task.command_centered_path=False;cfg.task.initial_hand_open_range=(.94,.94)
 cfg.task.reset_position_offset_task=(.14,0.,.10);cfg.task.reset_position_jitter_task=(0.,0.,0.)
 cfg.events.safe_hand.func=PushSafePoseReset
 cfg.commands.target_position.debug_vis=False;cfg.scene.ee_frame.debug_vis=False
 from isaaclab.sensors import ContactSensorCfg
 from hand_manipulation_test.assets.robot import ARM_BODY_NAMES,HAND_LINK_BODY_NAMES,ROBOT_CONTACT_BODY_NAMES,PALM_SENSOR_BODY_NAMES
 carrier_names=(*ARM_BODY_NAMES,'axia80_link',*HAND_LINK_BODY_NAMES)
 cfg.scene.cube_carrier_contacts=ContactSensorCfg(
    prim_path='{ENV_REGEX_NS}/TargetObject',history_length=2,update_period=0.,
    filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+n for n in carrier_names],
    max_contact_data_count_per_prim=512,debug_vis=False)
 if args.pad_exposure:
  from pxr import Gf,UsdGeom
  from isaaclab.sim.utils import clone
  from hand_manipulation_test.assets.contact_robot import spawn_contact_robot
  @clone
  def shifted_spawn(path,spawner,translation=None,orientation=None,**kwargs):
   root=spawn_contact_robot.__wrapped__(path,spawner,translation=translation,orientation=orientation,**kwargs)
   stage=get_current_stage();cache=UsdGeom.XformCache();root_path=str(root.GetPath())
   normal=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root_path+'/inspire_base_link')).TransformDir(Gf.Vec3d(0,1,0))
   for n in ('inspire_thumb_force_sensor_3','inspire_thumb_force_sensor_4'):
    collision=stage.GetPrimAtPath(root_path+'/'+n+'/collisions')
    world=cache.GetLocalToWorldTransform(collision);world.SetTranslateOnly(world.ExtractTranslation()+args.pad_exposure*normal)
    local=world*cache.GetLocalToWorldTransform(collision.GetParent()).GetInverse()
    xf=UsdGeom.Xformable(collision);xf.ClearXformOpOrder();xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local)
   return root
  cfg.scene.robot.spawn.func=shifted_spawn
 report={'table_center':cfg.task.table_center,'table_size':cfg.task.table_size,'cube_x':-.70,
         'direct_impedance':args.direct_impedance,'implicit_impedance':args.implicit_impedance,'measured_external_wrench':args.measured_external_wrench,'live_pad_anchor':args.live_pad_anchor,'stiffness':cfg.actions.arm_action.motion_stiffness,'pad_collision_exposure_m':args.pad_exposure,'carrier_names':carrier_names,'robot_table_contact_names':ROBOT_CONTACT_BODY_NAMES,'physical_mass_kg':cfg.task.cube_mass,'hand':(.94,.94),'trials':[]}
 env=gym.make('Isaac-Hand-Manipulation-Push-v1',cfg=cfg).unwrapped
 try:
  safe=env.event_manager.get_term_cfg('safe_hand').func;command=env.command_manager.get_term('target_position')
  arm=env.action_manager.get_term('arm_action');hand=env.action_manager.get_term('hand_action')
  robot=env.scene['robot'];cube=env.scene['target_object'];ids=torch.arange(4,device=env.device)
  if args.implicit_impedance:
   from hand_manipulation_test.impedance_math import implicit_cartesian_impedance_efforts
   stiffness=torch.full((6,),200.,device=env.device)
   damping=torch.ones(6,device=env.device)
   pad_ids=torch.tensor([robot.body_names.index(n) for n in PALM_SENSOR_BODY_NAMES],device=env.device)
   pad_centers_b=safe.collision_bounds.centers[pad_ids]
   external_ema=torch.zeros(4,6,device=env.device)
   previous_episode_length=torch.full((4,),-1,device=env.device,dtype=torch.long)
   def implicit_compute(*,jacobian_b,current_ee_pose_b,current_ee_vel_b,mass_matrix,gravity,**kwargs):
    pose_error=torch.cat(m.compute_pose_error(current_ee_pose_b[:,:3],current_ee_pose_b[:,3:],
                         arm._osc.desired_ee_pose_b[:,:3],arm._osc.desired_ee_pose_b[:,3:],rot_error_type='axis_angle'),-1)
    arm._probe_pose_error_b=pose_error
    external=None
    if args.measured_external_wrench:
     rootq=robot.data.root_link_quat_w
     control_w=robot.data.root_link_pos_w+m.quat_apply(rootq,current_ee_pose_b[:,:3])
     points_w=robot.data.body_link_pos_w[:,pad_ids]+m.quat_apply(robot.data.body_link_quat_w[:,pad_ids],pad_centers_b.expand(4,-1,-1))
     # Sensor force is applied to Cube; the robot receives the equal opposite reaction.
     reactions_w=-env.scene['cube_palm_contacts'].data.force_matrix_w[:,0]
     force_w=reactions_w.sum(1)
     moment_w=torch.cross(points_w-control_w[:,None],reactions_w,dim=-1).sum(1)
     force_w*=torch.minimum(force_w.new_ones((4,1)),20./force_w.norm(dim=-1,keepdim=True).clamp_min(1.e-8))
     moment_w*=torch.minimum(moment_w.new_ones((4,1)),2./moment_w.norm(dim=-1,keepdim=True).clamp_min(1.e-8))
     measured=torch.cat((m.quat_apply_inverse(rootq,force_w),m.quat_apply_inverse(rootq,moment_w)),-1)
     new_episode=env.episode_length_buf<previous_episode_length
     external_ema[new_episode]=0.
     external_ema.mul_(.5).add_(measured,alpha=.5)
     previous_episode_length.copy_(env.episode_length_buf)
     external=external_ema
    arm._probe_external_wrench_b=external_ema.clone()
    return implicit_cartesian_impedance_efforts(jacobian_b,mass_matrix,pose_error,current_ee_vel_b,
                stiffness,damping,cfg.sim.dt,gravity if cfg.actions.arm_action.gravity_compensation else None,external_wrench=external)
   arm._osc.compute=implicit_compute
  angles=torch.tensor((0.,0.,math.pi,math.pi),device=env.device)
  positions=torch.tensor([(-.70,-.03,1.08),(-.70,.03,1.08)]*2,device=env.device)
  selected_point_h=None
  def reset():
   command.set_reset_specs(ids,positions,angles,.20);env.reset(seed=42)
  reset()
  stage=get_current_stage();root=env.scene.env_prim_paths[0]+'/Robot'
  hpos=robot.data.body_link_pos_w[:,safe.hand_body_id];hq=robot.data.body_link_quat_w[:,safe.hand_body_id]
  geometry={}
  for name in ['inspire_palm_force_sensor']+[f'inspire_thumb_force_sensor_{i}' for i in range(1,5)]:
   body=stage.GetPrimAtPath(root+'/'+name);points=_points_in_frame(stage,body,body,mesh_vertices=True)
   local=torch.tensor([[float(v[0]),float(v[1]),float(v[2])] for v in points],device=env.device)
   body_id=robot.body_names.index(name)
   world=robot.data.body_link_pos_w[:,body_id,None]+m.quat_apply(robot.data.body_link_quat_w[:,body_id,None].expand(-1,len(local),-1),local.expand(4,-1,-1))
   h=m.quat_apply_inverse(hq[:,None].expand(-1,len(local),-1),world-hpos[:,None])
   # Actual mesh supporting surface along the shared H+Y palmar normal.
   projection=h[:,:,1];front=projection.amax(-1,keepdim=True)
   mask=projection>=front-.0005
   face_h=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
   face_b=(local[None]*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
   geometry[name]={'face_b':face_b.cpu().tolist(),'face_h':face_h.cpu().tolist(),'min_h':h.amin(1).cpu().tolist(),'max_h':h.amax(1).cpu().tolist(),
                   'face_local_w':(hpos+m.quat_apply(hq,face_h)-env.scene.env_origins).cpu().tolist()}
  report['geometry']=geometry
  report['initial_q']=robot.data.joint_pos[:,safe.arm_joint_ids].cpu().tolist()
  report['initial_certified']=safe.check_final(ids).cpu().tolist()
  print('FAR_THUMB_GEOMETRY',json.dumps(geometry),flush=True);save(report)
  original_record=command._record_metrics
  for pad in args.pad:
   for contact_height in args.contact_height:
    reset();initial=cube.data.root_pos_w.clone();direction=command.direction_w.clone()
    if args.implicit_impedance:external_ema.zero_();previous_episode_length.fill_(-1)
    selected_body_id=robot.body_names.index(f'inspire_thumb_force_sensor_{pad}')
    selected_face_b=torch.tensor(geometry[f'inspire_thumb_force_sensor_{pad}']['face_b'],device=env.device)
    hpos=robot.data.body_link_pos_w[:,safe.hand_body_id];hq=robot.data.body_link_quat_w[:,safe.hand_body_id]
    selected=torch.tensor(geometry[f'inspire_thumb_force_sensor_{pad}']['face_h'],device=env.device)
    selected[:2]=safe.palm_reference_h
    target_ref_c=arm.reference_c_quat_b.clone()
    target_ref_w=m.quat_mul(robot.data.root_link_quat_w,target_ref_c)
    h_ref=m.quat_mul(target_ref_w,m.quat_conjugate(safe.c_quat_h.expand(4,-1)))
    advance=torch.zeros(4,device=env.device);peak_forward=advance.clone();raw_peaks=torch.zeros(4,17,device=env.device)
    carrier_peaks=torch.zeros(4,len(carrier_names),device=env.device)
    alive=torch.ones(4,dtype=torch.bool,device=env.device);contacts=torch.zeros(4,dtype=torch.long,device=env.device)
    terminal=[None]*4;trace=[];initial_counters=safe.ik_iterations.clone()
    def record(state):
     original_record(state)
     done=state.failure|state.success
     for i in ids[done&alive].cpu().tolist():
      terminal[i]={'step':step,'failure':bool(state.failure[i]),'success':bool(state.success[i]),
                   'forward_m':float(((cube.data.root_pos_w[i]-initial[i])*direction[i]).sum()),
                   'cube_local':(cube.data.root_pos_w[i]-env.scene.env_origins[i]).cpu().tolist(),
                   'q':robot.data.joint_pos[i,safe.arm_joint_ids].cpu().tolist(),
                   'joint_names':robot.joint_names,'joint_pos_all':robot.data.joint_pos[i].cpu().tolist(),
                   'table_failure':bool(state.table_failure[i]),'footprint_failure':bool(state.footprint_failure[i]),
                   'fall_failure':bool(state.fall_failure[i]),'invalid_state':bool(state.invalid_state[i]),
                   'additional_failure':bool(state.additional_failure[i]),
                   'height_failure':bool(command._height_failure[i]),'finger_failure':bool(command._finger_failure[i]),
                   'wrist_failure':bool(command._wrist_failure[i]),
                   'robot_table_force_n':env.scene['table_contacts'].data.force_matrix_w_history[i,:,0].norm(dim=-1).amax(0).cpu().tolist(),
                   'central_palm_local':(safe.palm_reference_pose_w()[0][i]-env.scene.env_origins[i]).cpu().tolist(),
                   'actual_synergy':hand.actual_synergy[i].cpu().tolist(),
                   'pose_error_b':None if not args.implicit_impedance else arm._probe_pose_error_b[i].cpu().tolist(),
                   'external_wrench_b':None if not args.implicit_impedance else arm._probe_external_wrench_b[i].cpu().tolist()}

    command._record_metrics=record
    try:
     for step in range(args.steps):
      advance+=args.advance_step*(alive&(step>=args.settle_steps)).float()
      target_pad=initial-(.14-advance[:,None])*direction
      target_pad[:,2]+=target_pad.new_tensor((.025,.025,contact_height,contact_height))
      if args.live_pad_anchor:
       actual_face=robot.data.body_link_pos_w[:,selected_body_id]+m.quat_apply(robot.data.body_link_quat_w[:,selected_body_id],selected_face_b)
       live_h=m.quat_apply_inverse(robot.data.body_link_quat_w[:,safe.hand_body_id],actual_face-robot.data.body_link_pos_w[:,safe.hand_body_id])
       selected[2:]=live_h[2:]
      desired_c=target_pad+m.quat_apply(h_ref,(safe.c_offset_h-selected).expand(4,-1))
      persistent=robot.data.root_link_pos_w+m.quat_apply(robot.data.root_link_quat_w,arm.desired_c_pose_b[:,:3])
      _,current_q=control_point_pose_w(env)
      action=torch.zeros(4,8,device=env.device)
      action[:,:3]=m.quat_apply_inverse(current_q,desired_c-persistent)/action.new_tensor(cfg.actions.arm_action.translation_scale)
      action[:,6:]=hand.synergy_to_action(action.new_full((4,2),.94))
      action[~alive,:6]=0
      _,_,terminated,truncated,_=env.step(action.clamp(-1,1))
      state=command.state();history=env.scene['cube_palm_contacts'].data.force_matrix_w_history
      peaks=history[:,:,0].norm(dim=-1).amax(1)
      raw_peaks=torch.maximum(raw_peaks,torch.where(alive[:,None],peaks,torch.zeros_like(peaks)))
      carrier_now=env.scene['cube_carrier_contacts'].data.force_matrix_w_history[:,:,0].norm(dim=-1).amax(1)
      carrier_peaks=torch.maximum(carrier_peaks,torch.where(alive[:,None],carrier_now,torch.zeros_like(carrier_now)))
      forward=((cube.data.root_pos_w-initial)*direction).sum(-1)
      peak_forward=torch.maximum(peak_forward,torch.where(alive&~(terminated|truncated),forward,torch.zeros_like(forward)))
      contacts+= (alive&state.valid_push_contact).long()
      if step%10==0 or bool((terminated|truncated).any()):
       padbody=robot.body_names.index(f'inspire_thumb_force_sensor_{pad}')
       localface=torch.tensor(geometry[f'inspire_thumb_force_sensor_{pad}']['face_b'],device=env.device)
       actualface=robot.data.body_link_pos_w[:,padbody]+m.quat_apply(robot.data.body_link_quat_w[:,padbody],localface)
       trace.append({'carrier_forces':carrier_now.cpu().tolist(),'actual_pad_face_local':(actualface-env.scene.env_origins).cpu().tolist(),'actual_synergy':hand.actual_synergy.cpu().tolist(),
                     'arm_q':robot.data.joint_pos[:,safe.arm_joint_ids].cpu().tolist(),'palm_normal_cos':state.palm_alignment_cos.cpu().tolist(),'step':step,'alive':alive.cpu().tolist(),'cube_local':(cube.data.root_pos_w-env.scene.env_origins).cpu().tolist(),
                     'central_palm_local':(safe.palm_reference_pose_w()[0]-env.scene.env_origins).cpu().tolist(),
                     'force17':peaks.cpu().tolist(),'valid_contact':state.valid_push_contact.cpu().tolist(),
                     'robot_table_force_n':env.scene['table_contacts'].data.force_matrix_w_history[:,:,0].norm(dim=-1).amax(1).cpu().tolist(),
                     'grounded':state.grounded.cpu().tolist(),'forward_m':forward.cpu().tolist(),
                     'target_error':arm.position_target_error_c_m.cpu().tolist(),
                     'probe_pose_error_b':None if not args.implicit_impedance else arm._probe_pose_error_b.cpu().tolist(),
                     'external_wrench_b':None if not args.implicit_impedance else arm._probe_external_wrench_b.cpu().tolist()})
      alive&=~(terminated|truncated)
      if not bool(alive.any()):break
     trial={'carrier_peak_n':carrier_peaks.cpu().tolist(),'pad':pad,'contact_height':contact_height,'peak_force17':raw_peaks.cpu().tolist(),'valid_contact_steps':contacts.cpu().tolist(),
            'max_forward_m':peak_forward.cpu().tolist(),'terminal':terminal,'trace':trace,
            'normalstep_ik_unchanged':bool((safe.ik_iterations[alive]==initial_counters[alive]).all())}
     report['trials'].append(trial);save(report)
     print('FAR_THUMB_CONTACT',json.dumps({k:trial[k] for k in ('pad','contact_height','carrier_peak_n','peak_force17','valid_contact_steps','max_forward_m','terminal')}),flush=True)
    finally:command._record_metrics=original_record
 finally:env.close()
status=0
try:main()
except Exception:traceback.print_exc();status=1
finally:
 import omni.kit.app
 sys.stdout.flush();omni.kit.app.get_app().post_quit(status);app.close()
raise SystemExit(status)
