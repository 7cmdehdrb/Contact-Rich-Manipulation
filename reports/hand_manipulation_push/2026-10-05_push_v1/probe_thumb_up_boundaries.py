"""Actual PhysX-FK/whole-robot OBB certificates over conditional Push-v1 corners."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/hand_manipulation_test'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('thumb_up_boundaries.json'))
parser.add_argument('--table-width', type=float, default=.82)
parser.add_argument('--midpoint-offset', type=float, nargs='+', default=[.31])
parser.add_argument('--palm-height', type=float, nargs='+', default=[.10])
parser.add_argument('--backoff', type=float, nargs='+', default=[.10])
parser.add_argument('--exhaustive', action='store_true')
parser.add_argument('--random-samples', type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args, fast_shutdown=True)
app=launcher.app

def save(report):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp=args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2));tmp.replace(args.output)

def main():
    import gymnasium as gym
    import torch
    import hand_manipulation_test
    import isaaclab.utils.math as m
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from hand_manipulation_test.push_v1_math import push_v1_palm_rotation
    from hand_manipulation_test.mdp import contact_events
    from hand_manipulation_test.geometry import control_point_pose_w
    cfg=load_cfg_from_registry('Isaac-Hand-Manipulation-Push-v1','env_cfg_entry_point')
    cfg.scene.num_envs=32;cfg.scene.env_spacing=.5;cfg.sim.device=args.device
    cfg.commands.target_position.debug_vis=False;cfg.scene.ee_frame.debug_vis=False
    cfg.task.table_size=(args.table_width,1.,.04);cfg.scene.table.spawn.size=cfg.task.table_size
    # Bounds cover every tested conditional draw, while initial env construction
    # does not sample/reset. Each reset subsequently uses validated exact specs.
    span=.5*.3*math.sin(math.pi/18)+.002
    cfg.task.object_xy_offset_low=(min(args.midpoint_offset)-span-.00001,-.03)
    cfg.task.object_xy_offset_high=(max(args.midpoint_offset)+span+.00001,.03)
    cfg.task.command_midpoint_x_offset_m=args.midpoint_offset[0]
    report={'table_size':cfg.task.table_size,'table_center':cfg.task.table_center,
            'seeds':cfg.task.reset_joint_seed_offsets,'max_iterations':cfg.task.reset_max_iterations,
            'clearance_m':cfg.task.reset_clearance_margin_m,'position_tolerance_m':cfg.task.reset_position_tolerance_m,
            'orientation_tolerance_rad':cfg.task.reset_orientation_tolerance_rad,'fixture_env_spacing_m':cfg.scene.env_spacing,'cases':[]}
    env=gym.make('Isaac-Hand-Manipulation-Push-v1',cfg=cfg).unwrapped
    try:
        safe=env.event_manager.get_term_cfg('safe_hand').func
        command=env.command_manager.get_term('target_position')
        ids=torch.arange(32,device=env.device)
        specs=list(itertools.product((-10.,10.,170.,190.),(.2,.3),(-.002,.002),(-.03,.03)))
        angles=torch.tensor([s[0]*math.pi/180 for s in specs],device=env.device)
        lengths=torch.tensor([s[1] for s in specs],device=env.device)
        directions=command._directions(angles,ids)
        offset=torch.zeros(32,3,device=env.device)
        hand_fixture=None
        original_map=contact_events.inspire_synergy_to_joint_positions
        def hand_map(value):
            if hand_fixture is not None:
                value.copy_(hand_fixture)
                safe.initial_hand_synergy[ids]=value
            return original_map(value)
        contact_events.inspire_synergy_to_joint_positions=hand_map
        def sample(index):
            cube=env.scene['target_object'].data.root_pos_w[index]
            direction=command.direction_w[index]
            rot=push_v1_palm_rotation(direction)
            tangent=torch.stack((-direction[:,1],direction[:,0],torch.zeros_like(direction[:,0])),-1)
            up=torch.zeros_like(direction);up[:,2]=1
            palm=cube-offset[index,:1]*direction+offset[index,1:2]*tangent+offset[index,2:]*up
            control=palm+(rot@(safe.c_offset_h-safe.palm_reference_h).unsqueeze(-1)).squeeze(-1)
            local_y=control[:,1]-env.scene.env_origins[index,1]
            if not bool(((local_y>=cfg.task.reset_c_y_range[0])&(local_y<=cfg.task.reset_c_y_range[1])).all()):
                raise RuntimeError('exact fixture initial C outside production reset_c_y_range')
            safe.desired_palm_position_w[index]=palm;safe.desired_c_pos_w[index]=control
            safe.direction_w[index]=rot[:,:,1]
            safe.desired_c_quat_w[index]=m.quat_unique(m.quat_mul(m.quat_from_matrix(rot),safe.c_quat_h.expand(len(index),-1)))
        safe._sample_pose=sample
        def run(midpoint,backoff,height,jitter,hand_value,label):
            nonlocal hand_fixture
            command._specified[:]=False;command._pending[:]=False
            cfg.task.command_midpoint_x_offset_m=midpoint
            position=torch.zeros(32,3,device=env.device)
            position[:,0]=cfg.task.table_center[0]+midpoint-.5*lengths*directions[:,0]+torch.tensor([s[2] for s in specs],device=env.device)
            position[:,1]=torch.tensor([s[3] for s in specs],device=env.device)
            position[:,2]=cfg.task.cube_center_height_m
            offset[:]=offset.new_tensor((backoff,height*0,height))+offset.new_tensor(jitter)
            hand_fixture=None if hand_value is None else position.new_tensor(hand_value).expand(32,-1).clone()
            command.set_reset_specs(ids,position,angles,lengths)
            started=time.perf_counter();failure=None
            try: env.reset(seed=42+len(report['cases']))
            except RuntimeError as exc: failure=str(exc)
            torch.cuda.synchronize()
            accepted=safe.accepted_seed_index>=0
            certified=torch.zeros(32,dtype=torch.bool,device=env.device)
            if bool(accepted.any()):certified[accepted]=safe.check_final(ids[accepted])
            case={'label':label,'midpoint_offset':midpoint,'backoff':backoff,'height':height,'jitter':jitter,'hand':hand_value,
                  'accepted':accepted.cpu().tolist(),'certified':certified.cpu().tolist(),'seed':safe.accepted_seed_index.cpu().tolist(),
                  'iterations':safe.ik_iterations.cpu().tolist(),'position_error_m':safe.position_error_m.cpu().tolist(),
                  'orientation_error_rad':safe.orientation_error_rad.cpu().tolist(),
                  'initial_hand':safe.initial_hand_synergy.cpu().tolist(),'q':safe.robot.data.joint_pos[:,safe.arm_joint_ids].cpu().tolist(),
                  'cube_local':position.cpu().tolist(),'angle_deg':[s[0] for s in specs],'length_m':[s[1] for s in specs],
                  'wall_s':time.perf_counter()-started,'failure':failure,
                  'failed_seeds':{str(i):safe.seed_diagnostics(i) for i in ids[~certified].cpu().tolist()}}
            report['cases'].append(case);save(report)
            print('THUMB_UP_BOUNDARIES',json.dumps({k:case[k] for k in ('label','midpoint_offset','backoff','height','jitter','hand','seed','wall_s')}),
                  'certified',int(certified.sum()),'/32',flush=True)
            return bool(certified.all())
        good=[]
        for midpoint,backoff,height in itertools.product(args.midpoint_offset,args.backoff,args.palm_height):
            if run(midpoint,backoff,height,(0.,0.,0.),None,'nominal_endpoints'):
                good.append((midpoint,backoff,height))
        if args.exhaustive:
            for midpoint,backoff,height in good:
                for jitter in itertools.product((-.004,.004),(-.004,.004),(-.003,.003)):
                    for hand_value in itertools.product((.90,.98),repeat=2):
                        run(midpoint,backoff,height,jitter,hand_value,'jitter_hand_endpoints')
                for i in range(args.random_samples):
                    jitter=tuple((2*torch.rand(3)-1)*torch.tensor((.004,.004,.003)))
                    hand_value=tuple(.9+.08*torch.rand(2))
                    run(midpoint,backoff,height,tuple(float(x) for x in jitter),tuple(float(x) for x in hand_value),'random_jitter_hand')
        report['summary']={'cases':len(report['cases']),'certified_rows':sum(sum(c['certified']) for c in report['cases']),
                           'total_rows':32*len(report['cases']),'all_pass':all(all(c['certified']) for c in report['cases'])}
        save(report);print('THUMB_UP_BOUNDARIES_SUMMARY',json.dumps(report['summary']),flush=True)
    finally:env.close()

status=0
try:main()
except Exception:traceback.print_exc();status=1
finally:
    import omni.kit.app
    sys.stdout.flush();omni.kit.app.get_app().post_quit(status);app.close()
raise SystemExit(status)
