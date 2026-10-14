from pathlib import Path
import argparse, json, math, sys, traceback
sys.path.insert(0, '/home/min/7cmdehdrb/grad/src/hand_manipulation_test')
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--task',default='Isaac-Hand-Manipulation-Push-v0')
parser.add_argument('--random-resets',type=int,default=10)
parser.add_argument('--tilt-grid',action='store_true')
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args,fast_shutdown=True)
app=launcher.app

def main():
    import gymnasium as gym
    import torch
    import hand_manipulation_test
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    cfg=load_cfg_from_registry(args.task,'env_cfg_entry_point')
    cfg.scene.num_envs=4
    cfg.sim.device=args.device
    cfg.seed=42
    cfg.commands.target_position.debug_vis=False
    cfg.scene.ee_frame.debug_vis=False
    env=gym.make(args.task,cfg=cfg).unwrapped
    results={'task':args.task,'center':cfg.task.table_center,'cube_low':cfg.task.object_xy_range_low,
             'cube_high':cfg.task.object_xy_range_high,'cases':[], 'seed_offsets':cfg.task.reset_joint_seed_offsets}
    try:
        safe=env.event_manager.get_term_cfg('safe_hand').func
        ids=torch.arange(env.num_envs,device=env.device)
        command=env.command_manager.get_term('target_position')
        cube=env.scene['target_object']
        table=env.scene['table']
        robot=env.scene['robot']
        selected_position=None
        if 'Contact' in args.task:
            def place_cube(env,env_ids):
                env_ids=ids if env_ids is None else env_ids
                state=cube.data.default_root_state[env_ids].clone()
                if selected_position is None:
                    low=state.new_tensor(cfg.task.object_xy_range_low)
                    high=state.new_tensor(cfg.task.object_xy_range_high)
                    state[:,:2]=low+torch.rand((len(env_ids),2),device=env.device)*(high-low)
                    state[:,2]=cfg.task.cube_center_height_m
                else:
                    state[:,:3]=selected_position[env_ids]
                state[:,:3]+=env.scene.env_origins[env_ids]
                state[:,3:7]=state.new_tensor((1.,0.,0.,0.))
                state[:,7:]=0.
                cube.write_root_state_to_sim(state,env_ids=env_ids)
            env.event_manager.get_term_cfg('reset_cube').func=place_cube
        def check(label):
            nonlocal selected_position
            try:
                obs,_=env.reset()
            except RuntimeError as error:
                results['cases'].append({'case':label,'passed':False,'error':str(error)})
                print('PROBE_FAIL '+json.dumps(results['cases'][-1]),flush=True)
                return False
            assert bool(safe.check_final(ids).all())
            torch.testing.assert_close(table.data.root_pos_w-env.scene.env_origins,
                torch.tensor(cfg.task.table_center,device=env.device).expand(env.num_envs,-1),atol=2e-6,rtol=0.)
            case={'case':label,'passed':True,'accepted_seed':safe.accepted_seed_index.tolist(),
                  'iterations':safe.ik_iterations.tolist(),'position_error_m':safe.position_error_m.tolist(),
                  'orientation_error_rad':safe.orientation_error_rad.tolist(),
                  'finger_outward_cos':safe._finger_outward_cos(ids).tolist(),
                  'arm_q':robot.data.joint_pos[:,safe.arm_joint_ids].tolist(),
                  'seed_diagnostics':[safe.seed_diagnostics(i) for i in range(env.num_envs)]}
            results['cases'].append(case)
            print('PROBE_PASS '+json.dumps({k:v for k,v in case.items() if k not in ('arm_q','seed_diagnostics')}),flush=True)
            return True
        low,high=cfg.task.object_xy_range_low,cfg.task.object_xy_range_high
        positions=torch.tensor([(x,y,cfg.task.cube_center_height_m) for x in (low[0],high[0]) for y in (low[1],high[1])],device=env.device)
        if args.tilt_grid:
            import hand_manipulation_test.push_math as geometry
            canonical=geometry.push_palm_rotation
            for roll_degrees in (0.,30.,45.,60.):
                for height in (.03,.06,.08,.10,.13):
                    def tilted(direction, elevation=math.radians(roll_degrees)):
                        result=canonical(direction)
                        y=result[..., :,1]
                        z=result[..., :,2]
                        up=torch.zeros_like(z);up[...,2]=1.
                        lifted=math.cos(elevation)*z+math.sin(elevation)*up
                        z=torch.where((y[...,1]<0)[...,None],lifted,z)
                        x=torch.linalg.cross(y,z,dim=-1)
                        return torch.stack((x,y,z),dim=-1)
                    geometry.push_palm_rotation=tilted
                    cfg.task.reset_position_offset_task=(.14,0.,height)
                    command.set_reset_specs(ids,positions,math.pi,cfg.task.command_distance_range[1])
                    check(f'tilt/{roll_degrees}/height/{height}')
            results['tilt_grid']=True
            print('TILT_GRID_DONE',flush=True)
            return
        if 'Push' in args.task:
            for degrees in (-10.,0.,10.,170.,180.,190.):
                command.set_reset_specs(ids,positions,math.radians(degrees),cfg.task.command_distance_range[1])
                check(f'corners/{degrees}')
        else:
            selected_position=positions
            check('contact/corners')
        selected_position=None
        for index in range(args.random_resets):
            check(f'random/{index}')
        results['passed']=all(case['passed'] for case in results['cases'])
        if results['passed']:
            action=torch.zeros((env.num_envs,8),device=env.device)
            hand=env.action_manager.get_term('hand_action')
            action[:,6:]=hand.synergy_to_action(hand.actual_synergy)
            counters={key:value.clone() for key,value in {'ik':safe.ik_iterations,'obb':safe.obb_checks}.items()}
            for _ in range(50):
                _,_,terminated,truncated,_=env.step(action)
                assert not bool((terminated|truncated).any())
            torch.testing.assert_close(safe.ik_iterations,counters['ik'])
            torch.testing.assert_close(safe.obb_checks,counters['obb'])
            results['hold_50_steps']=True
        print('WORKSPACE_PROBE '+json.dumps(results),flush=True)
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        env.close()
    assert results['passed'],'At least one initial pose failed'

status=0
try: main()
except Exception:
    traceback.print_exc();status=1
finally:
    import omni.kit.app
    sys.stdout.flush();sys.stderr.flush()
    omni.kit.app.get_app().post_quit(status)
    app.close()
raise SystemExit(status)
