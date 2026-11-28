"""Play one IK-v1 policy left/right without resetting robot or objects between stages."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cli_args
from isaaclab.app import AppLauncher

TASK = 'inspire_sweep_baseline/Inspire-Sweep-Prof-Revision-v0_3_3-IK-v1'
RESET_REFERENCE_TASK = 'Inspire-Sweep-Prof-Revision-v0_3_3'
DEFAULT_CHECKPOINT = 'logs/TEMP/2026-11-28/inspire_sweep_prof_revision_v0_3_3_ik_v1/2026-11-27_19-25-38/model_16750.pt'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,default=Path(DEFAULT_CHECKPOINT))
parser.add_argument('--output-dir',type=Path)
parser.add_argument('--record',action='store_true',help='Save three synchronized videos of the continuous scene.')
parser.add_argument('--width',type=int,default=960)
parser.add_argument('--height',type=int,default=540)
parser.add_argument('--record-every',type=int,default=2,help='Render every N policy steps; 2 gives 25 fps.')
parser.add_argument('--stage-seconds',type=float,default=10.,help='Default preserves the checkpoint 10 s deadline; shorter values are lifecycle tests.')
parser.add_argument('--max-steps',type=int,default=2000)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--real-time',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record_every < 1 or args.max_steps < 1 or not 0 < args.stage_seconds < float('inf'):
    parser.error('record-every, max-steps and stage-seconds must be positive and finite')
if args.width<320 or args.height<180 or args.width%2 or args.height%2:
    parser.error('Recording resolution must be even and at least 320x180')
output = args.output_dir or Path('outputs/two_cylinder_sequence')/datetime.now().strftime('%Y%m%d_%H%M%S')
output.mkdir(parents=True,exist_ok=False)
checkpoint = args.checkpoint.resolve(strict=True)
if args.record:
    args.enable_cameras = True
cli_args.apply_renderer_compatibility(args)
app = AppLauncher(args).app
env = None
writers = {}
report = dict(status='initializing',task=TASK,checkpoint=str(checkpoint),seed=args.seed,
              hand_start_reference_task=RESET_REFERENCE_TASK,
              stage_seconds=args.stage_seconds,record=args.record,
              instruction='left / right; left is shelf -Y as seen from the robot (+X side)',
              transition='no scene/action/controller reset; refresh target, goal, initial-view OBBs and stage distance/contact caches')


def run():
    global env
    import importlib.metadata
    import math
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
    import inspire_sweep_baseline
    from inspire_sweep_baseline.runtime.prof_revision import ProfRevisionSequenceEnv
    from inspire_sweep_baseline.mdp.common.prof_revision_sequence import configure_sequence,START_XY,GOAL_XY,SEQUENCE_COUNT
    from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
    cfg = parse_env_cfg(TASK,device=args.device or 'cuda:0',num_envs=1)
    # Keep the checkpoint task, replacing only its hand-start distribution with
    # the unmodified non-IK variant's reset parameters. Its reset solver remains.
    reference_cfg = load_cfg_from_registry(RESET_REFERENCE_TASK,'env_cfg_entry_point')
    cfg.events.object_spawn.params['ik_params'] = deepcopy(reference_cfg.events.object_spawn.params['ik_params'])
    report['hand_start_params'] = deepcopy(cfg.events.object_spawn.params['ik_params'])
    # Save the precise intentional changes; original Gym task and classes remain untouched.
    cfg = configure_sequence(cfg)
    cfg.seed,cfg.log_dir = args.seed,str(output.resolve())
    cfg.episode_length_s = args.stage_seconds
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = min(cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity,2**20)
    if not math.isclose(round(args.stage_seconds/(cfg.sim.dt*cfg.decimation))*cfg.sim.dt*cfg.decimation,args.stage_seconds,abs_tol=1e-8):
        raise ValueError('stage-seconds must be a multiple of policy dt')
    cameras = dict(high_angle=((.02,-.85,1.85),(-.72,0.,1.09)),
                   rear_eye_level=((-1.6,0.,1.16),(-.1,0.,1.16)),
                   inside_top_down=((-0.7,0.,1.48),(-.7,0.,1.05)))
    if args.record:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        for name in cameras:
            setattr(cfg.scene,name,CameraCfg(prim_path='{ENV_REGEX_NS}/SequenceCamera_'+name,
                update_period=0.,width=args.width,height=args.height,data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18. if name=='high_angle' else 12. if name=='rear_eye_level' else 7.5,
                                                horizontal_aperture=24.,clipping_range=(.01,100.))))
    agent = load_cfg_from_registry(TASK,'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    raw = ProfRevisionSequenceEnv(cfg)
    env = RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
    runner = CurriculumOnPolicyRunner(env,agent.to_dict(),log_dir=None,device=raw.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=raw.device)
    obs = env.get_observations()
    assert obs['policy'].shape == (1,108) and obs['obstacles'].shape == (1,176)
    assert int(raw._active_object_mask.sum()) == SEQUENCE_COUNT
    assert raw._active_object_mask[0,:SEQUENCE_COUNT].all()
    baseline_resets = raw.sequence_physical_resets
    report.update(status='playing',checkpoint_iteration=runner.current_learning_iteration,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  start_xy_shelf_m=START_XY,goal_xy_shelf_m=GOAL_XY,
                  initial_objects_w=raw.scene['object_collection'].data.object_link_state_w[0,:SEQUENCE_COUNT].tolist(),
                  initial_joints=raw.scene['robot'].data.joint_pos[0].tolist(),
                  initial_eef_position_w=raw.scene['ee_frame'].data.target_pos_w[0,0].tolist(),
                  physical_resets_before_play=baseline_resets,dt=raw.step_dt)
    for name,asset in cfg.scene.object_collection.rigid_objects.items():
        if name in list(cfg.scene.object_collection.rigid_objects)[:SEQUENCE_COUNT]:
            assert asset.spawn.kind_override == 'cylinder'
    (output/'config.json').write_text(json.dumps(dict(environment=cfg.to_dict(),agent=agent.to_dict()),default=str,indent=2),encoding='utf-8')
    if args.record:
        import imageio.v2 as imageio
        for name,(eye,target) in cameras.items():
            raw.scene[name].set_world_poses_from_view(torch.tensor([eye],device=raw.device),torch.tensor([target],device=raw.device))
            writers[name] = imageio.get_writer(str(output/f'{name}.mp4'),fps=1/(raw.step_dt*args.record_every),
                codec='libx264',macro_block_size=1,pixelformat='yuv420p')
        # Robot at image bottom; screen-right is shelf +Y, screen-up is shelf -X.
        from isaaclab.utils.math import quat_apply,quat_mul
        def align_top():
            shelf = raw.scene['shelf'].data
            rotation = torch.tensor([[math.sqrt(.5),0.,0.,math.sqrt(.5)]],device=raw.device)
            raw.scene['inside_top_down'].set_world_poses(
                positions=shelf.root_pos_w+quat_apply(shelf.root_quat_w,torch.tensor([[0.,0.,1.48]],device=raw.device)),
                orientations=quat_mul(shelf.root_quat_w,rotation),convention='opengl')
        align_top()
        for _ in range(8):
            raw.sim.render()
        for name in cameras:
            raw.scene[name].update(raw.step_dt,force_recompute=True)
            imageio.imwrite(output/f'{name}_initial.png',raw.scene[name].data.output['rgb'][0,:,:,:3].cpu().numpy())
    trace = {key:[] for key in ('step','stage','stage_after','target_id','action','reward','done',
                                'joint_pos','joint_vel','object_state','goal','distance')}
    frames = 0
    with torch.inference_mode():
        for tick in range(args.max_steps):
            if not app.is_running():
                raise RuntimeError('Simulator closed before sequence completed')
            started = time.perf_counter()
            stage = raw.sequence_stage
            goal = raw.command_manager.get_command('target_goal_pos')[0,:3].clone()
            action = policy(obs)
            obs,reward,done,info = env.step(action)
            assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(action).all()
            assert raw.sequence_physical_resets == baseline_resets, 'Physical reset occurred during sequence'
            values = dict(step=tick,stage=stage+1,stage_after=raw.sequence_stage+1,target_id=stage,
                          action=action[0],reward=reward[0],done=done[0],
                          joint_pos=raw.scene['robot'].data.joint_pos[0],joint_vel=raw.scene['robot'].data.joint_vel[0],
                          object_state=raw.scene['object_collection'].data.object_link_state_w[0,:SEQUENCE_COUNT],goal=goal,
                          distance=(raw.sequence_records[-1]['distance_m'] if bool(done[0])
                                    else raw._stabilize_state.distance[0]))
            for name,value in values.items():
                trace[name].append(value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else value)
            if args.record and ((tick+1)%args.record_every==0 or raw.sequence_finished):
                align_top()
                raw.sim.render()
                for name in cameras:
                    raw.scene[name].update(raw.step_dt,force_recompute=True)
                    rgb=raw.scene[name].data.output['rgb'][0,:,:,:3].cpu().numpy().copy()
                    writers[name].append_data(rgb)
                    if bool(done[0]):
                        imageio.imwrite(output/f'{name}_stage{stage+1}_end.png',rgb)
                frames += 1
            if (tick+1)%100==0:
                print(f'[Sequence] step {tick+1}, stage {raw.sequence_stage+1}, time={(tick+1)*raw.step_dt:.2f}s',flush=True)
            if raw.sequence_finished:
                break
            if args.real_time:
                time.sleep(max(0.,raw.step_dt-(time.perf_counter()-started)))
    np.savez_compressed(output/'trace.npz',**{key:np.asarray(value) for key,value in trace.items()})
    report.update(status='physical_failure' if raw.sequence_failed else 'complete' if raw.sequence_finished else 'step_limit',
                  policy_steps=len(trace['step']),simulation_seconds=len(trace['step'])*raw.step_dt,video_frames=frames,
                  stage_results=raw.sequence_records,transitions=raw.sequence_transitions,
                  physical_resets_during_play=raw.sequence_physical_resets-baseline_resets,
                  final_objects_w=raw.scene['object_collection'].data.object_link_state_w[0,:SEQUENCE_COUNT].tolist(),
                  all_stages_success=len(raw.sequence_records)==SEQUENCE_COUNT and all(x['success'] for x in raw.sequence_records))
    print('TWO_CYLINDER_SEQUENCE_RESULT '+json.dumps({key:report[key] for key in ('status','policy_steps','all_stages_success','physical_resets_during_play')}),flush=True)


try:
    run()
except BaseException as exc:
    import traceback
    traceback.print_exc()
    report.update(status='error',error=repr(exc))
    raise
finally:
    for writer in writers.values():
        writer.close()
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if env is not None:
        env.close()
    app.close()
