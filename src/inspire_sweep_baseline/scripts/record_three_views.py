"""Synchronized three-camera checkpoint playback, independent of play.py."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--task', default='Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0',
                        choices=['Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0'])
    parser.add_argument('--num_envs', type=int, choices=[1], default=1)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--real-time', action='store_true')
    parser.add_argument('--preview-only', action='store_true', help='Save one synchronized PNG per camera, no videos.')
    parser.add_argument('--seed', type=int, default=42)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    import math
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.width < 720 or args.height < 480:
        parser.error('duration must be positive and finite; minimum resolution is 720x480')
    if args.width * 9 != args.height * 16:
        parser.error('camera output must use a 16:9 aspect ratio')
    if args.width % 2 or args.height % 2:
        parser.error('H.264 requires even image dimensions')
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir or Path('outputs/three_view_recordings') / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    writers = []
    report = dict(status='initializing', checkpoint=str(checkpoint), task=args.task, frames=0,
                  requested_seconds=args.seconds, seed=args.seed, resolution=[args.width,args.height])
    try:
        import csv
        import importlib.metadata
        import imageio.v2 as imageio
        import numpy as np
        import torch
        import gymnasium as gym
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        import inspire_sweep_baseline  # noqa: F401
        from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
        cfg = parse_env_cfg(args.task, device=args.device or 'cuda:0', num_envs=1)
        cfg.seed = args.seed
        cfg.log_dir = str(output.resolve())
        # World coordinates: shelf centered at x=-0.7, work surface z=1.05;
        # robot is on the +X side. Cameras are independent render products.
        views = {
            'high_angle': ((0.0,-0.95,1.95),(-0.78,0.03,1.08)),
            'rear_eye_level': ((-1.60,0.,1.16),(-0.10,0.,1.16)),
            'inside_top_down': ((-0.72,0.,1.48),(-0.72,0.,1.05)),
        }
        focal_lengths = {'high_angle':18., 'rear_eye_level':12., 'inside_top_down':7.5}
        for name in views:
            setattr(cfg.scene, name, CameraCfg(
                prim_path='{ENV_REGEX_NS}/RecordCamera_'+name, update_period=0.,
                width=args.width, height=args.height, data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=focal_lengths[name], horizontal_aperture=24.,
                                               clipping_range=(0.01,100.))))
        agent = load_cfg_from_registry(args.task,'rsl_rl_cfg_entry_point')
        agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
        agent.seed = args.seed
        agent.device = cfg.sim.device
        env = RslRlVecEnvWrapper(gym.make(args.task,cfg=cfg),clip_actions=agent.clip_actions)
        runner = CurriculumOnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        env.reset()
        raw = env.unwrapped
        dt = raw.step_dt
        steps = round(args.seconds/dt)
        if steps < 1 or not math.isclose(steps*dt,args.seconds,abs_tol=1e-8):
            raise ValueError('--seconds must be a positive multiple of control step dt')
        for name,(eye,target) in views.items():
            raw.scene[name].set_world_poses_from_view(
                torch.tensor([eye],device=raw.device),torch.tensor([target],device=raw.device))
        from isaaclab.utils.math import quat_mul, quat_apply
        roll = torch.tensor([[math.sqrt(.5),0.,0.,-math.sqrt(.5)]],device=raw.device)
        def align_top_camera():
            # Follow the shelf frame, including any actual physics tilt/yaw.
            # Camera +X maps to shelf -Y; +Y maps to shelf +X; -Z looks down.
            shelf = raw.scene['shelf'].data
            offset = torch.tensor([[0.,0.,1.48]],device=raw.device)
            raw.scene['inside_top_down'].set_world_poses(
                positions=shelf.root_pos_w + quat_apply(shelf.root_quat_w,offset),
                orientations=quat_mul(shelf.root_quat_w,roll),convention='opengl')
        align_top_camera()
        report.update(views=views,top_down_alignment='shelf frame: image right=-Y, up=+X',
                      focal_lengths_mm=focal_lengths,preview_only=args.preview_only)
        # Warm render products without advancing physics or video time.
        for _ in range(8): raw.sim.render()
        if args.preview_only:
            with torch.inference_mode():
                env.step(policy(env.get_observations()))
            align_top_camera()
            for _ in range(8): raw.sim.render()
            for name in views:
                camera = raw.scene[name]
                camera.update(dt,force_recompute=True)
                imageio.imwrite(output/f'{name}.png',camera.data.output['rgb'][0,:,:,:3].cpu().numpy())
            report.update(status='complete',frames=1,simulation_time_s=dt)
            return
        for name in views:
            writers.append(imageio.get_writer(str(output/f'{name}.mp4'),fps=1/dt,
                           codec='libx264',macro_block_size=1,pixelformat='yuv420p'))
        report.update(status='recording', dt=dt, fps=1/dt, expected_frames=steps,
                      views=views,checkpoint_iteration=runner.current_learning_iteration,
                      sampling='post-step, after auto-reset if done; all cameras same physics state')
        obs = env.get_observations()
        with (output/'frames.csv').open('w',newline='') as stream:
            timeline=csv.writer(stream)
            timeline.writerow(['frame','video_time_s','simulation_time_s','episode','reset_after_step'])
            episode=0
            for frame in range(steps):
                if not app.is_running(): raise RuntimeError('Simulator closed before recording completed')
                start=time.perf_counter()
                with torch.inference_mode():
                    obs,_,done,_=env.step(policy(obs))
                    policy.reset(done)
                reset=bool(done[0].item())
                episode += int(reset)
                align_top_camera()
                # One shared render; reading cameras never advances simulation.
                raw.sim.render()
                images=[]
                for name in views:
                    camera=raw.scene[name]
                    camera.update(dt,force_recompute=True)
                    rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy()
                    if rgb.shape != (args.height,args.width,3) or rgb.dtype != np.uint8:
                        raise RuntimeError(f'Unexpected RGB format: {name} {rgb.shape} {rgb.dtype}')
                    images.append(rgb.copy())
                for writer,rgb in zip(writers,images): writer.append_data(rgb)
                timeline.writerow([frame,frame*dt,(frame+1)*dt,episode,int(reset)])
                report['frames']=frame+1
                if (frame+1)%100==0:
                    stream.flush()
                    print(f'[Record] {frame+1}/{steps} frames per camera; {(frame+1)*dt:.2f}s',flush=True)
                if args.real_time: time.sleep(max(0,dt-(time.perf_counter()-start)))
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        try:
            for writer in writers: writer.close()
        finally:
            (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
            if env is not None: env.close()
            app.close()


if __name__ == '__main__':
    main()
