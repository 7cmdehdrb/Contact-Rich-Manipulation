"""Free-space PhysX cancellation and known-load checks for Relative-v0.

Move scene clutter away only in this diagnostic process, then drive real
physics without task resets. Inspect UNCLIPPED residuals, so observation caps
cannot hide a mass/inertia cancellation error.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
env = None
report = {'status':'running'}

def stats(values):
    values = torch.stack(values)
    return {'samples':len(values),
            'force_max_norm_N':float(values[...,:3].norm(dim=-1).max()),
            'moment_max_norm_Nm':float(values[...,3:].norm(dim=-1).max())}

try:
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorRelativeEnvCfg
    cfg = WrenchRefactorRelativeEnvCfg()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, args.device or 'cuda:0', 42
    cfg.observations.policy.enable_corruption = False
    with torch.inference_mode():
        env = gym.make('Inspire-Wrench-Refactor-Relative-v0', cfg=cfg).unwrapped
        obs, _ = env.reset(seed=42)
        assert not obs['policy'][:,-6:].any()
        robot = env.scene['robot']
        term = env.observation_manager.cfg.policy.wrist_wrench
        shelf = env.scene['shelf']
        state = shelf.data.root_state_w.clone()
        state[:,:3] = env.scene.env_origins + state.new_tensor([0.,1000.,0.]); state[:,7:] = 0
        shelf.write_root_state_to_sim(state)
        objects = env.scene['object_collection']
        state = objects.data.object_link_state_w.clone()
        state[...,:3] = env.scene.env_origins[:,None] + state.new_tensor([0.,2000.,2.])
        state[...,0] += torch.arange(objects.num_objects, device=env.device)[None] * .2
        state[...,7:] = 0
        objects.write_object_link_state_to_sim(state)
        action = torch.zeros(4,8,device=env.device); action[:,-2:] = 1
        child = robot.body_names.index('inspire_base_link')

        def advance(force=None, moment=None):
            env.action_manager.process_action(action)
            for _ in range(cfg.decimation):
                env._sim_step_counter += 1
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                if force is not None:
                    forces = torch.zeros((4,robot.num_bodies,3),device=env.device)
                    moments = torch.zeros_like(forces)
                    forces[:,child], moments[:,child] = force, moment
                    positions = robot.data.body_com_pos_w.clone()
                    advance.application_pos = positions[:,child].clone()
                    robot.root_physx_view.apply_forces_and_torques_at_position(
                        force_data=forces.reshape(-1,3), torque_data=moments.reshape(-1,3),
                        position_data=positions.reshape(-1,3), indices=robot._ALL_INDICES, is_global=True)
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
            env.common_step_counter += 1
            env.sensor_data_fresh[:] = True
            sample = term.func._read(env)
            # The live observation manager must use the same compensated sample.
            expected = sample.pure_eef.clamp(min=term.clip[0],max=term.clip[1])
            torch.testing.assert_close(env.observation_manager.compute_group('policy')[:,-6:],expected)
            return sample

        for _ in range(50): advance()
        raw, residual = [], []
        for _ in range(40):
            s = advance(); raw.append(s.raw_world.clone()); residual.append(s.pure_world.clone())
        report['static_raw'], report['static_residual'] = stats(raw), stats(residual)
        reader = term.func._reader
        assert robot.body_names.index('axia80_link') not in reader.body_ids
        report['hand_mass_kg'] = float(reader.masses[0].sum())
        report['sensor_mass_kg'] = float(robot.root_physx_view.get_masses()[0,robot.body_names.index('axia80_link')])
        residual, speeds = [], []
        for i in range(160):
            phase = i * .04
            action[:,0] = .15 * math.sin(phase)
            action[:,1] = .1 * math.cos(phase)
            action[:,2] = -.1 * math.sin(phase)
            action[:,3] = .15 * math.sin(phase)
            action[:,5] = -.15 * math.sin(phase)
            action[:,6] = .55 + .4 * math.cos(phase)
            action[:,7] = .60 + .3 * math.cos(phase)
            residual.append(advance().pure_world.clone())
            speeds.append(robot.data.joint_vel.abs().max())
        report['moving_residual'], report['max_joint_speed_rad_s'] = stats(residual),float(torch.stack(speeds).max())
        action.zero_(); action[:,-2:] = 1
        for _ in range(60): advance()
        force = torch.tensor([[5.,0.,0.],[0.,5.,0.],[0.,0.,5.],[2.,-3.,4.]],device=env.device)
        moment = torch.tensor([[0.,.2,0.],[.15,0.,0.],[0.,0.,.18],[.1,-.12,.08]],device=env.device)
        errors = []
        for i in range(70):
            s = advance(force,moment)
            eef = env.scene['ee_frame'].data.target_pos_w[:,0]
            expected = -torch.cat((force,moment+torch.cross(advance.application_pos-eef,force,dim=-1)),dim=-1)
            if i >= 30: errors.append(s.pure_world-expected)
        report['known_external_error'] = stats(errors)
        env.sensor_data_fresh[:] = False
        assert not term.func(env).any()
        for name, force_tol, moment_tol in [('static_residual',.15,.02),('moving_residual',.5,.05),('known_external_error',.15,.02)]:
            assert report[name]['force_max_norm_N'] < force_tol, (name,report[name])
            assert report[name]['moment_max_norm_Nm'] < moment_tol, (name,report[name])
        assert report['static_raw']['force_max_norm_N'] > .5
        report['status'] = 'passed'
finally:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if report['status'] == 'running': report['status'] = 'failed'
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if env is not None: env.close()
    app.close()
