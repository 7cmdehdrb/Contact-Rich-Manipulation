"""Read-only Prof Revision policy evaluation, including terminal transitions.

Run with Isaac Lab Python. Original observation noise, rewards and reset remain
active. Diagnostic hooks copy data and call each reward exactly once.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RewardTap:
    def __init__(self, original, storage, index):
        self.original, self.storage, self.index = original, storage, index

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, env, **params):
        value = self.original(env, **params)
        self.storage[:, self.index] = value
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='Inspire-Sweep-Prof-Revision-v0_3')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--steps', type=int, help='Override horizon for a smoke run.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference', action='store_true', help='Save post-step audit without diagnostic hooks.')
    parser.add_argument('--wrench-diagnostics',action='store_true',help='Copy same-step raw/support wrench and all-hand normal loads.')
    parser.add_argument('--impact-diagnostics', action='store_true', help='Record object/hand normal loads and physical wrench at each physics substep.')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    if 'logs' in output.parts or min(args.num_envs, args.episodes) < 1:
        parser.error('Use positive sizes and an output outside logs.')
    output.mkdir(parents=True, exist_ok=False)
    from cli_args import apply_renderer_compatibility
    apply_renderer_compatibility(args)
    app = AppLauncher(args).app
    env = None
    meta = dict(status='initializing', task=args.task, seed=args.seed,
                num_envs=args.num_envs, episodes_per_env=args.episodes,
                checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                inference='deterministic; original observation corruption retained',
                sample_timing='after original reward computation, before automatic reset; terminal step included',
                force_definition='force ON target FROM hand; normal plus friction; world axes; both physics substeps',
                success_definition='original termination manager success; 10s timeout, tracked visible OBB center XY <= .03m, no simultaneous physical failure',
                eligibility='first requested number of COMPLETE episodes per environment; all other rows excluded',
                surface_definition='target-only palm/dorsal normal load from final physics substep; unmapped edges remain unclassified',
                finger_normal_definition='distal pad +Z vs pad-to-live-visible-OBB-center; signed cosine, diagnostic only',
                reference=args.reference,wrench_diagnostics=args.wrench_diagnostics,
                impact_diagnostics=args.impact_diagnostics,
                target_body_force_definition='target-only normal force matrix by HAND_CONTACT_BODY_NAMES; final physics substep; no friction, cancellation within each body pair possible')
    rows, audits = {}, {}
    def append(destination, name, value):
        destination.setdefault(name, []).append(value.detach().cpu().numpy().copy())
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import inspire_sweep_baseline
        from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils.math import quat_apply
        from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
        from rsl_rl.runners import OnPolicyRunner
        from inspire_sweep_baseline.assets.robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES, HAND_CONTACT_BODY_NAMES, PALM_SENSOR_BODY_NAMES
        from inspire_sweep_baseline.assets.hand_kinematics import HAND_DRIVE_JOINT_NAMES
        from inspire_sweep_baseline.mdp.reward_functions.sweep_contact_preference import ContactSideSweepProgress
        from inspire_sweep_baseline.mdp.common.contact_sweep_clutter import all_object_tilt_angles
        from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name

        cfg = parse_env_cfg(args.task, device=args.device or 'cuda:0', num_envs=args.num_envs)
        cfg.seed, cfg.log_dir = args.seed, str(output)
        old_capacity = cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = min(old_capacity, 2**21)
        agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point'), importlib.metadata.version('rsl-rl-lib'))
        agent.seed, agent.device = args.seed, cfg.sim.device
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
        raw = env.unwrapped
        runner_type = {'OnPolicyRunner': OnPolicyRunner, 'CurriculumOnPolicyRunner': CurriculumOnPolicyRunner}[agent.class_name]
        runner = runner_type(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=raw.device)
        env.reset()
        obs = env.get_observations()
        dump_yaml(str(output/'environment.yaml'), raw.cfg)
        dump_yaml(str(output/'agent.yaml'), agent)
        package = Path(inspire_sweep_baseline.__file__).parent
        manifest = {}
        for path in package.rglob('*.py'):
            relative = path.relative_to(package)
            manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            target = output/'source_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(__file__, output/Path(__file__).name)
        (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        names = list(raw.reward_manager.active_terms)
        terms = [raw.reward_manager.get_term_cfg(n) for n in names]
        weights = torch.tensor([t.weight for t in terms], device=raw.device)
        meta.update(dt=raw.step_dt, physics_dt=raw.physics_dt, decimation=cfg.decimation,
                    checkpoint_iteration=runner.current_learning_iteration, reward_terms=names,
                    reward_weights=[t.weight for t in terms], reward_params=[t.params for t in terms],
                    reward_callables=[f'{getattr(t.func,"__module__",type(t.func).__module__)}.{getattr(t.func,"__qualname__",type(t.func).__qualname__)}' for t in terms],
                    termination_terms=list(raw.termination_manager.active_terms),
                    gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    isaacsim_version=importlib.metadata.version('isaacsim'),
                    capacity_override=[old_capacity, cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity],
                    source_files=len(manifest), hand_contact_body_names=list(HAND_CONTACT_BODY_NAMES),
                    tactile_channels=list(PALM_SENSOR_BODY_NAMES), hand_joint_names=list(HAND_JOINT_NAMES))
        initial = dict(joint_pos=raw.scene['robot'].data.joint_pos,
                       object_state=raw.scene['object_collection'].data.object_state_w,
                       shelf_state=raw.scene['shelf'].data.root_state_w,
                       goal=raw.command_manager.get_command('target_goal_pos'),
                       active_objects=raw._active_object_mask, target=raw._obb_live_pos_w)
        initial.update({'observation_'+key: value for key, value in obs.items()})
        np.savez_compressed(output/'initial_scene.npz', **{k:v.detach().cpu().numpy() for k,v in initial.items()})
        episode = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.long)
        robot = raw.scene['robot']
        arm_ids = [robot.joint_names.index(n) for n in ARM_JOINT_NAMES]
        hand_ids = [robot.joint_names.index(n) for n in HAND_JOINT_NAMES]
        drive_ids = [robot.joint_names.index(n) for n in HAND_DRIVE_JOINT_NAMES]
        wrist_id = robot.body_names.index('wrist_3_link')
        eef_body = robot.body_names.index('axia80_link')
        contacts = raw._stabilize_contacts
        target_sensor = raw.scene[object_hand_sensor_name(raw.scene['object_collection'].object_names[0])]
        substeps = []
        impact_substeps = {}
        if not args.reference:
            surface = ContactSideSweepProgress(raw.reward_manager.get_term_cfg('sweep_progress'), raw)
            original_capture = contacts.capture
            def capture(source):
                original_capture(source)
                substeps.append(torch.cat((contacts.force, contacts.load[:,None], contacts.valid[:,None].float()), -1).clone())
                if args.impact_diagnostics:
                    matrices = torch.stack([source.scene[object_hand_sensor_name(name)].data.force_matrix_w[:, 0]
                                            for name in source.scene['object_collection'].object_names], 1)
                    wrench_term = source.observation_manager.cfg.policy.wrist_wrench.func
                    sample = wrench_term._read(source)
                    values = dict(object_hand_normal_vector=matrices.sum(2),
                                  object_hand_normal_load=matrices.norm(dim=-1).sum(2),
                                  whole_hand_normal_load=source.scene['whole_hand_contact'].data.net_forces_w.norm(dim=-1).sum(1),
                                  pure_wrench=sample.pure_world, raw_wrench=sample.raw_world,
                                  support_wrench=sample.support_world)
                    for key, value in values.items():
                        impact_substeps.setdefault(key, []).append(value.clone())
            contacts.capture = capture
            raw_values = torch.zeros((raw.num_envs,len(names)), device=raw.device)
            for i,t in enumerate(terms):
                t.func = RewardTap(t.func, raw_values, i)
            original_compute = raw.reward_manager.compute
            def compute(dt):
                total = original_compute(dt)
                state, data = raw._stabilize_state, robot.data
                objects, shelf = raw.scene['object_collection'].data, raw.scene['shelf'].data
                frame = raw.scene['ee_frame'].data
                eef = frame.target_pos_w[:,0]
                offset = eef-data.body_link_pos_w[:,eef_body]
                eef_velocity = data.body_link_lin_vel_w[:,eef_body]+torch.cross(data.body_link_ang_vel_w[:,eef_body],offset,dim=-1)
                finger_quat = data.body_link_quat_w[:,raw._finger_body_ids]
                z = torch.zeros_like(raw._finger_pos_w); z[...,2]=1.
                normals = quat_apply(finger_quat,z)
                toward = raw._obb_live_pos_w[:,None]-raw._finger_pos_w
                cosine = (normals*toward).sum(-1)/toward.norm(dim=-1).clamp_min(1e-12)
                palm, dorsal = surface._surface_loads(raw)
                wrench = raw.observation_manager.cfg.policy.wrist_wrench.func
                tactile = raw.observation_manager.cfg.policy.tactile
                if len(substeps)!=cfg.decimation:
                    raise RuntimeError('Incomplete substep sample')
                records = dict(episode=episode, step=raw.episode_length_buf,
                    reward_raw=raw_values, reward_rate=raw.reward_manager._step_reward,
                    reward_terms=raw_values*weights*dt, reward_total=total,
                    target=raw._obb_live_pos_w, target_link=objects.object_link_pos_w[:,0],
                    goal=raw.command_manager.get_command('target_goal_pos'), sweep_direction=raw.sweep_dir,
                    target_size=raw._obb_target_size, target_quaternion=raw._obb_live_quat_w,
                    object_positions=objects.object_pos_w, object_velocity=objects.object_lin_vel_w,
                    object_tilt=all_object_tilt_angles(raw), active_objects=raw._active_object_mask,
                    initial_object_positions=raw._initial_object_positions_w,
                    eef=eef, eef_velocity=eef_velocity,
                    wrist_velocity=data.body_link_lin_vel_w[:,wrist_id],
                    arm_joint_velocity=data.joint_vel[:,arm_ids], hand_joint_velocity=data.joint_vel[:,hand_ids],
                    hand_joint_position=data.joint_pos[:,hand_ids], hand_drive_position=data.joint_pos[:,drive_ids],
                    finger_position=raw._finger_pos_w, finger_gap=raw._finger_gaps,
                    finger_normal_cosine=cosine, formation=raw._formation,
                    force=contacts.force_sum/contacts.samples, contact=contacts.contact,
                    contact_valid=contacts.all_valid, contact_recognized=state.contact_recognized,
                    contact_substeps=torch.stack(substeps,1), palm_load=palm, dorsal_load=dorsal,
                    target_normal_by_body=target_sensor.data.force_matrix_w[:,0],
                    tactile=tactile.func(raw, **tactile.params), wrench=wrench(raw),
                    distance=state.distance, positive=state.positive, retreat=state.retreat,
                    alignment=state.alignment, reach_distance=state.reach_distance,
                    reach_excess=state.reach_excess, reach_delta=state.reach_delta,
                    action=raw.action_manager.action, previous_action=raw.action_manager.prev_action,
                    shelf_position=shelf.root_pos_w, shelf_velocity=shelf.root_vel_w,
                    shelf_initial_position=raw._revision_initial_shelf_pos,
                    sensor_fresh=raw.sensor_data_fresh, done=raw.termination_manager.dones)
                for name in raw.termination_manager.active_terms:
                    records['termination_'+name]=raw.termination_manager.get_term(name)
                if args.wrench_diagnostics:
                    sample=wrench._read(raw)
                    records.update(wrench_raw_world=sample.raw_world,wrench_support_world=sample.support_world,
                        wrench_pure_world=sample.pure_world,
                        hand_net_normal=raw.scene['whole_hand_contact'].data.net_forces_w.sum(1),
                        hand_normal_by_body=raw.scene['whole_hand_contact'].data.net_forces_w,
                        target_normal=target_sensor.data.force_matrix_w[:,0].sum(1))
                if args.impact_diagnostics:
                    for key, values in impact_substeps.items():
                        if len(values) != cfg.decimation:
                            raise RuntimeError('Incomplete impact substep sample')
                        records['substep_' + key] = torch.stack(values, 1)
                    impact_substeps.clear()
                for key,value in records.items():
                    append(rows,key,value)
                substeps.clear()
                return total
            raw.reward_manager.compute=compute
        steps = args.steps or raw.max_episode_length*args.episodes
        started=time.perf_counter()
        for step in range(steps):
            with torch.inference_mode():
                obs, reward, done, _ = env.step(policy(obs))
                policy.reset(done)
            if step < 20:
                audit=dict(reward=reward,done=done,joint_pos=robot.data.joint_pos,
                           object_pos=raw.scene['object_collection'].data.object_pos_w)
                audit.update({'observation_'+k:v for k,v in obs.items()})
                for k,v in audit.items(): append(audits,k,v)
            episode.add_(done.long())
            if (step+1)%100==0:
                print(f'[ProfPlay] seed={args.seed} steps={step+1}/{steps} completed={int(episode.clamp_max(args.episodes).sum())} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        np.savez_compressed(output/'audit.npz', **{k:np.stack(v) for k,v in audits.items()})
        if rows:
            arrays={k:np.stack(v) for k,v in rows.items()}
            error=float(np.abs(arrays['reward_terms'].sum(-1)-arrays['reward_total']).max())
            rate_error=float(np.abs(arrays['reward_rate']*raw.step_dt-arrays['reward_terms']).max())
            if error>2e-5 or rate_error>2e-5: raise RuntimeError(f'Reward mismatch: {error}, {rate_error}')
            np.savez_compressed(output/'trace.npz', **arrays)
            meta.update(max_reward_sum_error=error,max_rate_error=rate_error)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=meta['checkpoint_sha256']:
            raise RuntimeError('Source checkpoint changed')
        meta.update(status='complete',steps=steps,elapsed_s=time.perf_counter()-started,
                    completed_per_env=episode.cpu().tolist())
    except BaseException as exc:
        import traceback
        meta.update(status='failed', error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        (output/'metadata.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
        if env is not None: env.close()
        app.close()


if __name__=='__main__':
    main()
