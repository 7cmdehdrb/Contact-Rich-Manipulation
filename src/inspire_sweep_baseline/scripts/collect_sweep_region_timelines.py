"""Collect reset-safe first-episode reward, tactile and wrist-wrench timelines.

Uses registered environments and deterministic checkpoint inference. Hooks only
copy already computed reward terms and physical samples; no new RNG draws.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--joint-diagnostics', action='store_true',
                        help='Also copy the physical hand joints used by the velocity reward.')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    if 'logs' in output.parts or min(args.steps, args.num_envs) < 1:
        parser.error('Positive sizes and an output outside logs required')
    output.mkdir(parents=True, exist_ok=False)
    from cli_args import apply_renderer_compatibility
    apply_renderer_compatibility(args)
    app = AppLauncher(args).app
    env = None
    meta = dict(status='initializing', task=args.task, seed=args.seed,
                num_envs=args.num_envs, requested_steps=args.steps,
                checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                policy='deterministic inference; original observation corruption retained',
                sampling='physical/reward: after reward compute, before auto-reset; policy: actual post-step observation, terminal rows invalid',
                force='ON target FROM hand, normal plus friction, mean across both physics substeps, world axes',
                tactile='17 paired states: none=0, palm=1, dorsal=-1, both=2; all contacted bodies, threshold .05 N',
                wrench='gravity/inertia compensated parent-on-child support reaction at EEF, EEF axes; opposite sign to external hand load',
                reward='RewardManager per-term contribution = weighted rate * control dt',
                initial_cohort='episode=0 only; terminal sample included; subsequent episodes excluded')
    rows = {}
    def append(name, value):
        rows.setdefault(name, []).append(value.detach().cpu().numpy().copy())
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import inspire_sweep_baseline  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab.utils.math import quat_apply_inverse
        from isaaclab.utils.io import dump_yaml
        from rsl_rl.runners import OnPolicyRunner
        from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
        from inspire_sweep_baseline.mdp.common.stabilize_sweep_contact import centroid
        from inspire_sweep_baseline.mdp.reward_functions.sweep_contact_preference import sweep_region
        from inspire_sweep_baseline.assets.robot import PALM_SENSOR_BODY_NAMES

        cfg = parse_env_cfg(args.task, device=args.device or 'cuda:0', num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.log_dir = str(output)
        old_capacity = cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = min(old_capacity, 2**21)
        agent = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
        agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
        agent.seed = args.seed
        agent.device = cfg.sim.device
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
        names = list(raw.reward_manager.active_terms)
        terms = [raw.reward_manager.get_term_cfg(name) for name in names]
        meta.update(dt=raw.step_dt, physics_dt=raw.physics_dt, decimation=cfg.decimation,
                    checkpoint_iteration=runner.current_learning_iteration,
                    reward_terms=names, reward_weights=[t.weight for t in terms],
                    reward_callables=[str(type(t.func)) if not hasattr(t.func, '__qualname__') else t.func.__module__+'.'+t.func.__qualname__ for t in terms],
                    reward_params=[t.params for t in terms], tactile_channels=list(PALM_SENSOR_BODY_NAMES),
                    gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    capacity_override=[old_capacity, cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity])
        hand_joint_ids = None
        if args.joint_diagnostics:
            hand_reward = raw.reward_manager.get_term_cfg('hand_joint_velocity')
            hand_joint_ids = hand_reward.params['asset_cfg'].joint_ids
            meta['hand_joint_names'] = [raw.scene['robot'].joint_names[i] for i in hand_joint_ids]
            meta['hand_joint_velocity_units'] = 'rad/s; exact reward input at last physics substep, before reset'
            meta['hand_joint_velocity_weight'] = hand_reward.weight
            meta['hand_joint_velocity_max_cost'] = hand_reward.params['max_cost']
        initial = dict(joint_pos=raw.scene['robot'].data.joint_pos,
                       object_state=raw.scene['object_collection'].data.object_state_w,
                       goal=raw.command_manager.get_command('target_goal_pos'),
                       sweep_dir=raw.sweep_dir, bounds=raw._revision_bounds_t,
                       active_objects=raw._active_object_mask)
        bank = {k: v.detach().cpu().numpy().copy() for k, v in initial.items()}
        np.savez_compressed(output/'initial_scene.npz', **bank)
        meta['initial_hashes'] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in bank.items()}
        episode = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
        initial_regions = sweep_region(quat_apply_inverse(raw.scene['shelf'].data.root_quat_w, raw.sweep_dir)).clone()
        meta['initial_region_counts'] = {str(i): int((initial_regions == i).sum()) for i in range(1,5)}
        # Capture load and normal force at each original physics sample. No
        # additional simulation or reward evaluation is performed.
        contacts = raw._stabilize_contacts
        capture_original = contacts.capture
        loads, normals = [], []
        def capture(source):
            capture_original(source)
            loads.append(contacts.load.clone())
            normals.append(contacts.view.get_contact_force_matrix(dt=raw.physics_dt).sum(1).clone())
        contacts.capture = capture
        surface = raw.reward_manager.get_term_cfg('sweep_progress').func
        surface_original = surface._surface_loads
        surface_values = {}
        def surface_loads(source):
            palm, dorsal = surface_original(source)
            surface_values.update(palm=palm.clone(), dorsal=dorsal.clone())
            return palm, dorsal
        surface._surface_loads = surface_loads
        obs_cfg = raw.observation_manager.cfg.policy
        tactile = obs_cfg.tactile
        wrench = obs_cfg.wrist_wrench.func
        slices = {}
        offset = 0
        for name, shape in zip(raw.observation_manager.active_terms['policy'], raw.observation_manager.group_obs_term_dim['policy']):
            width = int(np.prod(shape))
            slices[name] = slice(offset, offset+width)
            offset += width
        meta['policy_slices'] = {k: [v.start,v.stop] for k,v in slices.items()}
        original_compute = raw.reward_manager.compute
        def compute(dt):
            value = original_compute(dt)
            if len(loads) != cfg.decimation:
                raise RuntimeError('Missing physics contact samples')
            force = contacts.force_sum / contacts.samples
            target = centroid(raw)
            goal = raw.command_manager.get_command('target_goal_pos')[:, :3]
            done = raw.termination_manager.dones.clone()
            data = dict(episode=episode, step=raw.episode_length_buf, region=initial_regions,
                        reward_terms=raw.reward_manager._step_reward*dt, reward_total=value,
                        target=target, goal=goal, force_w=force,
                        normal_w=torch.stack(normals).mean(0), normal_load=torch.stack(loads).mean(0),
                        contact=contacts.contact, contact_valid=contacts.all_valid,
                        sensor_fresh=raw.sensor_data_fresh,
                        tactile_raw=tactile.func(raw, **tactile.params),
                        wrench_eef=wrench(raw), wrench_world=wrench.world_wrench(raw),
                        palm_load=surface_values['palm'], dorsal_load=surface_values['dorsal'],
                        done=done, success=raw.termination_manager.get_term('success'))
            if hand_joint_ids is not None:
                data['hand_joint_velocity'] = raw.scene['robot'].data.joint_vel[:, hand_joint_ids]
                data['hand_joint_position'] = raw.scene['robot'].data.joint_pos[:, hand_joint_ids]
            for name in raw.termination_manager.active_terms:
                data['termination_'+name] = raw.termination_manager.get_term(name)
            for key in ('contact_count','contact_recognized','sweep_angle_xy','sweep_direction_coefficient','sweep_gate','sweep_positive','sweep_direction_negative','reach_delta'):
                if hasattr(raw._stabilize_state, key):
                    data[key] = getattr(raw._stabilize_state,key)
            for key,tensor in data.items():
                append(key,tensor)
            episode.add_(done.long())
            loads.clear(); normals.clear()
            return value
        raw.reward_manager.compute = compute
        started = time.perf_counter()
        for step in range(args.steps):
            with torch.inference_mode():
                obs, reward, done, _ = env.step(policy(obs))
                policy.reset(done)
            append('policy_tactile', obs['policy'][:, slices['tactile']])
            append('policy_wrench', obs['policy'][:, slices['wrist_wrench']])
            append('policy_valid', ~done.bool())
            if (step+1) % 50 == 0:
                print(f'[RegionTrace] {args.task} seed={args.seed} {step+1}/{args.steps} elapsed={time.perf_counter()-started:.1f}s', flush=True)
        if any(len(v) != args.steps for v in rows.values()):
            raise RuntimeError('Trace length mismatch')
        arrays = {k:np.stack(v) for k,v in rows.items()}
        error = float(np.abs(arrays['reward_terms'].sum(-1)-arrays['reward_total']).max())
        if error > 2e-5:
            raise RuntimeError(f'Reward reconciliation failed: {error}')
        np.savez_compressed(output/'trace.npz', **arrays)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise RuntimeError('Source checkpoint changed')
        meta.update(status='complete', steps=args.steps, elapsed_s=time.perf_counter()-started,
                    max_reward_sum_error=error)
    except BaseException as exc:
        import traceback
        meta.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        (output/'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
