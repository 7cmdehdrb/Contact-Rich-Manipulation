"""Collect checkpoint evaluation data without changing training/play environments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli_args


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='Inspire-Sweep-Simple-Finger-v0')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=128)
    parser.add_argument('--episodes-per-env', type=int, default=20,
                        help='Log the first N complete episodes of EACH env; no direction quotas.')
    parser.add_argument('--max-steps', type=int, help='Optional policy-step cap; marks unfinished runs partial.')
    parser.add_argument('--chunk-steps', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--agent', default='rsl_rl_cfg_entry_point')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.episodes_per_env, args.chunk_steps) < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('Environment/episode/chunk counts and --max-steps must be positive')
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (args.output_dir or Path('outputs/sweep_evaluation') / args.task / (checkpoint.stem + '_' + stamp)).resolve()
    if any(p.lower() == 'logs' for p in output.parts) or output == checkpoint.parent or checkpoint.parent in output.parents:
        parser.error('Use an output directory outside logs and the checkpoint directory')
    output.mkdir(parents=True, exist_ok=False)
    raw = output / 'raw'
    raw.mkdir()
    report = {
        'schema_version': 1, 'status': 'running', 'task': args.task, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': file_hash(checkpoint), 'seed': args.seed,
        'num_envs': args.num_envs, 'episodes_per_env': args.episodes_per_env,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
        'sampling': 'one sample after reward, before reset, per policy step; last physics substep',
        'position': 'target object root, same point as target_goal_pos (not BBOX/COM)',
        'velocity': 'world-axis COM linear velocity, m/s; speed is rotation invariant',
        'frame': 'shelf pose frozen at each episode start; meters; quaternions wxyz',
        'cohort': 'first N episodes per env, including early failures; no direction resampling',
        'control_steps_completed': 0,
    }
    try:
        report['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        report['git_status'] = subprocess.check_output(['git', 'status', '--short'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report['git_commit'] = None
    metadata_path = output / 'metadata.json'

    def save_report():
        temporary = metadata_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(metadata_path)

    save_report()
    app = env = probe = None
    try:
        cli_args.apply_renderer_compatibility(args)
        app = AppLauncher(args).app
        import gymnasium as gym
        import torch
        from packaging.version import Version
        from rsl_rl.runners import OnPolicyRunner, DistillationRunner
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
        from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg,
                                        handle_deprecated_rsl_rl_checkpoint)
        import inspire_sweep_baseline  # noqa: F401
        from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
        from inspire_sweep_baseline.runtime.sweep_evaluation import SweepEvaluationProbe

        rsl_version = importlib.metadata.version('rsl-rl-lib')
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, args.agent), rsl_version)
        cfg.seed = agent.seed = args.seed
        if args.device is not None:
            agent.device = args.device
        cfg.log_dir = str(output)
        # Keep the registered EnvCfg/Managers; no play-only debug overlays/exports.
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
        runners = {'OnPolicyRunner': OnPolicyRunner, 'CurriculumOnPolicyRunner': CurriculumOnPolicyRunner,
                   'DistillationRunner': DistillationRunner}
        runner = runners[agent.class_name](env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(handle_deprecated_rsl_rl_checkpoint(str(checkpoint), rsl_version))
        # Apply the loaded curriculum before the first logged reset/spawn.
        obs, _ = env.reset()
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if Version(rsl_version) >= Version('4.0.0'):
            reset_policy = policy.reset
        else:
            network = runner.alg.policy if Version(rsl_version) >= Version('2.3.0') else runner.alg.actor_critic
            reset_policy = network.reset
        dump_yaml(str(output / 'env.yaml'), cfg)
        dump_yaml(str(output / 'agent.yaml'), agent)
        probe = SweepEvaluationProbe(env.unwrapped, raw, args.episodes_per_env, args.chunk_steps)
        report.update(control_dt_s=env.unwrapped.step_dt, physics_dt_s=env.unwrapped.physics_dt,
                      max_episode_length=env.unwrapped.max_episode_length,
                      checkpoint_iteration=runner.current_learning_iteration,
                      contact_definition=probe.contact_kind, contact_params=probe.contact_cfg.params,
                      goal_radius_m=getattr(cfg, 'goal_success_radius', None), rsl_rl_version=rsl_version,
                      torch_version=torch.__version__, runtime_class=type(env.unwrapped).__name__)
        save_report()
        print(f'[Evaluation] Saving to {output}', flush=True)
        with torch.inference_mode():
            while app.is_running() and not probe.finished:
                if args.max_steps is not None and report['control_steps_completed'] >= args.max_steps:
                    break
                probe.prepare_step()
                obs, _, dones, _ = env.step(policy(obs))
                reset_policy(dones)
                report['control_steps_completed'] += 1
                if report['control_steps_completed'] % args.chunk_steps == 0:
                    report['completed_episodes_per_env'] = probe.completed.cpu().tolist()
                    save_report()
                    print(f"[Evaluation] step={report['control_steps_completed']} "
                          f"episodes={int(probe.completed.sum())}/{args.num_envs * args.episodes_per_env}", flush=True)
        report['status'] = 'complete' if probe.finished else 'partial'
        if file_hash(checkpoint) != report['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during collection')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        try:
            if probe is not None:
                probe.close()
                report.update(completed_episodes_per_env=probe.completed.cpu().tolist(),
                              raw_samples=probe.samples, raw_chunks=probe.chunks)
        finally:
            report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            save_report()
            print(f"[Evaluation] {report['status']}: {output}", flush=True)
            if env is not None:
                env.close()
            if app is not None:
                app.close()


if __name__ == '__main__':
    main()
