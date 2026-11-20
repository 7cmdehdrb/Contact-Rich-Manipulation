"""Check all six derived configurations and one task's real wrapper/reset/step path.

Writes no training logs. Run with the Isaac Lab Python interpreter --headless.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Inspire-Sweep-Only-v1-Test-ActionPenalty')
parser.add_argument('--output', type=Path, default=Path('/tmp/sweep_only_action_penalty.json'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def validate():
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from inspire_sweep_baseline.assets.hand_kinematics import HAND_DRIVE_UPPER_LIMITS
    from inspire_sweep_baseline.mdp.reward_functions.stabilize_sweep import action_limit_excess
    from inspire_sweep_baseline.mdp.reward_functions.sweep_simple import finger_action_rate_l2
    from inspire_sweep_baseline.mdp.observation_functions.stabilize_sweep import legacy_clipped_previous_action

    result = {'configuration_pairs': [], 'task': args.task}
    for suffix in ('v0', 'v0-Test', 'v1-Test', 'v1_2-Test', 'v2-Test', 'v2_2-Test'):
        parent_id = f'Inspire-Sweep-Only-{suffix}'
        child_id = parent_id + '-ActionPenalty'
        parent = load_cfg_from_registry(parent_id, 'env_cfg_entry_point')
        before = parent.to_dict()
        child = load_cfg_from_registry(child_id, 'env_cfg_entry_point')
        assert type(child).__bases__ == (type(parent),)
        after = child.to_dict()
        # Exhaustive comparison: only these four leaves may differ.
        after['actions']['arm_action']['class_type'] = before['actions']['arm_action']['class_type']
        after['observations']['policy']['actions']['func'] = before['observations']['policy']['actions']['func']
        for group in ('arm', 'hand'):
            term = after['rewards'].pop(f'{group}_action_limit')
            assert term['weight'] == -.03 and term['params'] == {'action_group': group}
            assert getattr(child.rewards, f'{group}_action_limit').func is action_limit_excess
        assert before == after, child_id
        assert parent.to_dict() == before
        assert load_cfg_from_registry(parent_id, 'env_cfg_entry_point').to_dict() == before
        parent_runner = load_cfg_from_registry(parent_id, 'rsl_rl_cfg_entry_point')
        runner = load_cfg_from_registry(child_id, 'rsl_rl_cfg_entry_point')
        assert type(runner).__bases__ == (type(parent_runner),)
        rp, rc = parent_runner.to_dict(), runner.to_dict()
        assert rc.pop('clip_actions') is None and rp.pop('clip_actions') == math.pi / 2
        assert rc.pop('experiment_name') == rp.pop('experiment_name') + '_action_penalty'
        assert rc == rp
        assert gym.spec(parent_id).entry_point == gym.spec(child_id).entry_point
        result['configuration_pairs'].append(child_id)

    cfg = load_cfg_from_registry(args.task, 'env_cfg_entry_point')
    runner = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs = 2
    cfg.seed = 42
    cfg.debug_surface_points = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.commands.target_goal_pos.debug_vis = False
    env = gym.make(args.task, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner.clip_actions)
    base = env.unwrapped
    try:
        manager = base.action_manager
        arm = manager.get_term('arm_action')
        hand = manager.get_term('gripper_action')
        upper = torch.tensor(HAND_DRIVE_UPPER_LIMITS, device=base.device)
        previous = torch.full((2, 12), -2., device=base.device)
        raw = torch.stack([torch.full((12,), -200., device=base.device),
                           torch.full((12,), 200., device=base.device)])
        original = raw.clone()
        # Huge commands exercise only processing/costs, without advancing physics.
        manager.process_action(previous)
        manager.process_action(raw)
        torch.testing.assert_close(raw, original)
        torch.testing.assert_close(manager.action, raw)
        torch.testing.assert_close(manager.prev_action, previous)
        torch.testing.assert_close(arm.raw_actions, raw[:, :6].clamp(-math.pi / 2, math.pi / 2))
        torch.testing.assert_close(hand.processed_actions, raw[:, 6:].clamp_min(0).minimum(upper))
        for group, sl in (('arm', slice(0, 6)), ('hand', slice(6, 12))):
            torch.testing.assert_close(finger_action_rate_l2(base, group),
                                       (raw[:, sl] - previous[:, sl]).square().sum(-1))
            assert (action_limit_excess(base, group) > 0).all()
        torch.testing.assert_close(legacy_clipped_previous_action(base), raw.clamp(-math.pi / 2, math.pi / 2))

        ids = torch.tensor([0], device=base.device)
        base.reset(env_ids=ids)
        torch.testing.assert_close(manager.action[0], torch.zeros(12, device=base.device))
        torch.testing.assert_close(manager.prev_action[0], torch.zeros(12, device=base.device))
        torch.testing.assert_close(manager.action[1], raw[1])
        torch.testing.assert_close(manager.prev_action[1], previous[1])
        base.reset()

        command = torch.zeros(2, 12, device=base.device)
        command[:, 6:] = upper * 1.1
        for _ in range(3):
            obs, reward, dones, _ = wrapped.step(command)
            assert not dones.any(), 'Unexpected early reset during action-path check'
            assert obs['policy'].shape == (2, 66) and obs['obstacles'].shape == (2, 64)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            torch.testing.assert_close(manager.action, command)
            torch.testing.assert_close(hand.processed_actions, upper.expand(2, -1))
            torch.testing.assert_close(action_limit_excess(base, 'hand'), torch.full((2,), .01, device=base.device))
            index = base.reward_manager.active_terms.index('hand_action_limit')
            # _step_reward stores weighted rates; total reward multiplies by step_dt.
            torch.testing.assert_close(base.reward_manager._step_reward[:, index],
                                       torch.full((2,), -.0003, device=base.device))
        result['real_wrapper_steps'] = 3
        result['raw_history_execution_bounds_partial_reset'] = 'passed'
    finally:
        wrapped.close()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


try:
    validate()
finally:
    app.close()
