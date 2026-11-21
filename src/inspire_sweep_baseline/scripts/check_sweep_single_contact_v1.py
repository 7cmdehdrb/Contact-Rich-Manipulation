"""Check the complete v0/v1 contract and live manager/reset behavior on GPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('outputs/sweep_single_contact_v1'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def validate():
    import gymnasium as gym
    import numpy as np
    import torch
    import inspire_sweep_baseline  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from inspire_sweep_baseline.mdp.common.stabilize_sweep_contact import centroid
    from inspire_sweep_baseline.mdp.common.sweep_v0_state import SingleContactV1SweepState
    from inspire_sweep_baseline.mdp.reward_functions.stabilize_sweep import incremental_approach_reach, arm_joint_failure
    from inspire_sweep_baseline.mdp.reward_functions.sweep_contact_preference import SignedContactSideSweepProgress
    from reward_timeline_probe import RewardTimelineProbe

    parent_id, task = 'Inspire-Sweep-Single-Contact-v0', 'Inspire-Sweep-Single-Contact-v1'
    parent = load_cfg_from_registry(parent_id, 'env_cfg_entry_point')
    before = parent.to_dict()
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    assert type(cfg).__bases__ == (type(parent),)
    assert cfg.rewards.sweep_progress.func is SignedContactSideSweepProgress
    assert cfg.rewards.reaching.func is incremental_approach_reach
    assert cfg.rewards.reaching.weight == 4.9 and cfg.rewards.reaching.params == {}
    assert cfg.rewards.arm_joint_failure.func is arm_joint_failure and cfg.rewards.arm_joint_failure.weight == -3.
    after = cfg.to_dict()
    for name in ('sweep_progress', 'reaching'):
        after['rewards'][name] = before['rewards'][name]
    del after['rewards']['arm_joint_failure']
    assert after == before, 'Unrequested parent/child configuration difference'
    assert parent.to_dict() == before
    assert load_cfg_from_registry(parent_id, 'env_cfg_entry_point').to_dict() == before
    runner = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    parent_runner = load_cfg_from_registry(parent_id, 'rsl_rl_cfg_entry_point')
    assert type(runner).__bases__ == (type(parent_runner),)
    rp, rc = parent_runner.to_dict(), runner.to_dict()
    assert rc.pop('experiment_name') == 'inspire_sweep_single_contact_v1'
    rp.pop('experiment_name')
    assert rc == rp
    assert gym.spec(task).entry_point.endswith(':SweepSingleContactV1Env')
    cfg.scene.num_envs, cfg.seed = 4, 42
    cfg.debug_surface_points = cfg.scene.ee_frame.debug_vis = cfg.commands.target_goal_pos.debug_vis = False
    args.output.mkdir(parents=True, exist_ok=True)
    wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=runner.clip_actions)
    env = wrapped.unwrapped
    result = dict(task=task, parent=parent_id, configuration='only three requested reward differences',
                  ppo='experiment name only', num_envs=4, seed=42)
    rows = {}
    try:
        observations, _ = env.reset()
        state = env._stabilize_state
        assert type(state) is SingleContactV1SweepState
        assert not hasattr(state, 'contact_count')
        parent_terms = [name for name, term in vars(parent.rewards).items() if term is not None]
        assert env.reward_manager.active_terms == parent_terms + ['arm_joint_failure']
        assert len(env.reward_manager.active_terms) == 16
        assert not env.curriculum_manager.active_terms
        probe = RewardTimelineProbe(env, rows, args.output, result)
        probe.save_initial(observations)
        assert state.reach_radius.gt(0).all()
        initial = state.reach_excess.clone()
        action = torch.zeros(4, 12, device=env.device)
        for tick in range(8):
            old = state.reach_excess.clone()
            observations, reward, done, _ = wrapped.step(action)
            assert not done.any()
            assert observations['policy'].shape == (4, 66) and observations['obstacles'].shape == (4, 64)
            assert torch.isfinite(observations['policy']).all() and torch.isfinite(reward).all()
            expected = ((env.scene['ee_frame'].data.target_pos_w[:, 0]
                         - centroid(env) - state.reach_offset).norm(dim=-1) - .05).clamp_min(0.)
            torch.testing.assert_close(state.reach_excess, expected, atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(state.reach_delta, old - expected, atol=1e-7, rtol=1e-5)
            index = env.reward_manager.active_terms.index('reaching')
            torch.testing.assert_close(env.reward_manager._step_reward[:, index] * env.step_dt, 4.9 * state.reach_delta)
            torch.testing.assert_close(reward, env.reward_manager._step_reward.sum(-1) * env.step_dt)
            cached = incremental_approach_reach(env).clone()
            torch.testing.assert_close(incremental_approach_reach(env), cached)
        assert tick == 7
        torch.testing.assert_close(torch.as_tensor(np.stack(rows['reach_delta']).sum(0), device=env.device),
                                   initial - state.reach_excess, atol=1e-7, rtol=1e-5)
        other = {name: value[1:].clone() for name, value in vars(state).items() if isinstance(value, torch.Tensor)}
        env.reset(env_ids=torch.tensor([0], device=env.device))
        for name, value in other.items():
            torch.testing.assert_close(getattr(state, name)[1:], value)
        # Inject a termination-manager event, not synthetic velocities, to check
        # exact once-only cost through RewardManager and the automatic reset.
        termination = env.termination_manager.get_term_cfg('arm_joint_velocity')
        original = termination.func
        termination.func = lambda env, **kwargs: torch.tensor([True, False, False, False], device=env.device)
        before_terminal = state.reach_excess.clone()
        observations, reward, done, _ = wrapped.step(action)
        assert done.tolist() == [True, False, False, False]
        joint = env.reward_manager.active_terms.index('arm_joint_failure')
        assert rows['reward_contribution'][-1][0, joint] == -3.
        assert rows['joint_failure_event'][-1].tolist() == [True, False, False, False]
        np.testing.assert_allclose(rows['reach_previous_excess'][-1], before_terminal.cpu().numpy(), atol=1e-7)
        np.testing.assert_allclose(rows['reach_net'][-1],
                                   4.9 * (rows['reach_previous_excess'][-1] - rows['reach_excess'][-1]), atol=1e-7)
        assert state.reach_delta[0].item() == 0. and state.reach_step[0].item() == -1
        termination.func = original
        # Normal timeout also auto-resets; no joint-failure cost is charged.
        env.episode_length_buf[0] = env.max_episode_length - 1
        wrapped.step(action)
        assert env.reset_time_outs[0] and not env.reset_terminated[0]
        assert rows['reward_contribution'][-1][0, joint] == 0.
        wrapped.step(action)
        assert rows['reward_contribution'][-1][:, joint].tolist() == [0., 0., 0., 0.]
        trace = {k: np.stack(v) for k, v in rows.items()}
        for name, values in trace.items():
            assert np.isfinite(values).all(), name
        names = env.reward_manager.active_terms
        np.testing.assert_allclose(trace['reach_net'], trace['reward_contribution'][:, :, names.index('reaching')], atol=1e-6)
        np.testing.assert_allclose(trace['sweep_positive'] + trace['sweep_direction_negative'],
                                   trace['reward_contribution'][:, :, names.index('sweep_progress')], atol=1e-6)
        np.savez_compressed(args.output / 'trace.npz', **trace)
        result.update(status='passed', device=str(env.device), steps=11, partial_reset='passed',
                      terminal_joint_event='-3 once', normal_timeout='no joint cost',
                      first_step_and_telescoping='passed', parent_reinstantiation='unchanged')
    finally:
        wrapped.close()
    (args.output / 'smoke.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


try:
    validate()
finally:
    app.close()
