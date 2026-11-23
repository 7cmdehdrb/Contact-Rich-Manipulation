"""Live contract, frozen observation, rigid tracking, reset and export checks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('outputs/prof_revision_validation/smoke.json'))
parser.add_argument('--num-envs', type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def validate():
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from inspire_sweep_baseline.mdp.common.prof_revision_geometry import update_geometry
    from inspire_sweep_baseline.mdp.observation_functions.prof_revision import target_obb_in_eef
    task = 'Inspire-Sweep-Prof-Revision-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.scene.num_envs, cfg.seed = args.num_envs, 42
    assert type(cfg).__bases__ == (ManagerBasedRLEnvCfg,)
    runner_cfg = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from importlib.metadata import version
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, version('rsl-rl-lib'))
    parent = load_cfg_from_registry('Inspire-Sweep-Single-Contact-v0', 'env_cfg_entry_point')
    a, b = parent.to_dict(), cfg.to_dict()
    # Compare function/params/weights, not just final numeric rewards.
    for name in ('sweep_progress', 'sweep_retreat', 'reaching'):
        b['rewards'][name] = a['rewards'][name]
    del b['rewards']['contact_formation']
    assert b['rewards'] == {k:v for k,v in a['rewards'].items() if v is not None}, 'reward contract drift'
    assert b['actions'] == a['actions']
    assert b['commands'] == a['commands']
    for name in ('object_fall_or_tilt','shelf_failure','arm_joint_velocity','timeout'):
        assert b['terminations'][name] == a['terminations'][name]
    for name in ('object_geometry','object_physics'):
        assert b['events'][name] == a['events'][name]
    assert b['events']['object_spawn']['params'] == a['events']['object_spawn']['params']
    assert b['sim'] == a['sim']
    a['scene']['num_envs'] = b['scene']['num_envs']
    assert b['scene'] == a['scene'], 'scene/physics/asset contract drift'
    parent_runner = load_cfg_from_registry('Inspire-Sweep-Single-Contact-v0', 'rsl_rl_cfg_entry_point')
    parent_runner = handle_deprecated_rsl_rl_cfg(parent_runner, version('rsl-rl-lib'))
    ap, bp = parent_runner.to_dict(), runner_cfg.to_dict()
    for key in ('actor','critic'):
        for field in ('class_name','encoder_hidden_dims'):
            bp[key][field] = ap[key][field]
    bp['experiment_name'] = ap['experiment_name']
    assert bp == ap, 'PPO contract drift'
    cfg.scene.ee_frame.debug_vis = cfg.commands.target_goal_pos.debug_vis = False
    wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=runner_cfg.clip_actions)
    env = wrapped.unwrapped
    assert type(env).__bases__ == (ManagerBasedRLEnv,)
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (args.num_envs, 108)
        assert obs['obstacles'].shape == (args.num_envs, 176)
        expected_terms = ['joint_pos','joint_vel','ee_pose','hand_configuration','finger_poses',
                          'target_obs_state','goal_pos','actions','tactile','wrist_wrench']
        expected_dims = [6,6,7,6,35,10,3,12,17,6]
        assert env.observation_manager.active_terms['policy'] == expected_terms
        assert env.observation_manager.group_obs_term_dim['policy'] == [(v,) for v in expected_dims]
        from inspire_sweep_baseline.assets.hand_kinematics import HAND_DRIVE_JOINT_NAMES
        hand_ids = [env.scene['robot'].joint_names.index(name) for name in HAND_DRIVE_JOINT_NAMES]
        torch.testing.assert_close(obs['policy'][:,19:25], env.scene['robot'].data.joint_pos[:,hand_ids])
        torch.testing.assert_close(obs['policy'][:,25:60], env._finger_pose_e.flatten(1))
        for boxes in (obs['policy'][:,60:70], obs['obstacles'].reshape(args.num_envs,16,11)[env._obb_valid][:,1:]):
            torch.testing.assert_close(boxes[:,6:].norm(dim=-1), torch.ones_like(boxes[:,0]))
        assert runner_cfg.actor.encoder_hidden_dims == [128,64,32]
        assert len(env.reward_manager.active_terms) == 16
        assert not env.curriculum_manager.active_terms
        assert env._obb_target_size.gt(0).all()
        snapshot = env._obb_target_pos_w.clone()
        finger_frame_snapshot = env._finger_pose_e[0].detach().cpu().tolist()
        action = torch.zeros(args.num_envs, 12, device=env.device)
        rewards = []
        for tick in range(12):
            old = env._stabilize_state.reach_excess.clone()
            obs, reward, done, _ = wrapped.step(action)
            assert not done.any()
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(reward).all()
            from isaaclab.utils.math import quat_apply
            robot = env.scene['robot'].data
            expected_fingers = robot.body_link_pos_w[:,env._finger_body_ids] + quat_apply(
                robot.body_link_quat_w[:,env._finger_body_ids], env._finger_offsets.expand(args.num_envs,-1,-1))
            torch.testing.assert_close(env._finger_pos_w, expected_fingers)
            torch.testing.assert_close(env._stabilize_state.reach_delta, old-env._stabilize_state.reach_excess)
            torch.testing.assert_close(env._obb_target_pos_w, snapshot)
            assert env._formation.ge(0).all() and env._formation.le(1).all()
            torch.testing.assert_close(reward, env.reward_manager._step_reward.sum(-1)*env.step_dt)
            rewards.append(float(reward.mean()))
        # Moving only the object must not alter the Actor OBB snapshot.
        before = target_obb_in_eef(env).clone()
        states = env.scene['object_collection'].data.object_link_state_w.clone()
        states[:, 0, 0] += .01
        env.scene['object_collection'].write_object_link_state_to_sim(states)
        update_geometry(env)
        torch.testing.assert_close(target_obb_in_eef(env), before)
        tracked = env._obb_live_pos_w.clone()
        torch.testing.assert_close(tracked[:, 0]-snapshot[:, 0], torch.full_like(tracked[:, 0], .01), atol=.002, rtol=0.)
        # A target rotation must carry the same initial local visible fragment.
        from isaaclab.utils.math import quat_mul
        angle = .25
        rotation = states.new_tensor([__import__('math').cos(angle/2),0.,0.,__import__('math').sin(angle/2)])
        states[:,0,3:7] = quat_mul(rotation.expand(args.num_envs,-1), states[:,0,3:7])
        env.scene['object_collection'].write_object_link_state_to_sim(states)
        update_geometry(env)
        expected_center = states[:,0,:3]+quat_apply(states[:,0,3:7],env._obb_target_local_pos)
        torch.testing.assert_close(env._obb_live_pos_w,expected_center)
        torch.testing.assert_close(target_obb_in_eef(env),before)
        # Partial reset must not touch other rows' frozen/live geometry or history.
        other = {name: value[1:].clone() for name, value in vars(env).items()
                 if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == args.num_envs
                 and name.startswith(('_obb_', '_finger_', '_formation'))}
        env.reset(env_ids=torch.tensor([0], device=env.device))
        for name, value in other.items():
            torch.testing.assert_close(getattr(env, name)[1:], value)
        # Exercise real automatic timeout and inclusive success against a held predicate.
        env.episode_length_buf[:] = env.max_episode_length-1
        obs, reward, done, info = wrapped.step(action)
        assert done.all()
        assert env.episode_length_buf.eq(0).all()
        assert env._stabilize_state.contact_history.eq(False).all()
        assert torch.isfinite(obs['policy']).all()
        from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
        runner = CurriculumOnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=env.device)
        observations = wrapped.get_observations()
        for model in (runner.alg.actor, runner.alg.critic):
            export = model.as_jit().to(env.device)
            raw = torch.cat((observations['policy'], observations['obstacles']), -1)
            eager = export(raw)
            scripted = torch.jit.script(export)
            torch.testing.assert_close(scripted(raw), eager)
            assert raw.shape[-1] == 284
        result = dict(task=task, num_envs=args.num_envs, steps=12, reward_mean=rewards,
                      policy_dim=108, obstacle_dim=176, latent_dim=124,
                      partial_reset=True, automatic_timeout=True, frozen_actor=True,
                      tracked_obb=True, jit_export=True, finger_frames_eef=finger_frame_snapshot)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print('PROF_REVISION_SMOKE_PASSED '+str(args.output), flush=True)
    finally:
        env.close()


try:
    validate()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
