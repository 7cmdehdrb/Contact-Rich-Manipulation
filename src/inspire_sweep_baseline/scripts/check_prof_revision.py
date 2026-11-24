"""Live contract, frozen observation, rigid tracking, reset and export checks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', choices=('Inspire-Sweep-Prof-Revision-v0', 'Inspire-Sweep-Prof-Revision-v1',
                                      'Inspire-Sweep-Prof-Revision-v0_1', 'Inspire-Sweep-Prof-Revision-v0_2', 'Inspire-Sweep-Prof-Revision-v0_3'),
                    default='Inspire-Sweep-Prof-Revision-v0')
parser.add_argument('--checkpoint', type=Path, help='Optional fixed policy for child-task natural arm-cutoff validation.')
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
    from inspire_sweep_baseline.mdp.common.prof_revision_geometry import update_geometry, distance_angle_formation
    from inspire_sweep_baseline.mdp.observation_functions.prof_revision import target_obb_in_eef
    task = args.task
    v1 = task.endswith('-v1')
    failure_only = task.endswith('-v0_1')
    margin_task = task.endswith('-v0_2')
    tcp_task = task.endswith('-v0_3')
    child = v1 or failure_only or margin_task or tcp_task
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.log_dir = str(args.output.parent.resolve())
    cfg.scene.num_envs, cfg.seed = args.num_envs, 42
    if child:
        from inspire_sweep_baseline.envs.prof_revision_envs import ProfRevisionEnvCfg
        assert type(cfg).__bases__ == (ProfRevisionEnvCfg,)
        baseline = ProfRevisionEnvCfg()
        baseline.scene.num_envs, baseline.seed, baseline.log_dir = args.num_envs, 42, cfg.log_dir
        a, b = baseline.to_dict(), cfg.to_dict()
        if v1:
            assert b.pop('formation_mode') == 'distance_angle_sum'
            assert b['rewards']['contact_formation']['weight'] == .3675
            b['rewards']['contact_formation']['weight'] = a['rewards']['contact_formation']['weight']
        else:
            assert not hasattr(cfg, 'formation_mode')
            assert b['rewards']['contact_formation'] == a['rewards']['contact_formation']
        if not tcp_task:
            assert b['rewards'].pop('arm_joint_failure')['weight'] == -3.
        if margin_task or tcp_task:
            from inspire_sweep_baseline.mdp.reward_functions.stabilize_sweep import sweep_progress
            term = b['rewards']['sweep_progress']
            assert cfg.rewards.sweep_progress.func is sweep_progress
            assert term['weight'] == 100. and term['params'] == {}
            b['rewards']['sweep_progress'] = a['rewards']['sweep_progress']
        if margin_task:
            margin = b['rewards'].pop('arm_joint_velocity_margin')
            assert margin['weight'] == -3. and margin['params']['free_speed'] == .8
            assert margin['params']['limit_speed'] == b['terminations']['arm_joint_velocity']['params']['threshold']
            assert margin['params']['asset_cfg'] == b['terminations']['arm_joint_velocity']['params']['asset_cfg']
        if tcp_task:
            assert b['terminations']['arm_joint_velocity'] is None
            tcp = b['terminations'].pop('tcp_linear_velocity')
            assert tcp['params']['threshold'] == 4. and not tcp['time_out']
            assert tcp['params']['asset_cfg']['body_names'] == ['wrist_3_link']
            assert b['terminations']['success']['time_out']
            b['terminations']['arm_joint_velocity'] = a['terminations']['arm_joint_velocity']
            b['terminations']['success'] = a['terminations']['success']
        assert b == a, 'unexpected child environment contract change'
        reach = load_cfg_from_registry('Inspire-Sweep-Prof-Revision-Reach5cm-v0', 'env_cfg_entry_point')
        reach.scene.num_envs, reach.seed, reach.log_dir = args.num_envs, 42, cfg.log_dir
        r = reach.to_dict()
        assert r['reach_near_radius'] == .05
        r['reach_near_radius'] = a['reach_near_radius']
        assert r == a, 'Reach5cm parent behavior drift'
    else:
        assert type(cfg).__bases__ == (ManagerBasedRLEnvCfg,)
    runner_cfg = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from importlib.metadata import version
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, version('rsl-rl-lib'))
    if child:
        baseline_runner = load_cfg_from_registry('Inspire-Sweep-Prof-Revision-v0', 'rsl_rl_cfg_entry_point')
        baseline_runner = handle_deprecated_rsl_rl_cfg(baseline_runner, version('rsl-rl-lib'))
        a, b = baseline_runner.to_dict(), runner_cfg.to_dict()
        assert b['experiment_name'] == 'inspire_sweep_prof_revision_' + ('v1' if v1 else 'v0_2' if margin_task else 'v0_3' if tcp_task else 'v0_1')
        b['experiment_name'] = a['experiment_name']
        assert b == a, 'unexpected child PPO change'
    parent = load_cfg_from_registry('Inspire-Sweep-Single-Contact-v0', 'env_cfg_entry_point')
    a, b = parent.to_dict(), cfg.to_dict()
    # Compare function/params/weights, not just final numeric rewards.
    for name in ('sweep_progress', 'sweep_retreat', 'reaching'):
        b['rewards'][name] = a['rewards'][name]
    del b['rewards']['contact_formation']
    if child and not tcp_task:
        del b['rewards']['arm_joint_failure']
    if margin_task:
        del b['rewards']['arm_joint_velocity_margin']
    assert b['rewards'] == {k:v for k,v in a['rewards'].items() if v is not None}, 'reward contract drift'
    assert b['actions'] == a['actions']
    assert b['commands'] == a['commands']
    for name in ('object_fall_or_tilt','shelf_failure','arm_joint_velocity','timeout'):
        if not (tcp_task and name == 'arm_joint_velocity'):
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
        assert len(env.reward_manager.active_terms) == (18 if margin_task else 17 if child and not tcp_task else 16)
        if child and not tcp_task:
            assert env.reward_manager.active_terms[-1] == 'arm_joint_failure'
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
            assert env._formation.ge(0).all() and env._formation.le(5 if v1 else 1).all()
            if failure_only or margin_task or tcp_task:
                expected = .05 / (.05 + env._finger_gaps.sum(-1))
                torch.testing.assert_close(env._formation, expected)
                idx = env.reward_manager.active_terms.index('contact_formation')
                torch.testing.assert_close(env.reward_manager._step_reward[:,idx], .245*expected)
            if v1:
                # Independently reconstruct normals and direction in world coordinates.
                normals_w = quat_apply(robot.body_link_quat_w[:,env._finger_body_ids],
                                       robot.body_link_pos_w.new_tensor([0.,0.,1.]).expand(args.num_envs,5,3))
                direction_w = env._obb_live_pos_w[:,None]-env._finger_pos_w
                raw, proximity, alignment, contribution = distance_angle_formation(
                    env._finger_gaps,normals_w,direction_w,env.cfg.formation_distance_scale)
                torch.testing.assert_close(raw,env._formation)
                formation_index = env.reward_manager.active_terms.index('contact_formation')
                torch.testing.assert_close(env.reward_manager._step_reward[:,formation_index],.3675*raw)
                failure_index = env.reward_manager.active_terms.index('arm_joint_failure')
                assert env.reward_manager._step_reward[:,failure_index].eq(0.).all()
            if margin_task:
                ids = env.termination_manager.get_term_cfg('arm_joint_velocity').params['asset_cfg'].joint_ids
                speed = robot.joint_vel[:,ids].abs().amax(-1)
                expected_margin = -3.*((speed-.8)/.2).clamp(0.,1.).square()
                idx = env.reward_manager.active_terms.index('arm_joint_velocity_margin')
                torch.testing.assert_close(env.reward_manager._step_reward[:,idx], expected_margin)
                idx = env.reward_manager.active_terms.index('sweep_progress')
                torch.testing.assert_close(env.reward_manager._step_reward[:,idx]*env.step_dt,
                                           100.*env._stabilize_state.positive)
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
        if tcp_task:
            manager = env.termination_manager
            assert 'arm_joint_velocity' not in manager.active_terms
            tcp_cfg = manager.get_term_cfg('tcp_linear_velocity')
            ids = tcp_cfg.params['asset_cfg'].body_ids
            assert [env.scene['robot'].body_names[i] for i in ids] == ['wrist_3_link']
            assert args.num_envs >= 4, 'TCP boundary fixture requires at least four environments'
            original_compute = manager.compute
            tcp_checks = []
            # Inject only the sampled velocity buffer at the real manager boundary.
            # This validates wiring/reset, not physical production of a 4 m/s motion.
            def capture_tcp():
                velocity = env.scene['robot'].data.body_link_lin_vel_w
                saved = velocity.clone()
                saved_center = env._obb_live_pos_w.clone()
                velocity[:,ids,:] = 0.
                velocity[0,ids,0] = 3.999
                velocity[1,ids,0] = 4.
                velocity[2,ids,0] = -4.001
                velocity[3,ids,0] = 3.
                velocity[3,ids,1] = 4.  # Norm 5, every component <= 4.
                env._obb_live_pos_w.copy_(env.command_manager.get_command('target_goal_pos')[:, :3])
                try:
                    result = original_compute()
                    flags = manager.get_term('tcp_linear_velocity').clone()
                    success = manager.get_term('success').clone()
                    assert flags[:4].tolist() == [False,True,True,True]
                    due = env.episode_length_buf >= env.max_episode_length
                    assert success[0] == due[0]
                    assert not success[1:4].any()
                    tcp_checks.append(dict(flags=flags[:4].tolist(), success=success[:4].tolist(),
                                           timeout=due[:4].tolist()))
                    return result
                finally:
                    velocity.copy_(saved)
                    env._obb_live_pos_w.copy_(saved_center)
            manager.compute = capture_tcp
            try:
                wrapped.reset()
                _,_,done,_ = wrapped.step(action)
                assert not done[0] and done[1:4].all()
                assert env.episode_length_buf[1:4].eq(0).all()
                env.episode_length_buf[:] = env.max_episode_length-1
                _,_,done,_ = wrapped.step(action)
                assert done.all() and env.episode_length_buf.eq(0).all()
            finally:
                manager.compute = original_compute
            wrapped.reset()
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
        natural_cutoffs = 0
        if child and not tcp_task and args.checkpoint:
            from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_checkpoint
            checkpoint = handle_deprecated_rsl_rl_checkpoint(str(args.checkpoint), version('rsl-rl-lib'))
            runner.load(checkpoint, load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
            policy = runner.get_inference_policy(device=env.device)
            failure_index = env.reward_manager.active_terms.index('arm_joint_failure')
            # Read rewards/flags BEFORE auto-reset. Fixed v0 policy tests wiring, not v1 performance.
            original_compute = env.reward_manager.compute
            terminal_samples = []
            def capture_reward(dt):
                reward = original_compute(dt)
                flags = env.termination_manager.get_term('arm_joint_velocity').clone()
                cost = env.reward_manager._step_reward[:,failure_index]*dt
                torch.testing.assert_close(cost,-3.*flags.float())
                robot = env.scene['robot']
                ids = env.termination_manager.get_term_cfg('arm_joint_velocity').params['asset_cfg'].joint_ids
                max_speed = robot.data.joint_vel[:,ids].abs().amax(-1)
                assert torch.equal(flags,max_speed>1.)
                if margin_task:
                    idx = env.reward_manager.active_terms.index('arm_joint_velocity_margin')
                    margin_cost = env.reward_manager._step_reward[:,idx]*dt
                    torch.testing.assert_close(margin_cost,-3.*((max_speed-.8)/.2).clamp(0.,1.).square()*dt)
                    torch.testing.assert_close((margin_cost+cost)[flags],torch.full_like(cost[flags],-3.-3.*dt))
                torch.testing.assert_close(reward,env.reward_manager._step_reward.sum(-1)*dt)
                terminal_samples.append((flags.clone(),cost.clone(),max_speed.clone()))
                return reward
            env.reward_manager.compute = capture_reward
            try:
                with torch.inference_mode():
                    observations,_ = wrapped.reset()
                    for _ in range(220):
                        observations,reward,done,info = wrapped.step(policy(observations))
                        assert all(torch.isfinite(v).all() for v in observations.values())
                        assert torch.isfinite(reward).all()
                        natural_cutoffs += int(terminal_samples[-1][0].sum())
                        if natural_cutoffs >= args.num_envs:
                            break
                assert natural_cutoffs >= args.num_envs, 'no natural arm cutoffs observed'
            finally:
                env.reward_manager.compute = original_compute
        result = dict(task=task, num_envs=args.num_envs, steps=12, reward_mean=rewards,
                      policy_dim=108, obstacle_dim=176, latent_dim=124,
                      partial_reset=True, automatic_timeout=True, frozen_actor=True,
                      tracked_obb=True, jit_export=True, finger_frames_eef=finger_frame_snapshot)
        if v1:
            result.update(reward_terms=env.reward_manager.active_terms,
                          full_parent_cfg_equal_except_requested=True, reach5cm_preserved=True,
                          formation_weight=.3675, formation_distance=proximity.cpu().tolist(),
                          formation_angle=alignment.cpu().tolist(), formation_contribution=contribution.cpu().tolist(),
                          natural_arm_cutoffs=natural_cutoffs,
                          cutoff_checkpoint=str(args.checkpoint) if args.checkpoint else None)
        if failure_only or margin_task or tcp_task:
            result.update(reward_terms=env.reward_manager.active_terms,
                          full_parent_cfg_equal_except_requested=True,
                          formation_mode='parent_distance_sum', formation_weight=.245,
                          natural_arm_cutoffs=natural_cutoffs,
                          cutoff_checkpoint=str(args.checkpoint) if args.checkpoint else None)
        if margin_task:
            result.update(margin_weight=-3., margin_free_speed=.8, margin_limit_speed=1., neutral_contact_sides=True)
        if tcp_task:
            result.update(tcp_body='wrist_3_link', tcp_threshold=4., tcp_boundary_checks=tcp_checks,
                          termination_terms=env.termination_manager.active_terms)
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
