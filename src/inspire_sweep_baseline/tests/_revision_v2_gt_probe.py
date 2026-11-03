"""Actual GT routing, parent CFG equivalence, live movement and reset smoke."""

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--output', type=Path, default=Path('outputs/revision_v2_gt_validation'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
env = None
try:
    import gymnasium as gym
    import torch
    from tensordict import TensorDict
    from importlib.metadata import version
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.revision_envs import RevisionV2TestEnvCfg, RevisionV2TestGTEnvCfg
    from inspire_sweep_baseline.agent.revision_ppo_cfg import RevisionV2TestPPORunnerCfg, RevisionV2TestGTPPORunnerCfg
    from inspire_sweep_baseline.agent.revision_models import RevisionModel, RevisionGTModel
    from inspire_sweep_baseline.mdp.observation_functions.sweep import object_position_in_root

    parent = RevisionV2TestEnvCfg().to_dict()
    cfg = RevisionV2TestGTEnvCfg()
    assert type(cfg).__bases__ == (RevisionV2TestEnvCfg,)
    assert cfg.to_dict() == parent
    assert RevisionV2TestEnvCfg().to_dict() == parent
    base_runner, runner = RevisionV2TestPPORunnerCfg(), RevisionV2TestGTPPORunnerCfg()
    before, after = base_runner.to_dict(), runner.to_dict()
    assert {key for key in before if before[key] != after[key]} == {'experiment_name','obs_groups','actor'}
    assert before['obs_groups']['critic'] == after['obs_groups']['critic']
    assert after['obs_groups']['actor'] == ['policy','privileged','obstacles']
    assert [key for key in before['actor'] if before['actor'][key] != after['actor'][key]] == ['class_name']
    assert RevisionV2TestPPORunnerCfg().to_dict() == before
    runner = handle_deprecated_rsl_rl_cfg(runner, version('rsl-rl-lib'))
    task = 'Isaac-Inspire-Revision-Test-v2-GT'
    registered = gym.spec(task)
    assert registered.kwargs['env_cfg_entry_point'].endswith(':RevisionV2TestGTEnvCfg')
    assert registered.kwargs['rsl_rl_cfg_entry_point'].endswith(':RevisionV2TestGTPPORunnerCfg')
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 42
    cfg.log_dir = str(args.output)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=42)
    obs = TensorDict(obs, batch_size=[args.num_envs])
    assert {k:v.shape[-1] for k,v in obs.items()} == {'policy':57,'obstacles':64,'privileged':3,'critic_obstacles':64}
    def make_model(cls, model_cfg, role, width):
        params = model_cfg.to_dict()
        params.pop('class_name')
        return cls(obs, runner.obs_groups, role, width, **params).to(raw.device)
    actor = make_model(RevisionGTModel, runner.actor, 'actor', 8)
    critic = make_model(RevisionModel, runner.critic, 'critic', 1)
    assert actor.mlp[0].in_features == critic.mlp[0].in_features == 76
    with torch.inference_mode():
        initial_target = obs['policy'][:,21:24].clone()
        initial_obstacles = obs['obstacles'].clone()
        live_before = obs['privileged'].clone()
        objects = raw.scene['object_collection']
        state = objects.data.object_link_state_w.clone()
        state[:,0,0] += .01
        objects.write_object_link_state_to_sim(state)
        raw.sim.forward()
        raw.scene.update(raw.physics_dt)
        moved = TensorDict(raw.observation_manager.compute(), batch_size=[args.num_envs])
        torch.testing.assert_close(moved['privileged'], object_position_in_root(raw))
        assert not torch.allclose(moved['privileged'], live_before)
        torch.testing.assert_close(moved['policy'][:,21:24], initial_target)
        torch.testing.assert_close(moved['obstacles'], initial_obstacles)
        torch.testing.assert_close(actor.get_latent(moved)[:,57:60], moved['privileged'])
        for _ in range(32):
            actions = actor(moved)
            assert torch.isfinite(actions).all() and torch.isfinite(critic(moved)).all()
            next_obs, reward, _, _, _ = env.step(actions)
            moved = TensorDict(next_obs, batch_size=[args.num_envs])
            assert torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in moved.values())
            torch.testing.assert_close(moved['privileged'], object_position_in_root(raw))
        ids = torch.arange(0,args.num_envs,2,device=raw.device)
        next_obs, _ = raw.reset(env_ids=ids)
        moved = TensorDict(next_obs, batch_size=[args.num_envs])
        torch.testing.assert_close(moved['privileged'], object_position_in_root(raw))
        torch.testing.assert_close(actor.get_latent(moved)[:,57:60], moved['privileged'])
        exported = torch.jit.script(actor.as_jit())
        packed = torch.cat([moved[key] for key in runner.obs_groups['actor']], -1)
        assert packed.shape[-1] == 124
        torch.testing.assert_close(exported(packed), actor(moved))
    report = {'task':task,'num_envs':args.num_envs,'steps':32,'env_cfg_diff':[],
              'runner_diff':['experiment_name','obs_groups.actor','actor.class_name'],
              'actor_raw_width':124,'actor_latent_width':76,'critic_raw_width':124,'critic_latent_width':76,
              'live_target_tracks_motion':True,'initial_target_preserved':True,'initial_obstacles_preserved':True,
              'partial_reset':True,'jit_export_equal':True}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'probe.json').write_text(json.dumps(report,indent=2))
    print('REVISION_V2_GT_PROBE '+json.dumps(report), flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
