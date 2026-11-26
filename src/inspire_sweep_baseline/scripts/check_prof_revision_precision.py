"""Live precision reward, parent contract, partial reset and terminal telemetry checks."""
import argparse
import ast
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--output', type=Path, default=Path('outputs/prof_revision_v0_3_2_validation/smoke.json'))
parser.add_argument('--before', type=Path, default=Path('outputs/prof_revision_v0_3_2_validation/before'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def before_module(relative, package, name):
    module = types.ModuleType(package + '.' + name)
    module.__package__ = package
    sys.modules[module.__name__] = module
    path = args.before / relative
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    return module


def source_audit(root):
    checked = []
    for before in args.before.rglob('*.py'):
        relative = before.relative_to(args.before)
        old, new = (ast.parse(p.read_text(encoding='utf-8')) for p in (before,root/relative))
        old_defs = {n.name:n for n in old.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        new_defs = {n.name:n for n in new.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        for name,node in old_defs.items():
            if str(relative).replace('\\','/') == 'runtime/prof_revision.py' and name == 'ProfRevisionEnv':
                # The only parent changes allowed are state-factory indirection and the empty telemetry hook.
                original_init = 'self._stabilize_state = self._state_type(count, device)'
                changed_init = 'self._stabilize_state = self._create_state(count, device, cfg)'
                expected = ast.unparse(node).replace(original_init,changed_init)
                actual = new_defs[name]
                actual.body = [n for n in actual.body if not isinstance(n,ast.FunctionDef)
                               or n.name not in ('_create_state','_additional_step_metrics')]
                # ast.unparse indentation differs inside capture; remove precisely that expression by AST.
                class RemoveHook(ast.NodeTransformer):
                    def visit_Expr(self, expression):
                        if ast.unparse(expression) == 'self._prof_step_metrics.update(self._additional_step_metrics())':
                            return None
                        return expression
                actual = RemoveHook().visit(actual)
                assert ast.dump(ast.parse(expected)) == ast.dump(ast.Module(body=[actual],type_ignores=[]))
            else:
                assert ast.dump(node) == ast.dump(new_defs[name]), (relative,name)
        if relative.name == '__init__.py':
            old_calls = [ast.dump(n) for n in old.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
            new_calls = [ast.dump(n) for n in new.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
            assert new_calls[:len(old_calls)] == old_calls
            assert len(new_calls) == len(old_calls)+1
        checked.append(str(relative))
    return checked


def validate():
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.envs import prof_revision_envs as configs
    from inspire_sweep_baseline.agent import prof_revision_ppo_cfg as runners
    from inspire_sweep_baseline.runtime.prof_revision import ProfRevisionV0_3_2Env
    from inspire_sweep_baseline.mdp.common.prof_revision_state import ProfRevisionPrecisionState
    from inspire_sweep_baseline.mdp.reward_functions.prof_revision import precision_sweep_progress, precision_sweep_retreat

    args.output.parent.mkdir(parents=True,exist_ok=True)
    root = Path(inspire_sweep_baseline.__file__).parent
    audited = source_audit(root)
    original = before_module('envs/prof_revision_envs.py','inspire_sweep_baseline.envs','_precision_before')
    original_runner = before_module('agent/prof_revision_ppo_cfg.py','inspire_sweep_baseline.agent','_precision_before')
    parent_types = ['ProfRevisionEnvCfg','ProfRevisionReach5cmEnvCfg','ProfRevisionV1EnvCfg',
                    'ProfRevisionV0_1EnvCfg','ProfRevisionV0_2EnvCfg','ProfRevisionV0_3EnvCfg',
                    'ProfRevisionHardV0_3EnvCfg','ProfRevisionV0_3_1EnvCfg']
    for name in parent_types:
        assert getattr(original,name)().to_dict() == getattr(configs,name)().to_dict(),name
    for name in vars(original_runner):
        if name.startswith('ProfRevision') and name.endswith('Cfg'):
            assert getattr(original_runner,name)().to_dict() == getattr(runners,name)().to_dict(),name

    task = 'Inspire-Sweep-Prof-Revision-v0_3_2'
    cfg = load_cfg_from_registry(task,'env_cfg_entry_point')
    assert type(cfg).__bases__ == (configs.ProfRevisionV0_3_1EnvCfg,)
    parent = configs.ProfRevisionV0_3_1EnvCfg()
    a,b = parent.to_dict(),cfg.to_dict()
    constants = dict(sweep_force_fade_inner_m=.03,sweep_force_fade_outer_m=.06,
                     sweep_precision_radius_m=.06,sweep_precision_max_multiplier=2.)
    for name,value in constants.items():
        assert b.pop(name) == value
    changed = {'sweep_progress':100.,'sweep_retreat':-40.,'reaching':9.8,
               'contact_formation':.1715,'obstacle_displacement':-300.}
    assert list(a['rewards']) == list(b['rewards'])
    for name,weight in changed.items():
        assert b['rewards'][name]['weight'] == weight
        if name not in ('sweep_progress','sweep_retreat'):
            expected = dict(a['rewards'][name],weight=weight)
            assert b['rewards'][name] == expected
        else:
            assert b['rewards'][name]['params'] == {}
        b['rewards'][name] = a['rewards'][name]
    assert cfg.rewards.sweep_progress.func is precision_sweep_progress
    assert cfg.rewards.sweep_retreat.func is precision_sweep_retreat
    assert b == a, 'Unrequested environment contract change'
    runner = load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    a,b = runners.ProfRevisionV0_3_1PPORunnerCfg().to_dict(),runner.to_dict()
    assert b['experiment_name'] == 'inspire_sweep_prof_revision_v0_3_2'
    b['experiment_name'] = a['experiment_name']
    assert a == b
    assert type(runner).__bases__ == (runners.ProfRevisionV0_3_1PPORunnerCfg,)
    for key,bad in [('sweep_force_fade_inner_m',-.01),('sweep_force_fade_outer_m',.03),
                    ('sweep_precision_radius_m',0.),('sweep_precision_max_multiplier',float('nan'))]:
        try:
            configs.ProfRevisionV0_3_2EnvCfg(**{key:bad})
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid precision cfg accepted: '+key)
    cfg.scene.num_envs,cfg.seed = args.num_envs,42
    cfg.log_dir = str(args.output.parent.resolve())
    (args.output.parent/'contract.json').write_text(json.dumps(dict(
        source_audit=audited,parent_cfgs_preserved=parent_types,full_child_contract=True,
        ppo_only_log_changed=True,environment=cfg.to_dict(),runner=runner.to_dict()),
        indent=2,default=str)+'\n',encoding='utf-8')
    env = gym.make(task,cfg=cfg).unwrapped
    try:
        assert type(env) is ProfRevisionV0_3_2Env
        state = env._stabilize_state
        assert type(state) is ProfRevisionPrecisionState
        obs,_ = env.reset(seed=42)
        assert obs['policy'].shape == (args.num_envs,108)
        assert obs['obstacles'].shape == (args.num_envs,176)
        assert len(env.reward_manager.active_terms) == 16 and not env.curriculum_manager.active_terms
        policy = None
        if args.checkpoint:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
            from importlib.metadata import version
            from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
            wrapped = RslRlVecEnvWrapper(env,clip_actions=runner.clip_actions)
            runner = handle_deprecated_rsl_rl_cfg(runner,version('rsl-rl-lib'))
            learner = CurriculumOnPolicyRunner(wrapped,runner.to_dict(),log_dir=None,device=env.device)
            learner.load(str(args.checkpoint),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
            policy = learner.get_inference_policy(device=env.device)
            obs = wrapped.get_observations()
        # RslRlVecEnvWrapper performs an initial reset; freeze the snapshot afterwards.
        snapshot = env._obb_target_pos_w.clone()

        def h(d):
            u = (d/.06).clamp(0.,1.)
            return d+.06*(u-u**3+u**4/2.)

        original_compute = env.reward_manager.compute
        saved = {}
        counts = dict(steps=0,terminal_rows=0,positive_rows=0,retreat_rows=0)
        previous = state.distance.clone()

        def capture(dt):
            distance = (env.command_manager.get_command('target_goal_pos')[:,:2]-env._obb_live_pos_w[:,:2]).norm(dim=-1)
            torch.testing.assert_close(state.distance,distance)
            delta = h(previous)-h(distance)
            u = ((distance-.03)/.03).clamp(0.,1.)
            alpha = 3*u*u-2*u*u*u
            factor = 1.-alpha+alpha*state.alignment
            contact = env._stabilize_contacts
            gate = (state.contact_history.sum(-1)>=2) & contact.contact & contact.all_valid
            positive = delta.clamp_min(0.)*gate*factor
            retreat = (-delta).clamp_min(0.)
            torch.testing.assert_close(state.precision_positive,positive,atol=2e-7,rtol=1e-5)
            torch.testing.assert_close(state.precision_retreat,retreat,atol=2e-7,rtol=1e-5)
            torch.testing.assert_close(state.positive,(previous-distance).clamp_min(0.)*state.contact_recognized*state.alignment)
            torch.testing.assert_close(state.retreat,(distance-previous).clamp_min(0.))
            result = original_compute(dt)
            expectations = dict(sweep_progress=100.*positive,sweep_retreat=-40.*retreat,
                                reaching=9.8*state.reach_delta,contact_formation=.1715*env._formation*dt)
            positions = env.scene['object_collection'].data.object_pos_w
            mask = env._active_object_mask.clone()
            mask.scatter_(1,env.target_id,False)
            q = ((positions-env._initial_object_positions_w).square().sum(-1)*mask).sum(-1).clamp(max=.04)
            expectations['obstacle_displacement'] = -300.*q*dt
            for name,expected in expectations.items():
                index = env.reward_manager.active_terms.index(name)
                torch.testing.assert_close(env.reward_manager._step_reward[:,index]*dt,expected,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(result,env.reward_manager._step_reward.sum(-1)*dt)
            saved.update(progress=positive.mean().clone(),retreat=retreat.mean().clone(),
                         raw_positive=state.positive.mean().clone(),alignment=state.alignment.mean().clone())
            counts['steps'] += 1
            counts['terminal_rows'] += int(env.termination_manager.dones.sum())
            counts['positive_rows'] += int((positive>0.).sum())
            counts['retreat_rows'] += int((retreat>0.).sum())
            return result

        env.reward_manager.compute = capture
        with torch.inference_mode():
            for tick in range(args.steps):
                previous = state.distance.clone()
                action = policy(obs) if policy else torch.zeros(args.num_envs,12,device=env.device)
                obs,reward,terminated,truncated,info = env.step(action)
                assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
                torch.testing.assert_close(info['log']['ProfRevisionPrecision/precision_progress_m'],saved['progress'])
                torch.testing.assert_close(info['log']['ProfRevision/sweep_progress_m'],saved['raw_positive'])
                torch.testing.assert_close(info['log']['ProfRevision/force_alignment'],saved['alignment'])
                done = terminated|truncated
                torch.testing.assert_close(env._obb_target_pos_w[~done],snapshot[~done])
                snapshot = env._obb_target_pos_w.clone()
            # Partial reset must preserve every state/cache buffer of untouched rows.
            ids = torch.arange(0,args.num_envs,2,device=env.device)
            keep = torch.ones(args.num_envs,dtype=torch.bool,device=env.device)
            keep[ids] = False
            before = {n:v[keep].clone() for n,v in vars(state).items() if isinstance(v,torch.Tensor)}
            env.reset(env_ids=ids)
            for name,value in before.items():
                torch.testing.assert_close(getattr(state,name)[keep],value)
            assert not state.contact_history[ids].any()
            # Force timeout and confirm final reward/telemetry survives the automatic cache reset.
            previous = state.distance.clone()
            env.episode_length_buf[:] = env.max_episode_length-1
            _,reward,terminated,truncated,info = env.step(torch.zeros(args.num_envs,12,device=env.device))
            assert (terminated|truncated).all() and env.episode_length_buf.eq(0).all()
            assert state.precision_positive.eq(0.).all() and state.precision_retreat.eq(0.).all()
            torch.testing.assert_close(info['log']['ProfRevisionPrecision/precision_retreat_m'],saved['retreat'])
            torch.testing.assert_close(info['log']['ProfRevisionPrecision/precision_progress_m'],saved['progress'])
        result = dict(task=task,num_envs=args.num_envs,**counts,source_audit=audited,
                      parent_cfgs_preserved=parent_types,full_child_contract=True,ppo_only_log_changed=True,
                      reward_dt=True,partial_reset=True,terminal_telemetry=True,original_buffers_preserved=True,
                      checkpoint=str(args.checkpoint) if args.checkpoint else None)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('PROF_REVISION_PRECISION_PASSED '+str(args.output),flush=True)
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
