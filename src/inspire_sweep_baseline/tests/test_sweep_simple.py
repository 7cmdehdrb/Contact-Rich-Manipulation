"""Simple family equivalence: live configs, inherited lifecycle and registration.

Official Isaac configclass and manager terms execute on CPU. Scene/Env base
and simulator asset leaves are doubles; these checks do not execute PhysX.
"""
import ast
import copy
import importlib
import sys
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import assert_ordered_equal, configs
from test_omni_configuration import _AssetNode, _execute_definitions, omni_config

OLD = 'wrench_refactor_relative_ik_test'
PREFIX = 'WrenchRefactorRelativeIKV0_2'
VARIANTS = [('SweepSimple', '_2_4'), ('SweepSimpleProf', '_2_5'), ('SweepSimpleV1', '_2_6')]


@pytest.fixture
def family(configs, omni_config, monkeypatch):
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', importlib.import_module('_omni_official_utils.noise'))
    robot = sys.modules['_baseline_cpu_pkg.assets.robot']
    tactile = importlib.import_module('_baseline_cpu_pkg.assets.tactile_regions')
    contacts = importlib.import_module('_baseline_cpu_pkg.mdp.common.contact_sweep_clutter_fusion_contacts')
    ns = dict(vars(omni_config.baseline))
    ns.update(vars(robot))
    ns.update(Path=Path, sim=ns['sim_utils'],
              RevisionRobotSpawnerCfg=lambda **kw: _AssetNode(**kw, sensor_height=.0254),
              WrenchRefactorShelfSpawnerCfg=_AssetNode, RevisionObjectSpawnerCfg=_AssetNode,
              REVISION_OBJECT_KINDS=('cylinder','cube','mug','bottle','003_cracker_box','004_sugar_box','005_tomato_soup_can','002_master_chef_can'),
              TACTILE_PARENT_BODY_NAMES=tactile.TACTILE_PARENT_BODY_NAMES,
              parent_sensor_name=tactile.parent_sensor_name, object_hand_sensor_name=contacts.object_hand_sensor_name,
              make_robot_cfg=omni_config.baseline.make_robot_cfg)
    ns['_tool_frame'] = lambda frames, body='inspire_base_link', **kw: _AssetNode(target_frames=[
        _AssetNode(prim_path=f'Robot/{body}', offset=_AssetNode(pos=xyz, rot=(.5,-.5,-.5,-.5))) for name,xyz in frames])
    for stem, prefix in ((OLD, PREFIX+'Test'), ('sweep_simple', 'SweepSimple')):
        path = MODULE_ROOT/f'assets/scenes/{stem}.py'
        ns['__file__'] = str(path)
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_robot']
        exec(compile(tree, str(path), 'exec'), ns)
        ns.update(vars(_execute_definitions(path, ns, {prefix+'SceneCfg'})))
        for role in ('commands','observations','rewards','events','terminations','curricula'):
            ns.update({k:v for k,v in vars(configs(role,stem)).items() if k.startswith(('WrenchRefactor','SweepSimple'))})
        # Actions are compared structurally in full below, including all defaults.
        ns[prefix+'ActionsCfg'] = lambda: _AssetNode(arm_action=_AssetNode())
        ns[prefix+'OSCActionCfg'] = NS(OffsetCfg=_AssetNode)
    ns['SweepSimpleFingerActionsCfg'] = lambda: _AssetNode(arm_action=_AssetNode())
    ns['SweepSimpleFingerV1ActionsCfg'] = lambda: _AssetNode(arm_action=_AssetNode())
    ns['SweepSimpleFingerWrenchActionsCfg'] = lambda: _AssetNode(arm_action=_AssetNode())
    ns.update(GroupedResetObjectSpawnerCfg=_AssetNode, ViewerCfg=_AssetNode)
    ns.update(vars(_execute_definitions(MODULE_ROOT/f'envs/{OLD}_envs.py', ns)))
    return _execute_definitions(MODULE_ROOT/'envs/sweep_simple_envs.py', ns)


@pytest.mark.parametrize('new_name,old_version', VARIANTS)
def test_complete_env_config_matches_and_is_isolated(family, new_name, old_version):
    old_cls = getattr(family, PREFIX+old_version+'EnvCfg')
    new_cls = getattr(family, new_name+'EnvCfg')
    before = old_cls().to_dict()
    new = new_cls()
    expected = copy.deepcopy(before)
    if expected['rewards']['hand_contact'] is not None:
        expected['rewards']['hand_contact']['func'] = new.to_dict()['rewards']['hand_contact']['func']
        assert new.rewards.hand_contact.func.__name__ == 'GoalContactMaintenance'
    assert_ordered_equal(new.to_dict(), expected)
    new.rewards.reaching.weight = 999.
    new.events.object_spawn.params['ik_params']['height_offset'] = 99.
    new.scene.robot.actuators['arm'].effort_limit_sim['shoulder_pan_joint'] = 1.
    assert_ordered_equal(old_cls().to_dict(), before)
    assert_ordered_equal(new_cls().to_dict(), expected)


def test_prof_v1_only_changes_reach_and_required_hierarchy(family):
    base, prof, reach, both = (getattr(family, name+'EnvCfg') for name in
                             ('SweepSimple','SweepSimpleProf','SweepSimpleV1','SweepSimpleProfV1'))
    assert base.__bases__ == (family.ManagerBasedRLEnvCfg,)
    assert prof.__bases__ == reach.__bases__ == (base,)
    assert both.__bases__ == (prof,)
    expected = prof().to_dict()
    expected['rewards']['reaching'] = reach().to_dict()['rewards']['reaching']
    assert_ordered_equal(both().to_dict(), expected)
    child = both()
    child.rewards.goal_distance.params['threshold_n'] = 1.
    assert prof().rewards.goal_distance.params['threshold_n'] == .01
    assert both().rewards.goal_distance.params['threshold_n'] == .01


@pytest.mark.parametrize('role,suffix,version', [
    ('commands','CommandsCfg','Test'), ('observations','ObservationsCfg','Test'),
    ('events','EventsCfg','_2_2FastTest'), ('curricula','CurriculumCfg','_2_4'),
    ('terminations','TerminationsCfg','_2_4'), ('rewards','RewardsCfg','_2_4'),
])
def test_manager_order_values_and_callable_identity(configs, monkeypatch, role, suffix, version):
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', importlib.import_module('_omni_official_utils.noise'))
    old = getattr(configs(role, OLD), PREFIX+version+suffix)()
    new = getattr(configs(role,'sweep_simple'),'SweepSimple'+suffix)()
    expected = old.to_dict()
    if role == 'rewards':
        expected['hand_contact']['func'] = new.to_dict()['hand_contact']['func']
        assert new.hand_contact.func.__name__ == 'GoalContactMaintenance'
    assert_ordered_equal(expected,new.to_dict())
    for name, term in vars(new).items():
        if hasattr(term,'func') and not (role == 'rewards' and name == 'hand_contact'):
            assert term.func is getattr(old,name).func


def test_scene_actions_observations_commands_are_identical_definitions():
    """Compare full live bodies, including nested classes and all callable names."""
    for folder in ('assets/scenes','mdp/actions','mdp/observations','mdp/commands'):
        old = ast.parse((MODULE_ROOT/f'{folder}/{OLD}.py').read_text().replace(PREFIX+'Test','SweepSimple'))
        new = ast.parse((MODULE_ROOT/f'{folder}/sweep_simple.py').read_text())
        def definitions(tree):
            return [ast.dump(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
                    and getattr(n,'name','') != 'GroupedResetObjectSpawnerCfg'
                    and not getattr(n,'name','').startswith('SweepSimpleFinger')]
        assert definitions(old) == definitions(new), folder


def test_ppo_complete_config_and_network_checkpoint_contract(omni_config):
    cc = omni_config.baseline.configclass
    @cc
    class Base:
        distribution_cfg: object = None
        policy: object = None
    class Distribution(_AssetNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    Base.GaussianDistributionCfg = Distribution
    ns = dict(configclass=cc, MISSING=MISSING, RslRlMLPModelCfg=Base,
              RslRlOnPolicyRunnerCfg=Base, RslRlPpoAlgorithmCfg=_AssetNode)
    ns.update(vars(_execute_definitions(MODULE_ROOT/f'agent/{OLD}_ppo_cfg.py', ns)))
    new = _execute_definitions(MODULE_ROOT/'agent/sweep_simple_ppo_cfg.py', ns)
    for name, version in [*VARIANTS, ('SweepSimpleProfV1','_2_5'), ('SweepSimpleHard','_2_6'), ('SweepSimpleProfHard','_2_5')]:
        expected = ns[PREFIX+version+'PPORunnerCfg']().to_dict()
        actual = getattr(new,name+'PPORunnerCfg')().to_dict()
        assert actual['experiment_name'].startswith('inspire_sweep_simple_')
        actual['experiment_name'] = expected['experiment_name']
        assert_ordered_equal(actual, expected)


def test_registry_and_no_historical_cfg_or_runtime_ancestry():
    entries = {}
    gym = NS(registry=entries, register=lambda **kw: entries.update({kw['id']:kw}))
    _execute_definitions(MODULE_ROOT/'__init__.py', {'gym':gym})
    for suffix, name in [('v0','SweepSimple'),('Prof-v0','SweepSimpleProf'),('v1','SweepSimpleV1'),('Prof-v1','SweepSimpleProfV1'), ('Hard-v0','SweepSimpleHard'), ('Prof-Hard-v0','SweepSimpleProfHard')]:
        entry = entries['Inspire-Sweep-Simple-'+suffix]
        assert entry['entry_point'] == 'inspire_sweep_baseline.runtime.sweep_simple:SweepSimpleEnv'
        assert entry['kwargs'] == dict(
            env_cfg_entry_point=f'inspire_sweep_baseline.envs.sweep_simple_envs:{name}EnvCfg',
            rsl_rl_cfg_entry_point=f'inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:{name}PPORunnerCfg')
    paths = list(MODULE_ROOT.rglob('sweep_simple*.py'))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ClassDef) and path.parent.name not in ('agent', 'action_functions'):
                assert all(not isinstance(b,ast.Name) or not b.id.startswith('Wrench') for b in node.bases), path
            if isinstance(node,ast.ImportFrom) and node.module:
                assert node.module not in (f'envs.{OLD}_envs', f'runtime.{OLD}')
                if node.module.endswith(OLD):
                    assert not any(a.name.endswith('Cfg') for a in node.names), path


def runtime_classes():
    class Base:
        def __init__(self, cfg, render_mode=None, **kwargs):
            self.cfg, self.device, self.num_envs = cfg, cfg.sim.device, cfg.scene.num_envs
            self.extras = {'log':{}}
            self.calls = []
        def reset(self, seed=None, env_ids=None, options=None):
            self.calls.append(('reset',seed))
            return seed
        def step(self, action):
            self.calls.append(('step',action))
            return action
        def _reset_idx(self, ids):
            self.calls.append(('reset_idx',ids.tolist()))
            self.episode_length_buf[ids] = 0
            self.extras['log'] = {}
        def load_managers(self):
            self.calls.append(('load_managers',))
    result = []
    for stem, name in ((OLD,PREFIX+'_2_4Env'),('sweep_simple','SweepSimpleEnv')):
        tree = ast.parse((MODULE_ROOT/f'runtime/{stem}.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n,ast.ClassDef)]
        ns = dict(torch=torch, ManagerBasedRLEnv=Base)
        exec(compile(tree,stem,'exec'),ns)
        result.append(ns[name])
    return result


def assert_runtime_equal(left,right):
    for name in ('target_id','target_width','sweep_dir','sensor_data_fresh','_active_object_mask',
                 '_initial_obstacle_observation','_initial_object_positions_w','_revision_spawn_positions',
                 '_final_actual_population','_revision_layout_complete','_metric_sums','_metric_counts',
                 '_metric_peaks','_metric_final_distance','_metric_min_distance'):
        torch.testing.assert_close(getattr(left,name),getattr(right,name))
    for name in ('buffer','totals','cursor','filled'):
        a,b = getattr(left._timeout_outcomes,name),getattr(right._timeout_outcomes,name)
        if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b)
        else: assert a == b
    assert left.calls == right.calls
    assert_ordered_equal(left.extras,right.extras)


def test_runtime_lifecycle_rng_partial_resets_and_rolling_metrics_equivalent():
    cfg = NS(scene=NS(num_envs=4),sim=NS(device='cpu'),seed=42,
             timeout_metric_window=3,goal_reward_height_limit=.14,reset_ik_backend='warp',
             max_obstacles=6,obstacle_observation_slots=16,events=NS(robot_initial_pose=None),
             goal_success_radius=.03)
    envs = [cls(copy.deepcopy(cfg)) for cls in runtime_classes()]
    for env in envs:
        env._goal_reset_rng = torch.Generator().manual_seed(42)
        env.episode_length_buf = torch.tensor([500,500,100,0])
        env.max_episode_length = 500
        env.termination_manager = NS(dones=torch.tensor([True,True,True,False]),
            get_term=lambda _: torch.tensor([True,False,False,False]))
        env.action_manager = NS(_action=torch.zeros(4,8),_prev_action=torch.zeros(4,8))
        env.scene = {'object_collection': NS(data=NS(object_pos_w=torch.arange(204).reshape(4,17,3).float()))}
        env._metric_sums[:] = 2.
        env._metric_counts[:] = 1.
        env._metric_peaks[:] = 3.
        env._metric_min_distance[:] = .01
        env._metric_final_distance[:] = .02
    assert_runtime_equal(*envs)
    for seed in (42,None,73):
        for env in envs:
            env.reset(seed=seed)
            env.step('action')
        assert_runtime_equal(*envs)
        for name in ('_revision_layout_rng','_goal_reset_rng'):
            assert torch.equal(getattr(envs[0],name).get_state(),getattr(envs[1],name).get_state())
    for ids in ([3,2,0,1], [0,1,2,3], [2], [0,1]):
        for env in envs: env._reset_idx(torch.tensor(ids))
        assert_runtime_equal(*envs)
        assert envs[0]._timeout_outcomes.totals.tolist() == [1,2]
    for env in envs:
        env.episode_length_buf[0] = 100
        env._reset_idx(torch.tensor([0]))
    assert_runtime_equal(*envs)
    assert envs[0]._timeout_outcomes.totals.tolist() == [0,1]
    assert envs[1]._initial_object_positions_w[0].equal(envs[1].scene['object_collection'].data.object_pos_w[0])
    assert envs[1].action_manager._action[:,-2:].eq(1).all()


def test_unmodified_runtime_methods_and_ring_match_exactly():
    old = ast.parse((MODULE_ROOT/f'runtime/{OLD}.py').read_text())
    new = ast.parse((MODULE_ROOT/'runtime/sweep_simple.py').read_text())
    old_methods = {}
    for cls in old.body:
        if isinstance(cls,ast.ClassDef) and cls.name in (PREFIX+'TestEnv',PREFIX+'_2_4Env'):
            old_methods.update({n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)})
    new_cls = next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='SweepSimpleEnv')
    for method in new_cls.body:
        if isinstance(method,ast.FunctionDef) and method.name not in ('__init__','reset','_reset_idx'):
            assert ast.dump(method) == ast.dump(old_methods[method.name]), method.name
    def ring(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_RecentEpisodeOutcomes')
    assert ast.dump(ring(old)) == ast.dump(ring(new))


@pytest.mark.parametrize('child_name,parent_name', [
    ('SweepSimpleHard','SweepSimpleV1'), ('SweepSimpleProfHard','SweepSimpleProfV1'),
])
def test_hard_changes_only_goal_events_and_deadzone(family, child_name, parent_name):
    parent, child = (getattr(family, name+'EnvCfg') for name in (parent_name,child_name))
    before = parent().to_dict()
    actual = child().to_dict()
    expected = copy.deepcopy(before)
    expected['events']['object_geometry']['func'] = actual['events']['object_geometry']['func']
    expected['events']['object_spawn']['params'].update(
        min_goal_distance=.07, max_goal_distance=.15, excluded_x_half_angle=0.)
    expected['curriculum']['target_goal_direction_deadzone'] = None
    assert_ordered_equal(actual, expected)
    assert child.__bases__ == (parent,)
    assert child().events.object_geometry.func.__name__ == 'initialize_hard_geometry'
    assert child().events.object_spawn.func is parent().events.object_spawn.func
    instance = child()
    instance.events.object_spawn.params['min_goal_distance'] = .99
    assert_ordered_equal(parent().to_dict(), before)
    assert child().events.object_spawn.params['min_goal_distance'] == .07


def test_goal_contact_full_count_boundary_exit_cache_and_reset(configs):
    from test_wrench_refactor_relative_ik_test import contact_count_fixture
    env, force, _ = contact_count_fixture(configs, n=5)
    cfg = configs('rewards','sweep_simple').SweepSimpleRewardsCfg().hand_contact
    term = cfg.func(cfg,env)
    pos = env.scene['object_collection'].data.object_pos_w
    goal = torch.zeros(5,3)
    env.command_manager = NS(get_command=lambda _:goal)
    pos[:,0,:2] = 0.
    pos[:,0,0] = pos.new_tensor([0.,.029,.03,.031,float('nan')])
    force.zero_()
    env.sensor_data_fresh[:] = False
    env.scene['ee_frame'].data.target_pos_w[:,0,2] = 2.  # above the 15 cm ceiling
    result = term(env,**cfg.params)
    assert result.tolist() == [20.,20.,0.,0.,0.]
    assert term._count.tolist() == [20,20,0,0,0]
    torch.testing.assert_close(term(env),result)
    assert not term._history.any()  # goal occupancy must not fabricate tactile history
    pos[0,0,0] = .04
    env.common_step_counter += 1
    assert term(env).tolist() == [0.,20.,0.,0.,0.]
    assert term._count.tolist() == [18,20,0,0,0]
    term(env)
    assert term._count[0] == 18  # duplicate evaluation does not decay again
    term.reset(torch.tensor([1]))
    assert term._count.tolist() == [18,0,0,0,0]
    env.common_step_counter += 1
    assert term(env).tolist() == [0.,20.,0.,0.,0.]
    assert term._count[0] == 16
    # Reacquisition outside the goal follows real 2-of-3 history and the remaining count.
    env.sensor_data_fresh[:] = True
    env.scene['ee_frame'].data.target_pos_w[:,0,2] = 1.10
    force[:,0,0,0] = .01
    env.common_step_counter += 1
    term(env)
    assert term._count[0] == 14
    env.common_step_counter += 1
    assert term(env)[0] == 15


def test_goal_contact_scope_including_prof_and_historical_parents(family):
    for name in ('SweepSimple','SweepSimpleV1','SweepSimpleHard'):
        assert getattr(family,name+'EnvCfg')().rewards.hand_contact.func.__name__ == 'GoalContactMaintenance'
    for name in ('SweepSimpleProf','SweepSimpleProfV1','SweepSimpleProfHard'):
        assert getattr(family,name+'EnvCfg')().rewards.hand_contact is None
    for version in ('_2_4','_2_6'):
        assert getattr(family,PREFIX+version+'EnvCfg')().rewards.hand_contact.func.__name__ == 'TargetContactDecayMaintenance'
