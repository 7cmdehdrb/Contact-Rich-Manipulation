"""Wrench contracts: production managers, continuous success, geometry and reset banks."""

import ast
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import configs, assert_ordered_equal
from test_omni_configuration import omni_config, _execute_definitions
from test_fusion_holding_osc import osc
from test_revision_layout import bounds


@pytest.mark.parametrize('params', [{}, {'stages': ((40, 3, 3),), 'min_goal_distance': .025}])
def test_layout_startup_passes_real_manager_argument_validation(configs, params):
    import inspect

    manager_path = MODULE_ROOT.parents[2] / 'IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py'
    tree = ast.parse(manager_path.read_text())
    manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManagerBase')
    resolver = next(n for n in manager.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_resolve_common_term_cfg')
    namespace = {'inspect': inspect, 'ManagerTermBaseCfg': NS, 'ManagerTermBase': object}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(manager_path), 'exec'), namespace)
    events = configs('event_functions', 'wrench')
    term = NS(func=events.initialize_wrench_layouts, params=params)
    owner = NS(_env=NS(sim=NS(is_playing=lambda: False)))
    namespace['_resolve_common_term_cfg'](owner, 'layout_bank', term, min_argc=2)


def test_manager_contracts_and_isolation(configs):
    for role, old_name, new_name in [
        ('rewards', 'RevisionV2RewardsCfg', 'WrenchRewardsCfg'),
        ('curricula', 'RevisionV1CurriculumCfg', 'WrenchCurriculumCfg'),
        ('terminations', 'RevisionTerminationsCfg', 'WrenchTerminationsCfg'),
        ('events', 'RevisionEventsCfg', 'WrenchEventsCfg'),
    ]:
        old = getattr(configs(role, 'revision'), old_name)()
        new_cls = getattr(configs(role, 'wrench'), new_name)
        new = new_cls()
        assert new_cls.__bases__ == (object,)
        before = old.to_dict()
        old_terms = {k:v for k,v in vars(old).items() if v is not None}
        new_terms = {k:v for k,v in vars(new).items() if v is not None}
        aliases = {'target_goal_timeout_bonus': 'target_goal_success_bonus',
                   'target_goal_success_at_timeout': 'target_goal_success'}
        assert [aliases.get(k,k) if role in ('rewards','terminations') else k for k in old_terms] == [k for k in new_terms if k != 'layout_bank']
        exceptions = {
            'rewards': {'target_goal_timeout_bonus'},
            'curricula': {'target_goal_direction_deadzone'},
            'terminations': {'target_goal_success_at_timeout','target_goal_miss_at_timeout'},
            'events': {'object_spawn','object_physics_randomization'},
        }[role]
        for name, term in old_terms.items():
            if name not in exceptions:
                assert_ordered_equal(term.to_dict(), new_terms[name].to_dict())
        if role == 'rewards':
            assert new.target_goal_success_bonus.weight == old.target_goal_timeout_bonus.weight == 75
        if role == 'events':
            assert new.object_physics_randomization.mode == new.layout_bank.mode == 'startup'
            assert new.object_physics_randomization.func is old.object_physics_randomization.func
            assert new.object_physics_randomization.params == old.object_physics_randomization.params
            assert new.object_spawn.params == {**old.object_spawn.params,'gray_obstacles':3,'white_obstacles':3}
        if role == 'terminations':
            assert not new.target_goal_success.time_out
            assert new.target_goal_miss_at_timeout.time_out
        assert old.to_dict() == before


def test_observations_actions_commands_and_ppo_preserve_source_contract():
    for role in ('observations', 'actions', 'commands'):
        old = (MODULE_ROOT/f'mdp/{role}/revision.py').read_text()
        new = (MODULE_ROOT/f'mdp/{role}/wrench.py').read_text()
        new = new.replace('Independent Wrench', 'Independent Revision')
        tree = ast.parse(old)
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                new = new.replace(n.name.replace('Revision','Wrench'), n.name)
        new_tree = ast.parse(new)
        # New v1 is intentionally derived; this comparison guards the independent v0 declarations.
        new_tree.body = [n for n in new_tree.body if not (
            isinstance(n, ast.ClassDef) and n.name.startswith(('WrenchV0_1', 'WrenchV1', 'WrenchV2'))
            or isinstance(n, ast.ImportFrom) and n.module == 'action_functions.wrench')]
        assert ast.dump(ast.parse(old)) == ast.dump(new_tree)
    old_tree = ast.parse((MODULE_ROOT/'agent/revision_ppo_cfg.py').read_text())
    new_tree = ast.parse((MODULE_ROOT/'agent/wrench_ppo_cfg.py').read_text().replace(
        'WrenchModelCfg','RevisionModelCfg').replace('WrenchPPORunnerCfg','RevisionPPORunnerCfg').replace(
        'inspire_wrench_v0','inspire_revision_v0'))
    old_classes = {n.name: ast.dump(n) for n in old_tree.body if isinstance(n,ast.ClassDef)}
    for n in new_tree.body:
        if isinstance(n, ast.ClassDef) and not n.name.startswith(('WrenchV0_1', 'WrenchV1', 'WrenchV2')):
            assert ast.dump(n) == old_classes[n.name]


def test_no_old_environment_ancestors_or_cfg_instances():
    files = [*MODULE_ROOT.glob('mdp/*/wrench.py'), MODULE_ROOT/'envs/wrench_envs.py',
             MODULE_ROOT/'assets/scenes/wrench.py', MODULE_ROOT/'runtime/wrench.py',
             MODULE_ROOT/'agent/wrench_ppo_cfg.py']
    allowed = {'ManagerBasedRLEnvCfg','ManagerBasedRLEnv','InteractiveSceneCfg',
               'OperationalSpaceControllerActionCfg','ActionTermCfg','CommandTermCfg',
               'ObsGroup','RslRlMLPModelCfg','RslRlOnPolicyRunnerCfg','ManagerTermBase','ActionTerm'}
    for path in files:
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ClassDef) and n.name.startswith(('WrenchV0_1', 'WrenchV1', 'WrenchV2')))]
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                assert all(ast.unparse(b) in allowed for b in n.bases), (path,n.name)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
                assert not n.func.id.startswith(('RevisionEnv','RevisionV','FinalEnv','FinalV','RevisionScene'))
    registered = {}
    gym = NS(registry=registered, register=lambda **kw: registered.update({kw['id']:kw}))
    _execute_definitions(MODULE_ROOT/'__init__.py', {'gym':gym})
    assert registered['Inspire-Wrench-v0']['entry_point'].endswith('runtime.wrench:WrenchEnv')


def settling_env(count=7, dt=.02):
    pos = torch.zeros(count,1,3,dtype=torch.float64)
    pos[...,0] = .03
    vel = torch.zeros_like(pos)
    vel[...,2] = .005  # speed is 3D, including Z
    env = NS(num_envs=count,device='cpu',step_dt=dt,common_step_counter=0,
             target_id=torch.zeros(count,1,dtype=torch.long),
             scene={'object_collection': NS(data=NS(object_pos_w=pos,object_lin_vel_w=vel))},
             command_manager=NS(get_command=lambda _: torch.zeros(count,3,dtype=torch.float64)),
             termination_manager=NS(terminated=torch.zeros(count,dtype=torch.bool)),
             record_goal_accuracy_metrics=lambda: None)
    return env,pos,vel


@pytest.mark.parametrize('dt', [.01,.02,.07,.1])
def test_success_duration_boundaries_interruptions_cache_partial_reset(configs,dt):
    cls = configs('termination_functions','wrench').SettledGoalSuccess
    env,pos,vel = settling_env(dt=dt)
    term = cls(NS(params={}),env)
    required = math.ceil(3/dt-1e-9)
    pos[1,0,0] += 1e-9
    vel[2,0,2] += 1e-9
    pos[3,0,0] = float('nan')
    vel[4,0,0] = float('inf')
    env.termination_manager.terminated[6] = True
    for step in range(1,required):
        env.common_step_counter = step
        assert not term(env).any()
        term(env)  # duplicate manager query never advances timer
    env.common_step_counter = required
    assert term(env).tolist() == [True,False,False,False,False,True,False]
    term.reset(torch.tensor([0]))
    assert term(env).tolist() == [False,False,False,False,False,True,False]
    env.common_step_counter += 1
    vel[5,0,0] = .1
    assert not term(env).any()
    assert term._steps[0] == 1 and term._steps[5] == 0
    term.reset()
    assert not term._steps.any()


def test_success_manager_order_reward_and_timeout(configs):
    m = configs('termination_functions','wrench')
    rewards = configs('reward_functions','wrench')
    env,pos,vel = settling_env(3)
    term = m.SettledGoalSuccess(NS(params={}),env)
    term._steps[:] = 149
    env.common_step_counter = 150
    env.termination_manager.terminated[1] = True
    vel[2,0,2] = .006
    success = term(env)
    env.termination_manager.get_term = lambda _: success
    assert (rewards.settled_success_bonus(env) * env.step_dt * 75).tolist() == [75,0,0]
    m.time_out = lambda _: torch.ones(3,dtype=torch.bool)
    assert m.unsuccessful_timeout(env).tolist() == [False,True,True]


def test_curriculum_all_boundaries_and_bank_coverage(configs):
    c = configs('curricula','wrench').WrenchCurriculumCfg()
    term = NS(params={})
    env = NS(cfg=NS(max_gray_obstacles=4,max_white_obstacles=5),event_manager=NS(
        get_term_cfg=lambda _:term,set_term_cfg=lambda *_:None))
    layout = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_layout')
    expected = [(0,30),(9999,30),(10000,25),(11999,25),(12000,25),(12999,25),
                (13000,20),(13999,20),(14000,20),(15999,20),(16000,15),(18999,15),
                (19000,10),(21999,10),(22000,5),(24999,5),(25000,0),(90000,0)]
    for iteration, angle in expected:
        env.learning_iteration = iteration
        d = c.target_goal_direction_deadzone
        assert d.func(env,None,**d.params) == angle
        p = c.obstacle_population
        p.func(env,None,**p.params)
        assert (angle,term.params['gray_obstacles'],term.params['white_obstacles']) in layout.LAYOUT_STAGES


def test_red_expansion_and_gray_rear_retreat(load_mdp):
    obb = importlib.import_module('_baseline_cpu_pkg.mdp.common.bbox_scene_layout')
    # Sweep goes +X; the hand-entry rectangle is behind it (-X).
    target,goal = torch.tensor([[0.,.20]]),torch.tensor([[.1,.20]])
    half,yaw = torch.tensor([[.02,.02]]),torch.zeros(1)
    centers = torch.tensor([[[-.10,.20],[-.17,.20],[.05,.24]]])
    h = torch.full((1,3,2),.001)
    a = torch.zeros(1,3)
    old = obb.zone_masks(centers,h,a,target,goal,half,yaw,obb.LayoutConfig(gray_padding=.065))
    new = obb.zone_masks(centers,h,a,target,goal,half,yaw,obb.LayoutConfig(gray_padding=.065,red_margin=.023,rear_offset=.02))
    assert not old[0][0,2] and new[0][0,2]  # expanded RED
    assert old[2][0,1] and new[1][0,1]  # retreat extends rear Gray by 2 cm
    assert new[1][0,0] and not new[0][0,0]  # hand rectangle stays Gray
    assert obb.LayoutConfig().rear_offset == 0  # parent geometry preserved


def test_bank_selects_own_geometry_and_never_rebuilds_on_reset(load_mdp,monkeypatch):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_layout')
    b = bounds(4)
    b[1] *= 1.1
    gen = torch.Generator().manual_seed(42)
    bank = m.WrenchLayoutBank(b,gen,size=3,batch_size=7,stages=((30,3,3),(0,4,5)))
    monkeypatch.setattr(m,'wrench_layout',lambda *a,**kw: pytest.fail('OBB search executed during reset'))
    ids = torch.tensor([1,3])
    before = tuple(v.clone() for v in bank.banks[(30,3,3)])
    first = bank.sample(ids,torch.Generator().manual_seed(4),3,3,math.pi/6)
    second = bank.sample(ids,torch.Generator().manual_seed(4),3,3,math.pi/6)
    for a,z in zip(first,second): torch.testing.assert_close(a,z)
    for a,z in zip(before,bank.banks[(30,3,3)]): torch.testing.assert_close(a,z)
    local,yaw,active,delta,actual,complete = first
    torch.testing.assert_close((local[...,2]+b[ids,:,0,2])[active], torch.full_like(local[...,2][active],1.05))
    assert (actual <= torch.tensor([3,3])).all()
    assert torch.equal(complete,(actual == torch.tensor([3,3])).all(-1))
    with pytest.raises(ValueError,match='Uninitialized'):
        bank.sample(ids,gen,1,1,0.)


def test_reset_updates_only_selected_rows_and_preserves_bank(configs,monkeypatch):
    m = configs('event_functions','wrench')
    layout = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_layout')
    n=4
    b=bounds(n)
    rng=torch.Generator().manual_seed(42)
    bank=layout.WrenchLayoutBank(b,rng,size=2,stages=((30,3,3),))
    state=torch.zeros(n,10,13)
    state[...,3]=1
    gravity_calls=[]
    objects=NS(num_objects=10,root_physx_view=NS(
        set_disable_gravities=lambda flags,rows: gravity_calls.append((flags.clone(),rows.clone())),
        wake_up=lambda rows: None))
    def write(values,env_ids): state[env_ids]=values
    objects.write_object_link_state_to_sim=write
    root=NS(root_pos_w=torch.zeros(n,3),root_quat_w=torch.tensor([[1.,0,0,0]]).repeat(n,1))
    env=NS(num_envs=n,device='cpu',_revision_bounds_t=b,_revision_layout_rng=rng,
           _wrench_layout_bank=bank,_wrench_gravity_flags=torch.ones(n*10,1,dtype=torch.uint8),
           scene={'object_collection':objects,'robot':NS(data=root),'shelf':NS(data=root)},
           _active_object_mask=torch.zeros(n,10,dtype=torch.bool),
           target_id=torch.zeros(n,1,dtype=torch.long),target_width=torch.zeros(n,1),
           sweep_dir=torch.zeros(n,3),_initial_obstacle_observation=torch.zeros(n,64),
           _final_actual_population=torch.zeros(n,2,dtype=torch.long),
           _revision_layout_complete=torch.zeros(n,dtype=torch.bool),
           _revision_spawn_positions=torch.zeros(n,10,3),sensor_data_fresh=torch.ones(n,dtype=torch.bool))
    monkeypatch.setattr(m,'apply_final_visibility',lambda *a:None)  # independently covered existing helper
    monkeypatch.setattr(layout,'wrench_layout',lambda *a,**kw:pytest.fail('search called on reset'))
    ids=torch.tensor([1,3]); other=torch.tensor([0,2])
    before=state.clone()
    m.reset_wrench_scene(env,ids,gray_obstacles=3,white_obstacles=3)
    torch.testing.assert_close(state[other],before[other])
    assert env.sensor_data_fresh.tolist()==[True,False,True,False]
    assert env._active_object_mask[ids,0].all()
    assert (state[ids,:,7:]==0).all()
    flags,rows=gravity_calls[-1]
    assert set((rows%n).tolist())=={1,3}
    assert (flags.reshape(10,n)[:,other]==1).all()
    torch.testing.assert_close(flags[rows,0],(~env._active_object_mask[ids]).T.flatten().to(torch.uint8))
    first_obs=env._initial_obstacle_observation.clone()
    m.reset_wrench_scene(env,ids,gray_obstacles=3,white_obstacles=3)
    assert not torch.equal(first_obs[ids],env._initial_obstacle_observation[ids])
    assert not env._initial_obstacle_observation[other].any()


def test_new_height_limit_gates_all_three_rewards(configs,monkeypatch):
    from test_revision_v2 import height_env
    funcs=configs('reward_functions','revision')
    env=height_env([.119999,.12,.120001])
    env.cfg.goal_reward_height_limit=env.cfg.contact_height_limit=.12
    env.sensor_data_fresh=torch.ones(3,dtype=torch.bool)
    monkeypatch.setattr(funcs,'exponential_planar_goal_proximity',lambda *a,**kw:torch.ones(3))
    monkeypatch.setattr(funcs,'target_measurements',lambda *a:(torch.zeros(3),torch.zeros(3),torch.ones(3,dtype=torch.bool)))
    monkeypatch.setattr(funcs,'target_hand_contact_force_w',lambda _:torch.ones(3,3))
    for fn in (funcs.goal_distance_v2,funcs.goal_hold_v2,funcs.target_contact_below_height):
        assert fn(env).tolist()==[1,0,0]


def test_complete_env_composition_against_test_v2(configs,omni_config):
    """Execute actual EnvCfg/post-init; only Sim assets and separately checked groups are doubles."""
    from test_omni_configuration import _AssetNode
    cc=omni_config.baseline.configclass
    class Scene(_AssetNode):
        def __init__(self,**kw):
            super().__init__(**kw)
            self.robot=_AssetNode(actuators={'arm':_AssetNode()},spawn=_AssetNode())
            self.shelf=_AssetNode(spawn=_AssetNode())
            self.object_collection=_AssetNode(rigid_objects={str(i):_AssetNode(spawn=_AssetNode()) for i in range(10)})
            self.ee_frame=_AssetNode(target_frames=[_AssetNode(prim_path='Robot/axia80_link',offset=_AssetNode(pos=(0,0,.14),rot=(1,0,0,0)))])
        def configure_population_capacity(self,*args): assert args==(4,5,16)
    def actions(): return _AssetNode(arm_action=_AssetNode())
    def commands(): return _AssetNode(target_goal_pos=_AssetNode())
    ns=dict(vars(omni_config.baseline))
    ns.update(GOAL_SUCCESS_RADIUS=.03,HOLD_OUTER_RADIUS=.08,CONTACT_HEIGHT_LIMIT=.12,
              RevisionSceneCfg=Scene,RevisionV1TestSceneCfg=Scene,RevisionV2TestSceneCfg=Scene,
              WrenchSceneCfg=Scene,RevisionActionsCfg=actions,WrenchActionsCfg=actions,
              RevisionCommandsCfg=commands,WrenchCommandsCfg=commands,
              RevisionObservationsCfg=_AssetNode,WrenchObservationsCfg=_AssetNode,
              RevisionOSCActionCfg=NS(OffsetCfg=_AssetNode),WrenchOSCActionCfg=NS(OffsetCfg=_AssetNode))
    for role,names in {
        'rewards':('RevisionRewardsCfg','RevisionV1RewardsCfg','RevisionV2RewardsCfg','WrenchRewardsCfg'),
        'curricula':('RevisionCurriculumCfg','RevisionV1CurriculumCfg','WrenchCurriculumCfg'),
        'terminations':('RevisionTerminationsCfg','WrenchTerminationsCfg'),
        'events':('RevisionEventsCfg','WrenchEventsCfg'),
    }.items():
        for name in names: ns[name]=getattr(configs(role,'wrench' if name.startswith('Wrench') else 'revision'),name)
    old_module=_execute_definitions(MODULE_ROOT/'envs/revision_envs.py',ns)
    new_module=_execute_definitions(MODULE_ROOT/'envs/wrench_envs.py',ns,{'WrenchEnvCfg'})
    old,new=old_module.RevisionV2TestEnvCfg(),new_module.WrenchEnvCfg()
    before=old.to_dict()
    allowed={'goal_reward_height_limit','contact_height_limit','red_margin','rear_offset',
             'layout_bank_size','layout_build_batch_size','rewards','terminations','events','curriculum'}
    old_dict,new_dict=old.to_dict(),new.to_dict()
    for key in old_dict.keys()|new_dict.keys():
        if key not in allowed: assert_ordered_equal(old_dict[key],new_dict[key])
    assert new.goal_reward_height_limit==new.contact_height_limit==.12
    assert new.events.object_spawn.params==old.events.object_spawn.params
    assert old_module.RevisionV2TestEnvCfg().to_dict()==before


@pytest.mark.parametrize('degrees,gray,white',[(30,3,3),(25,3,4),(25,4,4),(20,4,4),(20,4,5),(15,4,5),(10,4,5),(5,4,5),(0,4,5)])
def test_wrench_geometry_for_every_curriculum_stage(load_mdp,degrees,gray,white):
    m=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_layout')
    obb=importlib.import_module('_baseline_cpu_pkg.mdp.common.bbox_scene_layout')
    b=bounds(16)
    local,yaw,active,delta,result=m.wrench_layout(b,torch.Generator().manual_seed(17),gray,white,
                                               excluded_x_half_angle=math.radians(degrees))
    half,offset=obb.footprint_from_bounds(b,yaw)
    centers=local[...,:2]+offset
    signs=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    corners=torch.einsum('bmki,bmij->bmkj',half[:,:,None]*signs,obb.axes(yaw))+centers[:,:,None]
    assert (corners[...,0].abs()[active]<=.195001).all()
    assert (corners[...,1].abs()[active]<=.495001).all()
    for i in range(10):
        for j in range(i+1,10):
            both=active[:,i]&active[:,j]
            assert obb.separated(centers[:,i],half[:,i],yaw[:,i],centers[:,j],half[:,j],yaw[:,j],.02-1e-6)[both].all()
    to_board=lambda xy:torch.stack((xy[...,1],.20-xy[...,0]),-1)
    red,g,w=obb.zone_masks(to_board(centers[:,1:]),half[:,1:],yaw[:,1:]-math.pi/2,
        to_board(centers[:,0]),to_board(centers[:,0]+delta),half[:,0],yaw[:,0]-math.pi/2,
        obb.LayoutConfig(gray_padding=.065,red_margin=.023,rear_offset=.02))
    assert not (red&active[:,1:]).any()
    assert g[:,:4][active[:,1:5]].all()
    assert w[:,4:][active[:,5:]].all()
    assert ((delta[:,1].abs()/delta.norm(dim=-1))>=math.sin(math.radians(degrees))-1e-6).all()
