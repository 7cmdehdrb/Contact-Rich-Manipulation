"""v1.4 schedule boundaries, live success predicate, inheritance and full clutter."""

from _source_units import import_unit

import importlib
import math
import runpy
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions, omni_config
from test_clutter_fusion_v1_0_population import population_modules
from test_clutter_layout import validate_scenes

ROOT = MODULE_ROOT / "contact_sweep_clutter_fusion_v1_4"
schedule = runpy.run_path(str(MODULE_ROOT / "mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py"))["staged_reward_threshold"]


@pytest.fixture
def cfgs(omni_config):
    cc = omni_config.baseline.configclass
    terms = omni_config.terms
    placeholder = lambda *args, **kwargs: None
    ns = dict(configclass=cc, CurriculumTermCfg=terms.RewardTermCfg,
              delayed_reward_schedule=placeholder, goal_deadzone_schedule=placeholder,
              obstacle_population_schedule=placeholder, staged_reward_threshold=schedule)
    ns.update(vars(_execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion_v1_0/curriculum_cfg.py", ns)))
    curriculum = _execute_definitions(ROOT / "curriculum_cfg.py", ns).ContactSweepClutterFusionV1_4CurriculumCfg()
    return curriculum


@pytest.mark.parametrize("name,checks", [
    ("arm_free_speed", [(0,1.),(499,1.),(500,.98),(749,.98),(750,.955),(3250,.705),(3299,.705),(3300,.7)]),
    ("eef_free_speed", [(0,1.5),(500,1.48),(750,1.455),(13250,.205),(13299,.205),(13300,.2)]),
    ("pushing_free_speed", [(0,.1),(499,.1),(500,.09),(750,.0775),(1750,.0275),(1899,.0275),(1900,.02)]),
    ("object_free_acceleration", [(0,100.),(9999,100.),(10000,50.),(11999,50.),(12000,50.),(12249,50.),(12250,50.-40/24),(15000,30.),(17999,10.+40/24),(18000,10.)]),
    ("wrench_free_force_change", [(0,5000.),(9999,5000.),(10000,200.),(11999,200.),(12000,200.),(12250,200.-100/24),(15000,150.),(18000,100.)]),
    ("wrench_free_moment_change", [(0,500.),(9999,500.),(10000,30.),(11999,30.),(12000,30.),(12250,30.-10/24),(15000,25.),(18000,20.)]),
    ("goal_hold_tolerance", [(0,0.),(4999,0.),(5000,.01),(5249,.01),(5250,.00975),(9999,.00525),(10000,.005)]),
])
def test_actual_schedule_boundaries_and_rewind(cfgs, name, checks):
    term = getattr(cfgs, name)
    reward = SimpleNamespace(params={}, weight=7.)
    env = SimpleNamespace(reward_manager=SimpleNamespace(
        get_term_cfg=lambda _: reward, set_term_cfg=lambda *_: None))
    for iteration, expected in checks + [(90000, checks[-1][1])] + list(reversed(checks)):
        env.learning_iteration = iteration
        for ids in (None, [0], [1,3]):
            assert term.func(env, ids, **term.params) == pytest.approx(expected)
            assert reward.params[term.params["parameter"]] == pytest.approx(expected)
            assert reward.weight == 7.


def test_population_and_deadzone_preserved(cfgs, population_modules):
    p = cfgs.obstacle_population.params
    assert p == dict(initial_gray=1, initial_white=1, final_gray=5, final_white=5,
                     gray_start_iteration=4500, white_start_iteration=3000, interval=3000)
    assert cfgs.goal_deadzone.params == dict(start_degrees=30., hold_iterations=3000,
                                           interval=500, decrement_degrees=5.)
    event = SimpleNamespace(params={})
    env = SimpleNamespace(cfg=SimpleNamespace(max_gray_obstacles=5, max_white_obstacles=5),
                          event_manager=SimpleNamespace(get_term_cfg=lambda _:event, set_term_cfg=lambda *_:None))
    for i,g,w in [(0,1,1),(3000,1,2),(4500,2,2),(9000,3,4),(10500,4,4),(12000,4,5),(13499,4,5),(13500,5,5),(0,1,1)]:
        env.learning_iteration=i
        assert population_modules.population.obstacle_population_schedule(env,None,**p) == dict(gray_obstacles=g,white_obstacles=w)


def test_success_continuous_time_exit_live_tolerance_failure_and_partial_reset(load_mdp, make_env):
    load_mdp("goal_rewards")
    mdp = import_unit("contact_sweep_clutter_fusion_v1_4.mdp", package="_baseline_cpu_pkg")
    env = make_env(3)
    env.step_dt = .02
    reward = SimpleNamespace(params={"position_tolerance":0.})
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda _:reward)
    failures = torch.zeros(3, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(get_term=lambda _:failures)
    goal = env.command_manager.get_command("target_goal_pos")
    env.scene["object_collection"].data.object_pos_w[:,0,:] = goal
    success = mdp.GoalHoldSuccess(SimpleNamespace(),env)
    for _ in range(160):
        assert not success(env).any()
    reward.params["position_tolerance"] = .01
    for _ in range(149):
        assert not success(env).any()
    assert success(env).all()  # 150 * .02 = 3 s
    success.reset([1])
    assert success(env).tolist() == [True,False,True]
    env.scene["object_collection"].data.object_pos_w[0,0,0] += .02
    assert not success(env)[0]
    env.scene["object_collection"].data.object_pos_w[0,0,0] -= .02
    assert not success(env)[0]
    # Tightening the live radius invalidates the accumulated occupancy.
    env.scene["object_collection"].data.object_pos_w[2,0,0] += .007
    reward.params["position_tolerance"] = .005
    assert not success(env)[2]
    failures[:] = True
    assert not success(env,failure_terms=("push_fast",)).any()
    success.reset()
    assert not success._hold_steps.any()


def test_direct_inheritance_registration_and_final_override(omni_config):
    cc = omni_config.baseline.configclass
    @cc
    class Parent:
        terminations = SimpleNamespace(push_fast=SimpleNamespace(params={"speed_condition":.8}))
        def __post_init__(self):
            self.terminations.push_fast.params["speed_condition"] = .8
    ns = dict(configclass=cc, ContactSweepClutterFusionV1_0EnvCfg=Parent,
              ContactSweepClutterFusionV1_4RewardsCfg=SimpleNamespace,
              ContactSweepClutterFusionV1_4TerminationsCfg=lambda: SimpleNamespace(push_fast=SimpleNamespace(params={})),
              ContactSweepClutterFusionV1_4CurriculumCfg=SimpleNamespace)
    cls = _execute_definitions(ROOT / "env_cfg.py",ns).ContactSweepClutterFusionV1_4EnvCfg
    child=cls()
    assert cls.__bases__ == (Parent,)
    assert (child.max_gray_obstacles,child.max_white_obstacles)==(5,5)
    assert child.terminations.push_fast.params["speed_condition"] == 1.
    assert Parent().terminations.push_fast.params["speed_condition"] == .8
    registry={}
    _execute_definitions(MODULE_ROOT / "__init__.py",dict(gym=SimpleNamespace(
        registry=registry,register=lambda **kw:registry.update({kw["id"]:kw}))))
    spec=registry["Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_4"]
    assert spec["entry_point"].endswith("runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env")
    assert spec["kwargs"]["env_cfg_entry_point"].endswith("envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_4EnvCfg")


def test_five_gray_five_white_geometry(population_modules, validate_scenes):
    mod=population_modules
    torch.manual_seed(20261023)
    count=128
    target_xy=mod.samplers._sample_uniform_disk(count,.02,"cpu")
    direction=mod.samplers._sample_planar_unit_vectors(count,"cpu",0.)
    target=mod.coordinates.shelf_xy_to_generator_cm(target_xy.numpy())
    distance=mod.samplers._goal_ray_distance(target_xy,direction,.15,(.14,.44))
    goal=mod.coordinates.shelf_xy_to_generator_cm((target_xy+distance[:,None]*direction).numpy())
    generator=mod.layout.population_layout_generator(SimpleNamespace(cfg=SimpleNamespace(seed=19)),5,5)
    batch=generator.generate_batch(target,goal)
    assert batch.obstacles.shape == (count,10,2)
    report=validate_scenes(batch)
    assert report["valid"],report


def test_rewards_and_terminations_inherit_without_parent_mutation(omni_config, load_mdp):
    cc=omni_config.baseline.configclass
    placeholder=lambda *args,**kwargs:None
    ns=dict(configclass=cc,math=math,RewTerm=omni_config.terms.RewardTermCfg,
            SceneEntityCfg=lambda name,**kw:SimpleNamespace(name=name,**kw),
            HAND_JOINT_NAMES=["hand"],arm_entity=lambda:SimpleNamespace(name="robot"),
            goal_rewards=load_mdp("goal_rewards"),
            dex_rewards=SimpleNamespace(DexReachProgress=placeholder,DexPlanarGoalProgress=placeholder),
            scheduled_goal_hold_xy=placeholder)
    for name in ("combined_active_object_cost","eef_linear_speed_cost","goal_hold_xy","shelf_failure",
                 "ShelfCollisionPenalty","ObjectAccelerationPenalty","WrenchMagnitudeChangePenalty",
                 "hand_contact_any","obstacle_contact_force_cost"):
        ns[name]=placeholder
    parent_dir=MODULE_ROOT/"contact_sweep_clutter_fusion_v1_0"
    parent_cls=_execute_definitions(parent_dir/"rewards_cfg.py",ns).ContactSweepClutterFusionV1_0RewardsCfg
    child_cls=_execute_definitions(ROOT/"rewards_cfg.py",ns).ContactSweepClutterFusionV1_4RewardsCfg
    parent,child=parent_cls(),child_cls()
    for name in vars(parent):
        if name not in ("goal_hold","hand_contact"):
            a,b=getattr(parent,name),getattr(child,name)
            assert a.to_dict()==b.to_dict() if hasattr(a,"to_dict") else a==b
    assert child.goal_hold.weight==pytest.approx(10*parent.goal_hold.weight)
    assert child.hand_contact.weight==pytest.approx(5*parent.hand_contact.weight)
    assert child.goal_hold.params=={"position_tolerance":0.}
    assert parent_cls().goal_hold.params=={"position_tolerance":.01}
    ns.update(DoneTerm=omni_config.terms.TerminationTermCfg,
              TerminationTermCfg=omni_config.terms.TerminationTermCfg,time_out=placeholder,
              any_active_object_fall_or_tilt=placeholder,GoalHoldSuccess=placeholder,
              terminations=SimpleNamespace(push_fast=placeholder,shelf_collision=placeholder,arm_velocity=placeholder),
              goal_terminations=SimpleNamespace(shelf_displacement=placeholder))
    _execute_definitions(parent_dir/"terminations_cfg.py",ns)
    terms=_execute_definitions(ROOT/"terminations_cfg.py",ns).ContactSweepClutterFusionV1_4TerminationsCfg()
    assert terms.push_fast.params=={"speed_condition":1.}
    assert not terms.success.time_out
    assert terms.success.params["hold_time"]==3.
    assert list(vars(terms)).index("success")>list(vars(terms)).index("shelf_displacement")
