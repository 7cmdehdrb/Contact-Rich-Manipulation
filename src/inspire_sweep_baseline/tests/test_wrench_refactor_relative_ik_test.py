"""Independent fusion contract, inclusive boundary and schedule regressions."""
import ast
import copy
from types import SimpleNamespace as NS

import pytest
import torch
from conftest import MODULE_ROOT
from test_final_configuration import configs, assert_ordered_equal
from test_omni_configuration import omni_config
from test_fusion_holding_v2 import goal_env

STEM = 'wrench_refactor_relative_ik_test'
PREFIX = 'WrenchRefactorRelativeIKV0_2Test'


def test_complete_reward_contract_and_isolation(configs):
    parent_cls = configs('rewards', 'contact_sweep_clutter_fusion_holding').FusionHoldingV2RewardsCfg
    new_cls = getattr(configs('rewards', STEM), PREFIX + 'RewardsCfg')
    before = parent_cls().to_dict()
    new = new_cls()
    expected = copy.deepcopy(before)
    expected.pop('arm_speed_excess')
    expected['goal_hold']['params']['position_tolerance'] = .03
    expected['shelf_failure']['func'] = new.to_dict()['shelf_failure']['func']
    assert_ordered_equal(new.to_dict(), expected)
    assert new_cls.__bases__ == (object,)
    new.goal_hold.params['position_tolerance'] = .8
    assert parent_cls().to_dict() == before
    assert new_cls().goal_hold.params['position_tolerance'] == .03


def test_observations_match_symmetric_ik_exactly(configs, monkeypatch):
    import importlib
    import sys
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', importlib.import_module('_omni_official_utils.noise'))
    old = configs('observations', 'wrench_refactor').WrenchRefactorRelativeObservationsCfg()
    new = getattr(configs('observations', STEM), PREFIX + 'ObservationsCfg')()
    assert_ordered_equal(old.to_dict(), new.to_dict())
    assert new.critic is None and new.critic_obstacles is None


def test_events_preserve_scene_reset_order_and_exact_ik(configs):
    old = configs('events', 'wrench_refactor').WrenchRefactorEventsCfg().to_dict()
    new = getattr(configs('events', STEM), PREFIX + 'EventsCfg')().to_dict()
    assert list(new) == [*old, 'robot_initial_pose']
    assert_ordered_equal({k: new[k] for k in old}, old)
    assert new['robot_initial_pose']['params'] == {
        'front_clearance': (.08, .10), 'lateral_offset': (.10, .18),
        'height_jitter': .02, 'height_offset': .01}
    assert new['robot_initial_pose']['mode'] == 'reset'


def test_terminations_exact_ik_failures_without_success(configs):
    old = configs('terminations', 'wrench_refactor').WrenchRefactorTerminationsCfg().to_dict()
    del old['target_goal_success']
    old['arm_joint_velocity']['params']['threshold'] = 4.
    new = getattr(configs('terminations', STEM), PREFIX + 'TerminationsCfg')().to_dict()
    assert_ordered_equal(new, old)


def test_complete_curriculum_contract(configs):
    old = configs('curricula', 'contact_sweep_clutter_fusion_holding').FusionHoldingV2CurriculumCfg().to_dict()
    ik = configs('curricula', 'wrench_refactor').WrenchRefactorRelativeCurriculumCfg().to_dict()
    expected = {k: v for k, v in old.items() if k not in (
        'obstacle_population', 'goal_deadzone', 'goal_position_scale', 'goal_position_tolerance', 'arm_free_speed')}
    expected.update(ik)
    new = getattr(configs('curricula', STEM), PREFIX + 'CurriculumCfg')().to_dict()
    assert new == expected


@pytest.mark.parametrize('iteration,angle,cutoff', [
    (0,30,4), (9999,30,4), (10000,25,4), (10249,25,4),
    (10250,25,3.95), (13000,20,3.4), (19999,10,2.05),
    (20000,10,2), (25000,0,2), (90000,0,2),
])
def test_schedules_keep_radius_fixed_and_restore_on_rewind(configs, iteration, angle, cutoff):
    rewards = getattr(configs('rewards', STEM), PREFIX + 'RewardsCfg')()
    curricula = getattr(configs('curricula', STEM), PREFIX + 'CurriculumCfg')()
    terminations = getattr(configs('terminations', STEM), PREFIX + 'TerminationsCfg')()
    # RewardManager normally instantiates stateful terms before curricula run.
    for term in vars(rewards).values():
        if term is not None and isinstance(term.func, type):
            term.func = NS(reset=lambda: None)
    event = NS(params={})
    env = NS(learning_iteration=iteration,
        reward_manager=NS(get_term_cfg=lambda name: getattr(rewards,name), set_term_cfg=lambda name,cfg: setattr(rewards,name,cfg)),
        termination_manager=NS(get_term_cfg=lambda name: getattr(terminations,name), set_term_cfg=lambda name,cfg: setattr(terminations,name,cfg)),
        event_manager=NS(get_term_cfg=lambda name:event, set_term_cfg=lambda *args:None))
    for current in (90000, iteration):
        env.learning_iteration = current
        for term in vars(curricula).values(): term.func(env,None,**term.params)
    import math
    assert event.params['excluded_x_half_angle'] == pytest.approx(math.radians(angle))
    assert terminations.arm_joint_velocity.params['threshold'] == pytest.approx(cutoff)
    assert rewards.goal_hold.params['position_tolerance'] == .03
    assert rewards.goal_distance.params['position_tolerance'] == 0.


def test_hold_contact_inclusive_boundary_and_no_three_second_budget(configs):
    rewards = getattr(configs('rewards', STEM), PREFIX + 'RewardsCfg')()
    env, goal = goal_env()
    goal.params = rewards.goal_hold.params.copy()
    pos = env.scene['object_collection'].data.object_pos_w
    pos[:,0,0] = torch.tensor([.03-1e-6, .03, .03+1e-6, 0.])
    for _ in range(501):
        hold = rewards.goal_hold.func(env,**rewards.goal_hold.params)
        contact = rewards.hand_contact.func(env,**rewards.hand_contact.params)
    assert (hold > 0).tolist() == [True,True,False,True]
    assert contact.tolist() == [1.,1.,0.,1.]
    assert hold[1] == pytest.approx(.5)
    assert hold[3] == pytest.approx(1.)
    assert not hasattr(rewards, 'target_goal_success_bonus')


def test_shelf_event_preserves_holding_scope_not_displacement(configs):
    fn = configs('reward_functions', STEM).holding_shelf_failure
    shelf = NS(root_pos_w=torch.zeros(4,3), root_vel_w=torch.zeros(4,6))
    wrist = torch.zeros(4,1,3); wrist[:,:,2] = 1.1
    env = NS(step_dt=.02, scene={'shelf':NS(data=shelf), 'wrist_frame':NS(data=NS(target_pos_w=wrist))})
    shelf.root_vel_w[1,0] = .11
    wrist[2,0,2] = 1.07
    shelf.root_pos_w[3,0] = .1  # displacement alone remains free for this reward
    assert (fn(env) * -3.0 * env.step_dt).tolist() == pytest.approx([0.,-3.,-3.,0.])


def test_no_task_configuration_or_runtime_ancestry():
    paths = [MODULE_ROOT / f'mdp/{role}/{STEM}.py' for role in
             ('actions','commands','observations','events','rewards','terminations','curricula')]
    paths += [MODULE_ROOT / f'assets/scenes/{STEM}.py', MODULE_ROOT / f'runtime/{STEM}.py',
              MODULE_ROOT / f'envs/{STEM}_envs.py', MODULE_ROOT / f'agent/{STEM}_ppo_cfg.py']
    allowed = {'ManagerBasedRLEnvCfg','ManagerBasedRLEnv','InteractiveSceneCfg',
        'OperationalSpaceControllerActionCfg','ActionTermCfg','CommandTermCfg','ObsGroup',
        'RslRlMLPModelCfg','RslRlOnPolicyRunnerCfg'}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ClassDef):
                assert all(isinstance(b,ast.Name) and b.id in allowed for b in node.bases), path
            if isinstance(node,ast.ImportFrom):
                for alias in node.names:
                    if alias.name.endswith('Cfg') and node.level:
                        assert alias.name.startswith(PREFIX) or 'SpawnerCfg' in alias.name, (path,alias.name)
