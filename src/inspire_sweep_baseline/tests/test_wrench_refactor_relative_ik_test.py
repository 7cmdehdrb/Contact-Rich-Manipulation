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
    old['timeout']['func'] = new['timeout']['func']  # telemetry wrapper preserves time_out
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
            if isinstance(node,ast.ClassDef) and not node.name.startswith('WrenchRefactorRelativeIKV0_2_') and node.name != 'GroupedResetObjectSpawnerCfg':
                assert all(isinstance(b,ast.Name) and b.id in allowed for b in node.bases), path
            if isinstance(node,ast.ImportFrom):
                for alias in node.names:
                    if alias.name.endswith('Cfg') and node.level:
                        assert alias.name.startswith(('WrenchRefactorRelativeIKV0_2Test', 'WrenchRefactorRelativeIKV0_2_')) or 'SpawnerCfg' in alias.name, (path,alias.name)


def test_child_only_changes_requested_rewards_and_schedules(configs):
    module = configs('rewards', STEM)
    parent = module.WrenchRefactorRelativeIKV0_2TestRewardsCfg().to_dict()
    child = module.WrenchRefactorRelativeIKV0_2_1TestRewardsCfg().to_dict()
    expected = copy.deepcopy(parent)
    expected.update(object=None, eef_linear_speed=None, pushing_speed=None)
    expected['object_tilt'] = child['object_tilt']
    expected['obstacle_displacement'] = child['obstacle_displacement']
    assert child == expected
    assert child['object_tilt']['weight'] == -30.
    assert child['obstacle_displacement']['weight'] == -150.
    assert child['obstacle_displacement']['params'] == {'displacement_max_cost': .04}
    assert module.WrenchRefactorRelativeIKV0_2TestRewardsCfg().to_dict() == parent
    schedules = configs('curricula', STEM)
    expected = schedules.WrenchRefactorRelativeIKV0_2TestCurriculumCfg().to_dict()
    expected.update(eef_free_speed=None, pushing_free_speed=None)
    assert schedules.WrenchRefactorRelativeIKV0_2_1TestCurriculumCfg().to_dict() == expected


def test_split_object_cost_preserves_sum(configs):
    module = configs('rewards', STEM)
    parent = module.WrenchRefactorRelativeIKV0_2TestRewardsCfg()
    child = module.WrenchRefactorRelativeIKV0_2_1TestRewardsCfg()
    env, _ = goal_env()
    pos = env.scene['object_collection'].data.object_pos_w
    env._initial_object_positions_w = torch.zeros_like(pos)
    pos[:, 1:, 0] = torch.tensor([0., .01, .2, 1.])[:, None]
    quat = env.scene['object_collection'].data.object_quat_w
    angle = torch.tensor([0., .0872664626, .4, .9])[:, None]
    quat[..., 0] = (angle / 2).cos()
    quat[..., 1] = (angle / 2).sin()
    old = parent.object.weight * parent.object.func(env, **parent.object.params)
    new = sum(term.weight * term.func(env, **term.params)
              for term in (child.object_tilt, child.obstacle_displacement))
    torch.testing.assert_close(old, new)


def test_metrics_temporal_mean_peak_and_partial_reset():
    # Execute the real runtime class with only its Isaac Lab base substituted.
    tree = ast.parse((MODULE_ROOT / f'runtime/{STEM}.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    class Base:
        def _reset_idx(self, ids):
            pass
    ns = {'torch': torch, 'ManagerBasedRLEnv': Base}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<runtime-metrics>', 'exec'), ns)
    cls = ns['WrenchRefactorRelativeIKV0_2TestEnv']
    env = cls.__new__(cls)
    env._metric_eef_body = 0
    env.cfg = NS(goal_success_radius=.03)
    env._metric_sums = torch.zeros(2,4)
    env._metric_counts = torch.zeros(2,4)
    env._metric_peaks = torch.zeros(2,2)
    env._metric_final_distance = torch.zeros(2)
    env._metric_min_distance = torch.full((2,), float('inf'))
    env.sensor_data_fresh = torch.ones(2,dtype=torch.bool)
    env.extras = {'log': {}}
    env.action_manager = NS(_action=torch.zeros(2,8), _prev_action=torch.zeros(2,8))
    objects = NS(object_pos_w=torch.zeros(2,17,3), object_lin_vel_w=torch.zeros(2,17,3))
    robot = NS(body_link_pos_w=torch.zeros(2,1,3), body_link_lin_vel_w=torch.zeros(2,1,3),
               body_link_ang_vel_w=torch.zeros(2,1,3))
    env.scene = {'robot':NS(data=robot), 'object_collection':NS(data=objects),
                 'ee_frame':NS(data=NS(target_pos_w=torch.zeros(2,1,3)))}
    env._initial_object_positions_w = torch.zeros(2,17,3)
    env.command_manager = NS(get_command=lambda _: torch.zeros(2,3))
    for distances, eef, target in (([.02,.04],[1.,2.],[.5,1.]), ([.03,.01],[3.,4.],[1.5,2.])):
        objects.object_pos_w[:,0,0] = torch.tensor(distances)
        robot.body_link_lin_vel_w[:,0,0] = torch.tensor(eef)
        objects.object_lin_vel_w[:,0,0] = torch.tensor(target)
        env.record_goal_accuracy_metrics()
    env._reset_idx(torch.tensor([0]))
    log = env.extras['log']
    expected = {'target_goal_distance_xy_m_mean':.025, 'target_goal_distance_xy_m_min':.02,
                'target_goal_distance_xy_m_final':.03, 'eef_linear_speed_m_s_mean':2.,
                'eef_linear_speed_m_s_episode_max':3., 'target_linear_speed_m_s_mean':1.,
                'target_linear_speed_m_s_episode_max':1.5, 'target_within_goal_radius_fraction_mean':1.}
    assert {k:float(v) for k,v in log.items()} == pytest.approx({'Metrics/'+k:v for k,v in expected.items()})
    assert not env._metric_counts[0].any()
    assert (env._metric_counts[1] == 2).all()
    objects.object_pos_w[0,0,0] = float('nan')
    env.record_goal_accuracy_metrics()
    env._reset_idx(torch.tensor([0]))
    assert all(torch.isfinite(v) for v in env.extras['log'].values())


def test_v022_only_changes_contact_reward(configs):
    module = configs('rewards', STEM)
    before = module.WrenchRefactorRelativeIKV0_2_1TestRewardsCfg().to_dict()
    new = module.WrenchRefactorRelativeIKV0_2_2TestRewardsCfg()
    expected = copy.deepcopy(before)
    expected['hand_contact'] = new.hand_contact.to_dict()
    assert_ordered_equal(new.to_dict(), expected)
    assert new.hand_contact.func.__name__ == 'TargetContactMaintenance'
    assert new.hand_contact.weight == .1
    assert new.hand_contact.params == {'threshold_n': .01, 'eef_height_limit': .15}
    assert module.WrenchRefactorRelativeIKV0_2_1TestRewardsCfg().to_dict() == before
    assert new.eef_linear_speed is None and new.pushing_speed is None


def contact_count_fixture(configs, n=2):
    from test_wrench_v0_1 import reward_env
    env, pos, force = reward_env(n)
    # Contact uses its own 15 cm parameter, independent of other reward gates.
    del env.cfg
    pos[:, 0, :2] = 0.  # Being at the goal must not fabricate contact.
    term = configs('reward_functions', STEM).TargetContactMaintenance(NS(params={}), env)
    return env, force, term


def test_target_count_window_cap_and_five_consecutive_rejections(configs):
    env, force, term = contact_count_fixture(configs)
    force.zero_()
    assert not term(env).any()  # goal occupancy without force is zero
    for index in range(12):
        env.common_step_counter += 1
        force[:,0,0,0] = .01
        result = term(env)
        assert result.tolist() == [float(min(index,10))]*2
        torch.testing.assert_close(term(env),result)  # duplicate call never increments
    force.zero_()
    for index in range(6):
        env.common_step_counter += 1
        result = term(env)
        if index == 0:
            assert result.tolist() == [10.,10.]  # two preceding contacts still in window
        else:
            assert not result.any()
        assert term._count.tolist() == ([0,0] if index == 5 else [10,10])


def test_target_count_partial_reset_freshness_and_skipped_step(configs):
    env, force, term = contact_count_fixture(configs)
    for _ in range(4):
        term(env);env.common_step_counter += 1
    term.reset(torch.tensor([0]))
    assert term(env).tolist() == [0.,4.]
    env.common_step_counter += 1
    assert term(env).tolist() == [1.,5.]
    env.common_step_counter += 2  # missing window cannot preserve continuity
    assert not term(env).any()
    env.sensor_data_fresh[:] = False
    for _ in range(3):
        env.common_step_counter += 1
        assert not term(env).any()
    env.sensor_data_fresh[:] = True
    force[:] = float('nan')
    for _ in range(3):
        env.common_step_counter += 1
        assert not term(env).any()


def test_only_target_hand_force_counts(configs):
    env, force, term = contact_count_fixture(configs)
    env.scene['object_collection'].object_names.append('obstacle')
    env.scene['hand_contact_obstacle'] = NS(data=NS(force_matrix_w=torch.ones_like(force)*100))
    force.zero_()
    for _ in range(4):
        assert not term(env).any()
        env.common_step_counter += 1


@pytest.mark.parametrize('height', [.15-1e-6, .15, .15+1e-6, float('nan'), float('inf')])
def test_target_contact_height_boundary_and_immediate_gate(configs, height):
    env, force, term = contact_count_fixture(configs)
    for _ in range(4):
        term(env)
        env.common_step_counter += 1
    before = term._count.clone()
    # A nonzero shelf world height proves that the limit is relative, not world Z.
    env._revision_initial_shelf_pos[:,2] = .3
    env.scene['ee_frame'].data.target_pos_w[:,0,2] = .3 + 1.05 + height
    valid = height < .15
    for i in range(5):
        reward = term(env)
        if valid:
            torch.testing.assert_close(reward, (before+i+1).clamp(max=10).float())
        else:
            assert not reward.any()  # old 2-of-3 history cannot bypass the ceiling
            torch.testing.assert_close(term._count, torch.zeros_like(before) if i==4 else before)
        env.common_step_counter += 1


def test_v0221_only_changes_rejection_reset(configs):
    module = configs('rewards', STEM)
    parent = module.WrenchRefactorRelativeIKV0_2_2TestRewardsCfg().to_dict()
    child = module.WrenchRefactorRelativeIKV0_2_2_1TestRewardsCfg()
    expected = copy.deepcopy(parent)
    expected['hand_contact']['params']['reset_after_misses'] = 1
    assert_ordered_equal(child.to_dict(), expected)
    assert module.WrenchRefactorRelativeIKV0_2_2TestRewardsCfg().to_dict() == parent


def test_single_rejection_resets_count_but_keeps_two_of_three(configs):
    env, force, term = contact_count_fixture(configs)
    params = {'reset_after_misses': 1}
    for _ in range(5):
        term(env, **params)
        env.common_step_counter += 1
    assert term._count.tolist() == [4, 4]
    force.zero_()
    # One raw missed contact still passes the two-of-three window.
    assert term(env, **params).tolist() == [5., 5.]
    env.common_step_counter += 1
    assert not term(env, **params).any()
    assert term._count.tolist() == [0, 0]
    force[:, 0, 0, 0] = .01
    env.common_step_counter += 1
    assert not term(env, **params).any()
    env.common_step_counter += 1
    assert term(env, **params).tolist() == [1., 1.]


def test_v0222_only_adds_doubled_inverse_distance(configs):
    module = configs('rewards', STEM)
    parent = module.WrenchRefactorRelativeIKV0_2_2_1TestRewardsCfg().to_dict()
    child = module.WrenchRefactorRelativeIKV0_2_2_2TestRewardsCfg()
    reference = configs('rewards', 'wrench_refactor').WrenchRefactorRelativeRewardsCfg().target_inverse_distance
    expected = copy.deepcopy(parent)
    expected_term = reference.to_dict()
    expected_term['weight'] *= 2
    expected['target_inverse_distance'] = expected_term
    assert_ordered_equal(child.to_dict(), expected)
    assert module.WrenchRefactorRelativeIKV0_2_2_1TestRewardsCfg().to_dict() == parent
    from test_wrench_refactor_relative import motion_reward_env
    env, pos, forces, goal = motion_reward_env(2)
    forces.zero_()
    cfg = child.target_inverse_distance
    term = cfg.func(cfg, env)
    # Initial reverse movement versus progress toward the goal.
    pos[:, 0, 0] = torch.tensor([-.01, .01])
    torch.testing.assert_close(cfg.weight * term(env) * env.step_dt, pos.new_tensor([-.2594, 0.]))
    env.common_step_counter += 1
    pos[:, 0, :2] = goal[:, :2]
    assert not term(env).any()
    # Overshoot starting at the goal is charged even inside the 3 cm radius.
    env.common_step_counter += 1
    pos[:, 0, 0] += .01
    torch.testing.assert_close(cfg.weight * term(env) * env.step_dt, pos.new_full((2,), -.2594))
    env.common_step_counter += 1
    pos[:, 0, 2] += .1
    assert not term(env).any()  # Only XY distance contributes.


def test_v0223_contact_decay_and_no_holding(configs):
    rewards = configs('rewards', STEM)
    parent = rewards.WrenchRefactorRelativeIKV0_2_2_2TestRewardsCfg().to_dict()
    cfg = rewards.WrenchRefactorRelativeIKV0_2_2_3FastTestRewardsCfg()
    expected = copy.deepcopy(parent)
    expected['hand_contact'] = cfg.hand_contact.to_dict()
    expected['goal_hold'] = None
    assert_ordered_equal(cfg.to_dict(), expected)
    assert cfg.hand_contact.weight == .1
    assert cfg.hand_contact.params == {'threshold_n': .01, 'eef_height_limit': .15}
    assert rewards.WrenchRefactorRelativeIKV0_2_2_2TestRewardsCfg().to_dict() == parent
    curricula = configs('curricula', STEM)
    before = curricula.WrenchRefactorRelativeIKV0_2_1TestCurriculumCfg().to_dict()
    expected = copy.deepcopy(before)
    expected['goal_speed_scale'] = None
    assert curricula.WrenchRefactorRelativeIKV0_2_2_3FastTestCurriculumCfg().to_dict() == expected
    assert curricula.WrenchRefactorRelativeIKV0_2_1TestCurriculumCfg().to_dict() == before
    env, force, _ = contact_count_fixture(configs)
    term = cfg.hand_contact.func(cfg.hand_contact, env)
    for i in range(25):
        result = term(env, **cfg.hand_contact.params)
        assert result.tolist() == [float(min(i, 20))]*2
        torch.testing.assert_close(term(env), result)
        env.common_step_counter += 1
    force.zero_()
    assert term(env).tolist() == [20., 20.]  # raw miss still accepted by 2-of-3
    for i in range(12):
        env.common_step_counter += 1
        assert not term(env).any()  # rejected windows pay zero but retain decayed count
        assert term._count.tolist() == [max(20 - 2*(i+1), 0)]*2
        term(env)  # same-step cache must not decay twice
        assert term._count.tolist() == [max(20 - 2*(i+1), 0)]*2
    force[:, 0, 0, 0] = .01
    for i in range(6):
        env.common_step_counter += 1
        assert term(env).tolist() == [float(i)]*2
    term.reset(torch.tensor([0]))
    assert term._count.tolist() == [0, 5]
    env.common_step_counter += 1
    assert term(env).tolist() == [0., 6.]
    env.common_step_counter += 1
    assert term(env).tolist() == [1., 7.]
    env.scene['ee_frame'].data.target_pos_w[:, 0, 2] = 1.05 + .15
    env.common_step_counter += 1
    assert not term(env).any()  # height gate overrides past contacts immediately
    assert term._count.tolist() == [0, 5]


def test_contact_decay_reacquisition_keeps_remaining_count(configs):
    env, force, _ = contact_count_fixture(configs)
    cls = configs('reward_functions', STEM).TargetContactDecayMaintenance
    term = cls(NS(params={}), env)
    for _ in range(25):
        term(env)
        env.common_step_counter += 1
    force.zero_()
    assert term(env).tolist() == [20., 20.]  # first raw miss still passes 2-of-3
    env.common_step_counter += 1
    assert not term(env).any()
    assert term._count.tolist() == [18, 18]
    force[:, 0, 0, 0] = .01
    env.common_step_counter += 1
    assert not term(env).any()  # first returning contact: only 1-of-3
    assert term._count.tolist() == [16, 16]
    env.common_step_counter += 1
    assert term(env).tolist() == [17., 17.]  # reacquisition resumes from remaining count


def test_v0224_success_only_at_deadline_strict_xy_and_failure_priority(configs, monkeypatch):
    n = 8
    positions = torch.zeros(n, 17, 3)
    positions[:, 2, 0] = torch.tensor([.029999, .03, .030001, .01, .01, float('nan'), .0, .01])
    positions[:, 2, 2] = 50.  # XY only; failure term independently gates invalid episodes
    failures = torch.tensor([False, False, False, False, True, False, False, False])
    env = NS(num_envs=n, device='cpu', max_episode_length=500,
             episode_length_buf=torch.tensor([500, 500, 500, 499, 500, 500, 0, 501]),
             target_id=torch.full((n, 1), 2),
             scene={'object_collection': NS(data=NS(object_pos_w=positions))},
             command_manager=NS(get_command=lambda _: torch.zeros(n, 3)),
             termination_manager=NS(get_term=lambda name: failures if name == 'shelf_failure' else torch.zeros(n, dtype=torch.bool)))
    module = configs('termination_functions', STEM)
    # Use the installed Isaac Lab timeout function; the generic CPU fixture is a no-op.
    path = MODULE_ROOT.parents[2] / 'IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py'
    tree = ast.parse(path.read_text())
    timeout_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'time_out')
    ns = {'ManagerBasedRLEnv': object, 'torch': torch}
    exec(compile(ast.Module(body=[timeout_node], type_ignores=[]), str(path), 'exec'), ns)
    monkeypatch.setattr(module, 'time_out', ns['time_out'])
    assert module.success_at_timeout(env).tolist() == [True, False, False, False, False, False, False, True]
    cfg = configs('terminations', STEM).WrenchRefactorRelativeIKV0_2_2_4TerminationsCfg()
    assert cfg.success.time_out  # preserve timeout bootstrapping, diagnostic classification only
    assert list(cfg.to_dict())[-1] == 'success'  # failure terms computed first


def test_v0224_timeout_ratio_counts_episodes_before_reset_and_training_no_overlay():
    tree = ast.parse((MODULE_ROOT / f'runtime/{STEM}.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WrenchRefactorRelativeIKV0_2_2_4Env')
    class Base:
        def _reset_idx(self, ids):
            self.episode_length_buf[ids] = 0
            self.outcomes[ids] = False
            self.extras = {'log': {}}
    ns = {'torch': torch, 'WrenchRefactorRelativeIKV0_2_2_2FastTestEnv': Base}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<timeout-metrics>', 'exec'), ns)
    cls = ns[node.name]
    env = cls.__new__(cls)
    env.cfg = NS(debug_spawn_regions=False)
    env._update_spawn_regions = lambda ids: pytest.fail('training must not build visual regions')
    env._timeout_outcome_totals = torch.zeros(2, dtype=torch.long)
    env.max_episode_length = 500
    env.episode_length_buf = torch.tensor([500, 500, 125, 0])
    env.outcomes = torch.tensor([True, False, False, False])
    env.termination_manager = NS(get_term=lambda _: env.outcomes)
    env._reset_idx(torch.arange(4))
    assert env._timeout_outcome_totals.tolist() == [1, 2]
    assert env.extras['log']['Metrics/timeout_success_rate'] == .5
    env._reset_idx(torch.arange(4))  # explicit reset/zero length must not count stale outcomes
    assert env._timeout_outcome_totals.tolist() == [1, 2]
    env.episode_length_buf[1] = 500
    env.outcomes[1] = True
    env._reset_idx(torch.tensor([1]))
    assert env._timeout_outcome_totals.tolist() == [2, 3]
    assert env.extras['log']['Metrics/timeout_success_rate'] == pytest.approx(2/3)
