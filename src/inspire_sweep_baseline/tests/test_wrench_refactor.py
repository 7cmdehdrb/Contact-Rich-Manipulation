"""Refactor behavioral checks: distributions, partial reset, contact and models."""
import ast
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict

from conftest import MODULE_ROOT
from test_final_configuration import configs, assert_ordered_equal
from test_omni_configuration import omni_config
from test_wrench_v0_1 import reward_env


def pool_bounds(count):
    half = torch.linspace(.025, .055, 34).reshape(17, 2)
    b = torch.zeros(count, 17, 2, 3)
    b[:, :, 0, :2], b[:, :, 1, :2] = -half, half
    b[:, :, 1, 2] = .12
    b[:, :, :, 0] += .01
    b[:, :, :, 2] -= .025
    b[:, 0, :, 0] -= .01  # centered cylinder target
    return b


@pytest.mark.parametrize('degrees', [0, 5, 10, 15, 20, 25, 30])
def test_uniform_angle_and_conditional_distance(load_mdp, degrees):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    b = pool_bounds(50000)[:, 0]
    alpha = math.radians(degrees)
    target, goal = m.sample_commands(b, torch.Generator().manual_seed(42), excluded_x_half_angle=alpha)
    delta = goal - target
    distance = delta.norm(dim=-1)
    direction = delta / distance[:, None]
    theta = torch.atan2(delta[:, 1], delta[:, 0]).remainder(2 * math.pi)
    assert ((theta.remainder(math.pi) >= alpha - 2e-5) & (theta.remainder(math.pi) <= math.pi - alpha + 2e-5)).all()
    left = (delta[:, 1] > 0).float().mean().item()
    assert .49 < left < .51
    unit_angle = (theta.remainder(math.pi) - alpha) / (math.pi - 2 * alpha)
    histogram = torch.histc(unit_angle, bins=10, min=0, max=1)
    assert ((histogram / len(b) - .1).abs() < .005).all()
    safe = torch.tensor([.2, .5]) - b[:, :, :2].abs().amax(1) - .02
    maximum = ((safe - direction.sign() * target) / direction.abs().clamp_min(1e-12)).amin(-1)
    maximum = maximum.clamp_max(.12)
    unit_distance = (distance - .06) / (maximum - .06)
    assert unit_distance.mean().item() == pytest.approx(.5, abs=.005)
    assert unit_distance.std().item() == pytest.approx(1 / math.sqrt(12), abs=.003)
    assert (distance <= .12 + 1e-7).all() and (distance >= .06 - 1e-7).all()
    assert (goal.abs() <= safe + 1e-6).all()


@pytest.mark.parametrize('degrees', [0, 5, 10, 15, 20, 25, 30])
def test_left_only_uniform_angle_and_distance(load_mdp, degrees):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    b = pool_bounds(20000)[:, 0]
    alpha = math.radians(degrees)
    target, goal = m.sample_commands(b, torch.Generator().manual_seed(42),
        excluded_x_half_angle=alpha, sweep_direction='left')
    delta = goal - target
    assert (delta[:, 1] < 0).all()
    theta = torch.atan2(delta[:, 1], delta[:, 0]).remainder(2 * math.pi)
    u = (theta - math.pi - alpha) / (math.pi - 2 * alpha)
    assert ((u >= -1e-6) & (u <= 1 + 1e-6)).all()
    assert ((torch.histc(u, bins=10, min=0, max=1) / len(b) - .1).abs() < .008).all()
    distance = delta.norm(dim=-1)
    assert (distance >= .06 - 1e-7).all() and (distance <= .12 + 1e-7).all()
    # The shelf has room for the full 12 cm annulus at every allowed angle.
    assert ((distance - .06) / .06).mean().item() == pytest.approx(.5, abs=.008)


@pytest.mark.parametrize('degrees', [0, 30])
@pytest.mark.parametrize('sweep_direction', ['both', 'left'])
def test_pool_exact_counts_bottom_gaps_and_parking(load_mdp, degrees, sweep_direction):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    obb = importlib.import_module('_baseline_cpu_pkg.mdp.common.bbox_scene_layout')
    b = pool_bounds(128)
    req = torch.tensor([(g, w) for g in range(4) for w in range(4)] * 8)
    local, yaw, active, delta, actual = m.fixed_pool_layout(
        b, 1.05 - b[..., 0, 2], torch.Generator().manual_seed(10), req,
        excluded_x_half_angle=math.radians(degrees), sweep_direction=sweep_direction)
    if sweep_direction == 'left':
        assert (delta[:, 1] < 0).all()
    assert torch.equal(req, actual)
    assert torch.equal(active.sum(-1), req.sum(-1) + 1)
    assert active[:, 1:].any(0).all()  # every fixed physical slot can be selected
    torch.testing.assert_close((local[..., 2] + b[..., 0, 2])[active], torch.full_like(local[..., 2][active], 1.05))
    assert (local[..., 2][~active] >= 50).all()
    half, offset = obb.footprint_from_bounds(b, yaw)
    centers = local[..., :2] + offset
    signs = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    corners = torch.einsum('bmki,bmij->bmkj', half[:, :, None] * signs, obb.axes(yaw)) + centers[:, :, None]
    assert (corners[..., 0].abs()[active] <= .195001).all()
    assert (corners[..., 1].abs()[active] <= .495001).all()
    for i in range(17):
        for j in range(i + 1, 17):
            both = active[:, i] & active[:, j]
            assert obb.separated(centers[:, i], half[:, i], yaw[:, i], centers[:, j], half[:, j], yaw[:, j], .02 - 1e-6)[both].all()


def test_reset_is_partial_has_uniform_counts_and_never_changes_physics(configs, monkeypatch):
    m = configs('event_functions', 'wrench_refactor')
    n = 32
    b, ids = pool_bounds(n), torch.tensor([1, 4, 8, 30])
    state = torch.zeros(n, 17, 13)
    gravity, awake = [], []
    objects = NS(num_objects=17, root_physx_view=NS(set_disable_gravities=lambda flags, rows: gravity.append((flags.clone(), rows.clone())), wake_up=lambda rows: awake.append(rows.clone())))
    objects.write_object_link_state_to_sim = lambda values, env_ids: state.__setitem__(env_ids, values)
    root = NS(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([[1., 0, 0, 0]]).repeat(n, 1))
    env = NS(num_envs=n, device='cpu', _revision_bounds_t=b, _refactor_spawn_z=1.05 - b[..., 0, 2],
             _revision_layout_rng=torch.Generator().manual_seed(42), cfg=NS(red_margin=.023, rear_offset=.02),
             _wrench_gravity_flags=torch.ones(n*17, 1, dtype=torch.uint8),
             scene={'object_collection': objects, 'robot': NS(data=root), 'shelf': NS(data=root)},
             _active_object_mask=torch.zeros(n, 17, dtype=torch.bool), target_id=torch.zeros(n, 1, dtype=torch.long),
             target_width=torch.zeros(n, 1), sweep_dir=torch.zeros(n, 3), _initial_obstacle_observation=torch.zeros(n, 64),
             _final_actual_population=torch.zeros(n, 2, dtype=torch.long), _revision_layout_complete=torch.zeros(n, dtype=torch.bool),
             _revision_spawn_positions=torch.zeros(n, 17, 3), sensor_data_fresh=torch.ones(n, dtype=torch.bool))
    monkeypatch.setattr(m, 'apply_final_visibility', lambda *args: None)
    expected = torch.randint(0, 4, (len(ids), 2), generator=torch.Generator().manual_seed(42))
    m.reset_scene(env, ids)
    assert torch.equal(env._final_actual_population[ids], expected)
    other = torch.ones(n, dtype=torch.bool); other[ids] = False
    assert not state[other].any()
    assert not state[ids, :, 7:].any()
    assert env.sensor_data_fresh[other].all() and not env.sensor_data_fresh[ids].any()
    assert set((gravity[-1][1] % n).tolist()) == set(ids.tolist())
    assert torch.equal(awake[-1], gravity[-1][1])
    assert awake[-1].numel() == len(ids) * 17
    assert torch.equal(b, pool_bounds(n))
    rng = env._revision_layout_rng.get_state().clone()
    m.reset_scene(env, [])
    assert torch.equal(rng, env._revision_layout_rng.get_state())


def test_contact_height_goal_boundary_and_partial_reset(configs):
    env, pos, _ = reward_env(3)
    env.cfg.goal_reward_height_limit = env.cfg.contact_height_limit = .14
    pos[:, 0, 0] = torch.tensor([.03 - 1e-7, .03, .03 + 1e-7], dtype=pos.dtype)
    env.sensor_data_fresh[:] = False
    term = configs('reward_functions', 'wrench_refactor').RefactorContactMaintenance(NS(params={}), env)
    assert term(env).tolist() == [1., 1., 0.]
    for i in range(12):
        env.common_step_counter += 1
        term(env)
    assert term(env).tolist() == [10., 10., 0.]
    env.scene['ee_frame'].data.target_pos_w[:, 0, 2] = 1.05 + .14
    env.common_step_counter += 1
    assert not term(env).any()  # even occupancy cannot bypass the current height
    for i in range(4):
        env.common_step_counter += 1
        term(env)
    assert not term._count.any()
    env.scene['ee_frame'].data.target_pos_w[:, 0, 2] = 1.05 + .13
    env.common_step_counter += 1
    assert term(env).tolist() == [1., 1., 0.]
    term.reset(torch.tensor([0]))
    assert term(env).tolist() == [0., 1., 0.]


def test_uniform_startup_physics_independent_objects_and_idempotence(configs, monkeypatch):
    m = configs('event_functions', 'wrench_refactor')
    n = 1024
    calls = []
    materials = [NS(apply=lambda rows, s, d: calls.append((rows.clone(), s.clone(), d.clone()))) for _ in range(3)]
    state = NS(default_mass=torch.full((n*17,1), .3), default_inertia=torch.ones(n*17,9),
               mass_buffer=torch.zeros(n*17,1), inertia_buffer=torch.zeros(n*17,9),
               materials=materials, mass=torch.zeros(n), static_friction=torch.zeros(n,3), dynamic_friction=torch.zeros(n,3))
    view = NS(set_masses=lambda *args: None, set_inertias=lambda *args: None)
    env = NS(num_envs=n, cfg=NS(seed=42), _clutter_physics=state, scene={'object_collection':NS(root_physx_view=view)})
    monkeypatch.setattr(m, 'initialize_clutter_materials', lambda env: None)
    m.initialize_physics(env, None)
    assert .23*1.5 <= state.mass_buffer.min() and state.mass_buffer.max() <= .5*1.5
    assert state.mass_buffer.mean().item() == pytest.approx((.23+.5)/2*1.5, abs=.002)
    torch.testing.assert_close(state.inertia_buffer, state.default_inertia * state.mass_buffer / state.default_mass)
    assert not torch.equal(state.mass_buffer[:n], state.mass_buffer[n:2*n])
    assert [len(c[0]) for c in calls] == [n,n*17,n]
    before = state.mass_buffer.clone()
    m.initialize_physics(env, None)
    assert torch.equal(before,state.mass_buffer) and len(calls) == 3


def test_entry_is_speed_independent_inclusive_and_dt_independent(configs):
    for dt in (.01, .02, .05):
        env, pos, _ = reward_env(2)
        env.cfg.goal_success_radius = .03
        env.step_dt, env._sim_step_counter = dt, 1
        pos[:, 0, 0] = torch.tensor([.03, .030001], dtype=pos.dtype)
        cfg = configs('rewards', 'wrench_refactor').WrenchRefactorRewardsCfg().target_goal_first_entry_bonus
        term = cfg.func(cfg, env)
        assert (term(env) * dt * cfg.weight).tolist() == pytest.approx([15., 0.])
        env.common_step_counter += 1
        assert not term(env).any()


def test_normalized_output_torque_and_clamped_hand_commands(configs):
    limits = torch.tensor([[1., 2., 4., 8., 16., 32.]])
    arm = NS(_asset=NS(data=NS(joint_effort_limits=limits)), _joint_ids=list(range(6)),
             cfg=NS(effort_limit_scale=1., minimum_effort_limit=1e-6), joint_efforts=limits * .5)
    env = NS(num_envs=1, device='cpu', common_step_counter=1,
             action_manager=NS(action=torch.tensor([[100.]*6 + [100., -100.]]), prev_action=torch.zeros(1, 8), get_term=lambda _: arm))
    term = configs('reward_functions', 'wrench_refactor').RefactorActionRatePenalty(NS(params={}), env)
    assert term(env).item() == pytest.approx(.03*(6*.25 + 1))
    env.action_manager.prev_action[:] = env.action_manager.action
    env.action_manager.action *= 2
    env.common_step_counter += 1
    assert term(env).item() == 0.  # saturated command excursion is not executable motion
    arm.joint_efforts *= -1
    env.common_step_counter += 1
    assert term(env).item() == pytest.approx(.03*6)


def test_all_manager_terms_against_entire_wrench_family(configs, monkeypatch):
    old = configs('rewards', 'wrench')
    new = configs('rewards', 'wrench_refactor').WrenchRefactorRewardsCfg()
    preserved = ['target_reaching', 'target_goal_hold', 'object_tilt_penalty',
                 'obstacle_displacement_penalty', 'shelf_penalty', 'target_goal_success_bonus', 'wrist_wrench_change_penalty']
    for name in preserved:
        assert_ordered_equal(getattr(new, name).to_dict(), getattr(old.WrenchRewardsCfg(), name).to_dict())
    assert new.target_goal_distance.func is old.WrenchRewardsCfg().target_goal_distance.func
    assert new.target_goal_distance.weight == 5
    assert new.target_contact.weight == .1
    assert new.obstacle_contact_penalty.params == {'free_force': 30., 'max_cost': 100.}
    assert len([v for v in vars(new).values() if hasattr(v, 'weight')]) == 13
    term = configs('terminations', 'wrench_refactor').WrenchRefactorTerminationsCfg()
    assert term.arm_joint_velocity.params['threshold'] == 3.
    assert term.timeout.time_out and term.timeout.func.__name__ == 'time_out'
    assert not hasattr(term, 'target_speed')
    cur = configs('curricula', 'wrench_refactor').WrenchRefactorCurriculumCfg()
    assert list(vars(cur)) == ['target_goal_direction_deadzone']
    assert_ordered_equal(cur.target_goal_direction_deadzone.to_dict(), configs('curricula', 'wrench').WrenchCurriculumCfg().target_goal_direction_deadzone.to_dict())
    import sys
    noise = importlib.import_module('_omni_official_utils.noise')
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', noise)
    obs = configs('observations', 'wrench_refactor').WrenchRefactorObservationsCfg()
    assert list(vars(obs.policy)) == list(vars(obs.critic))
    assert obs.policy.tactile.params['threshold_n'] == obs.critic.tactile.params['threshold_n'] == .05
    assert obs.policy.target_obs_state.func.__name__ == 'InitialObjectPositionInRoot'
    assert obs.critic.target_obs_state.func.__name__ == 'object_position_in_root'
    assert obs.critic_obstacles.current_positions.params['adjacent_swap_probability'] == 0.


def test_cfg_and_runtime_have_no_task_parents():
    allowed = {'ManagerBasedRLEnvCfg', 'ManagerBasedRLEnv', 'InteractiveSceneCfg', 'OperationalSpaceControllerActionCfg',
               'ActionTermCfg', 'CommandTermCfg', 'ObsGroup', 'RslRlMLPModelCfg', 'RslRlOnPolicyRunnerCfg'}
    paths = list(MODULE_ROOT.glob('mdp/*/wrench_refactor.py'))
    paths = [p for p in paths if not p.parent.name.endswith('_functions') and p.parent.name != 'common']
    paths += [MODULE_ROOT / 'envs/wrench_refactor_envs.py', MODULE_ROOT / 'runtime/wrench_refactor.py',
              MODULE_ROOT / 'assets/scenes/wrench_refactor.py', MODULE_ROOT / 'agent/wrench_refactor_ppo_cfg.py']
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                v1_parents = {'WrenchRefactorRelativeV2_1EnvCfg':'WrenchRefactorRelativeV2EnvCfg',
                              'WrenchRefactorRelativeV2_1PPORunnerCfg':'WrenchRefactorRelativeV2PPORunnerCfg','WrenchRefactorV0_1EnvCfg':'WrenchRefactorEnvCfg',
                              'WrenchRefactorRelativeEnvCfg':'WrenchRefactorV3_1EnvCfg',
                              'WrenchRefactorRelativeV3EnvCfg':'WrenchRefactorRelativeEnvCfg',
                              'WrenchRefactorRelativeV3RewardsCfg':'WrenchRefactorRelativeRewardsCfg',
                              'WrenchRefactorRelativeRewardsCfg':'WrenchRefactorV2RewardsCfg',
                              'WrenchRefactorRelativeV2EnvCfg':'WrenchRefactorRelativeEnvCfg',
                              'WrenchRefactorRelativeV2ObservationsCfg':'WrenchRefactorRelativeObservationsCfg',
                              'WrenchRefactorRelativeV2ModelCfg':'WrenchRefactorRelativeModelCfg',
                              'WrenchRefactorRelativeCurriculumCfg':'WrenchRefactorCurriculumCfg',
                              'WrenchRefactorRelativeV2PPORunnerCfg':'WrenchRefactorRelativePPORunnerCfg',
                              'WrenchRefactorRelativeV3PPORunnerCfg':'WrenchRefactorRelativePPORunnerCfg',
                              'WrenchRefactorRelativePPORunnerCfg':'WrenchRefactorV3_1PPORunnerCfg',
                              'WrenchRefactorRelativeModelCfg':'WrenchRefactorModelCfg',
                              'WrenchRefactorRelativeObservationsCfg':'WrenchRefactorObservationsCfg',
                              'WrenchRefactorV0_1PPORunnerCfg':'WrenchRefactorPPORunnerCfg',
                              'WrenchRefactorV1_1EnvCfg':'WrenchRefactorV1EnvCfg',
                              'WrenchRefactorV1_1PPORunnerCfg':'WrenchRefactorV1PPORunnerCfg',
                              'WrenchRefactorV2_1EnvCfg':'WrenchRefactorV2EnvCfg',
                              'WrenchRefactorV2_1PPORunnerCfg':'WrenchRefactorV2PPORunnerCfg',
                              'WrenchRefactorV3_1EnvCfg':'WrenchRefactorV3EnvCfg',
                              'WrenchRefactorV3_1PPORunnerCfg':'WrenchRefactorV3PPORunnerCfg',
                              'WrenchRefactorV3TestEnvCfg':'WrenchRefactorV3EnvCfg',
                              'WrenchRefactorV3TestPPORunnerCfg':'WrenchRefactorV3PPORunnerCfg',
                              'WrenchRefactorV3EnvCfg':'WrenchRefactorV2EnvCfg',
                              'WrenchRefactorV3PPORunnerCfg':'WrenchRefactorV2PPORunnerCfg',
                              'WrenchRefactorV2EnvCfg':'WrenchRefactorEnvCfg',
                              'WrenchRefactorV2RewardsCfg':'WrenchRefactorRewardsCfg',
                              'WrenchRefactorV2PPORunnerCfg':'WrenchRefactorPPORunnerCfg','WrenchRefactorV1EnvCfg':'WrenchRefactorEnvCfg',
                              'WrenchRefactorV1RewardsCfg':'WrenchRefactorRewardsCfg',
                              'WrenchRefactorV1PPORunnerCfg':'WrenchRefactorPPORunnerCfg'}
                v1_parents.update({f'WrenchRefactorRelative{v}Test{d}EnvCfg': f'WrenchRefactorRelative{v}EnvCfg'
                                   for v in ('V2_1', 'V3') for d in ('Right', 'Pull', 'Left', 'Push')})
                v1_parents.update({f'WrenchRefactorRelative{v}{suffix}': f'WrenchRefactorRelativeV2_1{suffix}'
                                   for v in ('V2_2', 'V2_3') for suffix in ('EnvCfg', 'PPORunnerCfg')})
                if node.name in v1_parents:
                    assert [ast.unparse(b) for b in node.bases] == [v1_parents[node.name]]
                elif node.name in ('PolicyCfg', 'CriticCfg', 'ObstaclesCfg') and isinstance(node.bases[0], ast.Attribute):
                    assert ast.unparse(node.bases[0]) in (f'WrenchRefactorObservationsCfg.{node.name}',
                                                        'WrenchRefactorRelativeObservationsCfg.PolicyCfg')
                else:
                    assert all(isinstance(b, ast.Name) and b.id in allowed for b in node.bases), path


def test_complete_scene_and_env_composition(configs, omni_config, monkeypatch):
    """Execute real scene/post-init and EnvCfg, substituting only Sim leaf types."""
    import sys
    from pathlib import Path
    from test_omni_configuration import _AssetNode, _execute_definitions
    noise = importlib.import_module('_omni_official_utils.noise')
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', noise)
    robot = sys.modules['_baseline_cpu_pkg.assets.robot']
    tactile = importlib.import_module('_baseline_cpu_pkg.assets.tactile_regions')
    contacts = importlib.import_module('_baseline_cpu_pkg.mdp.common.contact_sweep_clutter_fusion_contacts')
    kinds = ('cylinder','cube','mug','bottle','003_cracker_box','004_sugar_box','005_tomato_soup_can','002_master_chef_can')
    ns = dict(vars(omni_config.baseline))
    ns.update(vars(robot))
    ns.update(Path=Path, __file__=str(MODULE_ROOT/'assets/scenes/wrench_refactor.py'), sim=ns['sim_utils'],
              RevisionRobotSpawnerCfg=lambda **kw:_AssetNode(**kw,sensor_height=.0254),
              WrenchRefactorShelfSpawnerCfg=_AssetNode, RevisionObjectSpawnerCfg=_AssetNode,
              REVISION_OBJECT_KINDS=kinds, TACTILE_PARENT_BODY_NAMES=tactile.TACTILE_PARENT_BODY_NAMES,
              parent_sensor_name=tactile.parent_sensor_name, object_hand_sensor_name=contacts.object_hand_sensor_name,
              make_robot_cfg=omni_config.baseline.make_robot_cfg)
    ns['_tool_frame'] = lambda frames,body='inspire_base_link',**kw:_AssetNode(target_frames=[
        _AssetNode(prim_path=f'Robot/{body}',offset=_AssetNode(pos=xyz,rot=(.5,-.5,-.5,-.5))) for name,xyz in frames])
    scene_module = _execute_definitions(MODULE_ROOT/'assets/scenes/wrench_refactor.py',ns)
    scene = scene_module.WrenchRefactorSceneCfg()
    roster = scene.object_collection.rigid_objects
    assert scene.shelf.spawn.mass_props.mass == 30.
    assert scene.hand_tactile.force_threshold == .05
    assert scene.robot.init_state.joint_pos["wrist_3_joint"] == -math.pi/2
    assert len(roster) == 17
    assert [obj.spawn.kind_override for obj in roster.values()] == ['cylinder', *[k for k in kinds for _ in range(2)]]
    assert Path(scene.robot.spawn.usd_path).as_posix().endswith('wrench_right_hand/ur5e_inspire_right.usda')
    for parent in tactile.TACTILE_PARENT_BODY_NAMES:
        sensor = getattr(scene,tactile.parent_sensor_name(parent))
        assert len(sensor.filter_prim_paths_expr) == 17
        assert sensor.max_contact_data_count_per_prim >= 17*16
    ns.update(WrenchRefactorSceneCfg=scene_module.WrenchRefactorSceneCfg,
              WrenchRefactorActionsCfg=lambda:_AssetNode(arm_action=_AssetNode()),
              WrenchRefactorOSCActionCfg=NS(OffsetCfg=_AssetNode))
    for role, name in [('commands','Commands'),('observations','Observations'),('rewards','Rewards'),('events','Events'),('terminations','Terminations'),('curricula','Curriculum')]:
        ns[f'WrenchRefactor{name}Cfg'] = getattr(configs(role,'wrench_refactor'),f'WrenchRefactor{name}Cfg')
    ns['WrenchRefactorRelativeRewardsCfg'] = configs('rewards','wrench_refactor').WrenchRefactorRelativeRewardsCfg
    ns['WrenchRefactorRelativeV3RewardsCfg'] = configs('rewards','wrench_refactor').WrenchRefactorRelativeV3RewardsCfg
    ns['WrenchRefactorV2RewardsCfg'] = configs('rewards','wrench_refactor').WrenchRefactorV2RewardsCfg
    ns['WrenchRefactorV1RewardsCfg'] = configs('rewards','wrench_refactor').WrenchRefactorV1RewardsCfg
    ns['WrenchRefactorRelativeObservationsCfg'] = configs('observations','wrench_refactor').WrenchRefactorRelativeObservationsCfg
    ns['WrenchRefactorRelativeV2ObservationsCfg'] = configs('observations','wrench_refactor').WrenchRefactorRelativeV2ObservationsCfg
    ns['WrenchRefactorRelativeCurriculumCfg'] = configs('curricula','wrench_refactor').WrenchRefactorRelativeCurriculumCfg
    ns['math'] = math
    ns['__package__'] = '_baseline_cpu_pkg.envs'
    env_module = _execute_definitions(MODULE_ROOT/'envs/wrench_refactor_envs.py', ns)
    cfg = env_module.WrenchRefactorEnvCfg()
    assert cfg.goal_reward_height_limit == cfg.contact_height_limit == .14
    assert cfg.commands.target_goal_pos.goal_radius == .03
    assert cfg.decimation*cfg.sim.dt == .02
    assert cfg.scene.robot.actuators['arm'].stiffness == cfg.scene.robot.actuators['arm'].damping == 0.
    assert cfg.actions.arm_action.body_offset.pos == (0.,0.,.1427)
    assert cfg.actions.arm_action.body_offset.rot == (.5,-.5,-.5,-.5)
    assert [e.mode for e in vars(cfg.events).values()] == ['prestartup','startup','reset','reset']
    # No simulator imports: substitute only the action implementation leaf.
    from types import ModuleType
    action_module = ModuleType('_baseline_cpu_pkg.mdp.action_functions.wrench_refactor')
    action_module.WrenchRefactorOSCAction = type('WrenchRefactorOSCAction', (), {})
    monkeypatch.setitem(sys.modules, action_module.__name__, action_module)
    for version in ('V2_1', 'V3'):
        parent_type = getattr(env_module, f'WrenchRefactorRelative{version}EnvCfg')
        baseline = parent_type().to_dict()
        for direction, low, high in [('Right',60,120),('Pull',-60,60),('Left',240,300),('Push',120,240)]:
            child_type = getattr(env_module, f'WrenchRefactorRelative{version}Test{direction}EnvCfg')
            assert child_type.__bases__ == (parent_type,)
            expected = copy.deepcopy(baseline)
            expected['events']['object_spawn']['params']['goal_angle_range'] = (math.radians(low),math.radians(high))
            assert child_type().to_dict() == expected
            assert parent_type().to_dict() == baseline
    parent = env_module.WrenchRefactorRelativeEnvCfg()
    before = parent.to_dict()
    assert parent.wrist_wrench_observation_limits == (150.,150.,150.,8.,8.,8.)
    assert parent.rewards.robot_action_rate_penalty.params == {'arm_max_cost':100., 'hand_max_cost':100.}
    assert env_module.WrenchRefactorV3_1EnvCfg().rewards.robot_action_rate_penalty.params == {}
    assert list(parent.scene.robot.actuators['arm'].velocity_limit_sim.values()) == [3.14] * 6
    assert list(parent.scene.robot.actuators['arm'].effort_limit_sim.values()) == [150.] * 3 + [28.] * 3
    assert parent.terminations.arm_joint_velocity.params['threshold'] == 4.0
    assert parent.curriculum.target_goal_direction_deadzone.to_dict() == cfg.curriculum.target_goal_direction_deadzone.to_dict()
    assert not hasattr(cfg.curriculum, 'arm_joint_velocity_threshold')
    assert list(cfg.scene.robot.actuators['arm'].velocity_limit_sim.values()) == [2.] * 3 + [4.] * 3
    child = env_module.WrenchRefactorRelativeV2EnvCfg()
    expected = parent.to_dict()
    expected['observations'] = child.observations.to_dict()
    assert child.to_dict() == expected
    child_before = child.to_dict()
    variant = env_module.WrenchRefactorRelativeV2_1EnvCfg()
    expected_variant = child.to_dict()
    assert variant.rewards.target_goal_distance.func is configs('reward_functions','wrench_refactor').displacement_cosine_goal_distance
    expected_variant['rewards']['target_goal_distance']['func'] = '_baseline_cpu_pkg.mdp.reward_functions.wrench_refactor:displacement_cosine_goal_distance'
    expected_variant['rewards']['target_goal_distance']['params'] = {'distance_scale': .05}
    assert variant.to_dict() == expected_variant
    assert child.to_dict() == child_before
    assert variant.rewards.target_inverse_distance.weight == -12.97
    assert parent.to_dict() == before
    assert env_module.WrenchRefactorRelativeV2EnvCfg.__bases__ == (env_module.WrenchRefactorRelativeEnvCfg,)
    v3 = env_module.WrenchRefactorRelativeV3EnvCfg()
    expected_v3 = parent.to_dict()
    expected_v3['rewards']['target_contact'] = None
    expected_v3['rewards']['target_goal_distance'] = v3.rewards.target_goal_distance.to_dict()
    for name in ('target_goal_first_entry_bonus', 'target_goal_hold', 'target_goal_success_bonus'):
        expected_v3['rewards'][name] = None
    expected_v3['terminations']['target_goal_success'] = None
    assert parent.rewards.target_inverse_distance.weight == child.rewards.target_inverse_distance.weight == v3.rewards.target_inverse_distance.weight == -12.97
    assert v3.to_dict() == expected_v3
    assert parent.to_dict() == before
    assert parent.scene.robot.spawn.sensor_mass == child.scene.robot.spawn.sensor_mass == v3.scene.robot.spawn.sensor_mass == .280
    assert not hasattr(env_module, 'WrenchRefactorRelativeV1EnvCfg')
    assert 'print_velocity' not in child.terminations.arm_joint_velocity.params




def test_same_object_multiple_contacts_both_surfaces_and_no_cancellation(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    # One parent, two separated footprints. Four contacts from ONE pair, with
    # opposite force vectors on channel 0: a net-force sensor would lose them.
    region = {'footprint_halfspaces': [[1,0,-.02],[-1,0,-.02],[0,1,-.02],[0,-1,-.02]],
              'parent_halfspaces': [[1,0,0,-.1],[-1,0,0,-.1],[0,1,0,-.1],[0,-1,0,-.1],[0,0,1,-.01],[0,0,-1,-.01]],
              'pad_bounds_m': {'min': [-.02,-.02,-.001], 'max': [.02,.02,.001]}}
    mapper = m.PairedContactRegions([region, region], [0, 1], 'cpu')
    positions = torch.zeros(1,17,3); positions[0,1,0] = .1
    quats = torch.zeros(1,17,4); quats[...,0] = 1
    points = torch.tensor([[0.,0,.01], [0.,0,.01], [.1,0,.01], [.1,0,-.01], [.05,0,.01]])
    forces = torch.tensor([[0.,0,1.], [0.,0,-1.], [0.,0,2.], [0.,0,-3.], [0.,0,99.]])
    output = torch.zeros(1,34)
    mapper.accumulate(output, points, forces, torch.nn.functional.normalize(forces, dim=-1), torch.zeros(5,dtype=torch.long), positions, quats)
    assert output[0,0].item() == 2
    assert output[0,1].item() == 2
    assert output[0,18].item() == 3
    assert output.sum() == 7


def test_right_metadata_front_and_back_reach_real_parent_surfaces(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    regions = json.loads((MODULE_ROOT / 'assets/data/wrench_right_hand/tactile_regions.json').read_text())['regions']
    for i, r in enumerate(regions):
        mapper = m.PairedContactRegions([r], [i], 'cpu')
        center = mapper.center[0]
        hull = mapper.hull[0]
        c = hull[:,:2] @ center + hull[:,3]
        nz = hull[:,2]
        upper = (-c[nz>1e-8] / nz[nz>1e-8]).min()
        lower = (-c[nz<-1e-8] / nz[nz<-1e-8]).max()
        assert upper > lower, r['name']
        points = torch.tensor([[*center,upper], [*center,lower]])
        normals = torch.tensor([[0.,0,1.], [0.,0,-1.]])
        output = torch.zeros(1,34)
        q = torch.zeros(1,17,4); q[...,0] = 1
        mapper.accumulate(output, points, normals, normals, torch.zeros(2,dtype=torch.long), torch.zeros(1,17,3), q)
        assert output[0,i] == 1 and output[0,i+17] == 1, r['name']


def test_equal_model_width_encoder_gradients_and_jit(tmp_path):
    spec = importlib.util.spec_from_file_location('_refactor_models', MODULE_ROOT / 'agent/wrench_refactor_models.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(42)
    raw = torch.randn(8,74)
    live = raw.clone(); live[:,21:24] += 1.
    obstacles = torch.randn(8,16,4); obstacles[...,0] = 1.
    obs = TensorDict({'policy': raw, 'critic': live, 'obstacles': obstacles.flatten(1), 'critic_obstacles': (obstacles+.1).flatten(1)}, batch_size=[8])
    routing = {'actor':['policy','obstacles'], 'critic':['critic','critic_obstacles']}
    actor = m.WrenchRefactorModel(obs,routing,'actor',8, distribution_cfg={'class_name':'GaussianDistribution','init_std':1.})
    critic = m.WrenchRefactorModel(obs,routing,'critic',1)
    assert actor.obs_dim == critic.obs_dim == 74
    assert actor.get_latent(obs).shape == critic.get_latent(obs).shape == (8,90)
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    algorithm = PPO(actor,critic,RolloutStorage('rl',8,4,obs,[8],'cpu'),num_learning_epochs=1,num_mini_batches=2,device='cpu')
    before = [copy.deepcopy(model.obstacle_encoder.state_dict()) for model in (actor,critic)]
    with torch.inference_mode():
        for _ in range(4):
            action = algorithm.act(obs)
            algorithm.process_env_step(obs,-action.square().sum(-1),torch.zeros(8,dtype=torch.long),{})
        algorithm.compute_returns(obs)
    assert all(math.isfinite(v) for v in algorithm.update().values())
    for model, initial in zip((actor,critic),before):
        assert any(not torch.equal(initial[k],v) for k,v in model.obstacle_encoder.state_dict().items())
    actor.eval()
    exported = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(exported(torch.cat((raw,obs['obstacles']),-1)), actor(obs))
    changed = obs.clone(); changed['critic'] += 100.; changed['critic_obstacles'] += 100.
    torch.testing.assert_close(actor(obs),actor(changed))


from test_fusion_holding_osc import osc


def test_eef_command_axes_are_frozen_and_keep_bounded_torque_control(osc, tensor_math):
    from test_omni_configuration import _execute_definitions
    shared = _execute_definitions(
        MODULE_ROOT / 'mdp/action_functions/contact_sweep_clutter_fusion_holding_osc.py',
        dict(torch=torch, math_utils=tensor_math, OperationalSpaceControllerAction=osc.action, Sequence=list))
    cls = _execute_definitions(MODULE_ROOT / 'mdp/action_functions/wrench_refactor.py',
        dict(UR10OperationalSpaceControllerAction=shared.UR10OperationalSpaceControllerAction)).WrenchRefactorOSCAction
    term = cls.__new__(cls)
    cfg = osc.v0().actions.arm_action
    term._osc = osc.controller(cfg.controller_cfg, 2, 'cpu')
    term._ee_pose_b = torch.tensor([[.3,-.2,.5,1.,0.,0.,0.], [.3,-.2,.5,math.sqrt(.5),0.,0.,math.sqrt(.5)]])
    initial = term._ee_pose_b.clone()
    term._compute_ee_pose = lambda: None
    term._raw_actions = torch.zeros(2,6); term._processed_actions = torch.zeros(2,6)
    term._position_scale, term._orientation_scale = .03, .15
    term._pose_rel_idx = 0
    term._pose_abs_idx = term._wrench_abs_idx = term._stiffness_idx = term._damping_ratio_idx = None
    raw = torch.tensor([[1.,0.,0.,1.,0.,0.]]).repeat(2,1)
    term.process_actions(raw)
    desired = term._osc.desired_ee_pose_b.clone()
    torch.testing.assert_close(desired[:,:3] - initial[:,:3], torch.tensor([[.03,0.,0.],[0.,.03,0.]]), atol=1e-6, rtol=0.)
    dq = tensor_math.quat_from_euler_xyz(torch.full((2,),.15), torch.zeros(2), torch.zeros(2))
    torch.testing.assert_close(desired[:,3:], tensor_math.quat_mul(initial[:,3:],dq), atol=1e-6, rtol=0.)
    term._ee_pose_b[:,0] += 1.
    torch.testing.assert_close(term._task_frame_pose_b,initial)
    term.process_actions(torch.zeros_like(raw))
    torch.testing.assert_close(term._osc.desired_ee_pose_b,term._ee_pose_b,atol=1e-6,rtol=0.)
    assert cls.apply_actions is shared.UR10OperationalSpaceControllerAction.apply_actions


def test_tactile_threshold_exactly_005_and_reset_mask(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    term = m.RightHandTactile.__new__(m.RightHandTactile)
    forces = torch.zeros(2,17,3)
    forces[:, :3, 0] = torch.tensor([.049999,.05,.050001])
    term._read_forces = lambda env, sensor_name: forces
    term._parents = []
    term._robot_ids = []
    env = NS(common_step_counter=1,_sim_step_counter=1,num_envs=2,device='cpu',
             sensor_data_fresh=torch.tensor([True,False]),
             scene={'robot':NS(data=NS(body_link_pos_w=torch.zeros(2,0,3),body_link_quat_w=torch.zeros(2,0,4)))})
    # No parent contacts in this fixture; avoid an empty parent stack by using
    # one valid sensor with a zero-count contact buffer.
    term._parents = [('parent',None)]
    env.physics_dt = .01
    env.scene['parent'] = NS(contact_physx_view=NS(get_contact_data=lambda dt:[None,None,None,None,torch.zeros(2,1)]))
    bits = term(env)
    assert bits[0,:3].tolist() == [0.,1.,1.]
    assert not bits[1].any() and not bits[:,17:].any()


def test_offset_eef_jacobian_and_velocity_use_same_rotated_lever(tensor_math):
    from test_omni_configuration import _execute_definitions
    class Base:
        @property
        def jacobian_b(self): return self.test_jacobian
    cls = _execute_definitions(MODULE_ROOT / 'mdp/action_functions/wrench_refactor.py',
        dict(vars(tensor_math), torch=torch, UR10OperationalSpaceControllerAction=Base)).WrenchRefactorOSCAction
    term = cls.__new__(cls)
    term.cfg = NS(body_offset=object())
    term._ee_body_idx = 0
    term._offset_pos = torch.tensor([[0.,0.,.1427]])
    # A 90-degree pitch maps local +Z to root +X. A base +Z angular
    # velocity then moves the EEF along +Y, not along the COM velocity.
    q = tensor_math.quat_from_euler_xyz(torch.zeros(1),torch.full((1,),math.pi/2),torch.zeros(1))
    data = NS(root_quat_w=torch.tensor([[1.,0.,0.,0.]]),body_link_quat_w=q[:,None],
              body_link_vel_w=torch.tensor([[[0.,0.,0.,0.,0.,2.]]]),
              body_com_vel_w=torch.full((1,1,6),999.),root_link_vel_w=torch.zeros(1,6))
    term._asset = NS(data=data)
    term.test_jacobian = torch.zeros(1,6,1); term.test_jacobian[:,5,0] = 1.
    term._jacobian_b = torch.zeros_like(term.test_jacobian)
    term._ee_vel_w = torch.zeros(1,6); term._ee_vel_b = torch.zeros(1,6)
    term._compute_ee_jacobian(); term._compute_ee_velocity()
    expected = torch.tensor([[0.,.2854,0.,0.,0.,2.]])
    torch.testing.assert_close(term._ee_vel_b,expected,atol=1e-6,rtol=0.)
    torch.testing.assert_close((term._jacobian_b*2).squeeze(-1),expected,atol=1e-6,rtol=0.)


def test_startup_event_signatures_reserve_required_environment_arguments(configs):
    # EventManager strips the first two required parameters before checking CFG params.
    import inspect
    m = configs('event_functions', 'wrench_refactor')
    for func in (m.initialize_geometry, m.initialize_physics):
        params=list(inspect.signature(func).parameters.values())
        assert [p.name for p in params[:2]] == ['env','env_ids']
        assert all(p.default is inspect.Parameter.empty for p in params[:2])


def test_hand_velocity_penalty_ignores_mimic_followers(configs):
    # Resolve configured names against the real articulation ordering (arm then hand).
    cfg = configs('rewards', 'wrench_refactor').WrenchRefactorRewardsCfg().robot_joint_velocity_penalty
    names = ['inspire_left_thumb_1_joint', 'inspire_left_thumb_2_joint',
             'inspire_left_thumb_3_joint', 'inspire_left_thumb_4_joint',
             'inspire_left_index_1_joint', 'inspire_left_index_2_joint',
             'inspire_left_middle_1_joint', 'inspire_left_middle_2_joint',
             'inspire_left_ring_1_joint', 'inspire_left_ring_2_joint',
             'inspire_left_little_1_joint', 'inspire_left_little_2_joint']
    selected = cfg.params['hand_cfg'].joint_names
    assert selected == [names[i] for i in (0,1,4,6,8,10)]
    ids = [6+names.index(name) for name in selected]
    vel = torch.zeros(1,18)
    env = NS(scene={'robot': NS(data=NS(joint_vel=vel))})
    params = dict(arm_cfg=NS(name='robot',joint_ids=list(range(6))),
                  hand_cfg=NS(name='robot',joint_ids=ids))
    followers = [i for i in range(6,18) if i not in ids]
    vel[:,followers] = 5.
    assert cfg.func(env,**params).item() == 0.
    vel[:,ids] = 1.
    assert cfg.func(env,**params).item() == pytest.approx(.06)
    vel[:,:6] = 1.
    assert cfg.func(env,**params).item() == pytest.approx(.24)
    assert cfg.weight == -1.


def test_v1_only_changes_contact_and_distance_rewards(configs):
    m = configs('rewards','wrench_refactor')
    parent = m.WrenchRefactorRewardsCfg()
    before = copy.deepcopy(parent.to_dict())
    child = m.WrenchRefactorV1RewardsCfg()
    assert m.WrenchRefactorV1RewardsCfg.__bases__ == (m.WrenchRefactorRewardsCfg,)
    assert child.target_contact is None
    assert child.target_goal_distance.weight == 25.
    assert child.target_goal_distance.params == {'distance_scale':.05,'threshold_n':.01}
    assert len([v for v in vars(child).values() if hasattr(v,'weight')]) == 12
    for name in before:
        if name not in ('target_contact','target_goal_distance'):
            assert_ordered_equal(child.to_dict()[name],before[name])
    child.target_goal_hold.params['full_reward_radius'] = .123
    assert_ordered_equal(parent.to_dict(),before)


@pytest.mark.parametrize('force,height,fresh,expected',[
    (.009999,1.18,True,False),(.01,1.18,True,True),(.010001,1.18,True,True),
    (.02,1.19,True,False),(.02,1.190001,True,False),(.02,1.18,False,False),
    (float('nan'),1.18,True,False),(.02,float('nan'),True,False),
])
def test_v1_distance_requires_current_contact_and_strict_height(configs,force,height,fresh,expected):
    cfg = configs('rewards','wrench_refactor').WrenchRefactorV1RewardsCfg().target_goal_distance
    env,pos,forces = reward_env()
    env.cfg.contact_height_limit = env.cfg.goal_reward_height_limit = .14
    env.scene['ee_frame'].data.target_pos_w[...,2] = height
    env.sensor_data_fresh[:] = fresh
    forces.zero_();forces[0,0,0,0] = force
    value = cfg.func(env,**cfg.params)*cfg.weight*.02
    assert value.item() == pytest.approx(.5*math.exp(-1) if expected else 0.)
    # Even inside the goal there is no contact-free exception.
    pos.zero_();forces.zero_()
    assert cfg.func(env,**cfg.params).item() == 0.
    # A single fresh contact step pays immediately; losing contact stops immediately.
    env.sensor_data_fresh[:] = True
    env.scene['ee_frame'].data.target_pos_w[...,2] = 1.18
    forces[0,0,0,0] = .02
    assert cfg.func(env,**cfg.params).item() == 1.
    forces.zero_()
    assert cfg.func(env,**cfg.params).item() == 0.


def test_v2_only_changes_distance_callable(configs):
    m = configs('rewards', 'wrench_refactor')
    parent = m.WrenchRefactorRewardsCfg()
    before = copy.deepcopy(parent.to_dict())
    child = m.WrenchRefactorV2RewardsCfg()
    assert child.__class__.__bases__ == (m.WrenchRefactorRewardsCfg,)
    assert child.target_contact is not None
    assert child.target_goal_distance.weight == 5.
    assert child.target_goal_distance.params == {'distance_scale': .05}
    expected = copy.deepcopy(before)
    expected['target_goal_distance']['func'] = child.target_goal_distance.to_dict()['func']
    assert_ordered_equal(child.to_dict(), expected)
    child.target_goal_distance.params['distance_scale'] = 1.
    assert_ordered_equal(parent.to_dict(), before)


@pytest.mark.parametrize('xy,alpha', [
    ((.02,0.),1.), ((.02,.02),2**-.5), ((0.,.02),0.),
    ((-.02,.02),-2**-.5), ((-.02,0.),-1.), ((0.,0.),0.),
    ((.2,0.),1.), ((float('nan'),0.),0.), ((float('inf'),0.),0.),
])
def test_v2_cumulative_displacement_cosine(configs,xy,alpha):
    cfg = configs('rewards','wrench_refactor').WrenchRefactorV2RewardsCfg().target_goal_distance
    env,pos,forces = reward_env()
    env.cfg.contact_height_limit = env.cfg.goal_reward_height_limit = .14
    env.scene['ee_frame'].data.target_pos_w[...,2] = 1.18
    env._initial_object_positions_w = torch.zeros_like(pos)
    goal = env.command_manager.get_command('target_goal_pos')
    goal[:,0] = .1
    env.command_manager.get_command = lambda _: goal
    pos[0,0,:2] = torch.tensor(xy)
    distance = math.hypot(xy[0]-.1,xy[1])
    expected = math.exp(-distance/.05)*alpha if math.isfinite(distance) else 0.
    assert cfg.func(env,**cfg.params).item() == pytest.approx(expected)
    # Static displacement is enough; no instantaneous velocity field is used.
    forces.zero_()
    assert cfg.func(env,**cfg.params).item() == pytest.approx(expected)
    env.sensor_data_fresh[:] = False
    assert cfg.func(env,**cfg.params).item() == pytest.approx(expected)
    forces[0,0,0,0] = .02
    env.scene['ee_frame'].data.target_pos_w[...,2] = 1.19
    assert cfg.func(env,**cfg.params).item() == 0.
    env.scene['ee_frame'].data.target_pos_w[...,2] = 1.18
    goal.zero_()  # Coincident start/goal has no defined direction.
    assert cfg.func(env,**cfg.params).item() == 0.


def test_active_refactor_osc_matches_final_and_revision():
    root = Path(__file__).parents[1] / 'inspire_sweep_baseline/mdp/actions'
    calls = []
    for family in ('wrench_refactor', 'final', 'revision'):
        tree = ast.parse((root / (family + '.py')).read_text())
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith('OSCActionCfg'))
        class_type = next(n.value for n in cfg.body if isinstance(n, ast.AnnAssign) and n.target.id == 'class_type')
        assert class_type.id == 'UR10OperationalSpaceControllerAction'
        actions = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith('ActionsCfg'))
        arm = next(n.value for n in actions.body if isinstance(n, ast.Assign) and n.targets[0].id == 'arm_action')
        arm.func = ast.Name(id='CommonOSC', ctx=ast.Load())
        calls.append(ast.dump(arm))
    assert calls[0] == calls[1] == calls[2]


@pytest.mark.parametrize('radius', [.06, .12])
def test_goal_annulus_exact_radius_boundaries(load_mdp, radius):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    target, goal = m.sample_commands(pool_bounds(64)[:,0], torch.Generator().manual_seed(11),
                                    min_goal_distance=radius, max_goal_distance=radius)
    torch.testing.assert_close((goal-target).norm(dim=-1), torch.full((64,),radius), atol=1e-7, rtol=0.)


@pytest.mark.parametrize('maximum', [.059, float('nan'), float('inf')])
def test_goal_annulus_rejects_invalid_maximum(load_mdp, maximum):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    with pytest.raises(ValueError, match='Maximum goal distance'):
        m.sample_commands(pool_bounds(1)[:,0], torch.Generator(), max_goal_distance=maximum)


def test_policy_action_rate_is_uncapped_and_preserves_hand_clamping(configs):
    func = configs('reward_functions', 'wrench_refactor').policy_action_rate_penalty
    action = torch.tensor([[0., 20., 0., 0., 0., 0., 2., -1.],
                           [0., 0., 0., 0., 0., -10., .5, .25]])
    previous = torch.zeros_like(action)
    env = NS(action_manager=NS(action=action, prev_action=previous))
    # Both physical scales normalize back to raw action, so translation and
    # rotation raw changes of the same size carry the same cost.
    torch.testing.assert_close(func(env), torch.tensor([12.03, 3.009375]))
    assert func(env)[0] > 3.  # No former Arm sum cap100.
    saved = action.clone()
    previous[0] = action[0]  # One env can stay unchanged independently.
    assert func(env)[0] == 0 and func(env)[1] > 3.
    torch.testing.assert_close(action, saved)
    previous.zero_()  # Reset action history, not stale torque history.
    torch.testing.assert_close(func(env), torch.tensor([12.03, 3.009375]))


@pytest.mark.parametrize('low,high', [(60,120),(-60,60),(240,300),(120,240)])
@pytest.mark.parametrize('deadzone', [0, 30])
def test_relative_test_sector_distribution(load_mdp, low, high, deadzone):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    bounds = pool_bounds(20000)[:, 0]
    lo, hi = math.radians(low), math.radians(high)
    target, goal = m.sample_commands(bounds, torch.Generator().manual_seed(42),
        excluded_x_half_angle=math.radians(deadzone), goal_angle_range=(lo,hi))
    delta = goal-target
    angle = (torch.atan2(delta[:,1],delta[:,0])-lo).remainder(2*math.pi)
    unit = angle/(hi-lo)
    assert (unit <= 1+1e-5).all()
    assert ((torch.histc(unit,bins=10,min=0,max=1)/len(bounds)-.1).abs()<.01).all()
    distance = delta.norm(dim=-1)
    assert ((distance >= .06-1e-6) & (distance <= .12+1e-6)).all()
    safe = torch.tensor([.2,.5])-bounds[:,:,:2].abs().amax(1)-.02
    assert (goal.abs() <= safe+1e-6).all()


def test_relative_test_registrations():
    import types
    registry = {}
    def register(**kwargs):
        registry[kwargs['id']] = kwargs
    tree = ast.parse((MODULE_ROOT/'__init__.py').read_text())
    block = next(n for n in tree.body if isinstance(n, ast.For) and '_cfg_version' in ast.unparse(n.target))
    exec(compile(ast.Module(body=[block],type_ignores=[]),'<registrations>','exec'),
         {'gym':types.SimpleNamespace(registry=registry,register=register)})
    assert len(registry) == 8
    for version in ('v2_1','v3'):
        for direction in ('Right','Pull','Left','Push'):
            item = registry[f'Inspire-Wrench-Refactor-Relative-{version}-Test-{direction}']
            assert item['entry_point'].endswith(':WrenchRefactorEnv')
            assert item['kwargs']['rsl_rl_cfg_entry_point'].endswith(f':WrenchRefactorRelative{version.upper()}PPORunnerCfg')
            assert item['kwargs']['env_cfg_entry_point'].endswith(f':WrenchRefactorRelative{version.upper()}Test{direction}EnvCfg')
