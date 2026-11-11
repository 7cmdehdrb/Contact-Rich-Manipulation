"""Semi-Prof contracts and physical-sample math; CPU, no simulated physics."""
import copy
import importlib
import math
from dataclasses import MISSING
from types import SimpleNamespace as NS

import pytest
import torch
import warp as wp

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_sweep_simple import family, configs, omni_config
from test_final_configuration import assert_ordered_equal
from test_omni_configuration import _execute_definitions, _AssetNode
from test_wrench_refactor_relative_ik_test import inverse_reach_fixture


@pytest.fixture(autouse=True)
def warp_cache():
    wp.config.kernel_cache_dir = '/tmp/inspire_semi_prof_warp_cache'


def module(name):
    return importlib.import_module('_baseline_cpu_pkg.' + name)


def test_complete_parent_contract_and_isolation(family):
    parent = family.SweepSimpleFingerV1EnvCfg
    child = family.SweepSimpleSemiProfEnvCfg
    assert child.__bases__ == (parent,)
    before = parent().to_dict()
    cfg = child()
    expected = copy.deepcopy(before)
    for name in ('goal_distance', 'hand_contact', 'target_inverse_distance', 'force_direction'):
        expected['rewards'][name] = cfg.to_dict()['rewards'][name]
    expected['rewards']['wrench_change']['params']['max_positive_fraction'] = None
    assert_ordered_equal(cfg.to_dict(), expected)
    assert cfg.rewards.goal_distance.weight == 120.
    assert cfg.rewards.target_inverse_distance.weight == -31.128
    assert cfg.rewards.force_direction.weight == .33
    assert cfg.rewards.hand_contact is None
    assert not hasattr(cfg.rewards, 'direction_deviation')
    assert cfg.rewards.reaching.func is family.SweepSimpleProfV1EnvCfg().rewards.reaching.func
    cfg.rewards.goal_distance.weight = 999
    cfg.terminations.arm_joint_velocity.params['threshold'] = 999
    assert_ordered_equal(parent().to_dict(), before)


@pytest.mark.parametrize('direction_rate', [0., 1.])
def test_reward_manager_wrench_activation_without_positive_budget(configs, direction_rate):
    """Run real RewardManager.compute across the previously untested activation."""
    cfg = configs('rewards', 'sweep_simple').SweepSimpleSemiProfRewardsCfg()
    env = NS(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0,
             _sim_step_counter=0, sensor_data_fresh=torch.ones(2, dtype=torch.bool))
    wrench = torch.zeros(2, 6)
    env.observation_manager = NS(cfg=NS(policy=NS(wrist_wrench=NS(
        func=lambda env: wrench, params={}))))
    term = cfg.wrench_change
    term.func = term.func(term, env)
    names = [name for name, value in vars(cfg).items() if value is not None and hasattr(value, 'weight')]
    for name in names:
        if name != 'wrench_change':
            # Wrench cost must also apply when NO positive reward was earned.
            getattr(cfg, name).func = lambda env, _name=name, **kw: torch.full(
                (env.num_envs,), direction_rate * float(_name == 'force_direction'))
    cls = _execute_definitions(
        PROJECT_ROOT / 'IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py',
        {'torch': torch, 'ManagerBase': object}, {'RewardManager'}).RewardManager
    manager = cls.__new__(cls)
    manager._env = env
    manager._term_names = names
    manager._term_cfgs = [getattr(cfg, name) for name in names]
    manager._class_term_cfgs = [term]
    manager._episode_sums = {name: torch.zeros(2) for name in names}
    manager._reward_buf = torch.zeros(2)
    manager._step_reward = torch.zeros(2, len(names))
    env.reward_manager = manager
    schedule = configs('curricula', 'sweep_simple').SweepSimpleCurriculumCfg().wrench_weight
    for iteration in (0, 1000, 1250, 3000, 6000):
        env.learning_iteration = iteration
        schedule.func(env, None, **schedule.params)
        for _ in range(2):
            env.common_step_counter += 1
            env._sim_step_counter += 2
            wrench[:, 0] = 1000. if env.common_step_counter % 2 else 0.
            assert torch.isfinite(manager.compute(env.step_dt)).all()
            spent = -manager._episode_sums['wrench_change']
            earned = manager._episode_sums['force_direction']
        assert term.func._term_name is None  # no budget lookup or order dependency
        assert term.func._positive_names is None
    assert (spent > .05 * earned).all()
    torch.testing.assert_close(spent, torch.full((2,), .02 * (.00375 + 2 * .03 + 2 * .1)))


def test_contact_history_no_banking_height_partial_reset_and_no_repay(configs):
    env, pos, eef, _, _ = inverse_reach_fixture(configs, 2)
    force = env.scene['hand_contact_target'].data.force_matrix_w
    pos[:, 0, 0] = .09
    eef[:, 0, 2] = 1.19999
    cfg = configs('rewards', 'sweep_simple').SweepSimpleSemiProfRewardsCfg().goal_distance
    term = cfg.func(cfg, env)
    force.zero_()
    pos[:, 0, 0] = .08
    assert not term(env, **cfg.params).any()  # free approach consumes record
    force[:, 0, 0, 0] = .01
    env.common_step_counter += 1
    assert not term(env, **cfg.params).any()  # only one contact
    env.common_step_counter += 1
    assert not term(env, **cfg.params).any()  # recontact, no free-motion backpay
    pos[:, 0, 0] = .07
    env.common_step_counter += 1
    value = term(env, **cfg.params)
    assert (value > 0).all()
    torch.testing.assert_close(term(env, **cfg.params), value)  # no double history update
    assert term._contact_history.sum(-1).tolist() == [3, 3]
    force.zero_()
    pos[:, 0, 0] = .06
    eef[1, 0, 2] = 1.2  # exactly ceiling: rejected, including while history accepted
    env.common_step_counter += 1
    value = term(env, **cfg.params)
    assert value[0] > 0 and value[1] == 0  # current miss may pass 2-of-3
    env.common_step_counter += 1
    pos[:, 0, 0] = .05
    assert not term(env, **cfg.params).any()  # second miss rejected
    force[:, 0, 0, 0] = .02
    eef[:, 0, 2] = 1.19
    for _ in range(2):
        env.common_step_counter += 1
        assert not term(env, **cfg.params).any()  # lost height/contact increments never banked
    env.common_step_counter += 1
    pos[:, 0, 0] = .04
    assert (term(env, **cfg.params) > 0).all()
    env.common_step_counter += 1
    pos[:, 0, 0] = .07
    assert not term(env, **cfg.params).any()
    env.common_step_counter += 1
    pos[:, 0, 0] = .04
    assert not term(env, **cfg.params).any()  # no oscillation farming
    before = term._best_progress.clone()
    term.reset(torch.tensor([0]))
    assert not term._contact_history[0].any() and term._contact_history[1].all()
    assert term._best_progress[0] == 0 and term._best_progress[1] == before[1]
    env.common_step_counter += 1
    pos[:, 0, 0] = .03
    value = term(env, **cfg.params)
    assert value[0] == 0 and value[1] > 0


def test_friction_packed_ranges_invalid_unused_and_cancellation(load_mdp):
    fn = module('mdp.common.sweep_simple_force').aggregate_friction
    forces = torch.tensor([[1., 2, 0], [-1., 1, 0], [0, -3., 0],
                           [float('nan'), 0, 0], [1e30, 0, 0], [.01, 0, 0]])
    counts = torch.tensor([[2, 1, 0], [1, 2, 1]])
    starts = torch.tensor([[0, 2, -1], [3, 5, 5]])
    out, valid = fn(forces, counts, starts)
    assert valid.tolist() == [[True, True, True], [False, False, True]]
    torch.testing.assert_close(out[0].sum(0), torch.zeros(3))
    torch.testing.assert_close(out[1, 2], torch.tensor([.01, 0, 0]))
    empty, ok = fn(torch.empty(0, 3), torch.zeros(1, 1), torch.full((1, 1), -1))
    assert ok.all() and not empty.any()


def force_env(load_mdp, n=2):
    normal = torch.zeros(n, 2, 3)
    friction = torch.zeros(n, 3)
    counts = torch.ones(n, 2, dtype=torch.long); counts[:, 1] = 0
    starts = torch.arange(n)[:, None].expand(-1, 2)
    dts = []
    def normals(dt):
        dts.append(dt)
        return normal
    def frictions(dt):
        dts.append(dt)
        return friction, None, counts, starts
    view = NS(sensor_count=n, filter_count=2, get_contact_force_matrix=normals,
              get_friction_data=frictions)
    env = NS(num_envs=n, device='cpu', common_step_counter=0, _sim_step_counter=0,
             physics_dt=.01, cfg=NS(decimation=2), scene={
                 'object_collection': NS(object_names=['target']),
                 'hand_contact_target': NS(contact_physx_view=view)})
    env._semi_prof_forces = module('mdp.common.sweep_simple_force').SemiProfForceSamples(env)
    return env, normal, friction, dts


def test_physics_mean_real_contact_stale_reset_and_substep_cancellation(load_mdp):
    env, normal, friction, dts = force_env(load_mdp)
    state = env._semi_prof_forces
    state.begin(env)
    normal[:, 0, 0] = 2
    friction[:, 1] = 1
    env._sim_step_counter += 1
    state.capture(env); state.capture(env)
    assert state.samples == 1
    normal[0, 0, 0] = -2
    friction[0, 1] = -1
    env._sim_step_counter += 1
    state.capture(env)
    assert not state.mean(env)[1].any()  # policy counter not yet advanced
    env.common_step_counter += 1
    force, valid = state.mean(env)
    torch.testing.assert_close(force, torch.tensor([[0., 0, 0], [2., 1, 0]]))
    assert valid.all()  # zero force rejected by reward, not contact existence
    assert dts == [.01] * 4
    state.reset(torch.tensor([0]))
    assert state.mean(env)[1].tolist() == [False, True]
    state.begin(env)
    assert not state.normal_sum.any() and not state.contact.any()
    normal.zero_(); friction.fill_(2.)
    for _ in range(2):
        env._sim_step_counter += 1; state.capture(env)
    env.common_step_counter += 1
    assert not state.mean(env)[1].any()  # friction alone cannot fabricate real contact


def direction_env(configs, n):
    env, pos, _, _, _ = inverse_reach_fixture(configs, n)
    env.command_manager = NS(get_command=lambda _: torch.zeros(n, 3))
    pos[:, 0, 0] = -.05
    env._revision_initial_shelf_quat = torch.tensor([[1., 0, 0, 0]]).repeat(n, 1)
    forces = torch.zeros(n, 3)
    valid = torch.ones(n, dtype=torch.bool)
    env._semi_prof_forces = NS(mean=lambda _: (forces, valid))
    return env, pos, forces, valid


def test_angles_threshold_current_target_and_reversible_one_cm(configs):
    env, pos, force, valid = direction_env(configs, 10)
    fn = configs('rewards', 'sweep_simple').SweepSimpleSemiProfRewardsCfg().force_direction.func
    degrees = torch.tensor([0, 15, 30, 45, 60, 75, 90, 120, 180, 0.])
    radians = degrees * math.pi / 180
    force[:, 0] = 2 * torch.cos(radians); force[:, 1] = 2 * torch.sin(radians)
    expected = (1-degrees/90).clamp_min(0).square()
    torch.testing.assert_close(fn(env), expected, atol=1e-6, rtol=1e-5)
    force[:] = torch.tensor([1., 0, 0])
    force[0, 0] = .99999; force[1, 0] = 10.
    assert fn(env)[0] == 0 and fn(env)[1] == fn(env)[2] == 1
    # Overshoot: current goal direction reverses, even though initial direction was +X.
    pos[:, 0, 0] = .05
    assert not fn(env).any()
    force[:] = torch.tensor([-1., 0, 0])
    assert (fn(env) == 1).all()
    # Wrong force is accepted only strictly inside 1cm, no visit latch.
    force[:] = torch.tensor([1., 0, 0])
    pos[:, 0, 0] = torch.tensor([.009999, .01, .010001, 0., .005, .005, .005, .005, .005, .005])
    force[4] = 0; force[5, 0] = .99; valid[6] = False
    force[7, 0] = float('nan'); pos[8, 0, 0] = float('nan')
    assert fn(env).tolist() == [1., 0., 0., 1., 0., 0., 0., 0., 0., 1.]
    pos[0, 0, 0] = .04
    assert fn(env)[0] == 0
    force[0, 0] = -1
    assert fn(env)[0] == 1


def test_planar_reference_full_force_penalizes_vertical_but_keeps_near_exception(configs):
    env, pos, force, _ = direction_env(configs, 1)
    fn = configs('rewards', 'sweep_simple').SweepSimpleSemiProfRewardsCfg().force_direction.func
    # Shelf X points world +Y. Reference is planar, actual force retains Z.
    env._revision_initial_shelf_quat[:] = torch.tensor([math.sqrt(.5), 0, 0, math.sqrt(.5)])
    pos[:] = torch.tensor([0., -.05, 0.])
    # Avoid testing exact 1N after an inexact float32 quaternion rotation;
    # the exact cutoff itself is covered above in the identity frame.
    force[:] = torch.tensor([0., 1.01, 0.])
    assert fn(env).item() == pytest.approx(1.)
    for z in (1.01, -1.01):
        force[:, 2] = z
        assert fn(env).item() == pytest.approx(.25, abs=1e-6)
    force[:, 2] = 1000.
    assert fn(env).item() < 1e-5
    force[:] = torch.tensor([0., .99, 1000.])
    assert fn(env).item() == 0
    force[:] = torch.tensor([0., 1.01, 1000.])
    pos[:, 0, 1] = -.00999
    assert fn(env).item() == 1  # agreed inside-goal angle exception still applies
    pos[:, 0, 1] = -.02
    assert fn(env).item() < 1e-5  # instantaneous, never latched


@pytest.mark.parametrize('fail', [False, True])
def test_runtime_hooks_physics_before_reward_and_restores_on_error(load_mdp, fail):
    events = []
    class Parent:
        def step(self, action):
            for _ in range(2):
                self._sim_step_counter += 1
                self.scene.update(dt=.01)
            self.common_step_counter += 1
            events.append('reward')
            assert self._semi_prof_forces.mean(self)[1].all()
            if fail:
                raise RuntimeError('test physics failure')
            self._reset_idx(torch.tensor([0]))
            return action
        def _reset_idx(self, ids):
            events.append('parent_reset')
            assert self._semi_prof_forces.valid[ids].all()
    cls = _execute_definitions(MODULE_ROOT/'runtime/sweep_simple.py', dict(
        SweepSimpleFingerV1Env=Parent, __package__='_baseline_cpu_pkg.runtime'),
        {'SweepSimpleSemiProfEnv'}).SweepSimpleSemiProfEnv
    base, normal, _, _ = force_env(load_mdp)
    normal[:, 0, 0] = 2
    env = cls.__new__(cls); env.__dict__.update(base.__dict__)
    original = lambda **kw: events.append('scene_update')
    env.scene = NS(update=original)
    env._semi_collecting = False
    if fail:
        with pytest.raises(RuntimeError): env.step(torch.ones(1))
    else:
        env.step(torch.ones(1))
        assert env._semi_prof_forces.valid.tolist() == [False, True]
    assert env.scene.update is original and not env._semi_collecting
    assert events[:3] == ['scene_update', 'scene_update', 'reward']


def test_registry_and_runner_only_experiment_diff(omni_config):
    cc = omni_config.baseline.configclass
    @cc
    class Base:
        distribution_cfg: object = None
        policy: object = None
    class Distribution(_AssetNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    Base.GaussianDistributionCfg = Distribution
    runners = _execute_definitions(MODULE_ROOT/'agent/sweep_simple_ppo_cfg.py', dict(
        configclass=cc, MISSING=MISSING, RslRlMLPModelCfg=Base,
        RslRlOnPolicyRunnerCfg=Base, RslRlPpoAlgorithmCfg=_AssetNode))
    expected = runners.SweepSimpleFingerV1PPORunnerCfg().to_dict()
    expected['experiment_name'] = 'inspire_sweep_simple_semi_prof_v0'
    assert_ordered_equal(runners.SweepSimpleSemiProfPPORunnerCfg().to_dict(), expected)
    entries = {}
    _execute_definitions(MODULE_ROOT/'__init__.py', {'gym': NS(
        registry=entries, register=lambda **kw: entries.update({kw['id']: kw}))})
    entry = entries['Inspire-Sweep-Simple-Semi-Prof-v0']
    assert entry['entry_point'].endswith(':SweepSimpleSemiProfEnv')
    assert entry['kwargs']['env_cfg_entry_point'].endswith(':SweepSimpleSemiProfEnvCfg')
    assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(':SweepSimpleSemiProfPPORunnerCfg')
