"""Pull-only goal sampling and complete parent/child wiring, without Isaac Sim."""
import copy
from dataclasses import MISSING
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import assert_ordered_equal
from test_omni_configuration import _AssetNode, _execute_definitions
from test_sweep_simple import configs, family, omni_config


@pytest.fixture
def layouts(load_mdp):
    return NS(
        zone=importlib.import_module('_baseline_cpu_pkg.mdp.common.sweep_v0_layout'),
        base=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout'),
        side=importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.sweep_contact_preference'),
    )


@pytest.mark.parametrize('seed', [0, 42, 1123])
def test_only_zone3_with_parent_spawn_distance_edges_and_rng(layouts, seed):
    half = torch.full((4096, 17, 3), .025, dtype=torch.float64)
    bounds = torch.stack((-half, half), -2)
    parent = layouts.base.CircleResetLayout(bounds, 1.05 + half[..., 2])
    child = layouts.zone.SingleContactZone3ResetLayout(bounds, 1.05 + half[..., 2])
    ids = torch.arange(0, len(bounds), 2)
    params = dict(spawn_radius=.02, edge_clearance=.02, excluded_x_half_angle=0.,
                  min_goal_distance=.06, max_goal_distance=.12)
    rng1, rng2 = (torch.Generator().manual_seed(seed) for _ in range(2))
    initial, _ = parent.sample_commands(ids, rng1, **params)
    target, goal = child.sample_commands(ids, rng2, **params)
    torch.testing.assert_close(initial, target, rtol=0, atol=0)
    assert torch.equal(rng1.get_state(), rng2.get_state())
    delta = goal - target
    direction = torch.cat((delta, delta.new_zeros(len(ids), 1)), -1)
    assert layouts.side.sweep_region(direction).eq(3).all()
    angles = torch.atan2(delta[:, 1], delta[:, 0])
    assert (angles.abs() <= math.pi / 6 + 1e-12).all()
    assert angles.min() < -.50 and angles.max() > .50
    assert (target.norm(dim=-1) <= .02 + 1e-12).all()
    distances = delta.norm(dim=-1)
    assert ((distances >= .06 - 1e-12) & (distances <= .12 + 1e-12)).all()
    assert (goal.abs() <= torch.tensor([.155, .455])).all()
    assert params['excluded_x_half_angle'] == 0. and 'goal_angle_range' not in params


def test_complete_circle_placement_and_partial_ids_keep_pull_goals(layouts):
    half = torch.full((8, 17, 3), .01)
    bounds = torch.stack((-half, half), -2)
    before = bounds.clone()
    layout = layouts.zone.SingleContactZone3ResetLayout(bounds, 1.05 + half[..., 2])
    ids = torch.tensor([7, 1, 4])
    local, yaw, active, delta, requested = layout.sample(
        ids, torch.Generator().manual_seed(42), excluded_x_half_angle=0.)
    assert local.shape == (3, 17, 3) and yaw.shape == active.shape == (3, 17)
    direction = torch.cat((delta, delta.new_zeros(3, 1)), -1)
    assert layouts.side.sweep_region(direction).eq(3).all()
    assert active[:, 0].all() and torch.equal(active[:, 1:].sum(-1), requested)
    assert ((requested >= 0) & (requested <= 6)).all()
    torch.testing.assert_close(bounds, before, rtol=0, atol=0)
    assert layouts.zone.SingleContactZone3ResetLayout.sample is layouts.base.CircleResetLayout.sample


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_sector_endpoints_and_center_are_zone3(layouts, monkeypatch, dtype):
    half = torch.full((5, 17, 3), .02, dtype=dtype)
    bounds = torch.stack((-half, half), -2)
    layout = layouts.zone.SingleContactZone3ResetLayout(bounds, 1.05 + half[..., 2])
    draws = torch.full((5, 4), .5, dtype=dtype)
    draws[:, 2] = torch.tensor([0., .001, .5, .999, 1.], dtype=dtype)
    monkeypatch.setattr(torch, 'rand', lambda *a, **kw: draws.clone())
    target, goal = layout.sample_commands(torch.arange(5), None, excluded_x_half_angle=0.)
    delta = goal - target
    assert layouts.side.sweep_region(torch.cat((delta, delta.new_zeros(5, 1)), -1)).eq(3).all()


@pytest.fixture
def sweep_configs(family, configs, layouts):
    # Run real configclass, manager terms, SceneCfg and all EnvCfg post-init
    # code. Simulator asset leaves and Action constructors use existing doubles.
    ns = dict(vars(family))
    ns.update(vars(_execute_definitions(MODULE_ROOT / 'assets/scenes/stabilize_sweep.py', ns)))
    for role in ('commands', 'observations', 'rewards', 'events', 'terminations'):
        ns.update(vars(configs(role, 'stabilize_sweep')))
    reward = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep')
    side = layouts.side
    ns.update(
        StabilizeSweepActionsCfg=lambda: _AssetNode(arm_action=_AssetNode(), gripper_action=_AssetNode()),
        StabilizeSweepOSCActionCfg=NS(OffsetCfg=_AssetNode),
        SweepOnlyClippedOSCAction=type('SweepOnlyClippedOSCAction', (), {}),
        NormalizedFingerJointPositionAction=type('NormalizedFingerJointPositionAction', (), {}),
        legacy_clipped_previous_action=importlib.import_module(
            '_baseline_cpu_pkg.mdp.observation_functions.stabilize_sweep').legacy_clipped_previous_action,
        linear_reach=reward.linear_reach, incremental_approach_reach=reward.incremental_approach_reach,
        arm_joint_failure=reward.arm_joint_failure,
        ContactSideSweepProgress=side.ContactSideSweepProgress,
        SignedContactSideSweepProgress=side.SignedContactSideSweepProgress,
        SingleContactZone3ResetLayout=layouts.zone.SingleContactZone3ResetLayout,
    )
    return _execute_definitions(MODULE_ROOT / 'envs/sweep_v0_envs.py', ns)


def test_complete_configuration_diff_and_parent_isolation(sweep_configs, layouts):
    parent_cls = sweep_configs.SweepSingleContactEnvCfg
    child_cls = sweep_configs.SweepSingleContactZone3EnvCfg
    assert child_cls.__bases__ == (parent_cls,)
    parent = parent_cls()
    before = parent.to_dict()
    child = child_cls()
    expected = copy.deepcopy(before)
    layout_cls = layouts.zone.SingleContactZone3ResetLayout
    expected['events']['object_geometry']['params']['layout_class'] = f'{layout_cls.__module__}:{layout_cls.__name__}'
    expected['rewards']['sweep_progress']['weight'] = 100.
    expected['rewards']['sweep_retreat']['weight'] = -20.
    assert_ordered_equal(child.to_dict(), expected)
    assert child.rewards.reaching.func is parent.rewards.reaching.func
    assert child.rewards.reaching.weight == .8
    assert child.events.object_spawn.func is parent.events.object_spawn.func
    assert child.events.object_geometry.params['layout_class'] is layout_cls
    assert 'layout_class' not in parent.events.object_geometry.params
    child.events.object_spawn.params['ik_params']['height_offset'] = 99.
    child.events.object_geometry.params['layout_class'] = None
    child.rewards.sweep_progress.weight = -999.
    assert_ordered_equal(parent.to_dict(), before)
    assert_ordered_equal(parent_cls().to_dict(), before)
    assert_ordered_equal(child_cls().to_dict(), expected)


def test_registry_reuses_parent_runtime_and_ppo_only_changes_log_name(sweep_configs):
    entries = {}
    _execute_definitions(MODULE_ROOT / '__init__.py', {'gym': NS(
        registry=entries, register=lambda **kw: entries.update({kw['id']: kw}))})
    parent = entries['Inspire-Sweep-Single-Contact-v0']
    child = entries['Inspire-Sweep-Single-Contact-Zone3-v0']
    assert child['entry_point'] == parent['entry_point']
    assert child['kwargs']['env_cfg_entry_point'].endswith(':SweepSingleContactZone3EnvCfg')
    assert child['kwargs']['rsl_rl_cfg_entry_point'].endswith(':SweepSingleContactZone3PPORunnerCfg')
    # Execute every real runner field; only external learner/model leaves are doubles.
    base = sweep_configs.configclass(type('Runner', (), {
        '__annotations__': {'policy': object}, 'policy': None,
    }))
    runners = _execute_definitions(MODULE_ROOT / 'agent/sweep_v0_ppo_cfg.py',
        {'configclass': sweep_configs.configclass, 'MISSING': MISSING,
         'RslRlOnPolicyRunnerCfg': base, 'RslRlPpoAlgorithmCfg': _AssetNode,
         'RslRlMLPModelCfg': NS(GaussianDistributionCfg=_AssetNode),
         'SweepOnlyModelCfg': _AssetNode})
    p = runners.SweepSingleContactPPORunnerCfg()
    c = runners.SweepSingleContactZone3PPORunnerCfg()
    assert type(c).__bases__ == (type(p),)
    expected = p.to_dict()
    expected['experiment_name'] = 'inspire_sweep_single_contact_zone3_v0'
    assert_ordered_equal(c.to_dict(), expected)
