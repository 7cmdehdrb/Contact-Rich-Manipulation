"""Relative-frame invariance, snapshot isolation, tactile states and shared PPO storage."""
import importlib
import importlib.util
import math
from types import SimpleNamespace as NS

import torch
from tensordict import TensorDict

from conftest import MODULE_ROOT
from test_final_configuration import configs
from test_omni_configuration import omni_config


def relative_env():
    q = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    root = NS(root_pos_w=torch.tensor([[3., 2., 1.], [-3., 1., 2.]]), root_quat_w=q.clone())
    eef = NS(target_pos_w=(root.root_pos_w + torch.tensor([1., 0., 0.]))[:, None],
             target_quat_w=q[:, None].clone())
    command = NS(target_init_state_w=torch.cat((root.root_pos_w + torch.tensor([2., 1., 0.]), q), -1),
                 command=root.root_pos_w + torch.tensor([3., 2., 0.]))
    env = NS(num_envs=2, device='cpu', scene={'robot':NS(data=root), 'ee_frame':NS(data=eef)},
             cfg=NS(observations=NS(policy=NS(enable_corruption=True))),
             command_manager=NS(get_term=lambda name:command, get_command=lambda name:command.command))
    rows = torch.zeros(2, 16, 4)
    rows[:, 0] = torch.tensor([1., 2., 1., 0.])
    rows[:, 1, 1:] = 123.  # hidden slots must stay exactly zero after translation
    env._initial_obstacle_observation = rows.flatten(1)
    return env, command, eef


def test_current_eef_transform_and_no_live_geometry(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    env, command, eef = relative_env()
    snapshot = env._initial_obstacle_observation.clone()
    term = m.InitialObjectPositionInEEF(NS(params={}), env)
    initial = term(env, coordinate_noise=0.)
    torch.testing.assert_close(initial, torch.tensor([[1., 1., 0.]]).repeat(2, 1))
    torch.testing.assert_close(m.goal_position_in_eef(env), torch.tensor([[2., 2., 0.]]).repeat(2, 1))
    eef.target_pos_w[:, 0, 0] += 1.
    eef.target_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    expected = torch.tensor([[1., 0., 0.]]).repeat(2, 1)
    torch.testing.assert_close(term(env), expected, atol=2e-6, rtol=0)
    rows = m.initial_obstacle_positions_in_eef(env).reshape(2, 16, 4)
    torch.testing.assert_close(rows[:, 0, 1:], expected, atol=2e-6, rtol=0)
    assert not rows[:, 1:].any()
    assert torch.equal(snapshot, env._initial_obstacle_observation)
    # Changing even the command's snapshot has no effect until that env resets.
    command.target_init_state_w[:, 0] += 5.
    torch.testing.assert_close(term(env), expected, atol=2e-6, rtol=0)
    term.reset(torch.tensor([0]))
    result = term(env, coordinate_noise=0.)
    torch.testing.assert_close(result[0], torch.tensor([1., -5., 0.]), atol=2e-6, rtol=0)
    torch.testing.assert_close(result[1], expected[1], atol=2e-6, rtol=0)


def test_target_noise_is_fixed_in_root_and_partial_reset_only(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    env, _, eef = relative_env()
    term = m.InitialObjectPositionInEEF(NS(params={}), env)
    torch.manual_seed(41)
    first = term(env)
    bias = first - torch.tensor([1., 1., 0.])
    assert (bias.abs() <= .010001).all() and bias.abs().sum() > 0
    torch.testing.assert_close(first, term(env))
    eef.target_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    rotated = term(env)
    torch.testing.assert_close(rotated, first * torch.tensor([-1., -1., 1.]))
    term.reset(torch.tensor([0]))
    updated = term(env)
    torch.testing.assert_close(updated[1], rotated[1])
    assert not torch.equal(updated[0], rotated[0])


def test_tactile_four_states_and_original_bit_noise(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    bits = torch.zeros(4, 34)
    bits[1, :17] = bits[2, 17:] = bits[3, :] = 1.
    encoded = m.encode_paired_tactile(bits)
    assert torch.equal(encoded, torch.tensor([0., 1., -1., 2.])[:, None].expand(-1, 17))
    torch.manual_seed(77)
    expected = m.encode_paired_tactile(m.tactile_observation_noise(bits, None))
    torch.manual_seed(77)
    assert torch.equal(m.paired_tactile_observation_noise(encoded, None), expected)


def test_observations_only_expose_shared_actor_inputs(configs, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', importlib.import_module('_omni_official_utils.noise'))
    module = configs('observations', 'wrench_refactor')
    parent = module.WrenchRefactorObservationsCfg()
    before = parent.to_dict()
    relative = module.WrenchRefactorRelativeObservationsCfg()
    assert relative.critic is relative.critic_obstacles is None
    assert parent.to_dict() == before
    assert list(vars(parent.policy)) == list(vars(relative.policy))
    assert relative.policy.ee_pose.to_dict() == parent.policy.ee_pose.to_dict()
    assert relative.policy.wrist_wrench.to_dict() == parent.policy.wrist_wrench.to_dict()


def test_shared_models_storage_gradients_and_export():
    from rsl_rl.storage import RolloutStorage
    spec = importlib.util.spec_from_file_location('_relative_models', MODULE_ROOT / 'agent/wrench_refactor_models.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = torch.randn(8, 16, 4); rows[..., 0] = 1.
    obs = TensorDict({'policy':torch.randn(8, 58), 'obstacles':rows.flatten(1)}, batch_size=[8])
    groups = {'actor':['policy','obstacles'], 'critic':['policy','obstacles']}
    actor = m.WrenchRefactorRelativeModel(obs, groups, 'actor', 8,
        distribution_cfg={'class_name':'GaussianDistribution', 'init_std':1.})
    critic = m.WrenchRefactorRelativeModel(obs, groups, 'critic', 1, obstacle_group='obstacles')
    assert actor.obs_dim == critic.obs_dim == 58
    for model in (actor, critic):
        model(obs).square().mean().backward()
        assert model.obstacle_encoder.mlp[0].weight.grad.abs().sum() > 0
    export = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(export(torch.cat((obs['policy'], obs['obstacles']), -1)), actor(obs))
    storage = RolloutStorage('rl', 8, 2, obs, [8], 'cpu')
    assert set(storage.observations.keys()) == {'policy', 'obstacles'}
    assert storage.observations['policy'].shape == (2, 8, 58)
