"""Regression for the former tanh/atanh failure in the real installed PPO loop."""
import copy
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions


def module(name):
    return importlib.import_module('_baseline_cpu_pkg.' + name)


@pytest.mark.parametrize('entropy,schedule',[(.005,'adaptive'),(0.,'adaptive'),(.005,'fixed')])
@pytest.mark.parametrize('mean,std', [(0.,1.),(10.,.01),(100.,1.)])
def test_finite_saturated_rollout_has_finite_update(load_mdp, mean, std, entropy, schedule):
    torch.manual_seed(42)
    model = module('agent.sweep_simple_models').SweepSimpleFingerModel
    dist = module('agent.sweep_simple_distributions').FingerGaussian
    numerical = module('agent.numerical_ppo').NumericalPPO
    obs = TensorDict({'policy': torch.ones(8,66), 'obstacles': torch.zeros(8,64)}, batch_size=[8])
    cfg = dict(actor=dict(class_name=model, hidden_dims=[8], distribution_cfg=dict(class_name=dist,init_std=std)),
               critic=dict(class_name=model, hidden_dims=[8]),
               algorithm=dict(class_name=numerical,num_learning_epochs=8,num_mini_batches=4,
                              schedule=schedule,desired_kl=.02,entropy_coef=entropy),
               obs_groups={'actor':['policy','obstacles'],'critic':['policy','obstacles']},
               num_steps_per_env=4,multi_gpu=None)
    alg = PPO.construct_algorithm(obs,NS(num_envs=8,num_actions=12),copy.deepcopy(cfg),'cpu')
    with torch.no_grad():
        for p in alg.actor.mlp.parameters(): p.zero_()
        # Actual MLP last Linear before the distribution.
        linear = [m for m in alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        linear.bias.fill_(mean)
    with torch.inference_mode():
        for step in range(4):
            actions = alg.act(obs)
            assert torch.isfinite(actions).all()
            alg.process_env_step(obs,torch.arange(8,dtype=torch.float32)/8,
                                 torch.zeros(8,dtype=torch.long),{})
        alg.compute_returns(obs)
    losses = alg.update()
    assert all(math.isfinite(v) for v in losses.values())
    assert torch.isfinite(alg.actor.distribution.log_std_param.grad).all()


@pytest.mark.parametrize('clip_actions', [True, False])
def test_large_raw_actions_feedback_and_cost_follow_environment_variant(load_mdp, clip_actions):
    """Real runtime clip, observation/cost functions and PPO storage/update.

    Only physics and observation concatenation are replaced by a small CPU
    parent. Execute two rollouts to exercise next-iteration feedback as well.
    """
    torch.manual_seed(42)
    bounds = module('assets.hand_kinematics')
    observe = (module('mdp.observation_functions.sweep_simple').bounded_previous_action if clip_actions
               else module('mdp.observation_functions.final').previous_action_command)
    cost = module('mdp.reward_functions.sweep_simple').finger_action_rate_l2
    model = module('agent.sweep_simple_models').SweepSimpleFingerModel
    dist = module('agent.sweep_simple_distributions').FingerGaussian
    numerical = module('agent.numerical_ppo').NumericalPPO
    obs = TensorDict({'policy': torch.zeros(8, 66), 'obstacles': torch.zeros(8, 64)}, batch_size=[8])

    class PhysicsParent:
        def step(self, action):
            self.action_manager.prev_action = self.action_manager.action.clone()
            self.action_manager.action = action.clone()
            next_obs = obs.clone()
            # Synthetic layout; use the actual Last Action calculation.
            next_obs['policy'][:, :12] = observe(self)
            # Match RewardManager's 0.02 s integration in the actual environments.
            reward = -.03 * (cost(self, 'arm') + cost(self, 'hand')) * .02
            return next_obs, reward, torch.zeros(8, dtype=torch.long), {'log': {'existing': torch.tensor(7.)}}

    runtime = _execute_definitions(MODULE_ROOT / 'runtime/sweep_simple.py', dict(
        SweepSimpleFingerEnv=PhysicsParent, __package__='_baseline_cpu_pkg.runtime'), {'SweepSimpleFingerV1Env'})
    env = runtime.SweepSimpleFingerV1Env()
    env.cfg = NS(clip_raw_actions=clip_actions)
    env.num_envs, env.num_actions = 8, 12
    lo, hi = bounds.finger_action_bounds('cpu')
    env._finger_action_lower, env._finger_action_upper = lo, hi
    env.action_manager = NS(action=torch.zeros(8, 12), prev_action=torch.zeros(8, 12))
    cfg = dict(actor=dict(class_name=model, hidden_dims=[8], distribution_cfg=dict(class_name=dist, report_clip_fraction=clip_actions)),
               critic=dict(class_name=model, hidden_dims=[8]),
               algorithm=dict(class_name=numerical, num_learning_epochs=2, num_mini_batches=2),
               obs_groups={'actor': ['policy', 'obstacles'], 'critic': ['policy', 'obstacles']},
               num_steps_per_env=4, multi_gpu=None)
    alg = PPO.construct_algorithm(obs, env, copy.deepcopy(cfg), 'cpu')
    with torch.no_grad():
        for p in alg.actor.mlp.parameters():
            p.zero_()
        last = [m for m in alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.bias.copy_(torch.tensor([100., -100.] * 6))
    previous_logs = None
    for iteration in range(2):
        with torch.inference_mode():
            for step in range(4):
                raw = alg.act(obs)
                saved = raw.clone()
                log_prob = alg.actor.get_output_log_prob(raw).clone()
                obs, reward, dones, extras = env.step(raw)
                torch.testing.assert_close(raw, saved)  # No in-place clipping of PPO samples.
                expected_command = raw.clamp(lo, hi) if clip_actions else raw
                torch.testing.assert_close(env.action_manager.action, expected_command)
                torch.testing.assert_close(obs['policy'][:, :12], expected_command if clip_actions else raw.tanh())
                assert obs['policy'].abs().max() <= hi.max()
                delta = expected_command - env.action_manager.prev_action
                expected_reward = -.03 * (delta[:, :6].square().sum(-1)
                                         + delta[:, 6:].square().sum(-1)) * .02
                torch.testing.assert_close(reward, expected_reward)
                if clip_actions and (iteration or step):
                    torch.testing.assert_close(reward, torch.zeros_like(reward))
                if not clip_actions and iteration == 0 and step == 0:
                    assert (reward < -.12).all()  # Exceeds both former caps after dt integration.
                alg.process_env_step(obs, reward, dones, extras)
                torch.testing.assert_close(alg.storage.actions[step], saved)
                torch.testing.assert_close(alg.storage.actions_log_prob[step].squeeze(-1), log_prob)
                assert extras['log']['existing'] == 7.
                if clip_actions:
                    assert extras['log']['PolicyAction/clip_fraction/translation_x'] == 1.
                else:
                    assert not any('clip_fraction' in key for key in extras['log'])
                assert extras['log']['PolicyAction/mean/translation_x'] > 99.
                if previous_logs is not None:
                    assert extras['log'] is not previous_logs
                previous_logs = extras['log']
            alg.compute_returns(obs)
        losses = alg.update()
        assert all(math.isfinite(v) for v in losses.values())
