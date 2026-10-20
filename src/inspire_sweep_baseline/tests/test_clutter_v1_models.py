"""Exercise asymmetric clutter models through actual PPO and policy export."""

import copy
from dataclasses import MISSING
import importlib
import sys
import types
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from conftest import MODULE_ROOT, PACKAGE_ROOT, PROJECT_ROOT
from test_clutter_models import _observations
from test_omni_configuration import _execute_definitions


@pytest.fixture
def v1_models(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    return importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter_v1.models")


@pytest.fixture
def runner_configs(monkeypatch):
    """Run production Cfg classes with Isaac Lab's real configclass decorator."""
    utils_path = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/utils"
    utilities = types.ModuleType("_clutter_v1_model_config_utils")
    utilities.__path__ = [str(utils_path)]
    monkeypatch.setitem(sys.modules, utilities.__name__, utilities)
    configclass = importlib.import_module(f"{utilities.__name__}.configclass").configclass
    official = types.ModuleType("isaaclab_rl.rsl_rl")
    official.__dict__.update(configclass=configclass, MISSING=MISSING, Literal=Literal)
    monkeypatch.setitem(sys.modules, official.__name__, official)
    framework_root = PROJECT_ROOT / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl"
    _execute_definitions(framework_root / "rl_cfg.py", official.__dict__, {
        "RslRlMLPModelCfg", "RslRlRNNModelCfg", "RslRlPpoAlgorithmCfg",
        "RslRlBaseRunnerCfg", "RslRlOnPolicyRunnerCfg",
        "RslRlPpoActorCriticCfg", "RslRlPpoActorCriticRecurrentCfg",
    })
    for path in ("contact_sweep/ppo_cfg.py", "contact_sweep_clutter/ppo_cfg.py",
                 "contact_sweep_clutter_v1/ppo_cfg.py"):
        _execute_definitions(MODULE_ROOT / path, official.__dict__)
    parent = official.ContactSweepClutterPPORunnerCfg()
    child = official.ContactSweepClutterV1PPORunnerCfg()
    spec = importlib.util.spec_from_file_location("_clutter_v1_model_compat", framework_root / "utils.py")
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    parent = compatibility.handle_deprecated_rsl_rl_cfg(parent, "5.0.1")
    child = compatibility.handle_deprecated_rsl_rl_cfg(child, "5.0.1")
    return parent, child


def _privileged_observations(count=8):
    obs = _observations(count)
    obs["privileged"] = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 20.0
    return obs


def _algorithm(obs, config):
    cfg = config.to_dict()
    cfg["num_steps_per_env"] = 4
    cfg["multi_gpu"] = None
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 2
    return PPO.construct_algorithm(
        obs, SimpleNamespace(num_envs=obs.batch_size[0], num_actions=8), cfg, "cpu",
    )


def test_runner_retains_parent_ppo_and_actor_but_routes_live_xyz_to_critic(v1_models, runner_configs):
    parent, config = runner_configs
    assert config.algorithm.to_dict() == parent.algorithm.to_dict()
    assert config.actor.to_dict() == parent.actor.to_dict()
    for name in ("num_steps_per_env", "max_iterations", "save_interval", "clip_actions"):
        assert getattr(config, name) == getattr(parent, name)
    assert config.experiment_name == "inspire_contact_sweep_clutter_v1"

    obs = _privileged_observations()
    obs["policy"].requires_grad_()
    obs["privileged"].requires_grad_()
    alg = _algorithm(obs, config)
    assert isinstance(alg.critic, v1_models.PrivilegedObstacleEncoderModel)
    assert alg.actor.obs_dim == 57 and alg.critic.obs_dim == 60
    assert alg.actor.get_latent(obs).shape == (8, 73)
    assert alg.critic.get_latent(obs).shape == (8, 76)
    assert alg.actor.obstacle_encoder is not alg.critic.obstacle_encoder
    actions, values = alg.actor(obs), alg.critic(obs)
    actor_grad = torch.autograd.grad(actions.sum(), (obs["policy"], obs["privileged"]), allow_unused=True)
    critic_grad = torch.autograd.grad(values.sum(), (obs["policy"], obs["privileged"]), allow_unused=True)
    assert actor_grad[0] is not None and actor_grad[1] is None
    assert critic_grad[0] is not None and critic_grad[1] is not None
    assert critic_grad[1].abs().sum() > 0
    changed = obs.clone()
    changed["privileged"] += 10.0
    torch.testing.assert_close(alg.actor(changed), actions, rtol=0, atol=0)
    assert not torch.equal(alg.critic(changed), values)
    # A deployed Actor can also run without receiving any privileged tensor.
    actor_obs = TensorDict({"policy": obs["policy"], "obstacles": obs["obstacles"]}, batch_size=[8])
    torch.testing.assert_close(alg.actor(actor_obs), actions, rtol=0, atol=0)


def test_ppo_updates_both_encoders_and_restores_asymmetric_checkpoint(v1_models, runner_configs, tmp_path):
    torch.manual_seed(42)
    obs = _privileged_observations()
    alg = _algorithm(obs, runner_configs[1])
    initial = [copy.deepcopy(model.obstacle_encoder.state_dict()) for model in (alg.actor, alg.critic)]
    with torch.inference_mode():
        for _ in range(4):
            actions = alg.act(obs)
            rewards = -torch.sum((actions - 0.25) ** 2, dim=-1)
            alg.process_env_step(obs, rewards, torch.zeros(8, dtype=torch.long), {})
        alg.compute_returns(obs)
    losses = alg.update()
    assert all(np.isfinite(value) for value in losses.values())
    for model, before in zip((alg.actor, alg.critic), initial):
        parameters = list(model.obstacle_encoder.parameters())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
        assert any(parameter.grad.abs().sum() > 0 for parameter in parameters)
        assert any(not torch.equal(before[name], value) for name, value in model.obstacle_encoder.state_dict().items())
        assert all(parameter in alg.optimizer.state for parameter in parameters)
    checkpoint = tmp_path / "model.pt"
    torch.save(alg.save(), checkpoint)
    restored = _algorithm(obs, runner_configs[1])
    assert restored.load(torch.load(checkpoint, weights_only=True), None, strict=True)
    for original, loaded in ((alg.actor, restored.actor), (alg.critic, restored.critic)):
        torch.testing.assert_close(original(obs), loaded(obs), rtol=0, atol=0)
        assert all(parameter in restored.optimizer.state for parameter in loaded.obstacle_encoder.parameters())


def test_exported_actor_needs_only_97d_initial_observations(v1_models, runner_configs, tmp_path):
    import onnx
    from onnx.reference import ReferenceEvaluator

    obs = _privileged_observations(1)
    alg = _algorithm(obs, runner_configs[1])
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = alg
    inputs = torch.cat((obs["policy"], obs["obstacles"]), dim=-1)
    assert inputs.shape == (1, 97)
    expected = alg.actor(obs).detach()
    runner.export_policy_to_jit(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(inputs), expected)
    scripted.reset()
    runner.export_policy_to_onnx(str(tmp_path))
    graph = onnx.load(tmp_path / "policy.onnx")
    onnx.checker.check_model(graph)
    actual = ReferenceEvaluator(graph).run(None, {"obs": inputs.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)
    obs["privileged"].fill_(float("nan"))
    torch.testing.assert_close(alg.actor(obs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_dimension", "wrong_policy"])
def test_critic_rejects_incompatible_observation_routes(v1_models, failure):
    obs = _privileged_observations()
    groups = ["policy", "privileged", "obstacles"]
    message = "exactly once"
    if failure == "missing":
        groups.remove("privileged")
    elif failure == "duplicate":
        groups.append("privileged")
    elif failure == "wrong_dimension":
        obs["privileged"] = torch.zeros(8, 13)
        message = "num_envs, 3"
    else:
        obs["policy"] = torch.zeros(8, 56)
        message = "57D"
    with pytest.raises(ValueError, match=message):
        v1_models.PrivilegedObstacleEncoderModel(obs, {"critic": groups}, "critic", 1)
