"""Exercise the clutter encoder through the installed RSL-RL PPO/export paths."""

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
from test_omni_configuration import _execute_definitions


@pytest.fixture
def models(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    return importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter.models")


def _observations(count=8):
    generator = torch.Generator().manual_seed(17)
    obstacles = torch.randn(count, 10, 4, generator=generator)
    obstacles[..., 0] = 0
    obstacles[:, :3, 0] = 1
    return TensorDict({
        "policy": torch.randn(count, 57, generator=generator),
        "obstacles": obstacles.flatten(1),
    }, batch_size=[count])


def _algorithm(models, obs):
    cfg = {
        "actor": {
            "class_name": models.ObstacleEncoderModel,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {"class_name": models.ObstacleEncoderModel},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 2},
        "obs_groups": {"actor": ["policy", "obstacles"], "critic": ["policy", "obstacles"]},
        "num_steps_per_env": 4,
        "multi_gpu": None,
    }
    return PPO.construct_algorithm(
        obs, SimpleNamespace(num_envs=obs.batch_size[0], num_actions=8), cfg, "cpu",
    )


def test_ppo_updates_both_encoders_and_checkpoint_restores_them(models, tmp_path):
    torch.manual_seed(42)
    obs = _observations()
    alg = _algorithm(models, obs)
    assert alg.actor.get_latent(obs).shape == (8, 73)
    assert alg.critic.get_latent(obs).shape == (8, 73)
    assert alg.actor.obstacle_encoder is not alg.critic.obstacle_encoder
    initial = [
        copy.deepcopy(model.obstacle_encoder.state_dict())
        for model in (alg.actor, alg.critic)
    ]
    with torch.inference_mode():
        for _ in range(4):
            actions = alg.act(obs)
            rewards = -torch.sum((actions - 0.25) ** 2, dim=-1)
            alg.process_env_step(obs, rewards, torch.zeros(8, dtype=torch.long), {})
        alg.compute_returns(obs)
    losses = alg.update()
    assert all(np.isfinite(value) for value in losses.values())
    for model, before in zip((alg.actor, alg.critic), initial):
        encoder_parameters = list(model.obstacle_encoder.parameters())
        assert all(parameter.grad is not None for parameter in encoder_parameters)
        assert all(torch.isfinite(parameter.grad).all() for parameter in encoder_parameters)
        assert any(parameter.grad.abs().sum() > 0 for parameter in encoder_parameters)
        assert any(
            not torch.equal(before[name], value)
            for name, value in model.obstacle_encoder.state_dict().items()
        )
        assert all(parameter in alg.optimizer.state for parameter in encoder_parameters)

    checkpoint = tmp_path / "model.pt"
    torch.save(alg.save(), checkpoint)
    restored = _algorithm(models, obs)
    assert restored.load(torch.load(checkpoint, weights_only=True), None, strict=True)
    for original, loaded in ((alg.actor, restored.actor), (alg.critic, restored.critic)):
        torch.testing.assert_close(original(obs), loaded(obs), rtol=0, atol=0)
        assert all(parameter in restored.optimizer.state for parameter in loaded.obstacle_encoder.parameters())


def test_dummy_coordinates_cannot_affect_embedding_or_action(models):
    obs = _observations()
    changed = obs.clone()
    rows = changed["obstacles"].unflatten(-1, (10, 4))
    rows[:, 3:, 1] = float("nan")
    rows[:, 3:, 2] = float("inf")
    rows[:, 3:, 3] = -1.0e30
    model = _algorithm(models, obs).actor
    torch.testing.assert_close(model.get_latent(obs), model.get_latent(changed), rtol=0, atol=0)
    torch.testing.assert_close(model(obs), model(changed), rtol=0, atol=0)

    # Presence must matter even if a real obstacle's position is the origin.
    present = torch.zeros(1, 40)
    absent = present.clone()
    present[0, 0] = 1
    encoder = models.MaskedObstacleEncoder()
    assert not torch.equal(encoder(present), encoder(absent))


def test_model_rejects_incompatible_observation_contract(models):
    obs = _observations()
    groups = {"actor": ["policy", "obstacles"]}
    with pytest.raises(ValueError, match="exactly once"):
        models.ObstacleEncoderModel(obs, {"actor": ["policy"]}, "actor", 8)
    obs["obstacles"] = torch.zeros(8, 36)
    with pytest.raises(ValueError, match="40"):
        models.ObstacleEncoderModel(obs, groups, "actor", 8)
    obs["obstacles"] = torch.zeros(8, 40)
    obs["policy"] = torch.zeros(8, 56)
    with pytest.raises(ValueError, match="57D"):
        models.ObstacleEncoderModel(obs, groups, "actor", 8)


def test_framework_jit_and_onnx_exports_include_encoder(models, tmp_path):
    import onnx
    from onnx.reference import ReferenceEvaluator

    obs = _observations(count=1)
    alg = _algorithm(models, obs)
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = alg
    inputs = torch.cat((obs["policy"], obs["obstacles"]), dim=-1)
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

    # Export must apply masking too, rather than requiring caller-side cleanup.
    dirty_inputs = inputs.clone()
    dirty_inputs[:, 57:].unflatten(-1, (10, 4))[:, 3:, 1:] = float("nan")
    torch.testing.assert_close(scripted(dirty_inputs), expected)
    dirty_actual = ReferenceEvaluator(graph).run(None, {"obs": dirty_inputs.numpy()})[0]
    np.testing.assert_allclose(dirty_actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_runner_preserves_parent_hyperparameters_and_constructs_custom_models(models, monkeypatch):
    # Import the actual Isaac Lab config decorator without starting Sim.
    utils_path = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/utils"
    utilities = types.ModuleType("_clutter_model_config_utils")
    utilities.__path__ = [str(utils_path)]
    monkeypatch.setitem(sys.modules, utilities.__name__, utilities)
    configclass = importlib.import_module(f"{utilities.__name__}.configclass").configclass
    official = types.ModuleType("isaaclab_rl.rsl_rl")
    official.__dict__.update(configclass=configclass, MISSING=MISSING, Literal=Literal)
    monkeypatch.setitem(sys.modules, official.__name__, official)
    framework_root = PROJECT_ROOT / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl"
    class_names = {
        "RslRlMLPModelCfg", "RslRlRNNModelCfg", "RslRlPpoAlgorithmCfg",
        "RslRlBaseRunnerCfg", "RslRlOnPolicyRunnerCfg",
        "RslRlPpoActorCriticCfg", "RslRlPpoActorCriticRecurrentCfg",
    }
    _execute_definitions(framework_root / "rl_cfg.py", official.__dict__, class_names)
    _execute_definitions(MODULE_ROOT / "contact_sweep/ppo_cfg.py", official.__dict__)
    parent = official.ContactSweepPPORunnerCfg()
    _execute_definitions(MODULE_ROOT / "contact_sweep_clutter/ppo_cfg.py", official.__dict__)
    config = official.ContactSweepClutterPPORunnerCfg()
    assert config.algorithm.to_dict() == parent.algorithm.to_dict()
    for name in ("num_steps_per_env", "max_iterations", "save_interval", "clip_actions"):
        assert getattr(config, name) == getattr(parent, name)
    config.actor.encoder_hidden_dims[0] = 31
    assert config.critic.encoder_hidden_dims == [32]
    config.actor.encoder_hidden_dims[0] = 32

    # Run the same compatibility conversion used by train.py and play.py, then
    # have PPO resolve the fully qualified model classes from this config.
    spec = importlib.util.spec_from_file_location("_clutter_model_compat", framework_root / "utils.py")
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    config = compatibility.handle_deprecated_rsl_rl_cfg(config, "5.0.1")
    cfg = config.to_dict()
    cfg["num_steps_per_env"] = 4
    cfg["multi_gpu"] = None
    alg = PPO.construct_algorithm(
        _observations(), SimpleNamespace(num_envs=8, num_actions=8), cfg, "cpu",
    )
    assert isinstance(alg.actor, models.ObstacleEncoderModel)
    assert isinstance(alg.critic, models.ObstacleEncoderModel)
    assert alg.actor.obs_normalization is alg.critic.obs_normalization is False
    assert alg.actor.get_latent(_observations()).shape == (8, 73)
