"""Exercise independent fusion CFGs and the 55D actor through real CPU PPO."""

from _source_units import source_text, unit_paths

import ast
import copy
import importlib
import sys

import numpy as np
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from conftest import MODULE_ROOT, PACKAGE_ROOT, PROJECT_ROOT
from test_clutter_v1_models import _algorithm, runner_configs
from test_omni_configuration import _execute_definitions


@pytest.fixture
def fusion_models(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    return importlib.import_module("inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_models")


@pytest.fixture
def fusion_runner_cfg(runner_configs):
    namespace = sys.modules["isaaclab_rl.rsl_rl"].__dict__
    result = _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion/ppo_cfg.py", namespace)
    config = result.ContactSweepClutterFusionPPORunnerCfg()
    # Apply the same framework compatibility pass as train.py and play.py.
    path = PROJECT_ROOT / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py"
    spec = importlib.util.spec_from_file_location("_fusion_model_compat", path)
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    return compatibility.handle_deprecated_rsl_rl_cfg(config, "5.0.1")


def _observations(count=8):
    generator = torch.Generator().manual_seed(17)
    obstacles = torch.randn(count, 10, 4, generator=generator)
    obstacles[..., 0] = 0.0
    obstacles[:, :3, 0] = 1.0
    return TensorDict({
        "policy": torch.randn(count, 55, generator=generator),
        "privileged": torch.randn(count, 3, generator=generator),
        "obstacles": obstacles.flatten(1),
    }, batch_size=[count])


def test_independent_runner_preserves_ppo_and_routes_55d_policy(
    fusion_models, fusion_runner_cfg, runner_configs,
):
    cfg = fusion_runner_cfg
    reference = runner_configs[1]
    assert cfg.algorithm.to_dict() == reference.algorithm.to_dict()
    for name in ("num_steps_per_env", "max_iterations", "save_interval", "clip_actions"):
        assert getattr(cfg, name) == getattr(reference, name)
    assert cfg.class_name == "CurriculumOnPolicyRunner"
    cfg.actor.encoder_hidden_dims[0] = 31
    assert cfg.critic.encoder_hidden_dims == [32]
    cfg.actor.encoder_hidden_dims[0] = 32
    obs = _observations()
    obs["policy"].requires_grad_()
    obs["privileged"].requires_grad_()
    alg = _algorithm(obs, cfg)
    assert isinstance(alg.actor, fusion_models.ContactSweepClutterFusionModel)
    assert isinstance(alg.critic, fusion_models.ContactSweepClutterFusionModel)
    assert alg.actor.obs_dim == 55 and alg.critic.obs_dim == 58
    assert alg.actor.get_latent(obs).shape == (8, 71)
    assert alg.critic.get_latent(obs).shape == (8, 74)
    actions, values = alg.actor(obs), alg.critic(obs)
    actor_grad = torch.autograd.grad(actions.sum(), obs["privileged"], allow_unused=True)[0]
    critic_grad = torch.autograd.grad(values.sum(), obs["privileged"])[0]
    assert actor_grad is None
    assert critic_grad.abs().sum() > 0
    obs["privileged"] = torch.full((8, 3), float("nan"))
    torch.testing.assert_close(alg.actor(obs), actions)


def test_real_ppo_updates_both_encoders_and_restores_checkpoint(
    fusion_models, fusion_runner_cfg, tmp_path,
):
    torch.manual_seed(42)
    obs = _observations()
    alg = _algorithm(obs, fusion_runner_cfg)
    before = [copy.deepcopy(model.obstacle_encoder.state_dict()) for model in (alg.actor, alg.critic)]
    with torch.inference_mode():
        for _ in range(4):
            actions = alg.act(obs)
            rewards = -(actions - 0.25).square().sum(dim=-1)
            alg.process_env_step(obs, rewards, torch.zeros(8, dtype=torch.long), {})
        alg.compute_returns(obs)
    assert all(np.isfinite(value) for value in alg.update().values())
    for model, initial in zip((alg.actor, alg.critic), before):
        params = list(model.obstacle_encoder.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(p.grad.abs().sum() > 0 for p in params)
        assert any(not torch.equal(initial[name], value) for name, value in model.obstacle_encoder.state_dict().items())
    path = tmp_path / "model.pt"
    torch.save(alg.save(), path)
    restored = _algorithm(obs, fusion_runner_cfg)
    assert restored.load(torch.load(path, weights_only=True), None, strict=True)
    torch.testing.assert_close(alg.actor(obs), restored.actor(obs), rtol=0, atol=0)
    torch.testing.assert_close(alg.critic(obs), restored.critic(obs), rtol=0, atol=0)


def test_actor_export_uses_95d_and_masks_unobserved_coordinates(
    fusion_models, fusion_runner_cfg, tmp_path,
):
    import onnx
    from onnx.reference import ReferenceEvaluator

    obs = _observations(1)
    alg = _algorithm(obs, fusion_runner_cfg)
    expected = alg.actor(obs).detach()
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = alg
    inputs = torch.cat((obs["policy"], obs["obstacles"]), dim=-1)
    assert inputs.shape == (1, 95)
    inputs[:, 55:].unflatten(-1, (10, 4))[:, 3:, 1:] = float("nan")
    runner.export_policy_to_jit(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(inputs), expected)
    scripted.reset()
    runner.export_policy_to_onnx(str(tmp_path))
    graph = onnx.load(tmp_path / "policy.onnx")
    onnx.checker.check_model(graph)
    actual = ReferenceEvaluator(graph).run(None, {"obs": inputs.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("group,dimension", [("policy", 57), ("privileged", 4), ("obstacles", 36)])
def test_model_rejects_old_or_incompatible_shapes(fusion_models, group, dimension):
    obs = _observations()
    obs[group] = torch.zeros(8, dimension)
    with pytest.raises(ValueError):
        fusion_models.ContactSweepClutterFusionModel(
            obs, {"critic": ["policy", "privileged", "obstacles"]}, "critic", 1,
        )


def test_cfgs_do_not_inherit_other_task_cfgs():
    root = MODULE_ROOT / "contact_sweep_clutter_fusion"
    allowed = {
        "InteractiveSceneCfg", "ManagerBasedRLEnvCfg", "RslRlOnPolicyRunnerCfg",
        "RslRlMLPModelCfg", "ObservationGroupCfg", "ObsGroup",
    }
    files = unit_paths("contact_sweep_clutter_fusion", "*_cfg.py")
    assert files
    for file in files:
        for node in ast.walk(ast.parse(source_text(file))):
            if isinstance(node, ast.ClassDef):
                assert all(isinstance(base, ast.Name) and base.id in allowed for base in node.bases), (file, node.name)
            if isinstance(node, ast.ImportFrom) and node.level == 2:
                # MDP term implementations may live beside an older task Cfg;
                # only importing its configuration classes breaks independence.
                assert not any(alias.name.endswith("Cfg") for alias in node.names
                               if (node.module or "").split(".")[0].startswith("contact_sweep")), file


def test_fourteen_obstacle_slots_train_and_export(fusion_models, fusion_runner_cfg):
    """The added slots reach PPO gradients and the exported actor, without truncation."""
    obs = _observations()
    obs['obstacles'] = torch.zeros(8, 56)
    obs['obstacles'].unflatten(-1, (14, 4))[..., 0] = 1.
    obs['obstacles'][:, -3:] = torch.randn(8, 3)
    fusion_runner_cfg.actor.obstacle_observation_slots = 14
    fusion_runner_cfg.critic.obstacle_observation_slots = 14
    alg = _algorithm(obs, fusion_runner_cfg)
    obs['obstacles'].requires_grad_()
    output = alg.actor(obs)
    grad = torch.autograd.grad(output.sum(), obs['obstacles'])[0]
    assert grad[:, -3:].abs().sum() > 0
    assert alg.actor.obstacle_encoder.mlp[0].in_features == 56
    exported = alg.actor.as_jit()
    assert exported.get_dummy_inputs()[0].shape == (1, 111)
    inputs = torch.cat((obs['policy'], obs['obstacles']), -1)
    torch.testing.assert_close(torch.jit.script(exported)(inputs), output)
    with torch.inference_mode():
        for _ in range(alg.storage.num_transitions_per_env):
            actions = alg.act(obs)
            alg.process_env_step(obs, -actions.square().sum(-1), torch.zeros(8, dtype=torch.long), {})
        alg.compute_returns(obs)
    assert all(np.isfinite(value) for value in alg.update().values())
