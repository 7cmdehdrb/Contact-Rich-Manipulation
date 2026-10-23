"""Sixteen-slot reset, PPO, masking and export without changing Holding v0."""

import copy
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_clutter_fusion_models import _observations, fusion_models, fusion_runner_cfg
from test_clutter_fusion_v1_0_contacts import contacts, scene_cfg
from test_clutter_fusion_v1_0_population import _population_env, population_modules
from test_clutter_v1_models import _algorithm, runner_configs
from test_omni_configuration import _execute_definitions, omni_config


@pytest.fixture
def holding_runner_configs(fusion_runner_cfg):
    namespace = sys.modules["isaaclab_rl.rsl_rl"].__dict__
    for package in (
        "contact_sweep_clutter_fusion_v2", "contact_sweep_clutter_fusion_v3",
        "contact_sweep_clutter_fusion_v1_0", "contact_sweep_clutter_fusion_holding_v0",
        "contact_sweep_clutter_fusion_holding_v1",
    ):
        _execute_definitions(MODULE_ROOT / package / "ppo_cfg.py", namespace)
    path = PROJECT_ROOT / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py"
    spec = importlib.util.spec_from_file_location("_holding_v1_model_compat", path)
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    return tuple(compatibility.handle_deprecated_rsl_rl_cfg(namespace[name](), "5.0.1")
                 for name in ("FusionHoldingV0PPORunnerCfg", "FusionHoldingV1PPORunnerCfg"))


def _sixteen_slot_observations(count=8):
    obs = _observations(count)
    generator = torch.Generator().manual_seed(23)
    rows = torch.randn(count, 16, 4, generator=generator)
    rows[..., 0] = 1.0
    obs["obstacles"] = rows.flatten(1)
    return obs


def test_holding_v1_runner_isolates_encoder_capacity_from_v0(holding_runner_configs):
    old, new = holding_runner_configs
    assert new.experiment_name == "inspire_contact_sweep_clutter_fusion_holding_v1"
    assert new.algorithm.to_dict() == old.algorithm.to_dict()
    assert new.obs_groups == old.obs_groups
    for before, after in ((old.actor, new.actor), (old.critic, new.critic)):
        expected = before.to_dict()
        expected["obstacle_observation_slots"] = 16
        assert after.to_dict() == expected
        assert before.obstacle_observation_slots == 10
        assert after.encoder_hidden_dims == [32] and after.embedding_dim == 16
    new.actor.encoder_hidden_dims[0] = 7
    assert new.critic.encoder_hidden_dims == old.actor.encoder_hidden_dims == [32]
    assert type(new)().actor.encoder_hidden_dims == [32]


def test_sixteenth_slot_reaches_both_models_and_encoder_has_exact_dimensions(
    fusion_models, holding_runner_configs,
):
    torch.manual_seed(7)
    obs = _sixteen_slot_observations()
    obs["obstacles"].requires_grad_()
    alg = _algorithm(obs, holding_runner_configs[1])
    for model, width in ((alg.actor, 71), (alg.critic, 74)):
        layers = [layer for layer in model.obstacle_encoder.mlp if isinstance(layer, torch.nn.Linear)]
        assert [(layer.in_features, layer.out_features) for layer in layers] == [(64, 32), (32, 16)]
        assert model.get_latent(obs).shape == (8, width)
        gradient = torch.autograd.grad(model(obs).sum(), obs["obstacles"])[0]
        assert gradient[:, -3:].abs().sum() > 0
        assert torch.isfinite(gradient).all()
    with pytest.raises(ValueError, match="num_envs, 64"):
        _algorithm(_observations(), holding_runner_configs[1])
    # v0 still consumes its original 40D obstacles and 95D export input.
    legacy = _algorithm(_observations(), holding_runner_configs[0])
    assert legacy.actor.as_jit().input_size == 95


def test_sixteen_slot_ppo_updates_and_restores_both_encoders(
    fusion_models, holding_runner_configs, tmp_path,
):
    torch.manual_seed(42)
    obs = _sixteen_slot_observations()
    alg = _algorithm(obs, holding_runner_configs[1])
    initial = [copy.deepcopy(model.obstacle_encoder.state_dict()) for model in (alg.actor, alg.critic)]
    with torch.inference_mode():
        for _ in range(alg.storage.num_transitions_per_env):
            actions = alg.act(obs)
            alg.process_env_step(obs, -(actions - .25).square().sum(-1), torch.zeros(8, dtype=torch.long), {})
        alg.compute_returns(obs)
    assert all(np.isfinite(value) for value in alg.update().values())
    for model, previous in zip((alg.actor, alg.critic), initial):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.obstacle_encoder.parameters())
        assert any(not torch.equal(previous[name], value) for name, value in model.obstacle_encoder.state_dict().items())
    checkpoint = tmp_path / "holding_v1.pt"
    torch.save(alg.save(), checkpoint)
    restored = _algorithm(obs, holding_runner_configs[1])
    assert restored.load(torch.load(checkpoint, weights_only=True), None, strict=True)
    torch.testing.assert_close(restored.actor(obs), alg.actor(obs), rtol=0, atol=0)
    torch.testing.assert_close(restored.critic(obs), alg.critic(obs), rtol=0, atol=0)


def test_sixteen_slot_export_accepts_119d_and_masks_absent_nan_coordinates(
    fusion_models, holding_runner_configs, tmp_path,
):
    import onnx
    from onnx.reference import ReferenceEvaluator

    obs = _sixteen_slot_observations(1)
    obs["obstacles"].unflatten(-1, (16, 4))[:, 10:, 0] = 0.
    alg = _algorithm(obs, holding_runner_configs[1])
    expected = alg.actor(obs).detach()
    obs["obstacles"].unflatten(-1, (16, 4))[:, 10:, 1:] = float("nan")
    torch.testing.assert_close(alg.actor(obs), expected)
    inputs = torch.cat((obs["policy"], obs["obstacles"]), -1)
    assert inputs.shape == (1, 119)
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = alg
    runner.export_policy_to_jit(str(tmp_path))
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(inputs), expected)
    runner.export_policy_to_onnx(str(tmp_path))
    graph = onnx.load(tmp_path / "policy.onnx")
    onnx.checker.check_model(graph)
    actual = ReferenceEvaluator(graph).run(None, {"obs": inputs.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_sixteen_physical_slots_build_contact_filters_and_reject_overflow(scene_cfg, contacts):
    scene = scene_cfg.child()
    scene.configure_population_capacity(8, 8, observation_slots=16)
    assert len(scene.object_collection.rigid_objects) == 17
    assert hasattr(scene, contacts.object_hand_sensor_name("obstacle_15"))
    for parent in scene_cfg.geometry.TACTILE_PARENT_BODY_NAMES:
        sensor = getattr(scene, scene_cfg.geometry.parent_sensor_name(parent))
        assert len(sensor.filter_prim_paths_expr) == 17
        assert sensor.max_contact_data_count_per_prim == 16 * 17
    with pytest.raises(ValueError, match="at most 16"):
        scene.configure_population_capacity(8, 9, observation_slots=16)


def test_full_sixteen_obstacle_reset_with_supplied_layout(population_modules, make_env, monkeypatch):
    """Exercise reset capacity separately from crowded Gray/White placement."""
    env = _population_env(make_env, count=1, max_gray=8, max_white=8)
    env.cfg.obstacle_observation_slots = 16
    env._initial_obstacle_observation = torch.zeros(1, 64)
    points = np.array([(x, y) for x in (-.12, 0., .12) for y in (-.40, -.24, -.08, .08, .24, .40)
                       if x != 0. or abs(y) > .1])
    assert points.shape == (16, 2)
    generator = SimpleNamespace(generate=lambda *_: population_modules.layout.shelf_xy_to_generator_cm(points[None]))
    monkeypatch.setattr(population_modules.events, "population_layout_generator", lambda *_: generator)
    torch.manual_seed(42)
    population_modules.events.randomize_population_clutter_scene(
        env, None, gray_obstacles=8, white_obstacles=8, observation_noise=0.,
        visibility_flip_probability=0., observation_swap_probability=0.,
    )
    assert env._active_object_mask.all()
    assert env.scene["object_collection"].disabled_rows.count_nonzero() == 0
    torch.testing.assert_close(env.scene["object_collection"].data.object_pos_w[0, 1:, :2],
                               torch.as_tensor(points, dtype=torch.float32) + torch.tensor([-.7, 0.]))
    assert env._initial_obstacle_observation.shape == (1, 64)
    assert torch.isfinite(env._initial_obstacle_observation).all()


@pytest.mark.parametrize("capacity,slots", [(5, 10), (5, 16), (8, 16)])
def test_reset_supports_sixteen_slots_and_preserves_unreset_rows(
    population_modules, make_env, capacity, slots,
):
    env = _population_env(make_env, count=3, max_gray=capacity, max_white=capacity)
    env.cfg.obstacle_observation_slots = slots
    env._initial_obstacle_observation = torch.zeros(3, 4 * slots)
    reset = population_modules.events.randomize_population_clutter_scene
    reset(env, None, observation_noise=0., visibility_flip_probability=0., observation_swap_probability=0.)
    snapshot = env._initial_obstacle_observation.clone()
    states = env.scene["object_collection"].data.object_link_state_w.clone()
    reset(env, [1], gray_obstacles=2, white_obstacles=3,
          observation_noise=0., visibility_flip_probability=0., observation_swap_probability=0.)
    assert env._active_object_mask[1].sum().item() == 6
    assert env._initial_obstacle_observation.shape == (3, 4 * slots)
    assert torch.isfinite(env._initial_obstacle_observation).all()
    assert torch.equal(env._initial_obstacle_observation[[0, 2]], snapshot[[0, 2]])
    assert torch.equal(env.scene["object_collection"].data.object_link_state_w[[0, 2]], states[[0, 2]])
    assert env._initial_obstacle_observation.unflatten(-1, (slots, 4))[:, 5:].count_nonzero() == 0
