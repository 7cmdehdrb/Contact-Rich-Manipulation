"""CPU checks for Revision's observation contract and trainable asymmetry."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from conftest import MODULE_ROOT


@pytest.fixture
def revision_observations(load_mdp):
    return importlib.import_module("_baseline_cpu_pkg.mdp.observation_functions.revision")


@pytest.fixture
def revision_models():
    spec = importlib.util.spec_from_file_location("_revision_models_cpu", MODULE_ROOT / "agent/revision_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observations(count=8):
    generator = torch.Generator().manual_seed(17)
    initial = torch.randn(count, 16, 4, generator=generator)
    initial[..., 0] = 0.0
    initial[:, :4, 0] = 1.0
    live = initial.clone()
    live[:, :5, 0] = 1.0
    live[..., 1:] += 0.1
    return TensorDict({
        "policy": torch.randn(count, 57, generator=generator),
        "privileged": torch.randn(count, 3, generator=generator),
        "obstacles": initial.flatten(1),
        "critic_obstacles": live.flatten(1),
    }, batch_size=[count])


ROUTING = {"actor": ["policy", "obstacles"], "critic": ["policy", "privileged", "critic_obstacles"]}


def _models(module, obs, distribution=False):
    actor = module.RevisionModel(
        obs, ROUTING, "actor", 8,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0} if distribution else None,
    )
    critic = module.RevisionModel(obs, ROUTING, "critic", 1)
    return actor, critic


def test_actual_arm_measurements_do_not_subtract_default(revision_observations, make_env):
    env = make_env(2)
    robot = env.scene["robot"].data
    robot.joint_pos[:] = 1.25
    robot.default_joint_pos[:] = 0.75
    robot.joint_vel[:] = -0.3
    robot.default_joint_vel[:] = 0.2
    cfg = SimpleNamespace(name="robot", joint_ids=[0, 2, 4, 6, 8, 10])
    assert torch.equal(revision_observations.arm_joint_positions(env, cfg), torch.full((2, 6), 1.25))
    assert torch.equal(revision_observations.arm_joint_velocities(env, cfg), torch.full((2, 6), -0.3))


def test_actor_visibility_flip_never_activates_unused_slots(revision_observations):
    positions = torch.tensor([[[1., 3., 1.], [1., 2., 1.], [float("nan"), 9., 1.]]])
    active = torch.tensor([[True, True, False]])
    visible = torch.tensor([[True, False, False]])
    packed = revision_observations.make_obstacle_observation(
        positions, active, visible=visible, visibility_flip_probability=1.0,
        adjacent_swap_probability=0.0,
    ).reshape(1, 16, 4)
    assert torch.equal(packed[0, 0], torch.tensor([1., 1., 2., 1.]))
    assert torch.count_nonzero(packed[0, 1:]) == 0
    assert torch.isfinite(packed).all()
    env = SimpleNamespace(_initial_obstacle_observation=packed.flatten(1))
    first = revision_observations.initial_obstacle_positions(env)
    first.fill_(9.)
    assert torch.equal(env._initial_obstacle_observation[0, :4], torch.tensor([1., 1., 2., 1.]))


def test_live_critic_excludes_target_and_inactive_candidates_and_tracks_motion(revision_observations, make_env):
    env = make_env(2)
    positions = torch.tensor([
        [[3., 0., 2.], [2., 4., 2.], [4., 3., 2.], [2., 9., 2.]],
        [[2., 1., 2.], [2., 2., 2.], [2., 3., 2.], [2., 4., 2.]],
    ])
    env.scene["object_collection"].data.object_pos_w = positions
    env.scene["robot"].data.root_pos_w[:] = torch.tensor([1., 0., 1.])
    env.target_id = torch.tensor([[0], [2]])
    env._active_object_mask = torch.tensor([[True, True, True, False], [False, True, True, True]])
    # Actor cache remains initial and can hide a real obstacle.
    env._initial_obstacle_observation = torch.zeros(2, 64)
    env._initial_obstacle_observation[:, :4] = torch.tensor([1., 1., 4., 1.])
    actor_before = revision_observations.initial_obstacle_positions(env).clone()
    first = revision_observations.current_obstacle_positions(env, adjacent_swap_probability=0.0).reshape(2, 16, 4)
    assert torch.equal(first[0, :2], torch.tensor([[1., 1., 4., 1.], [1., 3., 3., 1.]]))
    assert torch.equal(first[1, :2], torch.tensor([[1., 1., 4., 1.], [1., 1., 2., 1.]]))
    assert torch.count_nonzero(first[:, 2:]) == 0
    # A real obstacle leaving the shelf is still an actual obstacle until reset.
    positions[0, 1] = torch.tensor([12., -20., -5.])
    moved = revision_observations.current_obstacle_positions(env, adjacent_swap_probability=0.0).reshape(2, 16, 4)
    assert torch.equal(moved[0, 1], torch.tensor([1., 11., -20., -6.]))
    assert torch.equal(actor_before, revision_observations.initial_obstacle_positions(env))
    assert torch.equal(first[1], moved[1])
    # Partial role change: the new target is removed, the old target is an obstacle.
    env.target_id[0, 0] = 2
    swapped = revision_observations.current_obstacle_positions(env, adjacent_swap_probability=0.0).reshape(2, 16, 4)
    assert torch.equal(swapped[0, 0], torch.tensor([1., 2., 0., 1.]))
    assert torch.equal(swapped[1], first[1])


def test_root_frame_rotation_and_noise_free_critic(revision_observations, make_env):
    env = make_env(1)
    env.scene["object_collection"].data.object_pos_w = torch.tensor([[[0., 0., 0.], [0., 1., 2.]]])
    env.scene["robot"].data.root_quat_w[:] = torch.tensor([2**-0.5, 0., 0., 2**-0.5])
    env._active_object_mask = torch.tensor([[True, True]])
    value = revision_observations.current_obstacle_positions(env, adjacent_swap_probability=0.0)
    torch.testing.assert_close(value[0, :4], torch.tensor([1., 1., 0., 2.]), atol=1e-6, rtol=0.)
    assert torch.equal(value, revision_observations.current_obstacle_positions(env, adjacent_swap_probability=0.0))


def test_sort_swaps_only_valid_neighbors_and_capacity_is_not_silently_lost(revision_observations):
    positions = torch.tensor([[[0., 1., 0.], [0., 3., 0.], [0., 2., 0.]]])
    active = torch.ones((1, 3), dtype=torch.bool)
    result = revision_observations.make_obstacle_observation(
        positions, active, adjacent_swap_probability=1.0,
    ).reshape(1, 16, 4)
    assert result[0, :3, 2].tolist() == [2., 1., 3.]
    assert torch.count_nonzero(result[0, 3:]) == 0
    with pytest.raises(ValueError, match="exceed"):
        revision_observations.make_obstacle_observation(torch.zeros(1, 17, 3), torch.ones(1, 17, dtype=torch.bool))


def test_config_declares_requested_order_without_old_cfg_inheritance():
    tree = ast.parse((MODULE_ROOT / "mdp/observations/revision.py").read_text(encoding="utf-8"))
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert cfg.bases == []
    policy = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
    terms = [node.targets[0].id for node in policy.body if isinstance(node, ast.Assign)]
    assert terms == [
        "joint_pos", "joint_vel", "hand_configuration", "ee_pose", "target_obs_state",
        "goal_pos", "actions", "tactile", "wrist_wrench",
    ]
    dimensions = [6, 6, 2, 7, 3, 2, 8, 17, 6]
    assert sum(dimensions) == 57
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            assert all(isinstance(base, ast.Name) and base.id == "ObsGroup" for base in node.bases)


def test_models_have_73_76_inputs_and_no_actor_privileged_leak(revision_models):
    torch.manual_seed(4)
    obs = _observations()
    for value in obs.values():
        value.requires_grad_()
    actor, critic = _models(revision_models, obs)
    assert actor.obs_dim == 57 and critic.obs_dim == 60
    assert actor.get_latent(obs).shape == (8, 73)
    assert critic.get_latent(obs).shape == (8, 76)
    assert actor.obstacle_encoder is not critic.obstacle_encoder
    actions, values = actor(obs), critic(obs)
    actor_grad = torch.autograd.grad(actions.sum(), tuple(obs.values()), allow_unused=True)
    critic_grad = torch.autograd.grad(values.sum(), tuple(obs.values()), allow_unused=True)
    # TensorDict insertion order: policy, privileged, initial, live.
    assert actor_grad[0] is not None and actor_grad[1] is None
    assert actor_grad[2] is not None and actor_grad[3] is None
    assert critic_grad[0] is not None and critic_grad[1] is not None
    assert critic_grad[2] is None and critic_grad[3] is not None
    changed = obs.clone()
    changed["privileged"] = torch.full((8, 3), float("nan"))
    changed["critic_obstacles"] = torch.full((8, 64), float("nan"))
    assert torch.equal(actor(changed), actions)
    assert actor.mlp[0].in_features == 73 and critic.mlp[0].in_features == 76


def test_actor_export_121d_preserves_padding_and_has_no_privileged_input(revision_models):
    obs = _observations(1)
    actor, _ = _models(revision_models, obs)
    expected = actor(obs)
    exported = actor.as_jit()
    assert exported.get_dummy_inputs()[0].shape == (1, 121)
    inputs = torch.cat((obs["policy"], obs["obstacles"]), -1)
    inputs[:, 57:].unflatten(-1, (16, 4))[:, 4:, 1:] = float("nan")
    scripted = torch.jit.script(exported)
    torch.testing.assert_close(scripted(inputs), expected)
    scripted.reset()


def test_ppo_updates_separate_encoders_and_restores_checkpoint(revision_models, tmp_path):
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage

    torch.manual_seed(42)
    obs = _observations()
    actor, critic = _models(revision_models, obs, distribution=True)
    storage = RolloutStorage("rl", 8, 4, obs, [8], "cpu")
    alg = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=2, device="cpu")
    before = [copy.deepcopy(model.obstacle_encoder.state_dict()) for model in (actor, critic)]
    with torch.inference_mode():
        for _ in range(4):
            actions = alg.act(obs)
            alg.process_env_step(obs, -(actions - 0.25).square().sum(-1), torch.zeros(8, dtype=torch.long), {})
        alg.compute_returns(obs)
    assert all(torch.isfinite(torch.tensor(value)) for value in alg.update().values())
    for model, initial in zip((actor, critic), before):
        params = list(model.obstacle_encoder.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(p.grad.abs().sum() > 0 for p in params)
        assert any(not torch.equal(initial[name], value) for name, value in model.obstacle_encoder.state_dict().items())
    checkpoint = tmp_path / "revision_model.pt"
    torch.save(alg.save(), checkpoint)
    new_actor, new_critic = _models(revision_models, obs, distribution=True)
    restored = PPO(new_actor, new_critic, RolloutStorage("rl", 8, 4, obs, [8]), device="cpu")
    assert restored.load(torch.load(checkpoint, weights_only=True), None, strict=True)
    torch.testing.assert_close(new_actor(obs), actor(obs), atol=0., rtol=0.)
    torch.testing.assert_close(new_critic(obs), critic(obs), atol=0., rtol=0.)


@pytest.mark.parametrize("failure", ["wrong_policy", "old_obstacle_slots", "shared_critic_obstacles"])
def test_model_rejects_incompatible_contracts(revision_models, failure):
    obs = _observations()
    routing = copy.deepcopy(ROUTING)
    if failure == "wrong_policy":
        obs["policy"] = torch.zeros(8, 55)
    elif failure == "old_obstacle_slots":
        obs["critic_obstacles"] = torch.zeros(8, 80)
    else:
        routing["critic"][-1] = "obstacles"
    with pytest.raises(ValueError):
        revision_models.RevisionModel(obs, routing, "critic", 1)
