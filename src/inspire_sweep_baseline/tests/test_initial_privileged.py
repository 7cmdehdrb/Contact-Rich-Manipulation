"""Actor reset snapshot, live Scene Critic state and actual RSL-RL routing."""

from _source_units import source_text

import ast
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT


def _snapshot_command(env):
    command = SimpleNamespace(target_init_state_w=env.scene["object_collection"].data.object_link_state_w[:, 0].clone())
    env.command_manager.get_term = lambda name: command
    return command


def test_actor_initial_position_stays_fixed_and_critic_and_rewards_remain_live(load_mdp, make_env):
    observations = load_mdp("asymmetric_observations")
    rewards = load_mdp("rewards")
    env = make_env(3)
    _snapshot_command(env)
    term = observations.InitialObjectPositionInRoot(SimpleNamespace(), env)
    before = term(env)
    live_before = observations.live_object_state_in_root(env).clone()
    approach = env.scene["object_collection"].data.object_pos_w[:, 0] + torch.tensor([-0.02, -0.04, 0.09])
    env.scene["ee_frame"].data.target_pos_w[:, 0] = approach
    reaching_before = rewards.reaching_target(env)
    # Moving the target changes privileged data and Scene reward, not Actor input.
    env.scene["object_collection"].data.object_pos_w[0, 0, 1] += 0.06
    env.scene["object_collection"].data.object_link_state_w[0, 0, 7:13] = torch.tensor([1., 2., 3., 4., 5., 6.])
    assert torch.equal(term(env), before)
    assert observations.live_object_state_in_root(env).shape == (3, 13)
    live_after = observations.live_object_state_in_root(env)
    assert live_after[0, 1].item() == pytest.approx(live_before[0, 1].item() + 0.06)
    assert torch.equal(live_after[0, 7:], torch.tensor([1., 2., 3., 4., 5., 6.]))
    assert rewards.reaching_target(env)[0] < reaching_before[0]
    # A manager/noise modifier must not corrupt the term's episode latch.
    before.fill_(123.0)
    assert not torch.equal(term(env), before)


def test_partial_reset_captures_new_command_after_reset_and_only_changes_selected_rows(load_mdp, make_env):
    observations = load_mdp("asymmetric_observations")
    env = make_env(3)
    command = _snapshot_command(env)
    term = observations.InitialObjectPositionInRoot(SimpleNamespace(), env)
    original = term(env)
    term.reset(torch.tensor([1]))  # ObservationManager resets first.
    command.target_init_state_w[1, :3] = torch.tensor([0.2, 0.3, 1.05])
    # This other row's new command must not be captured without its own reset.
    command.target_init_state_w[0, :3] += 1.0
    updated = term(env)
    assert torch.equal(updated[[0, 2]], original[[0, 2]])
    assert torch.equal(updated[1], torch.tensor([0.2, 0.3, 1.05]))
    command.target_init_state_w[1, 0] = 0.9
    assert torch.equal(term(env), updated)
    term.reset()
    assert torch.equal(term(env)[1], command.target_init_state_w[1, :3])


def test_root_coordinate_frame_and_velocity_rotation_are_shared(load_mdp, make_env):
    observations = load_mdp("asymmetric_observations")
    env = make_env(1)
    robot = env.scene["robot"].data
    robot.root_pos_w[:] = torch.tensor([10., 20., 0.4])
    robot.root_quat_w[:] = torch.tensor([0., 0., 0., 1.])  # 180 degrees about Z.
    state = env.scene["object_collection"].data.object_link_state_w
    state[0, 0] = torch.tensor([9., 22., 1.4, 0., 0., 0., 1., 1., 2., 3., -4., 5., -6.])
    _snapshot_command(env)
    term = observations.InitialObjectPositionInRoot(SimpleNamespace(), env)
    expected_position = torch.tensor([[1., -2., 1.]])
    assert torch.allclose(term(env), expected_position)
    privileged = observations.live_object_state_in_root(env)
    assert torch.allclose(privileged[:, :3], expected_position)
    assert torch.allclose(privileged[:, 3:7], torch.tensor([[1., 0., 0., 0.]]))
    assert torch.allclose(privileged[:, 7:], torch.tensor([[-1., -2., 3., 4., -5., -6.]]))
    # The initial root-coordinate measurement is latched too.
    robot.root_pos_w[:, 0] += 1.0
    assert torch.allclose(term(env), expected_position)


def test_initial_measurement_noise_is_fixed_until_selected_episode_resets():
    # Run Isaac Lab's actual noise implementation, without starting the simulator.
    path = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py"
    spec = importlib.util.spec_from_file_location("_initial_privileged_noise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(
        noise_cfg=SimpleNamespace(func=module.constant_noise, bias=0.0, operation="add"),
        bias_noise_cfg=SimpleNamespace(func=module.uniform_noise, n_min=-0.01, n_max=0.01, operation="abs"),
        sample_bias_per_component=True,
    )
    noise = module.NoiseModelWithAdditiveBias(cfg, 3, "cpu")
    data = torch.ones(3, 3)
    first = noise(data)
    assert torch.equal(noise(data), first)
    assert torch.all((first - data).abs() <= 0.01)
    noise.reset(torch.tensor([1]))
    second = noise(data)
    assert torch.equal(second[[0, 2]], first[[0, 2]])
    assert not torch.equal(second[1], first[1])


def test_actual_rsl_rl_models_use_41d_actor_and_54d_critic_without_privileged_actor_leak():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    tree = ast.parse(source_text(MODULE_ROOT / "agents/initial_privileged_ppo_cfg.py"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    routing_node = next(node for node in runner.body if isinstance(node, ast.Assign) and node.targets[0].id == "obs_groups")
    routing = ast.literal_eval(routing_node.value)
    assert routing == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    policy = torch.randn(2, 41, requires_grad=True)
    privileged = torch.randn(2, 13, requires_grad=True)
    obs = TensorDict({"policy": policy, "privileged": privileged}, batch_size=[2])
    actor = MLPModel(obs, routing, "actor", 8, hidden_dims=[256, 128, 64], activation="elu")
    critic = MLPModel(obs, routing, "critic", 1, hidden_dims=[256, 128, 64], activation="elu")
    assert actor.obs_dim == 41
    assert critic.obs_dim == 54
    action = actor(obs)
    value = critic(obs)
    assert action.shape == (2, 8) and value.shape == (2, 1)
    actor_grad = torch.autograd.grad(action.sum(), (policy, privileged), allow_unused=True)
    critic_grad = torch.autograd.grad(value.sum(), (policy, privileged), allow_unused=True)
    assert actor_grad[0] is not None and actor_grad[1] is None
    assert critic_grad[0] is not None and critic_grad[1] is not None
    changed = TensorDict({"policy": policy, "privileged": privileged + 10.0}, batch_size=[2])
    assert torch.equal(actor(changed), action)
    assert not torch.equal(critic(changed), value)


def test_new_environment_inherits_realistic_physics_and_only_changes_observations():
    tree = ast.parse(source_text(MODULE_ROOT / "initial_privileged_env_cfg.py"))
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InspireSweepRealisticInitialPrivilegedEnvCfg")
    assert [node.id for node in cfg.bases] == ["InspireSweepRealisticPhysicsEnvCfg"]
    assert [node.target.id for node in cfg.body if isinstance(node, ast.AnnAssign)] == ["observations"]
    post = next(node for node in cfg.body if isinstance(node, ast.FunctionDef))
    attribute_writes = [node.targets[0] for node in post.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)]
    assert [node.attr for node in attribute_writes] == ["func", "params", "noise"]
    runner = ast.parse(source_text(MODULE_ROOT / "agents/initial_privileged_ppo_cfg.py"))
    runner = next(node for node in runner.body if isinstance(node, ast.ClassDef))
    assert runner.bases[0].id == "InspireSweepRealisticPhysicsPPORunnerCfg"
    assert [node.targets[0].id for node in runner.body if isinstance(node, ast.Assign)] == ["experiment_name", "obs_groups"]
