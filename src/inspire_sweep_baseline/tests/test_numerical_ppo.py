"""Exercise the actual RSL-RL rollout/update seam, including finite outliers."""
import copy
import importlib

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from conftest import PACKAGE_ROOT


@pytest.fixture
def safety(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    return importlib.import_module("inspire_sweep_baseline.agent.numerical_ppo")


def make_algorithm(cls=PPO, **kwargs):
    torch.manual_seed(42)
    obs = TensorDict({"policy": torch.ones(4, 3)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(obs, groups, "actor", 2, hidden_dims=[8], distribution_cfg={
        "class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar",
    })
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[8])
    alg = cls(actor, critic, RolloutStorage("rl", 4, 2, obs, [2]),
              num_learning_epochs=2, num_mini_batches=1, **kwargs)
    return alg, obs


def rollout(alg, obs):
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.long), {})
        alg.compute_returns(obs)


def test_finite_outlier_reproduces_upstream_std_nan():
    alg, obs = make_algorithm()
    with torch.inference_mode():
        for i in range(2):
            sample = obs.clone()
            if i == 1:
                sample["policy"][0] = 1e30
            alg.act(sample)
            alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.long), {})
        alg.compute_returns(obs)
    alg.update()
    with pytest.raises(RuntimeError, match="std >= 0.0"):
        alg.act(obs)


def test_guard_captures_finite_outlier_before_model_is_damaged(safety, tmp_path):
    alg, obs = make_algorithm(safety.NumericalPPO)
    alg.diagnostic_dir = tmp_path
    initial = copy.deepcopy(alg.save())
    obs["policy"][0, 1] = 1e30
    with pytest.raises(safety.NumericalTrainingError, match="observation/policy"):
        alg.act(obs)
    dump = torch.load(tmp_path / "numerical_failure.pt", weights_only=False)
    assert dump["reason"].startswith("observation/policy")
    assert dump["observations"]["policy"][0, 1] == 1e30
    for key, value in initial["actor_state_dict"].items():
        torch.testing.assert_close(alg.actor.state_dict()[key], value, rtol=0, atol=0)


def test_healthy_rollout_update_matches_upstream_exactly(safety):
    normal, obs = make_algorithm()
    guarded, _ = make_algorithm(safety.NumericalPPO)
    for alg in (normal, guarded):
        torch.manual_seed(91)
        rollout(alg, obs)
        alg.update()
    for kind in ("actor_state_dict", "critic_state_dict"):
        for name, value in normal.save()[kind].items():
            torch.testing.assert_close(guarded.save()[kind][name], value, rtol=0, atol=0)


def test_resume_restores_adaptive_lr_variable_together_with_adam(safety):
    source, _ = make_algorithm()
    source.optimizer.param_groups[0]["lr"] = 1e-5
    restored, _ = make_algorithm(safety.NumericalPPO)
    restored.load(source.save(), None, True)
    assert restored.learning_rate == restored.optimizer.param_groups[0]["lr"] == 1e-5


def test_weights_only_load_keeps_current_lr(safety):
    source, _ = make_algorithm()
    source.optimizer.param_groups[0]["lr"] = 1e-5
    restored, _ = make_algorithm(safety.NumericalPPO)
    restored.load(source.save(), {"actor": True, "critic": True}, True)
    assert restored.learning_rate == 1e-3


def test_optimizer_cannot_commit_nonfinite_gradient(safety, tmp_path):
    alg, obs = make_algorithm(safety.NumericalPPO)
    alg.diagnostic_dir = tmp_path
    rollout(alg, obs)
    parameter = next(alg.actor.parameters())
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    before = copy.deepcopy(alg.save())
    with pytest.raises(safety.NumericalTrainingError, match="gradient"):
        alg.update()
    handle.remove()
    for name, value in before["actor_state_dict"].items():
        torch.testing.assert_close(alg.actor.state_dict()[name], value, rtol=0, atol=0)
    assert (tmp_path / "numerical_failure.pt").exists()


def test_scalar_std_projection_preserves_checkpoint_key(safety):
    alg, _ = make_algorithm(safety.NumericalPPO)
    std = alg.actor.distribution.std_param
    std.grad = torch.ones_like(std) * 100
    with torch.no_grad():
        std.fill_(1e-5)
    alg.optimizer.step()
    assert (std >= alg.min_std).all()
    assert "distribution.std_param" in alg.save()["actor_state_dict"]


def test_real_curriculum_runner_trains_loads_and_exports(safety, tmp_path):
    module = importlib.import_module("inspire_sweep_baseline.agent.curriculum_runner")
    _, obs = make_algorithm()

    class Env:
        num_envs, num_actions, max_episode_length = 4, 2, 10
        device = "cpu"
        cfg = {}
        learning_iteration = 0
        episode_length_buf = torch.zeros(4, dtype=torch.long)

        @property
        def unwrapped(self):
            return self

        def get_observations(self):
            return obs

        def set_learning_iteration(self, iteration):
            self.learning_iteration = iteration

        def step(self, actions):
            return obs, -actions.square().sum(-1), torch.zeros(4, dtype=torch.long), {}

    config = {
        "num_steps_per_env": 2, "save_interval": 50,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [8], "distribution_cfg": {
            "class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [8]},
        "algorithm": {"class_name": "PPO", "rnd_cfg": None, "num_learning_epochs": 1, "num_mini_batches": 1},
    }
    runner = module.CurriculumOnPolicyRunner(Env(), config)
    assert isinstance(runner.alg, safety.NumericalPPO)
    assert config["algorithm"]["class_name"] == "PPO"
    runner.learn(2)
    runner.alg.optimizer.param_groups[0]["lr"] = 1e-5
    runner.save(str(tmp_path / "model.pt"))
    restored = module.CurriculumOnPolicyRunner(Env(), config)
    restored.load(str(tmp_path / "model.pt"))
    assert restored.alg.learning_rate == 1e-5
    restored.export_policy_to_jit(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(obs["policy"]), restored.alg.actor(obs))
