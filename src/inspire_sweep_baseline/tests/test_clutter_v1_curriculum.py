"""Schedule boundaries and actual RSL-RL learn/save/load on a CPU toy VecEnv."""

import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PACKAGE_ROOT
from test_clutter_v1_models import runner_configs, v1_models, _privileged_observations
from test_omni_configuration import _execute_definitions


@pytest.fixture
def curriculum(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    mdp = importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter_v1_curriculum.mdp")
    runner = importlib.import_module("inspire_sweep_baseline.agents.curriculum_runner")
    cfg = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_v1_curriculum/curriculum_cfg.py",
        {"configclass": lambda cls: cls, "CurriculumTermCfg": SimpleNamespace,
         "reward_schedule": mdp.reward_schedule},
    ).ContactSweepClutterV1CurriculumCfg()
    terms = [getattr(cfg, name) for name in vars(type(cfg)) if not name.startswith("_")]
    return SimpleNamespace(mdp=mdp, runner=runner, terms=terms)


class ScheduleEnv:
    def __init__(self, terms):
        self.terms = terms
        self.configs = {
            term.params["term_name"]: SimpleNamespace(
                weight=-1.0, params={"free_speed": 99.0, "max_cost": 100.0}, func=object(),
            ) for term in terms
        }
        self.reward_manager = SimpleNamespace(
            get_term_cfg=lambda name: self.configs[name],
            set_term_cfg=lambda name, cfg: self.configs.__setitem__(name, cfg),
        )
        self.learning_iteration = 0

    def set_learning_iteration(self, iteration):
        self.learning_iteration = iteration
        for term in self.terms:
            term.func(self, None, **term.params)

    def values(self):
        return (
            self.configs["pushing_speed"].params["free_speed"],
            self.configs["arm_speed_excess"].params["free_speed"],
            self.configs["arm_action_rate"].weight,
            self.configs["arm_joint_velocity"].weight,
        )


@pytest.mark.parametrize("iteration,expected", [
    (0, (.1, 1., -.03, -.03)),
    (2999, (.1, 1., -.03, -.03)),
    (3000, (.09, 1., -.035, -.035)),
    (4999, (.09, 1., -.035, -.035)),
    (5000, (.09, .95, -.035, -.035)),
    (6000, (.08, .95, -.04, -.04)),
    (9000, (.07, .95, -.045, -.045)),
    (10000, (.07, .9, -.045, -.045)),
    (12000, (.06, .9, -.05, -.05)),
    (14999, (.06, .9, -.05, -.05)),
    (15000, (.05, .85, -.055, -.055)),
    (17999, (.05, .85, -.055, -.055)),
    (18000, (.05, .85, -.06, -.06)),
    (19999, (.05, .85, -.06, -.06)),
    (20000, (.05, .8, -.06, -.06)),
    (90000, (.05, .8, -.06, -.06)),
])
def test_absolute_schedule_boundaries_and_repeated_resets(curriculum, iteration, expected):
    env = ScheduleEnv(curriculum.terms)
    functions = [cfg.func for cfg in env.configs.values()]
    env.set_learning_iteration(iteration)
    assert env.values() == expected
    for ids in ([0], [1, 2], slice(None)):
        for term in curriculum.terms:
            term.func(env, ids, **term.params)
        assert env.values() == expected
    assert [cfg.func for cfg in env.configs.values()] == functions
    assert all(cfg.params["max_cost"] == 100 for cfg in env.configs.values())
    # Loading an earlier checkpoint recomputes from its index, not current values.
    env.set_learning_iteration(0)
    assert env.values() == (.1, 1., -.03, -.03)


def test_runtime_updates_manager_immediately(curriculum):
    cls = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_v1_curriculum/env.py",
        {"ContactSweepClutterV1Env": object},
    ).ContactSweepClutterV1CurriculumEnv
    env = cls.__new__(cls)
    calls = []
    env.curriculum_manager = SimpleNamespace(
        compute=lambda env_ids: calls.append((env.learning_iteration, env_ids)),
    )
    env.set_learning_iteration(3000)
    assert calls == [(3000, None)]
    for invalid in (-1, 3.5):
        with pytest.raises(ValueError):
            env.set_learning_iteration(invalid)


class ToyVecEnv(ScheduleEnv):
    """Real PPO observations and transitions; no physics or Isaac mocks."""

    def __init__(self, terms):
        super().__init__(terms)
        self.unwrapped = self
        self.num_envs, self.num_actions = 8, 8
        self.device = "cpu"
        self.cfg = {}
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(8, dtype=torch.long)
        self.history = []
        self.obs = _privileged_observations()

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.history.append((self.learning_iteration, self.values()))
        self.episode_length_buf += 1
        return self.obs, -actions.square().sum(-1), torch.zeros(8, dtype=torch.long), {}


@pytest.mark.parametrize("horizon,start", [(2, 2999), (4, 4999)])
def test_actual_runner_iteration_boundaries_resume_and_playback(
    curriculum, v1_models, runner_configs, tmp_path, horizon, start,
):
    cfg = runner_configs[1].to_dict()
    cfg["num_steps_per_env"] = horizon
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    env = ToyVecEnv(curriculum.terms)
    runner = curriculum.runner.CurriculumOnPolicyRunner(env, deepcopy(cfg), device="cpu")
    runner.current_learning_iteration = start
    runner.learn(2, init_at_random_ep_len=True)
    assert [it for it, _ in env.history] == [start] * horizon + [start + 1] * horizon
    assert env.history[0][1] != env.history[horizon][1]
    assert runner.current_learning_iteration == start + 1
    # Buffer assignment must reach the real VecEnv, not shadow it on the adapter.
    assert runner.env.episode_length_buf is env.episode_length_buf
    assert (env.episode_length_buf >= 2 * horizon).all()

    checkpoint = str(tmp_path / "checkpoint.pt")
    runner.save(checkpoint)
    restored_env = ToyVecEnv(curriculum.terms)
    restored = curriculum.runner.CurriculumOnPolicyRunner(restored_env, deepcopy(cfg), device="cpu")
    restored.load(checkpoint)
    assert restored.current_learning_iteration == start + 1
    assert restored_env.values() == env.values()
    for _ in range(horizon * 3):
        restored.env.step(torch.zeros(8, 8))
    assert restored_env.learning_iteration == start + 1  # playback stays fixed
    restored_env.history.clear()
    restored.learn(2)
    assert [it for it, _ in restored_env.history] == [start + 1] * horizon + [start + 2] * horizon


def test_registration_and_launcher_runner_selection():
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    spec = registrations["Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1"]
    assert spec["entry_point"].endswith("curriculum.env:ContactSweepClutterV1CurriculumEnv")
    assert spec["kwargs"]["env_cfg_entry_point"].endswith(":ContactSweepClutterV1CurriculumEnvCfg")
    assert spec["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":ContactSweepClutterV1CurriculumPPORunnerCfg")
    for script in ("train.py", "play.py"):
        assert 'agent_cfg.class_name == "CurriculumOnPolicyRunner"' in (
            PACKAGE_ROOT / "scripts" / script
        ).read_text()
