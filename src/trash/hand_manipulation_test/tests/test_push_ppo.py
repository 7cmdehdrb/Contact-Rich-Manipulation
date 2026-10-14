"""Prevent drift from Sweep-Policy cfg_02 while adapting the installed RSL model API."""

import ast
import __future__
from dataclasses import MISSING
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Literal
from unittest.mock import patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = REPOSITORY_ROOT / "example/Sweep-Policy/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py"
PUSH_PATH = PACKAGE_ROOT / "agents/rsl_rl_push_ppo_cfg.py"

def _class_assignments(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {node.targets[0].id: node.value for node in cls.body if isinstance(node, ast.Assign)}


def test_push_ppo_matches_sweep_reference_through_rsl5_model_api():
    reference = _class_assignments(
        REPOSITORY_ROOT / "example/Sweep-Policy/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py",
        "UR5eSweepPPORunnerCfg",
    )
    push = _class_assignments(PACKAGE_ROOT / "agents/rsl_rl_push_ppo_cfg.py", "PushPPORunnerCfg")
    for key in ("num_steps_per_env", "max_iterations", "save_interval", "run_name"):
        assert ast.literal_eval(push[key]) == ast.literal_eval(reference[key])
    old_policy = {keyword.arg: ast.literal_eval(keyword.value) for keyword in reference["policy"].keywords}
    actor = {keyword.arg: keyword.value for keyword in push["actor"].keywords}
    critic = {keyword.arg: keyword.value for keyword in push["critic"].keywords}
    assert ast.literal_eval(actor["hidden_dims"]) == old_policy["actor_hidden_dims"]
    assert ast.literal_eval(critic["hidden_dims"]) == old_policy["critic_hidden_dims"]
    assert ast.literal_eval(actor["activation"]) == ast.literal_eval(critic["activation"]) == old_policy["activation"]
    assert ast.literal_eval(actor["obs_normalization"]) == old_policy["actor_obs_normalization"]
    assert ast.literal_eval(critic["obs_normalization"]) == old_policy["critic_obs_normalization"]
    distribution = actor["distribution_cfg"]
    assert ast.unparse(distribution.func) == "RslRlMLPModelCfg.GaussianDistributionCfg"
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in distribution.keywords} == {
        "init_std": old_policy["init_noise_std"]
    }
    algorithm = lambda node: {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
    assert algorithm(push["algorithm"]) == algorithm(reference["algorithm"])
    assert ast.literal_eval(push["obs_groups"]) == {"actor": ["policy"], "critic": ["policy"]}
    assert ast.literal_eval(push["clip_actions"]) is None


def test_real_configclass_runner_preserves_reference_defaults_and_gaussian_semantics():
    """Construct both schemas with installed Isaac configclasses, without Kit."""
    utils_root = REPOSITORY_ROOT / "IsaacLab/source/isaaclab/isaaclab/utils"
    rl_path = REPOSITORY_ROOT / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py"
    if not (utils_root / "configclass.py").is_file() or not rl_path.is_file():
        pytest.skip("Isaac Lab configuration source is unavailable")
    package = ModuleType("_push_ppo_configuration_fixture")
    package.__path__ = []
    serialization = ModuleType(package.__name__ + ".dict")

    def unexpected_serialization(*args, **kwargs):
        raise AssertionError("Serialization is not needed for schema comparison")

    serialization.class_to_dict = serialization.update_class_from_dict = unexpected_serialization
    spec = importlib.util.spec_from_file_location(package.__name__ + ".configclass", utils_root / "configclass.py")
    decorator = importlib.util.module_from_spec(spec)
    namespace = ModuleType(package.__name__ + ".schemas")
    with patch.dict(sys.modules, {package.__name__: package, serialization.__name__: serialization,
                                  decorator.__name__: decorator, namespace.__name__: namespace}):
        spec.loader.exec_module(decorator)
        namespace.__dict__.update(configclass=decorator.configclass, MISSING=MISSING, Literal=Literal,
                                  RslRlRndCfg=type("RslRlRndCfg", (), {}),
                                  RslRlSymmetryCfg=type("RslRlSymmetryCfg", (), {}))
        classes = {node.name: node for node in ast.parse(rl_path.read_text()).body if isinstance(node, ast.ClassDef)}
        for name in ("RslRlMLPModelCfg", "RslRlPpoAlgorithmCfg", "RslRlBaseRunnerCfg",
                     "RslRlOnPolicyRunnerCfg", "RslRlPpoActorCriticCfg"):
            tree = ast.Module(body=[classes[name]], type_ignores=[])
            exec(compile(tree, str(rl_path), "exec", flags=__future__.annotations.compiler_flag), namespace.__dict__)
        for path, name in ((REFERENCE_PATH, "UR5eSweepPPORunnerCfg"), (PUSH_PATH, "PushPPORunnerCfg"),
                           (PACKAGE_ROOT / "agents/rsl_rl_push_v1_ppo_cfg.py", "PushV1PPORunnerCfg")):
            cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name)
            exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace.__dict__)
        reference, push = namespace.UR5eSweepPPORunnerCfg(), namespace.PushPPORunnerCfg()
        assert vars(push.algorithm) == vars(reference.algorithm)
        for name in ("seed", "device", "logger", "num_steps_per_env", "max_iterations", "save_interval", "run_name", "resume"):
            assert getattr(push, name) == getattr(reference, name), name
        assert push.actor.hidden_dims == reference.policy.actor_hidden_dims
        assert push.critic.hidden_dims == reference.policy.critic_hidden_dims
        assert push.actor.activation == push.critic.activation == reference.policy.activation
        assert push.actor.obs_normalization == reference.policy.actor_obs_normalization
        assert push.critic.obs_normalization == reference.policy.critic_obs_normalization
        distribution = push.actor.distribution_cfg
        assert distribution.class_name == "GaussianDistribution"
        assert distribution.init_std == reference.policy.init_noise_std == 1.0
        assert distribution.std_type == reference.policy.noise_std_type == "scalar"
        assert not reference.policy.state_dependent_std
        assert push.critic.distribution_cfg is None
        assert push.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
        assert push.clip_actions is reference.clip_actions is None
        v1 = namespace.PushV1PPORunnerCfg()
        assert v1.experiment_name == "hand_manipulation_push_v1"
        assert push.experiment_name == "hand_manipulation_push"
        # Compare the actual instantiated configuration, including inherited
        # defaults and nested models, rather than just its one source override.
        def canonical(value):
            # configclass deep-copies deprecated MISSING placeholders into
            # distinct objects; compare their meaning instead of identity.
            if isinstance(value, type(MISSING)):
                return "<dataclass missing>"
            if isinstance(value, dict):
                return {name: canonical(item) for name, item in value.items()}
            if isinstance(value, (list, tuple)):
                return tuple(canonical(item) for item in value)
            if hasattr(value, "__dataclass_fields__"):
                return (type(value).__name__, canonical(vars(value)))
            return value

        inherited = canonical({name: value for name, value in vars(v1).items() if name != "experiment_name"})
        parent = canonical({name: value for name, value in vars(push).items() if name != "experiment_name"})
        assert inherited == parent
