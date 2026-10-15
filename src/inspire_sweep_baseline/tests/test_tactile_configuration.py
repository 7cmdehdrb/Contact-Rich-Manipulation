"""Execute inherited tactile wiring using the official CPU config semantics."""

import ast
from copy import deepcopy
from dataclasses import MISSING
import importlib
import inspect
import re
import sys
import types
from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_omni_configuration import _AssetNode, _execute_definitions, omni_config


@pytest.fixture
def tactile_config(omni_config, load_mdp, monkeypatch):
    baseline = omni_config.baseline
    configclass = baseline.configclass
    tactile = load_mdp("tactile_observations")
    geometry = importlib.import_module("_baseline_cpu_pkg.assets.tactile_regions")
    cfg_module = types.ModuleType("_tactile_cpu_cfg")
    cfg_module.__dict__.update(
        configclass=configclass, ContactSensorCfg=_AssetNode,
        SweepSceneCfg=baseline.SweepSceneCfg, ObservationsCfg=baseline.ObservationsCfg,
        InspireSweepEnvCfg=baseline.InspireSweepEnvCfg,
        ObsTerm=omni_config.terms.ObservationTermCfg,
        PALM_SENSOR_BODY_NAMES=tactile.PALM_SENSOR_BODY_NAMES,
        TactileContactBits=tactile.TactileContactBits,
        ProjectedTactileContactBits=tactile.ProjectedTactileContactBits,
        TACTILE_REGIONS=geometry.TACTILE_REGIONS,
        TACTILE_PARENT_BODY_NAMES=geometry.TACTILE_PARENT_BODY_NAMES,
        parent_sensor_name=geometry.parent_sensor_name,
    )
    monkeypatch.setitem(sys.modules, cfg_module.__name__, cfg_module)
    cfg = _execute_definitions(MODULE_ROOT / "tactile_env_cfg.py", cfg_module.__dict__)

    # Run the checked-in Isaac Lab runner configuration class bodies, including
    # mutable-default handling, then execute both production PPO subclasses.
    runner_module = types.ModuleType("_tactile_official_runner")
    runner_module.__dict__.update(configclass=configclass, MISSING=MISSING, Literal=Literal)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    official_runner_path = PROJECT_ROOT / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py"
    runner_classes = {node.name for node in ast.parse(official_runner_path.read_text()).body
                      if isinstance(node, ast.ClassDef)}
    official_runner = _execute_definitions(official_runner_path, runner_module.__dict__, runner_classes)
    baseline_runner_module = types.ModuleType("_tactile_cpu_baseline_runner")
    baseline_runner_module.__dict__.update(
        configclass=configclass, RslRlOnPolicyRunnerCfg=official_runner.RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg=official_runner.RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg=official_runner.RslRlPpoAlgorithmCfg,
    )
    monkeypatch.setitem(sys.modules, baseline_runner_module.__name__, baseline_runner_module)
    baseline_runner = _execute_definitions(
        MODULE_ROOT / "agents/rsl_rl_ppo_cfg.py", baseline_runner_module.__dict__,
    )
    tactile_runner_module = types.ModuleType("_tactile_cpu_runner")
    tactile_runner_module.__dict__.update(
        configclass=configclass, InspireSweepPPORunnerCfg=baseline_runner.InspireSweepPPORunnerCfg,
    )
    monkeypatch.setitem(sys.modules, tactile_runner_module.__name__, tactile_runner_module)
    tactile_runner = _execute_definitions(MODULE_ROOT / "agents/tactile_ppo_cfg.py", tactile_runner_module.__dict__)
    return SimpleNamespace(cfg=cfg, baseline=baseline, terms=omni_config.terms,
                           tactile=tactile, geometry=geometry,
                           baseline_runner=baseline_runner, runner=tactile_runner)


def _add_live_sensor_data(env, pad_names, geometry):
    robot = env.scene["robot"]
    robot.body_names = list(dict.fromkeys([
        "base_link", "axia80_link", "inspire_base_link", *pad_names, *geometry.TACTILE_PARENT_BODY_NAMES,
    ]))
    body_count = len(robot.body_names)
    robot.data.body_pos_w = torch.zeros(env.num_envs, body_count, 3)
    robot.data.body_quat_w = torch.tensor([1., 0., 0., 0.]).expand(env.num_envs, body_count, 4).clone()
    robot.data.body_link_pos_w = robot.data.body_pos_w.clone()
    robot.data.body_link_quat_w = robot.data.body_quat_w.clone()
    robot.data.body_incoming_joint_wrench_b = torch.zeros(env.num_envs, body_count, 6)
    robot.data.body_incoming_joint_wrench_b[:, robot.body_names.index("inspire_base_link"), 0] = 2.0
    env.scene["palm_tactile"] = SimpleNamespace(
        body_names=["inspire_palm_force_sensor"],
        data=SimpleNamespace(net_forces_w=torch.zeros(env.num_envs, 1, 3)),
    )
    env.scene["hand_tactile"] = SimpleNamespace(
        body_names=list(reversed(pad_names)),
        data=SimpleNamespace(net_forces_w=torch.zeros(env.num_envs, 17, 3)),
    )
    env.physics_dt = 0.01

    class EmptyContactView:
        def get_contact_data(self, dt):
            assert dt == env.physics_dt
            return (
                torch.zeros(0, 1), torch.zeros(0, 3), torch.zeros(0, 3), torch.zeros(0, 1),
                torch.zeros(env.num_envs, 3, dtype=torch.int32),
                torch.zeros(env.num_envs, 3, dtype=torch.int32),
            )

    for parent in geometry.TACTILE_PARENT_BODY_NAMES:
        env.scene[geometry.parent_sensor_name(parent)] = SimpleNamespace(
            body_names=[parent], contact_physx_view=EmptyContactView(),
            data=SimpleNamespace(
                net_forces_w=torch.zeros(env.num_envs, 1, 3),
                force_matrix_w=torch.zeros(env.num_envs, 1, 3, 3),
            ),
        )


def _assemble_policy(cfg, env, term_type):
    columns = []
    for term in vars(cfg.observations.policy).values():
        if not isinstance(term, term_type):
            continue
        params = deepcopy(term.params)
        if "asset_cfg" in params:
            params["asset_cfg"].joint_ids = list(range(6))
        function = term.func(term, env) if inspect.isclass(term.func) else term.func
        columns.append(function(env, **params))
    return torch.cat(columns, dim=-1)


def test_tactile_scene_selects_exactly_seventeen_existing_pad_bodies(tactile_config):
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    baseline = tactile_config.baseline.InspireSweepEnvCfg()
    assert isinstance(cfg, type(baseline))
    assert isinstance(cfg.scene, tactile_config.baseline.SweepSceneCfg)
    assert cfg.scene.palm_tactile is None
    assert baseline.scene.palm_tactile is not None
    assert "hand_tactile" not in vars(baseline.scene)
    assert set(vars(cfg.scene)) - set(vars(baseline.scene)) == {
        "hand_tactile", *(tactile_config.geometry.parent_sensor_name(parent)
                          for parent in tactile_config.geometry.TACTILE_PARENT_BODY_NAMES),
    }
    sensor = cfg.scene.hand_tactile
    assert sensor.prim_path.startswith("{ENV_REGEX_NS}/Robot/")
    pattern = sensor.prim_path.rsplit("/", 1)[-1]
    names = tactile_config.tactile.PALM_SENSOR_BODY_NAMES
    assert all(re.fullmatch(pattern, name) for name in names)
    assert not re.fullmatch(pattern, "inspire_left_index_1")
    assert not re.fullmatch(pattern, "inspire_thumb_force_sensor_5")
    assert not re.fullmatch(pattern, "axia80_link")
    assert sensor.update_period == 0.0
    assert sensor.force_threshold == 0.05
    assert sensor.debug_vis is False
    assert sensor.history_length == 0
    for parent in tactile_config.geometry.TACTILE_PARENT_BODY_NAMES:
        observer = getattr(cfg.scene, tactile_config.geometry.parent_sensor_name(parent))
        assert observer.prim_path == "{ENV_REGEX_NS}/Robot/" + parent
        assert observer.filter_prim_paths_expr == [
            "{ENV_REGEX_NS}/mass_cylinder", "{ENV_REGEX_NS}/Shelf.*", "/World/ground/.*",
        ]
        assert observer.history_length == 0 and observer.update_period == 0.0
        assert observer.max_contact_data_count_per_prim == 16


def test_tactile_changes_only_sensor_observation_and_preserves_baseline_physics(tactile_config):
    baseline = tactile_config.baseline.InspireSweepEnvCfg()
    baseline_snapshot = baseline.to_dict()
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    baseline_policy = baseline.observations.policy
    policy = cfg.observations.policy
    assert isinstance(policy, type(baseline_policy))
    assert list(vars(policy)) == list(vars(baseline_policy))
    for name, term in vars(baseline_policy).items():
        if name != "tactile":
            assert getattr(policy, name) == term
    assert policy.tactile.func is tactile_config.tactile.ProjectedTactileContactBits
    assert issubclass(policy.tactile.func, tactile_config.tactile.TactileContactBits)
    assert policy.tactile.params == {"sensor_name": "hand_tactile", "threshold_n": 0.05}
    for name in ("actions", "commands", "events", "rewards", "terminations", "sim", "viewer"):
        left, right = getattr(cfg, name), getattr(baseline, name)
        assert vars(left) == vars(right)
    for name, value in vars(baseline.scene).items():
        if name != "palm_tactile":
            assert getattr(cfg.scene, name) == value
    assert cfg.decimation == baseline.decimation
    assert cfg.episode_length_s == baseline.episode_length_s
    # Mutations stay local to the new environment instance.
    cfg.observations.policy.tactile.params["threshold_n"] = 0.1
    cfg.scene.hand_tactile.force_threshold = 0.1
    cfg.scene.robot.init_state.joint_pos["elbow_joint"] = -99.0
    assert baseline.scene.robot.init_state.joint_pos["elbow_joint"] == 2.2
    another = tactile_config.cfg.InspireSweepTactileEnvCfg()
    assert another.observations.policy.tactile.params["threshold_n"] == 0.05
    assert another.scene.hand_tactile.force_threshold == 0.05
    assert another.scene.robot.init_state.joint_pos["elbow_joint"] == 2.2
    assert baseline.to_dict() == baseline_snapshot
    assert tactile_config.baseline.InspireSweepEnvCfg().to_dict() == baseline_snapshot


def test_live_policy_is_57d_and_baseline_stays_41d(tactile_config, make_env):
    env = make_env(2)
    _add_live_sensor_data(env, tactile_config.tactile.PALM_SENSOR_BODY_NAMES, tactile_config.geometry)
    # Last body in the reversed PhysX order is the first policy pad (palm).
    env.scene["hand_tactile"].data.net_forces_w[0, -1, 0] = 0.08
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    baseline = tactile_config.baseline.InspireSweepEnvCfg()
    actual = _assemble_policy(cfg, env, tactile_config.terms.ObservationTermCfg)
    old = _assemble_policy(baseline, env, tactile_config.terms.ObservationTermCfg)
    assert actual.shape == (2, 57) and old.shape == (2, 41)
    assert torch.equal(actual[:, :34], old[:, :34])
    assert torch.equal(actual[:, 51:], old[:, 35:])
    assert actual[0, 34] == 1.0
    assert actual[0, 35:51].count_nonzero() == 0
    assert actual[1, 34:51].count_nonzero() == 0
    assert actual[:, 51].tolist() == [2.0, 2.0]


def test_configured_reader_combines_buried_palm_region_and_direct_pad_with_partial_reset(tactile_config, make_env):
    env = make_env(2)
    _add_live_sensor_data(env, tactile_config.tactile.PALM_SENSOR_BODY_NAMES, tactile_config.geometry)
    env.sensor_data_fresh[1] = False
    # Reversed actual pad order: the last policy channel has a direct contact.
    env.scene["hand_tactile"].data.net_forces_w[0, 0, 0] = 0.08
    env.scene["hand_tactile"].data.net_forces_w[1].fill_(torch.nan)
    env.scene["robot"].data.body_link_pos_w[1].fill_(torch.nan)
    env.scene["robot"].data.body_link_quat_w[1].fill_(torch.nan)
    palm = tactile_config.geometry.TACTILE_REGIONS[0]
    proof = palm["verification"]
    assert proof["pad_wholly_buried_in_parent_convex_hull"]

    class PalmContactView:
        def get_contact_data(self, dt):
            assert dt == env.physics_dt
            points = torch.full((3, 3), torch.nan)
            normals = torch.full((3, 3), torch.nan)
            points[0] = torch.tensor(proof["parent_surface_point_local_m"])
            normals[0] = torch.tensor(proof["parent_surface_outward_normal_local"])
            return (
                torch.tensor([[0.08], [torch.nan], [torch.nan]]), points, normals,
                torch.zeros(3, 1), torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.int32),
                torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
            )

    observer = env.scene[tactile_config.geometry.parent_sensor_name(palm["parent_name"])]
    observer.contact_physx_view = PalmContactView()
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    term = cfg.observations.policy.tactile
    bits = term.func(term, env)(env, **term.params)
    expected = torch.zeros(2, 17)
    expected[0, 0], expected[0, -1] = 1.0, 1.0
    assert torch.equal(bits, expected)


def test_configured_reader_does_not_read_direct_or_parent_physics_handles_after_reset(tactile_config, make_env):
    env = make_env(2)
    _add_live_sensor_data(env, tactile_config.tactile.PALM_SENSOR_BODY_NAMES, tactile_config.geometry)
    env.sensor_data_fresh.zero_()

    class StaleForceData:
        @property
        def net_forces_w(self):
            raise AssertionError("No fresh direct force sample")

    class StaleContactView:
        def get_contact_data(self, dt):
            raise AssertionError("No fresh parent contact sample")

    env.scene["hand_tactile"].data = StaleForceData()
    for parent in tactile_config.geometry.TACTILE_PARENT_BODY_NAMES:
        env.scene[tactile_config.geometry.parent_sensor_name(parent)].contact_physx_view = StaleContactView()
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    observations = _assemble_policy(cfg, env, tactile_config.terms.ObservationTermCfg)
    assert observations.shape == (2, 57)
    assert torch.isfinite(observations).all()
    assert observations[:, 34:].count_nonzero() == 0


def test_registration_uses_baseline_runtime_with_separate_tactile_runner():
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    namespace = {"gym": gym}
    _execute_definitions(MODULE_ROOT / "__init__.py", namespace)
    baseline = registrations["Isaac-Inspire-Sweep-Baseline-v0"]
    tactile = registrations["Isaac-Inspire-Sweep-Baseline-Tactile-v0"]
    assert tactile["entry_point"] == baseline["entry_point"]
    assert tactile["kwargs"]["env_cfg_entry_point"] == "inspire_sweep_baseline.tactile_env_cfg:InspireSweepTactileEnvCfg"
    assert tactile["kwargs"]["rsl_rl_cfg_entry_point"] == "inspire_sweep_baseline.agents.tactile_ppo_cfg:InspireSweepTactilePPORunnerCfg"


def test_tactile_ppo_keeps_all_baseline_settings_except_log_directory(tactile_config):
    baseline = tactile_config.baseline_runner.InspireSweepPPORunnerCfg()
    tactile = tactile_config.runner.InspireSweepTactilePPORunnerCfg()
    assert isinstance(tactile, type(baseline))
    expected = baseline.to_dict()
    expected["experiment_name"] = "inspire_sweep_tactile"
    assert tactile.to_dict() == expected
    tactile.policy.actor_hidden_dims[0] = 999
    assert baseline.policy.actor_hidden_dims == [256, 128, 64]


def test_real_rsl_mlp_accepts_tactile_policy_and_emits_eight_actions(tactile_config, make_env):
    models = pytest.importorskip("rsl_rl.models")
    tensordict = pytest.importorskip("tensordict")
    if not hasattr(models, "MLPModel"):
        pytest.skip("Installed RSL-RL version does not provide MLPModel")
    env = make_env(2)
    _add_live_sensor_data(env, tactile_config.tactile.PALM_SENSOR_BODY_NAMES, tactile_config.geometry)
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    policy = _assemble_policy(cfg, env, tactile_config.terms.ObservationTermCfg)
    runner = tactile_config.runner.InspireSweepTactilePPORunnerCfg()
    obs = tensordict.TensorDict({"policy": policy}, batch_size=[2])
    model = models.MLPModel(
        obs, runner.obs_groups, "actor", 8,
        hidden_dims=runner.policy.actor_hidden_dims, activation=runner.policy.activation,
        obs_normalization=runner.policy.actor_obs_normalization,
    )
    actions = model(obs)
    assert model.obs_dim == 57
    assert actions.shape == (2, 8) and torch.isfinite(actions).all()
