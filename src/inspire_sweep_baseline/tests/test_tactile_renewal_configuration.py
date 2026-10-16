"""Tactile Renewal inheritance, real observation assembly and force boundaries."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions, omni_config
from test_tactile_configuration import _add_live_sensor_data, _assemble_policy, tactile_config


@pytest.fixture
def renewal_tactile(tactile_config, omni_config, monkeypatch):
    module = types.ModuleType("_tactile_renewal_cpu_cfg")
    module.__dict__.update(
        configclass=tactile_config.baseline.configclass,
        ObsTerm=tactile_config.terms.ObservationTermCfg,
        SweepSceneCfg=tactile_config.baseline.SweepSceneCfg,
        TactileSweepSceneCfg=tactile_config.cfg.TactileSweepSceneCfg,
        TactileObservationsCfg=tactile_config.cfg.TactileObservationsCfg,
        ProjectedTactileContactBits=tactile_config.tactile.ProjectedTactileContactBits,
        InspireSweepOmniRenewalEnvCfg=omni_config.renewal.InspireSweepOmniRenewalEnvCfg,
        InspireDexSweepOmniRenewalEnvCfg=omni_config.dex_renewal.InspireDexSweepOmniRenewalEnvCfg,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = _execute_definitions(MODULE_ROOT / "tactile_omni_renewal_env_cfg.py", module.__dict__)
    runner_module = types.ModuleType("_tactile_renewal_cpu_runner")
    runner_module.__dict__.update(
        configclass=tactile_config.baseline.configclass,
        InspireSweepPPORunnerCfg=tactile_config.baseline_runner.InspireSweepPPORunnerCfg,
    )
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    runners = _execute_definitions(MODULE_ROOT / "agents/tactile_omni_renewal_ppo_cfg.py", runner_module.__dict__)
    return SimpleNamespace(cfg=cfg, runners=runners, old=tactile_config)


@pytest.mark.parametrize("dex", [False, True])
def test_only_sensor_and_tactile_observation_change(renewal_tactile, dex):
    module = renewal_tactile.cfg
    parent_type = module.InspireDexSweepOmniRenewalEnvCfg if dex else module.InspireSweepOmniRenewalEnvCfg
    child_type = module.InspireDexTactileSweepOmniRenewalEnvCfg if dex else module.InspireTactileSweepOmniRenewalEnvCfg
    parent, child = parent_type(), child_type()
    snapshot = parent.to_dict()
    assert isinstance(child, parent_type)
    for name, value in snapshot.items():
        if name not in ("scene", "observations"):
            assert child.to_dict()[name] == value, name
    for name, value in parent.scene.to_dict().items():
        if name != "palm_tactile":
            assert child.scene.to_dict()[name] == value, name
    assert child.scene.palm_tactile is None
    assert child.scene.hand_tactile.force_threshold == 0.01
    for region_parent in renewal_tactile.old.geometry.TACTILE_PARENT_BODY_NAMES:
        sensor = getattr(child.scene, renewal_tactile.old.geometry.parent_sensor_name(region_parent))
        assert sensor.filter_prim_paths_expr == ["{ENV_REGEX_NS}/mass_cylinder"]
    for name, term in vars(parent.observations.policy).items():
        if name != "tactile":
            assert getattr(child.observations.policy, name) == term
    assert child.observations.policy.tactile.params == {"sensor_name": "hand_tactile", "threshold_n": 0.01}
    child.observations.policy.tactile.params["threshold_n"] = 99.0
    child.scene.robot.init_state.joint_pos["elbow_joint"] = -99.0
    assert parent.to_dict() == snapshot
    assert child_type().observations.policy.tactile.params["threshold_n"] == 0.01
    assert child_type().scene.robot == parent.scene.robot


@pytest.mark.parametrize("dex", [False, True])
def test_57d_policy_inclusive_001n_per_pad_and_reset(renewal_tactile, make_env, dex):
    module, old = renewal_tactile.cfg, renewal_tactile.old
    cfg = (module.InspireDexTactileSweepOmniRenewalEnvCfg if dex else module.InspireTactileSweepOmniRenewalEnvCfg)()
    env = make_env(4)
    _add_live_sensor_data(env, old.tactile.PALM_SENSOR_BODY_NAMES, old.geometry)
    # Reverse PhysX body order must retain the documented channel order.
    env.scene["hand_tactile"].data.net_forces_w[:, -1, 0] = torch.tensor([0.0099, 0.01, 0.0101, torch.nan])
    env.sensor_data_fresh[-1] = False
    policy = _assemble_policy(cfg, env, old.terms.ObservationTermCfg)
    assert policy.shape == (4, 57)
    assert policy[:, 34].tolist() == [0., 1., 1., 0.]
    assert policy[:, 35:51].count_nonzero() == 0
    assert set(policy[:, 34:51].unique().tolist()) <= {0., 1.}


def test_buried_palm_uses_projected_net_force_at_001n(renewal_tactile, make_env):
    old = renewal_tactile.old
    env = make_env(3)
    _add_live_sensor_data(env, old.tactile.PALM_SENSOR_BODY_NAMES, old.geometry)
    palm = old.geometry.TACTILE_REGIONS[0]
    proof = palm["verification"]

    class PalmContactView:
        def get_contact_data(self, dt):
            assert dt == env.physics_dt
            return (
                torch.tensor([[0.0099], [0.01], [0.0101]]),
                torch.tensor(proof["parent_surface_point_local_m"]).repeat(3, 1),
                torch.tensor(proof["parent_surface_outward_normal_local"]).repeat(3, 1),
                torch.zeros(3, 1), torch.ones(3, 1, dtype=torch.int32),
                torch.arange(3, dtype=torch.int32).reshape(3, 1),
            )

    observer = env.scene[old.geometry.parent_sensor_name(palm["parent_name"])]
    # PhysX reserves extra capacity beyond live contacts.
    view = PalmContactView()
    class PaddedView:
        def get_contact_data(self, dt):
            force, point, normal, separation, counts, starts = view.get_contact_data(dt)
            return (
                torch.cat((force, torch.zeros(1, 1))),
                torch.cat((point, torch.zeros(1, 3))),
                torch.cat((normal, torch.zeros(1, 3))),
                torch.cat((separation, torch.zeros(1, 1))), counts, starts,
            )
    observer.contact_physx_view = PaddedView()
    cfg = renewal_tactile.cfg.InspireTactileSweepOmniRenewalEnvCfg()
    term = cfg.observations.policy.tactile
    bits = term.func(term, env)(env, **term.params)
    assert bits[:, 0].tolist() == [0., 1., 1.]
    assert bits[:, 1:].count_nonzero() == 0
    assert env.scene["hand_tactile"].data.net_forces_w.count_nonzero() == 0


@pytest.mark.parametrize("prefix,experiment", [
    ("InspireTactileSweepOmniRenewal", "inspire_tactile_sweep_omni_renewal"),
    ("InspireDexTactileSweepOmniRenewal", "inspire_dex_tactile_sweep_omni_renewal"),
])
def test_registration_and_ppo_preserve_parent_settings(renewal_tactile, prefix, experiment):
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    task = "Isaac-Inspire-" + ("Dex-" if "Dex" in prefix else "") + "Tactile-Sweep-Omni-Renewal-v0"
    registration = registrations[task]
    assert registration["entry_point"] == "inspire_sweep_baseline.env:InspireSweepEnv"
    assert registration["kwargs"]["env_cfg_entry_point"] == f"inspire_sweep_baseline.tactile_omni_renewal_env_cfg:{prefix}EnvCfg"
    assert registration["kwargs"]["rsl_rl_cfg_entry_point"] == f"inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:{prefix}PPORunnerCfg"
    runner = getattr(renewal_tactile.runners, prefix + "PPORunnerCfg")()
    expected = renewal_tactile.old.baseline_runner.InspireSweepPPORunnerCfg().to_dict()
    expected["experiment_name"] = experiment
    assert runner.to_dict() == expected
