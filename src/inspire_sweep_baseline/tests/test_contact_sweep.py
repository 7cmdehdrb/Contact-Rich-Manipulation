"""Independent configuration parity and reset lifecycle for the new reference."""

import ast
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions, omni_config
from test_tactile_configuration import tactile_config, _add_live_sensor_data, _assemble_policy
from test_tactile_renewal_configuration import renewal_tactile, renewal_tactile_v1


@pytest.fixture
def contact_config(renewal_tactile_v1, renewal_tactile, omni_config, monkeypatch, load_mdp):
    old = renewal_tactile.old
    physics = load_mdp("realistic_physics")
    obs = load_mdp("observations")
    terminations = load_mdp("terminations")
    namespace = dict(vars(omni_config.baseline))
    namespace.update(vars(omni_config.realistic))
    namespace.update(
        math=__import__("math"), obs=obs, dex_rewards=omni_config.dex.dex_rewards,
        goal_rewards=omni_config.omni.goal_rewards, goal_terminations=omni_config.omni.goal_terminations,
        terminations=terminations, time_out=omni_config.mdp.time_out,
        reset_scene_to_default=omni_config.mdp.reset_scene_to_default,
        HAND_JOINT_NAMES=sys.modules["_baseline_cpu_pkg.assets.robot"].HAND_JOINT_NAMES,
        PALM_SENSOR_BODY_NAMES=old.tactile.PALM_SENSOR_BODY_NAMES,
        TACTILE_PARENT_BODY_NAMES=old.geometry.TACTILE_PARENT_BODY_NAMES,
        parent_sensor_name=old.geometry.parent_sensor_name,
        ProjectedTactileContactBits=old.tactile.ProjectedTactileContactBits,
        JointPositionActionCfg=omni_config.mdp.JointPositionActionCfg,
        HandOpennessActionCfg=omni_config.mdp.HandOpennessActionCfg,
        SweepGoalCommandCfg=omni_config.mdp.SweepGoalCommandCfg,
        initialize_realistic_materials=physics.initialize_realistic_materials,
        randomize_scaled_episode_physics=physics.randomize_scaled_episode_physics,
        randomize_goal_object_scene=omni_config.omni.randomize_goal_object_scene,
        ShelfCollisionPenalty=omni_config.mdp.ShelfCollisionPenalty,
        InspireDexTactileSweepOmniRenewalV1EnvCfg=renewal_tactile_v1.InspireDexTactileSweepOmniRenewalV1EnvCfg,
    )
    for name in ("common", "scene_cfg", "observations_cfg", "actions_cfg", "commands_cfg",
                 "events_cfg", "rewards_cfg", "terminations_cfg", "env_cfg"):
        module = types.ModuleType(f"_contact_cpu_{name}")
        module.__dict__.update(namespace, __name__=module.__name__)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        result = _execute_definitions(MODULE_ROOT / "contact_sweep" / f"{name}.py", module.__dict__)
        namespace.update(vars(result))
    v3_module = types.ModuleType("_contact_reference_v3")
    v3_module.__dict__.update(namespace, __name__=v3_module.__name__)
    monkeypatch.setitem(sys.modules, v3_module.__name__, v3_module)
    v3 = _execute_definitions(MODULE_ROOT / "dex_tactile_omni_renewal_v3_env_cfg.py", v3_module.__dict__)
    return SimpleNamespace(new=namespace, v3=v3, old=old)


def test_independent_managers_match_v3_and_instances_do_not_share_state(contact_config):
    ns = contact_config.new
    new = ns["ContactSweepEnvCfg"]()
    old = contact_config.v3.InspireDexTactileSweepOmniRenewalV3EnvCfg()
    for name in ("observations", "actions", "commands", "events", "rewards", "terminations", "sim", "viewer"):
        new_part, old_part = getattr(new, name), getattr(old, name)
        if hasattr(new_part, "to_dict"):
            assert new_part.to_dict() == old_part.to_dict(), name
        else:
            assert vars(new_part) == vars(old_part), name
    assert list(vars(new.rewards)) == list(vars(old.rewards))
    assert list(vars(new.terminations)) == list(vars(old.terminations))
    for name in ("ee_frame", "finger_frame", "wrist_frame", "hand_tactile"):
        assert vars(getattr(new.scene, name)) == vars(getattr(old.scene, name))
    for parent in contact_config.old.geometry.TACTILE_PARENT_BODY_NAMES:
        name = contact_config.old.geometry.parent_sensor_name(parent)
        assert vars(getattr(new.scene, name)) == vars(getattr(old.scene, name))
    assert new.scene.object_collection.rigid_objects['mass_cylinder'].spawn.mass_props.mass == .5475
    assert new.scene.robot.actuators['arm'].velocity_limit_sim == old.scene.robot.actuators['arm'].velocity_limit_sim
    before = old.to_dict()
    new.rewards.goal_distance.weight = 1
    new.events.physics_randomization.params['mass_scale'] = 99
    assert old.to_dict() == before
    assert ns['ContactSweepEnvCfg']().rewards.goal_distance.weight == 94


def test_reference_observation_order_and_57d_values_match_v3(contact_config, make_env):
    ns, old = contact_config.new, contact_config.old
    new = ns['ContactSweepEnvCfg']()
    reference = contact_config.v3.InspireDexTactileSweepOmniRenewalV3EnvCfg()
    env = make_env(3)
    _add_live_sensor_data(env, old.tactile.PALM_SENSOR_BODY_NAMES, old.geometry)
    env.sensor_data_fresh[1] = False
    torch.manual_seed(42)
    a = _assemble_policy(new, env, old.terms.ObservationTermCfg)
    torch.manual_seed(42)
    b = _assemble_policy(reference, env, old.terms.ObservationTermCfg)
    assert a.shape == (3, 57)
    assert torch.equal(a, b)


def test_contact_classes_do_not_import_or_inherit_legacy_environment_configs():
    root = MODULE_ROOT / 'contact_sweep'
    forbidden = ('renewal', 'tactile_env_cfg', 'realistic_physics_env_cfg', 'omni_env_cfg')
    allowed_bases = {'InteractiveSceneCfg', 'ObsGroup', 'ManagerBasedRLEnvCfg', 'ManagerBasedRLEnv', 'RslRlOnPolicyRunnerCfg'}
    for file in root.glob('*.py'):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(part in (node.module or '') for part in forbidden), file
                assert not (node.level == 2 and node.module == 'env_cfg'), file
            if isinstance(node, ast.ClassDef):
                assert all(isinstance(base, ast.Name) and base.id in allowed_bases for base in node.bases), (file, node.name)
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs['id']: kwargs})
    _execute_definitions(MODULE_ROOT / '__init__.py', {'gym': gym})
    cfg = registrations['Isaac-Inspire-Contact-Sweep-v0']
    assert cfg['entry_point'] == 'inspire_sweep_baseline.contact_sweep.env:ContactSweepEnv'
    assert cfg['kwargs']['env_cfg_entry_point'] == 'inspire_sweep_baseline.contact_sweep.env_cfg:ContactSweepEnvCfg'
    assert cfg['kwargs']['rsl_rl_cfg_entry_point'] == 'inspire_sweep_baseline.contact_sweep.ppo_cfg:ContactSweepPPORunnerCfg'


def test_reference_ppo_has_its_own_framework_base_and_matches_v3(renewal_tactile, monkeypatch):
    namespace = dict(vars(renewal_tactile.old.baseline_runner))
    module = types.ModuleType('_contact_reference_runner')
    module.__dict__.update(namespace, __name__=module.__name__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = _execute_definitions(MODULE_ROOT / 'contact_sweep/ppo_cfg.py', module.__dict__)
    actual = cfg.ContactSweepPPORunnerCfg().to_dict()
    expected = renewal_tactile.runners.InspireDexTactileSweepOmniRenewalV3PPORunnerCfg().to_dict()
    expected['experiment_name'] = 'inspire_contact_sweep'
    assert actual == expected


def test_reference_runtime_resets_sensor_freshness_and_hand_history():
    class Framework:
        def _reset_idx(self, ids):
            self.reset_ids = ids.clone()
        def step(self, action):
            self.sensor_data_fresh[1] = False
            return self.sensor_data_fresh.clone(), action
    ns = {'torch': torch, 'ManagerBasedRLEnv': Framework}
    runtime = _execute_definitions(MODULE_ROOT / 'contact_sweep/env.py', ns).ContactSweepEnv
    env = runtime.__new__(runtime)
    env.sensor_data_fresh = torch.ones(3, dtype=torch.bool)
    env.action_manager = SimpleNamespace(_action=torch.zeros(3, 8), _prev_action=torch.zeros(3, 8))
    env._reset_idx(torch.tensor([1]))
    assert env.sensor_data_fresh.tolist() == [True, False, True]
    assert env.action_manager._action[1].tolist() == [0.] * 6 + [1., 1.]
    assert env.action_manager._prev_action[1].tolist() == [0.] * 6 + [1., 1.]
    env.sensor_data_fresh[:] = False
    fresh, _ = env.step(torch.zeros(3, 8))
    assert fresh.tolist() == [True, False, True]
