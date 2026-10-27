"""Final matches the live source contract without any task Cfg inheritance.

CPU tests use the official configclass/manager terms. Simulator imports,
asset construction and the full EnvCfg are covered by _final_probe.py.
"""

import ast
import importlib
import sys
from types import ModuleType

import pytest

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions, omni_config
from test_fusion_holding_osc import osc


def assert_ordered_equal(a, b):
    if isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            assert_ordered_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_ordered_equal(left, right)
    else:
        assert a == b


@pytest.fixture
def configs(omni_config, monkeypatch):
    managers = sys.modules['isaaclab.managers']
    for name in ('RewardTermCfg', 'TerminationTermCfg', 'EventTermCfg',
                 'ObservationTermCfg', 'ObservationGroupCfg'):
        monkeypatch.setattr(managers, name, getattr(omni_config.terms, name), raising=False)
    # CurriculumTermCfg only adds the callable annotation to ManagerTermBaseCfg.
    monkeypatch.setattr(managers, 'CurriculumTermCfg', omni_config.terms.ManagerTermBaseCfg, raising=False)
    envs = ModuleType('isaaclab.envs')
    mdp = ModuleType('isaaclab.envs.mdp')
    mdp.reset_scene_to_default = omni_config.mdp.reset_scene_to_default
    mdp.time_out = omni_config.mdp.time_out
    envs.mdp = mdp
    monkeypatch.setitem(sys.modules, 'isaaclab.envs', envs)
    monkeypatch.setitem(sys.modules, 'isaaclab.envs.mdp', mdp)
    return lambda role, module: importlib.import_module(f'_baseline_cpu_pkg.mdp.{role}.{module}')


@pytest.mark.parametrize('role,module,old_name,new_name', [
    ('rewards', 'contact_sweep_clutter_fusion_holding', 'FusionHoldingV4RewardsCfg', 'FinalRewardsCfg'),
    ('terminations', 'contact_sweep_clutter_fusion_holding', 'FusionHoldingV4TerminationsCfg', 'FinalTerminationsCfg'),
    ('curricula', 'contact_sweep_clutter_fusion_holding_osc_object_random', 'ObjectRandomV2CurriculumCfg', 'FinalCurriculumCfg'),
    ('events', 'contact_sweep_clutter_fusion_holding_osc_object_random', 'ObjectRandomEventsCfg', 'FinalEventsCfg'),
])
def test_live_manager_values_order_callables_and_isolation(configs, role, module, old_name, new_name):
    old = getattr(configs(role, module), old_name)()
    new_cls = getattr(configs(role, 'final'), new_name)
    new = new_cls()
    assert new_cls.__bases__ == (object,)
    assert_ordered_equal(old.to_dict(), new.to_dict())
    for key, term in vars(new).items():
        if hasattr(term, 'params'):
            term.params['isolation_probe'] = True
            assert 'isolation_probe' not in getattr(old, key).params
            assert 'isolation_probe' not in getattr(new_cls(), key).params


def test_final_has_no_old_task_cfg_or_runtime_ancestors_or_instances():
    paths = [*(MODULE_ROOT / f'mdp/{role}/final.py' for role in
               ('actions', 'commands', 'curricula', 'events', 'observations', 'rewards', 'terminations')), MODULE_ROOT / 'envs/final_envs.py',
             MODULE_ROOT / 'assets/scenes/final.py', MODULE_ROOT / 'agent/final_ppo_cfg.py',
             MODULE_ROOT / 'runtime/final.py']
    allowed = {'ManagerBasedRLEnvCfg', 'ManagerBasedRLEnv', 'InteractiveSceneCfg',
               'OperationalSpaceControllerActionCfg', 'ActionTermCfg', 'CommandTermCfg',
               'ObsGroup', 'RslRlMLPModelCfg', 'RslRlOnPolicyRunnerCfg'}
    for path in paths:
        tree = ast.parse(path.read_text())
        # v0 remains independent; later Final environments inherit it.
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ClassDef)
                     and node.name.startswith(('FinalV1', 'FinalV2', 'FinalGoalAccuracy')))]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert all(isinstance(b, ast.Name) and b.id in allowed for b in node.bases), path
            if isinstance(node, ast.ImportFrom) and node.level:
                for alias in node.names:
                    if alias.name.endswith('Cfg'):
                        assert alias.name.startswith('Final') or alias.name in {
                            'RealisticRobotSpawnerCfg', 'RealisticShelfSpawnerCfg', 'RealisticCylinderSpawnerCfg',
                        }, (path, alias.name)


def test_new_registry_entry_is_independent():
    from types import SimpleNamespace
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs['id']: kwargs})
    _execute_definitions(MODULE_ROOT / '__init__.py', {'gym': gym})
    entry = registrations['Isaac-Inspire-Final-v0']
    assert entry['entry_point'] == 'inspire_sweep_baseline.runtime.final:FinalEnv'
    assert entry['kwargs'] == {
        'env_cfg_entry_point': 'inspire_sweep_baseline.envs.final_envs:FinalEnvCfg',
        'rsl_rl_cfg_entry_point': 'inspire_sweep_baseline.agent.final_ppo_cfg:FinalPPORunnerCfg',
    }


def test_scene_assets_sensors_and_rebuild_match(omni_config):
    """Execute both live scene builders; only USD/Sim leaf types are doubles."""
    from types import SimpleNamespace
    from test_omni_configuration import _AssetNode
    common = importlib.import_module('_baseline_cpu_pkg.mdp.common.contact_sweep_clutter_fusion')
    contacts = importlib.import_module('_baseline_cpu_pkg.mdp.common.contact_sweep_clutter_fusion_contacts')
    population = importlib.import_module('_baseline_cpu_pkg.mdp.common.contact_sweep_clutter_fusion_population')
    tactile = importlib.import_module('_baseline_cpu_pkg.assets.tactile_regions')
    layout = importlib.import_module('_baseline_cpu_pkg.assets.scenes.contact_sweep_clutter_layout')
    robot = sys.modules['_baseline_cpu_pkg.assets.robot']
    ns = dict(vars(omni_config.baseline))
    ns.update(vars(common), **vars(omni_config.realistic))
    ns.update(vars(robot))
    ns.update(configclass=omni_config.baseline.configclass,
              make_robot_cfg=omni_config.baseline.make_robot_cfg,
              object_hand_sensor_name=contacts.object_hand_sensor_name,
              PARKING_HEIGHT=population.PARKING_HEIGHT, validate_capacity=population.validate_capacity,
              TACTILE_PARENT_BODY_NAMES=tactile.TACTILE_PARENT_BODY_NAMES,
              parent_sensor_name=tactile.parent_sensor_name,
              DEFAULT_LAYOUT_CONFIG=layout.DEFAULT_LAYOUT_CONFIG)
    ns['sim_utils'].MultiAssetSpawnerCfg = _AssetNode
    ns['sim'] = ns['sim_utils']
    # Execute prop_variants unchanged; no USD stage or network asset loading.
    prop_tree = ast.parse((MODULE_ROOT / 'assets/contact_sweep_clutter_fusion_holding_osc_object_random_spawners.py').read_text())
    prop_tree.body = [n for n in prop_tree.body if isinstance(n, ast.Assign) or
                      isinstance(n, ast.FunctionDef) and n.name == 'prop_variants']
    ns.update(ShelfPropCfg=_AssetNode, ISAAC_NUCLEUS_DIR='test-nucleus')
    exec(compile(prop_tree, '<prop_variants>', 'exec'), ns)
    ns.update(vars(_execute_definitions(MODULE_ROOT / 'assets/scenes/contact_sweep_clutter_fusion.py', ns)))
    ns.update(vars(_execute_definitions(MODULE_ROOT / 'assets/scenes/contact_sweep_clutter_fusion_holding_osc_object_random.py', ns)))
    final = _execute_definitions(MODULE_ROOT / 'assets/scenes/final.py', ns).FinalSceneCfg
    old = ns['ObjectRandomV1SceneCfg']
    for gray, white in ((8, 8), (3, 4), (0, 0)):
        left, right = old(num_envs=2, env_spacing=2.5), final(num_envs=2, env_spacing=2.5)
        for cfg in (left, right):
            cfg.configure_population_capacity(gray, white, 16)
        assert_ordered_equal(left.to_dict(), right.to_dict())
        # The env applies friction then the runtime reconstructs the scene again.
        for cfg in (left, right):
            cfg.object_collection.rigid_objects['mass_cylinder'].spawn.friction_defaults = _AssetNode(marker=1)
            cfg.configure_population_capacity(gray, white, 16)
        assert_ordered_equal(left.to_dict(), right.to_dict())


def test_ppo_and_encoder_values_match(omni_config):
    from dataclasses import MISSING
    from test_omni_configuration import _AssetNode
    cc = omni_config.baseline.configclass

    @cc
    class Base:
        distribution_cfg: object = None
        policy: object = None

    class Distribution(_AssetNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    Base.GaussianDistributionCfg = Distribution
    ns = dict(configclass=cc, MISSING=MISSING, RslRlMLPModelCfg=Base,
              RslRlOnPolicyRunnerCfg=Base, RslRlPpoAlgorithmCfg=_AssetNode)
    for suffix in ('', '_holding', '_holding_osc', '_holding_osc_object_random'):
        ns.update(vars(_execute_definitions(MODULE_ROOT / f'agent/contact_sweep_clutter_fusion{suffix}_ppo_cfg.py', ns)))
    old = ns['FusionHoldingOSCObjectRandomV3PPORunnerCfg']()
    new = _execute_definitions(MODULE_ROOT / 'agent/final_ppo_cfg.py', ns).FinalPPORunnerCfg()
    assert new.experiment_name == 'inspire_final_v0'
    data = new.to_dict()
    data['experiment_name'] = old.experiment_name
    assert_ordered_equal(old.to_dict(), data)


def test_observation_terms_and_noise_match(configs, monkeypatch):
    from test_omni_configuration import _AssetNode
    noise = ModuleType('isaaclab.utils.noise')
    noise.NoiseCfg = noise.ConstantNoiseCfg = noise.NoiseModelWithAdditiveBiasCfg = noise.UniformNoiseCfg = _AssetNode
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', noise)
    old = configs('observations', 'contact_sweep_clutter_fusion').ContactSweepClutterFusionObservationsCfg()
    new = configs('observations', 'final').FinalObservationsCfg()
    assert_ordered_equal(old.to_dict(), new.to_dict())


def test_runtime_lifecycle_matches_for_seed_partial_reset_and_iteration():
    """Exercise both live lifecycle classes on a CPU ManagerBasedRLEnv double."""
    import copy
    from types import SimpleNamespace as NS
    import torch

    class Base:
        def __init__(self, cfg, render_mode=None, **kwargs):
            self.cfg = cfg
            self.action_manager = NS(_action=torch.zeros(3, 8), _prev_action=torch.zeros(3, 8))
            self.scene = {'object_collection': NS(data=NS(object_pos_w=torch.arange(18).reshape(3, 2, 3).float()))}
            self.curriculum_calls = []
            self.curriculum_manager = NS(compute=lambda env_ids: self.curriculum_calls.append(env_ids))

        def seed(self, seed):
            return seed

        def reset(self, seed=None, env_ids=None, options=None):
            self._reset_idx(list(range(3)) if env_ids is None else env_ids)
            return seed, options

        def _reset_idx(self, env_ids):
            self.action_manager._action[env_ids] = 0
            self.action_manager._prev_action[env_ids] = 0

        def step(self, action):
            return action

    ns = dict(ManagerBasedRLEnv=Base, torch=torch, ClutterLayoutGenerator=lambda seed: NS(seed=seed))
    old_cls = _execute_definitions(MODULE_ROOT / 'runtime/contact_sweep_clutter_fusion.py', ns).ContactSweepClutterFusionV1_0Env
    new_cls = _execute_definitions(MODULE_ROOT / 'runtime/final.py', ns).FinalEnv
    scene = NS(num_envs=3, object_collection=NS(rigid_objects={'target': None, 'obstacle': None}),
               configure_population_capacity=lambda *args: None)
    cfg = NS(scene=scene, sim=NS(device='cpu'), seed=42, obstacle_observation_slots=16,
             max_gray_obstacles=8, max_white_obstacles=8)
    old, new = old_cls(copy.deepcopy(cfg)), new_cls(copy.deepcopy(cfg))
    for env in (old, new):
        env._population_layout_generators[0] = 'old'
        env._gray_padding_layout_generators = {0: 'old'}
        env.step(torch.ones(3, 8))
        env.reset(seed=123, env_ids=[1], options={'probe': True})
        env.set_learning_iteration(12000)
    for key in ('_active_object_mask', 'target_id', 'target_width', 'sweep_dir', 'sensor_data_fresh',
                '_initial_obstacle_observation', '_initial_object_positions_w'):
        torch.testing.assert_close(getattr(old, key), getattr(new, key))
    torch.testing.assert_close(old.action_manager._action, new.action_manager._action)
    torch.testing.assert_close(old.action_manager._prev_action, new.action_manager._prev_action)
    assert new.sensor_data_fresh.tolist() == [True, False, True]
    assert new._population_layout_seed == old._population_layout_seed == 123
    assert new._population_layout_generators == old._population_layout_generators == {}
    assert new._gray_padding_layout_generators == old._gray_padding_layout_generators == {}
    assert new.learning_iteration == old.learning_iteration == 12000
    assert new.curriculum_calls == old.curriculum_calls == [None]
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            new.set_learning_iteration(invalid)


def test_osc_hand_and_goal_cfgs_match(osc, omni_config):
    from types import SimpleNamespace as NS
    cc = omni_config.baseline.configclass
    arm = osc.v0().actions.arm_action
    base = type(arm)
    ns = dict(configclass=cc, OperationalSpaceControllerActionCfg=base,
              OperationalSpaceControllerCfg=type(arm.controller_cfg),
              ActionTermCfg=base.__bases__[0], ActionTerm=object,
              ARM_JOINT_NAMES=omni_config.baseline.ARM_JOINT_NAMES,
              UR10OperationalSpaceControllerAction=osc.action, EEFRelativeOSCAction=osc.action,
              HandOpennessAction=object)
    ns.update(vars(_execute_definitions(MODULE_ROOT / 'mdp/actions/contact_sweep_clutter_fusion_holding_osc.py', ns)))
    ns.update(vars(_execute_definitions(MODULE_ROOT / 'mdp/actions/hand_openness.py', ns)))
    ns.update(vars(_execute_definitions(MODULE_ROOT / 'mdp/common/contact_sweep_clutter_fusion_holding_osc_configuration.py', ns)))
    final = _execute_definitions(MODULE_ROOT / 'mdp/actions/final.py', ns)
    old_cfg = osc.parent()
    ns['configure_reach_osc'](old_cfg)
    new = final.FinalActionsCfg()
    eef = old_cfg.scene.ee_frame.target_frames[0]
    new.arm_action.body_name = eef.prim_path.rsplit('/', 1)[-1]
    new.arm_action.body_offset = final.FinalOSCActionCfg.OffsetCfg(pos=eef.offset.pos, rot=eef.offset.rot)
    assert_ordered_equal(old_cfg.actions.arm_action.to_dict(), new.arm_action.to_dict())
    assert_ordered_equal(ns['HandOpennessActionCfg']().to_dict(), new.gripper_action.to_dict())

    @cc
    class CommandBase:
        pass

    ns.update(CommandTermCfg=CommandBase, CommandTerm=object, SweepGoalCommand=object)
    old_goal = _execute_definitions(MODULE_ROOT / 'mdp/commands/sweep_goal.py', ns).SweepGoalCommandCfg(debug_vis=True)
    new_goal = _execute_definitions(MODULE_ROOT / 'mdp/commands/final.py', ns).FinalCommandsCfg().target_goal_pos
    assert_ordered_equal(old_goal.to_dict(), new_goal.to_dict())
