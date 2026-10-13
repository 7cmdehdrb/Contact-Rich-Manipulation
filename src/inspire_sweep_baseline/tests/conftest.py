"""CPU harness: use Isaac Lab's real tensor math without starting Isaac Sim.

Only manager base classes are replaced. Production functions, sensor transforms,
and the checked-in Isaac Lab quaternion implementation execute unchanged.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
MODULE_ROOT = PACKAGE_ROOT / "inspire_sweep_baseline"


def _module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def tensor_math():
    return _module_from_file(
        "_baseline_cpu_official_math",
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/utils/math.py",
    )


@pytest.fixture
def load_mdp(monkeypatch, tensor_math):
    """Import a production MDP module without the top-level Sim package."""

    class SceneEntityCfg:
        def __init__(self, name, joint_ids=slice(None), **kwargs):
            self.name, self.joint_ids = name, joint_ids
            self.__dict__.update(kwargs)

    class TermBase:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.device, self.num_envs = env.device, env.num_envs
            if hasattr(cfg, "asset_name"):
                self._asset = env.scene[cfg.asset_name]

    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = SceneEntityCfg
    managers.ActionTerm = managers.CommandTerm = managers.ManagerTermBase = TermBase
    managers.ActionTermCfg = managers.CommandTermCfg = object
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    root = types.ModuleType("isaaclab")
    root.__path__ = []
    utils.__path__ = []
    monkeypatch.setitem(sys.modules, "isaaclab", root)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", tensor_math)
    local_package = types.ModuleType("_baseline_cpu_pkg")
    local_package.__path__ = [str(MODULE_ROOT)]
    monkeypatch.setitem(sys.modules, local_package.__name__, local_package)
    assets_package = types.ModuleType(f"{local_package.__name__}.assets")
    assets_package.__path__ = [str(MODULE_ROOT / "assets")]
    monkeypatch.setitem(sys.modules, assets_package.__name__, assets_package)
    robot_constants = types.ModuleType(f"{assets_package.__name__}.robot")
    robot_file = MODULE_ROOT / "assets/robot.py"
    constants_ast = ast.parse(robot_file.read_text())
    constants_ast.body = [node for node in constants_ast.body if isinstance(node, ast.Assign)]
    robot_constants.__dict__.update(Path=Path, __file__=str(robot_file))
    exec(compile(constants_ast, str(robot_file), "exec"), robot_constants.__dict__)
    monkeypatch.setitem(sys.modules, robot_constants.__name__, robot_constants)
    package = types.ModuleType(f"{local_package.__name__}.mdp")
    package.__path__ = [str(MODULE_ROOT / "mdp")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    loaded = []

    def load(name):
        fullname = f"{package.__name__}.{name}"
        module = _module_from_file(fullname, MODULE_ROOT / "mdp" / f"{name}.py")
        loaded.append(fullname)
        return module

    yield load
    for name in loaded:
        sys.modules.pop(name, None)
    sys.modules.pop(f"{package.__name__}.sensors", None)


@pytest.fixture
def make_env():
    import torch

    def make(count=3):
        state = torch.zeros(count, 1, 13)
        state[:, 0, :3] = torch.tensor([-0.75, 0.0, 1.05])
        state[:, 0, 3] = 1.0
        data = SimpleNamespace(
            object_pos_w=state[..., :3],
            object_lin_vel_w=state[..., 7:10],
            object_link_state_w=state,
        )

        class Collection:
            num_objects = 1

            def __init__(self):
                self.data = data

            def write_object_link_state_to_sim(self, states, env_ids, object_ids):
                assert object_ids.tolist() == [0]
                self.data.object_link_state_w[env_ids] = states

        origins = torch.zeros(count, 3)
        robot_data = SimpleNamespace(
            joint_pos=torch.zeros(count, 18),
            default_joint_pos=torch.zeros(count, 18),
            joint_vel=torch.zeros(count, 18),
            default_joint_vel=torch.zeros(count, 18),
            root_pos_w=torch.zeros(count, 3),
            root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, -1).clone(),
        )
        frame_quat = robot_data.root_quat_w[:, None].clone()

        class Scene(dict):
            env_origins = origins

        scene = Scene(
            object_collection=Collection(),
            robot=SimpleNamespace(data=robot_data),
            ee_frame=SimpleNamespace(data=SimpleNamespace(
                target_pos_w=torch.zeros(count, 1, 3), target_quat_w=frame_quat.clone(),
            )),
            wrist_frame=SimpleNamespace(data=SimpleNamespace(target_pos_w=torch.zeros(count, 1, 3))),
            shelf=SimpleNamespace(data=SimpleNamespace(default_root_state=torch.cat(
                (torch.zeros(count, 3), robot_data.root_quat_w), dim=-1,
            ))),
        )
        env = SimpleNamespace(
            num_envs=count, device="cpu", scene=scene,
            target_id=torch.zeros(count, 1, dtype=torch.long),
            target_width=torch.full((count, 1), 0.04),
            sweep_dir=torch.tensor([0.0, 0.18, 0.0]).expand(count, -1).clone(),
            sensor_data_fresh=torch.ones(count, dtype=torch.bool),
            action_manager=SimpleNamespace(action=torch.zeros(count, 8), prev_action=torch.zeros(count, 8)),
        )
        goal = state[:, 0, :3].clone() + env.sweep_dir
        env.command_manager = SimpleNamespace(get_command=lambda name: goal)
        return env

    return make
