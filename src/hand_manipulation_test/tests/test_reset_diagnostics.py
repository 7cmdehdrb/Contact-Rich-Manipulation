"""Run the real reset seed loop against scripted IK and collision outcomes."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"
TEST_PACKAGE = "_hand_manipulation_reset_diagnostics_fixture"


class _ManagerTerm:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


class _Scene(dict):
    pass


class _Robot:
    def __init__(self, count, action_math):
        self.joint_names = (*action_math.UR5E_ARM_JOINT_NAMES, *action_math.INSPIRE_HAND_JOINT_NAMES)
        self.body_names = ("forearm_link", "inspire_base_link", "wrist_3_link")
        root = torch.zeros(count, 13)
        root[:, 3] = 1.0
        joints = torch.zeros(count, 18)
        joints[:, 6:] = action_math.inspire_synergy_to_joint_positions(torch.full((count, 2), 0.94))
        body_position = torch.zeros(count, 3, 3)
        # A stable row tag lets the collision adapter recover the selected ids.
        body_position[:, 0, 0] = torch.arange(count)
        body_quaternion = torch.zeros(count, 3, 4)
        body_quaternion[:, :, 0] = 1.0
        self.data = SimpleNamespace(
            default_root_state=root, default_joint_pos=joints,
            joint_pos=joints.clone(), joint_vel=torch.zeros_like(joints),
            soft_joint_pos_limits=torch.tensor((-6.0, 6.0)).expand(count, 18, 2).clone(),
            body_link_pos_w=body_position, body_link_quat_w=body_quaternion,
        )
        self.root_pose = root[:, :7].clone()
        self.root_velocity = root[:, 7:].clone()
        self.position_targets = []

    def find_joints(self, names, preserve_order):
        assert preserve_order
        return [self.joint_names.index(name) for name in names], list(names)

    def find_bodies(self, name, preserve_order):
        assert preserve_order
        return [self.body_names.index(name)], [name]

    def write_root_pose_to_sim(self, pose, env_ids):
        self.root_pose[env_ids] = pose

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.root_velocity[env_ids] = velocity

    def write_joint_state_to_sim(self, position, velocity, env_ids):
        self.data.joint_pos[env_ids] = position
        self.data.joint_vel[env_ids] = velocity

    def set_joint_position_target(self, targets, env_ids):
        self.position_targets.append((env_ids.clone(), targets.clone()))

    def set_joint_velocity_target(self, targets, env_ids):
        assert not targets.any()

    def set_joint_effort_target(self, targets, env_ids):
        assert not targets.any()


class _CollisionBounds:
    def __init__(self, stage, path, body_names, device):
        self.body_names = body_names
        self.palm_face_h = torch.tensor((0.00045, 0.04772, 0.12023))
        self.lookup = None

    def overlaps(self, body_positions, body_quaternions, other_position, other_quaternion, size, margin):
        assert margin == 0.004
        ids = body_positions[:, 0, 0].long().tolist()
        source = "cube" if tuple(size) == (0.06, 0.06, 0.06) else "table"
        overlaps = torch.zeros(len(ids), len(self.body_names), dtype=torch.bool)
        for row, env_id in enumerate(ids):
            for body in self.lookup(env_id).get(source, ()):
                overlaps[row, self.body_names.index(body)] = True
        return overlaps


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backend():
    # Only simulator dependencies are replaced. The constructor, __call__,
    # collision-mask bookkeeping and public diagnostics execute real source.
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PACKAGE_ROOT)]
    mdp = ModuleType(f"{TEST_PACKAGE}.mdp")
    mdp.__path__ = [str(PACKAGE_ROOT / "mdp")]
    assets = ModuleType(f"{TEST_PACKAGE}.assets")
    assets.__path__ = []
    for module in (package, mdp, assets):
        sys.modules[module.__name__] = module
    action_math = _load(f"{TEST_PACKAGE}.action_math", "action_math.py")
    _load(f"{TEST_PACKAGE}.contact_math", "contact_math.py")
    robot = ModuleType(f"{TEST_PACKAGE}.assets.robot")
    robot.ARM_JOINT_NAMES = action_math.UR5E_ARM_JOINT_NAMES
    robot.HAND_JOINT_NAMES = action_math.INSPIRE_HAND_JOINT_NAMES
    bounds = ModuleType(f"{TEST_PACKAGE}.collision_geometry")
    bounds.RobotCollisionBounds = _CollisionBounds
    sys.modules[robot.__name__] = robot
    sys.modules[bounds.__name__] = bounds
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = _ManagerTerm
    sim_utils = ModuleType("isaaclab.sim.utils")
    sim_utils.get_current_stage = lambda: None
    utils = ModuleType("isaaclab.utils")
    utils.math = ModuleType("isaaclab.utils.math")
    with patch.dict(sys.modules, {
        "isaaclab": ModuleType("isaaclab"), managers.__name__: managers,
        "isaaclab.sim": ModuleType("isaaclab.sim"), sim_utils.__name__: sim_utils,
        utils.__name__: utils, utils.math.__name__: utils.math,
    }):
        events = _load(f"{TEST_PACKAGE}.mdp.contact_events", "mdp/contact_events.py")
    return action_math, events


def _environment(backend, scripts, count=3):
    action_math, events = backend
    scene = _Scene()
    scene.env_origins = torch.zeros(count, 3)
    if count > 1927:
        scene.env_origins[1927] = torch.tensor((-48.75, 37.5, 0.0))
    scene.env_prim_paths = [f"/World/envs/env_{index}" for index in range(count)]
    robot = _Robot(count, action_math)
    scene["robot"] = robot
    identity = torch.tensor((1.0, 0.0, 0.0, 0.0)).expand(count, -1).clone()
    for name, local in (("target_object", (-0.45, 0.0, 1.08)), ("table", (-0.5, 0.0, 1.03))):
        scene[name] = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor(local).expand(count, -1).clone() + scene.env_origins,
            root_quat_w=identity.clone(),
        ))
    task = SimpleNamespace(
        reset_joint_seed_offsets=((0.0,) * 6, (0.2,) + (0.0,) * 5, (0.4,) + (0.0,) * 5),
        initial_hand_open_range=(0.94, 0.94), reset_joint_limit_margin_rad=0.035,
        reset_wrist_3_range_rad=(-2.8, 2.8), reset_clearance_margin_m=0.004,
        reset_position_tolerance_m=0.003, reset_orientation_tolerance_rad=0.05,
        cube_size=0.06, table_size=(0.36, 1.0, 0.04),
    )
    env = SimpleNamespace(num_envs=count, device="cpu", scene=scene, cfg=SimpleNamespace(
        task=task, actions=SimpleNamespace(arm_action=SimpleNamespace(
            body_name="inspire_base_link", body_offset_pos=(0.0, 0.05, 0.1),
            body_offset_quat=(1.0, 0.0, 0.0, 0.0),
        )),
    ))

    class ScriptedReset(events.ContactSafePoseReset):
        def __init__(self):
            super().__init__(SimpleNamespace(), env)
            self.scripts = scripts
            self.current_seed = torch.full((count,), -1, dtype=torch.long)
            self.solve_calls = []
            self.collision_bounds.lookup = self.outcome

        def outcome(self, env_id):
            return self.scripts[env_id][int(self.current_seed[env_id])]

        def _sample_pose(self, env_ids):
            self.current_seed[env_ids] = -1
            self.desired_c_pos_w[env_ids] = env.scene["target_object"].data.root_pos_w[env_ids]
            self.desired_c_pos_w[env_ids, 1] -= 0.14
            self.desired_c_quat_w[env_ids] = torch.tensor((1.0, 0.0, 0.0, 0.0))

        def _solve_seed(self, env_ids, hand_targets):
            self.current_seed[env_ids] += 1
            self.solve_calls.append(tuple(env_ids.tolist()))
            converged = []
            for env_id in env_ids.tolist():
                outcome = self.outcome(env_id)
                self.position_error_m[env_id] = outcome["position"]
                self.orientation_error_rad[env_id] = outcome.get("rotation", 0.01)
                self.ik_iterations[env_id] += outcome.get("iterations", 1)
                self.robot.data.joint_pos[env_id, 0] = outcome.get("joint_tag", 0.15)
                converged.append(outcome.get("converged", True))
            return torch.tensor(converged)

        def _check_pose_and_limits(self, env_ids):
            self.palm_alignment_cos[env_ids] = 1.0
            return torch.tensor([self.outcome(env_id).get("pose_valid", True) for env_id in env_ids.tolist()])

    return env, ScriptedReset(), robot


def test_final_nonconvergence_does_not_report_an_earlier_seed_table_collision(backend):
    env_id = 1927
    scripts = {env_id: (
        {"position": 0.001, "table": ("forearm_link",)},
        {"position": 0.007, "rotation": 0.1, "converged": False},
        {"position": 0.003000014, "rotation": 0.02, "converged": False},
    )}
    env, reset, robot = _environment(backend, scripts, count=2048)
    with pytest.raises(RuntimeError, match="no episode started") as failure:
        reset(env, torch.tensor((env_id,)))
    assert reset.attempts[env_id].item() == 3
    assert reset.accepted_seed_index[env_id].item() == -1
    assert reset.obb_checks[env_id].item() == 1
    assert reset.position_error_m[env_id].item() == pytest.approx(0.003000014, abs=1e-10)
    assert reset.position_error_m[env_id].item() > env.cfg.task.reset_position_tolerance_m
    assert not reset.last_collision_checked[env_id]
    assert not reset.last_cube_overlap[env_id].any()
    assert not reset.last_table_overlap[env_id].any()
    assert math.isnan(reset.palm_alignment_cos[env_id].item())
    seeds = reset.seed_diagnostics(env_id)
    assert [seed["seed_index"] for seed in seeds] == [0, 1, 2]
    assert [seed["collision_checked"] for seed in seeds] == [True, False, False]
    assert [seed["pose_valid"] for seed in seeds] == [True, False, False]
    assert seeds[0]["table_overlap_bodies"] == ["forearm_link"]
    for seed in seeds[1:]:
        assert seed["table_overlap_bodies"] == seed["cube_overlap_bodies"] == []
    assert [seed["position_error_m"] for seed in seeds] == pytest.approx([0.001, 0.007, 0.003000014])
    latest_message = str(failure.value).split("'seeds':", 1)[0]
    assert "'collision_checked': False" in latest_message
    assert "'table_overlap_bodies': []" in latest_message
    assert "'cube_overlap_bodies': []" in latest_message
    assert "forearm_link" in str(failure.value)
    assert robot.position_targets == []


def test_partial_reset_preserves_other_rows_and_stops_each_row_at_its_first_valid_seed(backend):
    scripts = {
        0: ({"position": 0.001, "joint_tag": 0.15},),
        1: ({"position": 0.001, "table": ("forearm_link",)}, {"position": 0.002, "joint_tag": 0.35}),
        2: ({"position": 0.002, "cube": ("inspire_base_link",)}, {"position": 0.001, "joint_tag": 0.25}),
    }
    env, reset, robot = _environment(backend, scripts)
    reset(env, torch.tensor((1,)))
    buffers = (
        "attempts", "accepted_seed_index", "ik_iterations", "obb_checks",
        "position_error_m", "orientation_error_rad", "palm_alignment_cos",
        "last_collision_checked", "last_cube_overlap", "last_table_overlap",
        "seed_position_error_m", "seed_orientation_error_rad", "seed_pose_valid",
        "seed_collision_checked", "seed_cube_overlap", "seed_table_overlap",
        "initial_hand_synergy", "desired_c_pos_w", "desired_c_quat_w",
    )
    untouched = {name: getattr(reset, name)[1].clone() for name in buffers}
    untouched_joint = robot.data.joint_pos[1].clone()
    reset.solve_calls.clear()
    reset(env, torch.tensor((0, 2)))
    assert reset.solve_calls == [(0, 2), (2,)]
    assert reset.attempts.tolist() == [1, 2, 2]
    assert reset.accepted_seed_index.tolist() == [0, 1, 1]
    assert reset.obb_checks.tolist() == [1, 2, 2]
    assert len(reset.seed_diagnostics(0)) == 1
    assert len(reset.seed_diagnostics(2)) == 2
    assert reset.seed_diagnostics(2)[0]["cube_overlap_bodies"] == ["inspire_base_link"]
    assert reset.seed_diagnostics(2)[1]["cube_overlap_bodies"] == []
    assert torch.isnan(reset.seed_position_error_m[0, 1:]).all()
    assert not reset.seed_collision_checked[0, 1:].any()
    for name, saved in untouched.items():
        torch.testing.assert_close(getattr(reset, name)[1], saved, equal_nan=True)
    torch.testing.assert_close(robot.data.joint_pos[1], untouched_joint)
    torch.testing.assert_close(robot.data.joint_pos[:, 0], torch.tensor((0.15, 0.35, 0.25)))
    torch.testing.assert_close(robot.position_targets[-1][0], torch.tensor((0, 2)))


def test_new_reset_clears_previous_selected_row_seed_history(backend):
    scripts = {0: (
        {"position": 0.001, "table": ("forearm_link",)},
        {"position": 0.001, "joint_tag": 0.25},
    )}
    env, reset, _ = _environment(backend, scripts)
    reset(env, torch.tensor((0,)))
    assert reset.seed_table_overlap[0, 0].any()
    reset.scripts[0] = ({"position": 0.002, "joint_tag": 0.15},)
    reset(env, torch.tensor((0,)))
    assert reset.attempts[0].item() == 1
    assert reset.accepted_seed_index[0].item() == 0
    assert len(reset.seed_diagnostics(0)) == 1
    assert not reset.seed_table_overlap[0].any()
    assert not reset.seed_cube_overlap[0].any()
    assert torch.isnan(reset.seed_position_error_m[0, 1:]).all()
    assert torch.isnan(reset.seed_orientation_error_rad[0, 1:]).all()
    assert not reset.seed_pose_valid[0, 1:].any()
    assert not reset.seed_collision_checked[0, 1:].any()


def test_converged_ik_with_invalid_pose_is_not_accepted_even_when_collision_free(backend):
    scripts = {0: (
        {"position": 0.001, "pose_valid": False, "joint_tag": 0.15},
        {"position": 0.002, "pose_valid": True, "joint_tag": 0.25},
    )}
    env, reset, robot = _environment(backend, scripts)
    reset(env, torch.tensor((0,)))
    assert reset.accepted_seed_index[0].item() == 1
    assert reset.obb_checks[0].item() == 2
    seeds = reset.seed_diagnostics(0)
    assert [seed["pose_valid"] for seed in seeds] == [False, True]
    assert all(seed["collision_checked"] for seed in seeds)
    assert all(not seed["cube_overlap_bodies"] and not seed["table_overlap_bodies"] for seed in seeds)
    assert robot.data.joint_pos[0, 0].item() == pytest.approx(0.25)
