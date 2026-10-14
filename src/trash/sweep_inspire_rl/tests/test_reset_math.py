"""CPU regression tests for one-object sampling, palm geometry and reset IK.

Isaac Lab itself requires an already running SimulationApp. These tests load
the reset's torch-only helpers and live-FK solver against a small analytic
six-DOF articulation, so sampling/IK regressions remain testable without Kit.
"""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).resolve().parents[1] / "sweep_inspire_rl" / "mdp" / "events.py"


def quat_mul(first, second):
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2), dim=-1)


def rotation_quaternion(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = torch.where(angle > 1.0e-8, torch.sin(angle / 2) / angle.clamp_min(1.0e-12), torch.full_like(angle, 0.5))
    return torch.cat((torch.cos(angle / 2), vector * scale), dim=-1)


def pose_error(position, quaternion, desired_position, desired_quaternion, **_):
    inverse = quaternion.clone()
    inverse[:, 1:] *= -1
    error_q = quat_mul(desired_quaternion, inverse)
    error_q = torch.where(error_q[:, :1] >= 0, error_q, -error_q)
    norm = torch.linalg.vector_norm(error_q[:, 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, error_q[:, :1])
    return desired_position - position, error_q[:, 1:] * (angle / norm.clamp_min(1.0e-12))


def load_reset_namespace():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {
        "torch": torch, "math": math, "ManagerTermBase": object, "EventTermCfg": object,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name), "Articulation": object,
        "RigidObjectCollection": object,
        "ARM_JOINT_NAMES": ("pan", "lift", "elbow", "wrist_1", "wrist_2", "wrist_3_joint"),
        "HAND_BASE_BODY_NAME": "inspire_base_link", "ROBOT_CONTACT_BODY_NAMES": (),
        "RIGHT_PALM_QUAT_WXYZ": (math.sqrt(.5), 0., -math.sqrt(.5), 0.),
        "CONTROL_POINT_OFFSET_H": (0., .05, .10),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)

    def combine(position, quaternion, offset):
        rotation = namespace["_rotation_from_quaternion"](quaternion)
        return position + (rotation @ offset.unsqueeze(-1)).squeeze(-1), quaternion

    namespace["math_utils"] = SimpleNamespace(combine_frame_transforms=combine, compute_pose_error=pose_error)
    return namespace


NS = load_reset_namespace()


class FakeData:
    def __init__(self, robot):
        self.robot = robot
        self.soft_joint_pos_limits = torch.tensor([-3., 3.]).expand(len(robot.q), 6, 2).clone()

    @property
    def joint_pos(self):
        return self.robot.q

    @property
    def default_joint_pos(self):
        return torch.zeros_like(self.robot.q)

    @property
    def body_pos_w(self):
        return self.robot.q[:, None, :3].clone()

    @property
    def body_quat_w(self):
        return rotation_quaternion(self.robot.q[:, 3:])[:, None, :]


class FakeRobot:
    def __init__(self, count=2):
        self.q = torch.zeros(count, 6)
        self.data = FakeData(self)

    def write_joint_position_to_sim(self, q, env_ids):
        self.q[env_ids] = q

    def write_joint_state_to_sim(self, q, velocity, env_ids):
        self.q[env_ids] = q


def fake_reset():
    reset = NS["RightPalmReachingPoseReset"].__new__(NS["RightPalmReachingPoseReset"])
    reset.robot = FakeRobot()
    reset.arm_joint_ids = list(range(6))
    reset.hand_body_id = 0
    reset.offset = torch.tensor([0., .05, .10])
    return reset


class ResetMathTests(unittest.TestCase):
    def test_right_palm_axes_and_side_offset(self):
        positions = torch.tensor([[-.65, .10, 1.05], [-.75, -.20, 1.05]])
        desired, quaternion = NS["right_palm_pre_push_pose"](positions, torch.tensor([.06, .09]))
        rotation = NS["_rotation_from_quaternion"](quaternion)
        expected = torch.tensor([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]).expand(2, -1, -1)
        torch.testing.assert_close(rotation, expected)
        torch.testing.assert_close(desired - positions, torch.tensor([[-.02, -.10, .12], [-.02, -.13, .12]]))
        self.assertTrue(torch.all(desired[:, 1] < positions[:, 1]))

    def test_independent_source_slots_jitter_yaw_and_origins(self):
        poses = [[x, y, 1.05, 1., 0., 0., 0.] for x in (-.75, -.60) for y in (-.20, 0., .20)]
        origins = torch.arange(128 * 3, dtype=torch.float32).reshape(128, 3)
        torch.manual_seed(12)
        states = NS["sample_single_object_states"](poses, origins)
        self.assertEqual(tuple(states.shape), (128, 1, 13))
        local = states[:, 0, :3] - origins
        distances = torch.max(torch.abs(local[:, None, :2] - torch.tensor(poses)[None, :, :2]), dim=-1).values
        self.assertTrue(torch.all(torch.amin(distances, dim=1) <= .0201))
        self.assertGreater(torch.unique(torch.argmin(distances, dim=1)).numel(), 1)
        torch.testing.assert_close(local[:, 2], torch.full((128,), 1.05), atol=3.e-5, rtol=0)
        torch.testing.assert_close(torch.linalg.vector_norm(states[:, 0, 3:7], dim=-1), torch.ones(128))
        self.assertGreater(torch.unique(states[:, 0, 3:7], dim=0).shape[0], 120)
        self.assertTrue(torch.all(states[..., 7:] == 0))

    def test_spawn_changes_only_requested_environments_and_fixed_right_command(self):
        class Collection:
            num_objects = 1
            object_names = ["target"]

            def write_object_link_state_to_sim(self, state, env_ids):
                self.written_state = state
                self.written_ids = env_ids

        objects = Collection()
        class Scene(SimpleNamespace):
            def __getitem__(self, key):
                return objects

        scene = Scene(env_origins=torch.zeros(4, 3))
        env = SimpleNamespace(device="cpu", scene=scene, target_id=torch.full((4, 1), 8),
                              target_width=torch.full((4, 1), .9), sweep_dir=torch.full((4, 3), -.4))
        ids = torch.tensor([3, 1])
        NS["spawn_single_sweep_object"](env, ids, [[-.65, 0., 1.05, 1., 0., 0., 0.]], .06)
        torch.testing.assert_close(objects.written_ids, ids)
        self.assertEqual(tuple(objects.written_state.shape), (2, 1, 13))
        torch.testing.assert_close(env.sweep_dir[ids], torch.tensor([[0., .18, 0.], [0., .18, 0.]]))
        self.assertTrue(torch.all(env.target_id[ids] == 0))
        self.assertTrue(torch.all(env.target_id[[0, 2]] == 8))
        torch.testing.assert_close(env.sweep_dir[[0, 2]], torch.full((2, 3), -.4))

    def test_complete_rotated_extent_and_margin_are_respected(self):
        quaternion = torch.tensor(NS["RIGHT_PALM_QUAT_WXYZ"])
        minimum, maximum = NS["collision_aabbs"](torch.zeros(3), quaternion, torch.tensor([0., .05, .10]), torch.tensor([.04, .02, .10]))
        torch.testing.assert_close(minimum, torch.tensor([-.20, .03, -.04]))
        torch.testing.assert_close(maximum, torch.tensor([0., .07, .04]), atol=1.e-7, rtol=0)
        origin = torch.zeros(1, 3)
        extent = torch.ones(1, 3)
        far_min = torch.tensor([[1.003, 0., 0.]])
        self.assertFalse(bool(NS["aabbs_overlap"](origin, extent, far_min, far_min + 1, 0)))
        self.assertTrue(bool(NS["aabbs_overlap"](origin, extent, far_min, far_min + 1, .004)))

    def test_live_fk_jacobian_includes_wrist_rotation_and_tcp_offset(self):
        reset = fake_reset()
        jacobian = reset._control_point_jacobian(torch.tensor([0, 1]))
        expected = torch.eye(6)
        offset = reset.offset
        for index in range(3):
            expected[:3, index + 3] = torch.linalg.cross(torch.eye(3)[index], offset)
        torch.testing.assert_close(jacobian, expected.expand(2, -1, -1), atol=6.e-5, rtol=1.e-3)
        torch.testing.assert_close(reset.robot.q, torch.zeros(2, 6))
        self.assertGreater(float(torch.linalg.vector_norm(jacobian[:, :3, 5])), 0)

    def test_dls_converges_full_pose_without_advancing_physics(self):
        reset = fake_reset()
        desired_q = torch.tensor([[.20, -.10, .30, .20, -.15, .40], [-.10, .20, .10, -.10, .30, -.20]])
        reset.robot.q = desired_q.clone()
        desired_pos, desired_quat = reset._control_pose(torch.tensor([0, 1]))
        reset.robot.q.zero_()
        reset._solve_pose(torch.tensor([0, 1]), desired_pos, desired_quat, 80, .045, .65, .001, .005, .035)
        actual_pos, actual_quat = reset._control_pose(torch.tensor([0, 1]))
        position_error, rotation_error = pose_error(actual_pos, actual_quat, desired_pos, desired_quat)
        self.assertTrue(torch.all(torch.linalg.vector_norm(position_error, dim=-1) < .001))
        self.assertTrue(torch.all(torch.linalg.vector_norm(rotation_error, dim=-1) < .005))
        self.assertTrue(torch.all(torch.abs(reset.robot.q[:, 5]) > .1))

    def test_clearance_rejects_low_fingertip_and_target_overlap(self):
        reset = fake_reset()
        reset.safety_ids = [0, 1]
        reset.body_local_centers = torch.zeros(2, 3)
        reset.body_half_extents = torch.full((2, 3), .01)
        reset.object_local_center = torch.tensor([0., 0., .08])
        reset.object_half_extent = torch.tensor([.03, .03, .08])
        robot_position = torch.tensor([
            [[-.65, -.20, 1.20], [-.65, -.20, 1.10]],
            [[-.65, -.20, 1.20], [-.65, -.20, 1.04]],
            [[-.65, 0., 1.15], [-.65, -.20, 1.10]],
        ])
        quaternion = torch.tensor([1., 0., 0., 0.]).expand(3, 2, 4)
        reset.robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=robot_position, body_quat_w=quaternion))
        object_state = torch.zeros(3, 1, 13)
        object_state[:, 0, :3] = torch.tensor([-.65, 0., 1.05])
        object_state[:, 0, 3] = 1
        reset.objects = SimpleNamespace(data=SimpleNamespace(object_link_state_w=object_state))
        reset.shelf = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(3, 3)))
        env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(3, 3)))
        accepted = reset._clearance_mask(env, torch.arange(3), .004, 1.05, (-.88, -.52, -.5, .5))
        torch.testing.assert_close(accepted, torch.tensor([True, False, False]))

    def test_failed_ik_restores_reset_subset_and_raises(self):
        reset = fake_reset()
        reset.wrist_index = 5
        reset.robot.q[:] = .05
        original_q = reset.robot.q.clone()
        object_state = torch.zeros(2, 1, 13)
        object_state[:, 0, :3] = torch.tensor([-.65, 0., 1.05])
        reset.objects = SimpleNamespace(data=SimpleNamespace(object_link_state_w=object_state))
        reset._solve_pose = lambda *args: None
        env = SimpleNamespace(
            device="cpu", num_envs=2,
            target_width=torch.full((2, 1), .06),
            sweep_dir=torch.tensor([[0., .18, 0.], [0., .18, 0.]]),
            target_init_pos_w=torch.zeros(2, 3), desired_reaching_pose_w=torch.zeros(2, 7),
            reaching_ik_success=torch.zeros(2, dtype=torch.bool),
            reaching_ik_attempt_count=torch.zeros(2, dtype=torch.long),
            reset_collision_free=torch.zeros(2, dtype=torch.bool),
        )
        with self.assertRaisesRegex(RuntimeError, "Unable to initialize collision-safe right-palm"):
            reset(env, torch.tensor([1]), position_noise=0, joint_seed_offsets=((0., 0., 0., 0., 0., 0.),))
        torch.testing.assert_close(reset.robot.q, original_q)
        self.assertFalse(bool(env.reaching_ik_success[1]))
        self.assertEqual(int(env.reaching_ik_attempt_count[1]), 1)


if __name__ == "__main__":
    unittest.main()
