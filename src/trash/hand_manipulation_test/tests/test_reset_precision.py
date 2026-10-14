"""Cached IK writeback precision and measured-FK refinement budget."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hand_manipulation_test.reset_kinematics import (
    NumpySerialArmKinematics,
    SerialArmKinematics,
    bounded_dls_correction,
    cached_ik_convergence_tolerances,
)


def _quantized_boundary_fixture():
    # A reachable revolute endpoint in the same float32 world-coordinate
    # scale as the distant replicated environments. All six columns remain
    # valid DLS inputs; the pose can be corrected without translating the root.
    frame = torch.zeros(6, 7)
    frame[:, 3] = 1.
    axes = torch.zeros(6, 3)
    axes[:, 2] = 1.
    model = SerialArmKinematics(frame, frame.clone(), axes, torch.tensor([1., 0., 0., 1., 0., 0., 0.]))
    cpu = NumpySerialArmKinematics(model)
    count = 1024
    seed = np.zeros((count, 6))
    seed[:, 0] = np.linspace(0., 2. * np.pi, count, endpoint=False).astype(np.float32)
    root = np.tile(np.array([50., 50., .79505, 0., 0., 0., 1.], dtype=np.float32), (count, 1))
    goal = seed.copy()
    goal[:, 0] += .003
    position, quaternion = cpu.pose(goal, root)
    target = np.concatenate((position, quaternion), axis=-1).astype(np.float32).astype(np.float64)
    cached_position, _ = cpu.pose(seed, root)
    stored_position, _ = model.pose(torch.from_numpy(seed.astype(np.float32)), torch.from_numpy(root))
    cached_error = np.linalg.norm(cached_position - target[:, :3], axis=-1)
    stored_error = np.linalg.norm(stored_position.numpy() - target[:, :3], axis=-1)
    failures = np.flatnonzero((cached_error <= .003) & (stored_error > .003))
    assert len(failures), "Fixture must expose a cached/float32 stopping-boundary disagreement"
    selected = failures[:8]
    return model, cpu, seed[selected], root[selected], target[selected]


def _solve(cpu, seed, root, target, tolerances):
    lower = np.full_like(seed, -10.)
    upper = -lower
    return cpu.solve(seed, root, target, lower, upper, max_iterations=80, damping=.045,
        step_size=.65, position_error_step=.06, orientation_error_step=.25, delta_limit=.18,
        position_tolerance=tolerances[0], orientation_tolerance=tolerances[1])


def test_internal_margin_converges_inside_strict_outer_gate_after_float32_world_writeback():
    model, cpu, seed, root, target = _quantized_boundary_fixture()
    legacy_q, legacy_counts, _ = _solve(cpu, seed, root, target, (.003, .05))
    assert np.all(legacy_counts == 1)
    stored_legacy, _ = model.pose(torch.from_numpy(legacy_q.astype(np.float32)), torch.from_numpy(root))
    assert np.all(np.linalg.norm(stored_legacy.numpy() - target[:, :3], axis=-1) > .003)
    q, counts, _ = _solve(cpu, seed, root, target, cached_ik_convergence_tolerances(.003, .05))
    stored, quaternion = model.pose(torch.from_numpy(q.astype(np.float32)), torch.from_numpy(root))
    assert np.all(np.linalg.norm(stored.numpy() - target[:, :3], axis=-1) < .003)
    inverse = quaternion.numpy() * np.array([1., -1., -1., -1.])
    rotation_error = cpu.rotation_vector(cpu.multiply(target[:, 3:], inverse))
    assert np.all(np.linalg.norm(rotation_error, axis=-1) < .05)
    assert np.all((counts > legacy_counts) & (counts <= 80))


def _finish_cached_seed_method():
    # Execute the production method with a CPU articulation fixture. Loading
    # only this method keeps an Isaac Sim application out of pure unit tests.
    source = Path(__file__).parents[1] / "hand_manipulation_test/mdp/push_events.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PushSafePoseReset")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_finish_cached_seed")
    namespace = {"torch": torch, "bounded_dls_correction": bounded_dls_correction}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


class _StoredPoseFixture:
    def __init__(self):
        task = SimpleNamespace(reset_position_tolerance_m=.003, reset_orientation_tolerance_rad=.05,
            reset_max_iterations=80, reset_damping=.045, reset_step_size=.65,
            reset_position_error_step_m=.06, reset_orientation_error_step_rad=.25,
            reset_joint_delta_limit_rad=.18)
        self._env = SimpleNamespace(cfg=SimpleNamespace(task=task))
        self.arm_joint_ids = list(range(6))
        self.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(3, 18)))
        self.robot.data.joint_pos[:, 6:] = .95
        self.target_position = torch.tensor([[.0030002, 0., 0.], [.0030002, 0., 0.], [.0025, 0., 0.]])
        self.position_error_m = torch.zeros(3)
        self.orientation_error_rad = torch.zeros(3)
        # Global counters already include previous seeds; the available budget
        # must be determined by this seed's counts rather than the global sum.
        self.ik_iterations = torch.tensor([159, 160, 110])
        self.reset_io_counters = {"real_fk_refinements": 0}
        self.jacobian_rows = []
        self.write_rows = []

    def _pose_errors(self, ids):
        return self.target_position[ids] - self.robot.data.joint_pos[ids, :3], torch.zeros(len(ids), 3)

    def _numerical_control_jacobian(self, ids):
        self.jacobian_rows.append(ids.clone())
        return torch.eye(6).expand(len(ids), -1, -1)

    def _bounded_arm(self, positions, ids):
        return positions.clamp(-2.8, 2.8)

    def _write_reset_joint_state(self, positions, velocities, ids):
        assert torch.count_nonzero(velocities) == 0
        self.write_rows.append(ids.clone())
        self.robot.data.joint_pos[ids] = positions


def test_single_real_fk_refinement_respects_remaining_seed_budget_and_preserves_other_rows_and_hand():
    fixture = _StoredPoseFixture()
    saved = fixture.robot.data.joint_pos.clone()
    result = _finish_cached_seed_method()(fixture, torch.arange(3), torch.tensor([79, 80, 30]))
    assert result.tolist() == [True, False, True]
    assert len(fixture.jacobian_rows) == len(fixture.write_rows) == 1
    assert fixture.write_rows[0].tolist() == [0]
    torch.testing.assert_close(fixture.robot.data.joint_pos[1:], saved[1:])
    torch.testing.assert_close(fixture.robot.data.joint_pos[:, 6:], saved[:, 6:])
    assert fixture.ik_iterations.tolist() == [160, 160, 110]
    assert fixture.reset_io_counters["real_fk_refinements"] == 1
    assert fixture.position_error_m[0] < .003 and fixture.position_error_m[1] > .003


def test_accepted_stored_rows_never_enter_refinement_or_write_simulator_state():
    fixture = _StoredPoseFixture()
    fixture.target_position[:] = torch.tensor([.0025, 0., 0.])
    result = _finish_cached_seed_method()(fixture, torch.arange(3), torch.tensor([79, 80, 30]))
    assert bool(result.all())
    assert not fixture.jacobian_rows and not fixture.write_rows
    assert fixture.reset_io_counters["real_fk_refinements"] == 0


def test_real_fk_correction_caps_near_singular_steps_and_handles_zero_error():
    jacobian = .01 * torch.eye(6).expand(2, -1, -1)
    errors = torch.tensor([[100., 100., 100.], [0., 0., 0.]])
    delta = bounded_dls_correction(jacobian, errors, errors, damping=.045, step_size=.65,
        position_error_step=.06, orientation_error_step=.25, delta_limit=.18)
    assert bool(torch.isfinite(delta).all())
    assert bool((delta.abs() <= .18).all())
    torch.testing.assert_close(delta[0, 3:], torch.full((3,), .18))
    torch.testing.assert_close(delta[1], torch.zeros(6))
