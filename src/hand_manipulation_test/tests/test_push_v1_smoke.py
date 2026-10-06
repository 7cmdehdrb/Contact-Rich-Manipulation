"""Check smoke CLI and independent frame/servo computations without Kit."""

import importlib.util
from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from hand_manipulation_test import action_math


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).parents[1] / "scripts/smoke_push_v1_env.py"
    spec = importlib.util.spec_from_file_location("_push_v1_smoke_script_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _AppArguments:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device")
        parser.add_argument("--verbose", action="store_true")


def test_smoke_parser_accepts_delegated_v1_contract_and_physical_modes(script):
    parser = script.build_parser(_AppArguments)
    args = parser.parse_args(["--task", "Isaac-Hand-Manipulation-Push-v1", "--headless",
                              "--num-envs", "4", "--skip-physical-fixture", "--check-timeout"])
    script.validate_args(args)
    assert args.num_envs == 4 and args.skip_physical_fixture and args.check_timeout
    args = parser.parse_args(["--physical-only", "--physical-steps", "350"])
    script.validate_args(args)
    assert args.physical_only and not args.skip_physical_fixture
    args = parser.parse_args(["--physical-only", "--physical-goal", "--physical-goal-distance", ".30",
                              "--physical-angle-offset-deg", "-10", "--fixture-verbose"])
    script.validate_args(args)
    assert args.physical_goal and args.physical_goal_distance == .30 and args.fixture_verbose
    assert args.physical_angle_offset_deg == -10.
    assert not args.verbose
    with pytest.raises(ValueError, match="cannot be combined"):
        script.validate_args(parser.parse_args(["--physical-only", "--skip-physical-fixture"]))
    with pytest.raises(SystemExit):
        parser.parse_args(["--task", "Isaac-Hand-Manipulation-Push-v0"])
    with pytest.raises(ValueError, match="physical-goal-distance"):
        script.validate_args(parser.parse_args(["--physical-goal-distance", ".4"]))
    with pytest.raises(ValueError, match="physical-angle-offset"):
        script.validate_args(parser.parse_args(["--physical-angle-offset-deg", "11"]))


@pytest.mark.parametrize("offset_deg", (-10., 0., 10.))
def test_deterministic_physical_commands_use_actual_base_direction_and_task_path_center(script, offset_deg):
    smoke = script.PushV1Smoke.__new__(script.PushV1Smoke)
    smoke.torch = torch
    smoke.env = SimpleNamespace(num_envs=4, device="cpu")
    smoke.ids = torch.arange(4)
    task = SimpleNamespace(table_center=(-.9, .15, 1.13), command_midpoint_x_offset_m=.31,
        object_xy_range_low=(-.62, .12), object_xy_range_high=(-.56, .18), cube_center_height_m=1.18,
        command_distance_range=(.2, .3), command_angle_jitter_rad=math.radians(10), command_centered_path=True)
    smoke.cfg = SimpleNamespace(task=task)
    yaw = torch.tensor([math.pi, math.pi+.02, math.pi-.01, math.pi])
    rotation = torch.zeros(4, 3)
    rotation[:, 2] = yaw
    quaternion = action_math.quaternion_from_rotation_vector(rotation)
    smoke.geometry = SimpleNamespace(base_link_pose_w=lambda env: (torch.zeros(4, 3), quaternion))
    smoke.math = SimpleNamespace(quat_apply=action_math.quaternion_rotate)
    calls = []
    smoke.command = SimpleNamespace(set_reset_specs=lambda *values: calls.append(values))
    smoke.select_both_sides(.3, math.radians(offset_deg))
    ids, positions, angles, distance = calls[0]
    torch.testing.assert_close(ids, smoke.ids)
    expected_angles = torch.tensor([0., math.pi, 0., math.pi])+math.radians(offset_deg)
    torch.testing.assert_close(angles, expected_angles)
    world_dx = torch.sin(expected_angles+yaw)
    torch.testing.assert_close(positions[:, 0], -.9+.31-.15*world_dx, atol=1e-7, rtol=0.)
    torch.testing.assert_close(positions[:, 1], torch.full((4,), .15))
    torch.testing.assert_close(positions[:, 2], torch.full((4,), 1.18))
    assert distance == .3
    with pytest.raises(ValueError, match="configured command band"):
        smoke.select_both_sides(.3, math.radians(11.))


def test_boundary_fixtures_cover_actual_conditional_sampler_without_leaving_table(script):
    task = SimpleNamespace(table_center=(-.75, 0., 1.03), table_size=(.36, 1., .04),
                           object_xy_range_low=(-.73, -.03), object_xy_range_high=(-.67, .03),
                           cube_size=.06, table_path_margin_m=.01, command_centered_path=True,
                           command_midpoint_x_offset_m=.05, command_initial_x_jitter_m=.002)
    padding = math.sqrt(3)*.03 + .01
    table_low = (-.75-.18+padding, -.5+padding)
    table_high = (-.75+.18-padding, .5-padding)
    specs = []
    for angle_deg in (-10., 0., 10., 170., 180., 190.):
        angle = math.radians(angle_deg)
        direction = (-math.sin(angle), math.cos(angle), 0.)
        low, high = script.sampled_start_bounds(task, direction, .3)
        assert all(low[i] <= high[i] for i in range(2))
        midpoint_x = task.table_center[0]+task.command_midpoint_x_offset_m-.15*direction[0]
        assert low[0] == pytest.approx(midpoint_x-.002)
        assert high[0] == pytest.approx(midpoint_x+.002)
        for start in product((low[0], high[0]), (low[1], high[1])):
            specs.append(start)
            goal = tuple(start[i]+.3*direction[i] for i in range(2))
            for point in (start, goal):
                assert all(table_low[i]-1e-12 <= point[i] <= table_high[i]+1e-12 for i in range(2))
            assert all(task.object_xy_range_low[i] <= start[i] <= task.object_xy_range_high[i] for i in range(2))
    assert len(specs) == 24
    # Broad bbox corners are accepted by the diagnostic hook but are not
    # emitted by the angle/length-conditioned production sampler.
    low, high = script.sampled_start_bounds(task, (0., -1., 0.), .3)
    assert (low[0], high[0]) == pytest.approx((-.702, -.698))
    with pytest.raises(ValueError, match="no complete Table path"):
        script.valid_start_bounds(task, (1., 0., 0.), 2.)
    task.command_initial_x_jitter_m = .1
    with pytest.raises(ValueError, match="conditional sampler domain"):
        script.sampled_start_bounds(task, (0., 1., 0.), .3)


def _frame_fixture(script):
    smoke = script.PushV1Smoke.__new__(script.PushV1Smoke)
    dtype = torch.float64
    smoke.torch = torch
    smoke.math = SimpleNamespace(quat_apply=action_math.quaternion_rotate,
                                 quat_apply_inverse=lambda q, v: action_math.quaternion_rotate(action_math.quaternion_conjugate(q), v),
                                 quat_conjugate=action_math.quaternion_conjugate,
                                 quat_mul=action_math.quaternion_multiply)
    root_position = torch.tensor([[50., -48.75, .79505]], dtype=dtype)
    root_quaternion = action_math.quaternion_from_rotation_vector(torch.tensor([[.1, -.2, 2.3]], dtype=dtype))
    position_b = torch.tensor([[-.35, .2, .4]], dtype=dtype)
    position = root_position + action_math.quaternion_rotate(root_quaternion, position_b)
    relative_q = action_math.quaternion_from_rotation_vector(torch.tensor([[.7, -.3, .5]], dtype=dtype))
    quaternion = action_math.quaternion_multiply(root_quaternion, relative_q)
    smoke.robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_position, root_link_quat_w=root_quaternion))
    smoke.geometry = SimpleNamespace(control_point_pose_w=lambda env: (position, quaternion))
    smoke.env = SimpleNamespace(num_envs=1, device="cpu")
    desired = torch.cat((position_b, relative_q), -1)
    smoke.arm = SimpleNamespace(desired_c_pose_b=desired.clone(), reference_c_quat_b=relative_q.clone())
    return smoke, position, quaternion, root_position, root_quaternion


def test_smoke_target_error_crosscheck_uses_actual_c_axes_and_unscaled_metres(script):
    smoke, position, quaternion, root_position, root_quaternion = _frame_fixture(script)
    expected = position.new_tensor([[.031, -.022, .055]])
    target_w = position + action_math.quaternion_rotate(quaternion, expected)
    smoke.arm.desired_c_pose_b[:, :3] = smoke.math.quat_apply_inverse(root_quaternion, target_w-root_position)
    torch.testing.assert_close(smoke.expected_target_error(), expected, atol=2e-14, rtol=0.)


def test_scripted_servo_updates_persistent_target_in_current_c_axes_with_hand_offset_once(script):
    smoke, position, quaternion, root_position, root_quaternion = _frame_fixture(script)
    increment = position.new_tensor([[.006, -.003, .004]])
    old_error = position.new_tensor([[.012, 0., 0.]])
    persistent_w = position + action_math.quaternion_rotate(quaternion, old_error)
    smoke.arm.desired_c_pose_b[:, :3] = smoke.math.quat_apply_inverse(root_quaternion, persistent_w-root_position)
    desired_c = persistent_w + action_math.quaternion_rotate(quaternion, increment)
    smoke.safe = SimpleNamespace(
        c_quat_h=action_math.quaternion_from_rotation_vector(position.new_tensor([[.15, 0., 0.]])),
        c_offset_h=position.new_tensor([0., .05, .10]), palm_reference_h=position.new_tensor([.00045, .047734, .12023]))
    hand_quaternion = action_math.quaternion_multiply(quaternion, action_math.quaternion_conjugate(smoke.safe.c_quat_h))
    desired_palm = desired_c-action_math.quaternion_rotate(hand_quaternion, (smoke.safe.c_offset_h-smoke.safe.palm_reference_h)[None])
    smoke.hold_action = lambda: position.new_tensor([[0., 0., 0., 0., 0., 0., .3, .7]])
    smoke.cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(
        translation_scale=(.012, .012, .008), rotation_scale=(.06, .06, .06))))
    smoke.math.axis_angle_from_quat = lambda q: torch.zeros(1, 3, dtype=position.dtype)
    action = script.OscPushFixture(smoke).action_toward(desired_palm, quaternion)
    torch.testing.assert_close(action, position.new_tensor([[.5, -.25, .5, 0., 0., 0., .3, .7]]), atol=1e-12, rtol=0.)


def test_scripted_rotation_action_uses_reset_reference_despite_actual_orientation_drift(script):
    smoke, position, quaternion, _, root_quaternion = _frame_fixture(script)
    smoke.safe = SimpleNamespace(c_quat_h=quaternion.new_tensor([[1., 0., 0., 0.]]),
                                 c_offset_h=position.new_zeros(3), palm_reference_h=position.new_zeros(3))
    smoke.cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(
        translation_scale=(.012,)*3, rotation_scale=(.06,)*3)))
    smoke.hold_action = lambda: position.new_zeros(1, 8)

    def rotation_vector(q):
        vector_length = q[:, 1:].norm(dim=-1, keepdim=True)
        return q[:, 1:] * (2*torch.atan2(vector_length, q[:, :1])/vector_length)

    smoke.math.axis_angle_from_quat = rotation_vector
    reset_world = action_math.quaternion_multiply(root_quaternion, smoke.arm.reference_c_quat_b)
    residual = position.new_tensor([[.018, -.012, .030]])
    desired = action_math.quaternion_multiply(reset_world, action_math.quaternion_from_rotation_vector(residual))
    fixture = script.OscPushFixture(smoke)
    first = fixture.action_toward(position, desired)
    actual_drift = action_math.quaternion_multiply(quaternion,
        action_math.quaternion_from_rotation_vector(position.new_tensor([[.3, -.2, .1]])))
    smoke.geometry.control_point_pose_w = lambda env: (position, actual_drift)
    second = fixture.action_toward(position, desired)
    torch.testing.assert_close(first[:, 3:6], residual/.06, atol=2e-14, rtol=0.)
    torch.testing.assert_close(second[:, 3:6], first[:, 3:6], atol=0., rtol=0.)


def test_scripted_reference_to_c_target_uses_live_thumb_offset_after_joint_deflection(script):
    smoke, position, quaternion, _, _ = _frame_fixture(script)
    live_offset = position.new_tensor([[.012, .047, .13]])
    smoke.safe = SimpleNamespace(c_quat_h=quaternion.new_tensor([[1., 0., 0., 0.]]),
        c_offset_h=position.new_tensor([0., .05, .10]), palm_reference_h=position.new_tensor([0., .05, .12]),
        contact_reference_offset_h=lambda: live_offset)
    smoke.cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(
        translation_scale=(.012,)*3, rotation_scale=(.06,)*3)))
    smoke.hold_action = lambda: position.new_tensor([[0., 0., 0., 0., 0., 0., .3, .7]])
    smoke.math.axis_angle_from_quat = lambda q: position.new_zeros(1, 3)
    desired_contact = position-action_math.quaternion_rotate(quaternion, smoke.safe.c_offset_h-live_offset)
    fixture = script.OscPushFixture(smoke)
    torch.testing.assert_close(fixture.action_toward(desired_contact, quaternion)[:, :3], position.new_zeros(1, 3), atol=2.e-12, rtol=0.)
    live_offset[:, 0] += .006
    action = fixture.action_toward(desired_contact, quaternion)
    torch.testing.assert_close(action[:, :3], position.new_tensor([[-.5, 0., 0.]]), atol=2.e-12, rtol=0.)
    torch.testing.assert_close(action[:, 6:], position.new_tensor([[.3, .7]]), atol=0., rtol=0.)


def test_physical_planner_reads_optional_actual_contact_reference_and_retains_central_fallback(script):
    central, thumb = torch.tensor([[0., 0., .1]]), torch.tensor([[.01, .02, .025]])
    quaternion = torch.tensor([[1., 0., 0., 0.]])
    safe = SimpleNamespace(palm_reference_pose_w=lambda: (central, quaternion))
    fixture = script.OscPushFixture(SimpleNamespace(safe=safe))
    assert fixture.contact_pose()[0] is central
    safe.contact_reference_pose_w = lambda: (thumb, quaternion)
    assert fixture.contact_pose()[0] is thumb


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("origin", ((0., 0., 0.), (-48.75, 37.5, .79505)))
def test_approach_start_centers_actual_pad_tangentially_preserves_each_normal_gap_and_inputs(script, dtype, origin):
    # Execute only the pure production method with measured point tensors;
    # there is no arm model, simulation state writer, or synthetic force.
    angles = torch.tensor((-10., 0., 10., 170., 180., 190.), dtype=dtype)*math.pi/180
    direction = torch.stack((-angles.sin(), angles.cos(), torch.zeros_like(angles)), -1)
    tangent = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(angles)), -1)
    initial = torch.tensor((-.70, 0., 1.08), dtype=dtype).expand(6, -1).clone()
    initial += initial.new_tensor(origin)
    gaps = initial.new_tensor((.14, .09, .12, .08, .16, .11))
    tangent_offsets = initial.new_tensor((.018, -.025, .012, -.019, .026, -.021))
    measured = initial-gaps[:, None]*direction+tangent_offsets[:, None]*tangent
    measured[:, 2] += .10
    before_initial, before_measured, before_direction = initial.clone(), measured.clone(), direction.clone()
    quaternion = initial.new_tensor((1., 0., 0., 0.)).expand(6, -1)
    fixture = SimpleNamespace(contact_pose=lambda: (measured, quaternion),
        smoke=SimpleNamespace(cfg=SimpleNamespace(task=SimpleNamespace(contact_palm_height_m=.025))))
    start = script.OscPushFixture.approach_start(fixture, initial, direction)
    tolerance = 7.e-6 if dtype == torch.float32 else 1.e-13
    torch.testing.assert_close(((start-initial)*direction).sum(-1), ((measured-initial)*direction).sum(-1), atol=tolerance, rtol=0.)
    torch.testing.assert_close(((start-initial)*tangent).sum(-1), initial.new_zeros(6), atol=tolerance, rtol=0.)
    torch.testing.assert_close(start[:, 2], initial[:, 2]+.025, atol=0., rtol=0.)
    assert float(((before_measured-initial)*tangent).sum(-1).abs().min()) > .011
    torch.testing.assert_close(initial, before_initial, atol=0., rtol=0.)
    torch.testing.assert_close(measured, before_measured, atol=0., rtol=0.)
    torch.testing.assert_close(direction, before_direction, atol=0., rtol=0.)
    assert start.data_ptr() != measured.data_ptr()


def test_goal_planner_lowers_then_tracks_and_releases_at_measured_palm_without_changing_cube(script):
    initial = torch.tensor([[-.48, 0., 1.08], [-.48, 0., 1.08]], dtype=torch.float64)
    direction = torch.tensor([[0., 1., 0.], [0., -1., 0.]], dtype=initial.dtype)
    goal = initial+.2*direction
    start = initial-.1*direction+initial.new_tensor([0., 0., .025])
    planner = script.GoalPushPlanner(torch, initial, goal, direction, start)
    cube = initial.clone()
    high_palm = start+start.new_tensor([0., 0., .03])
    active = torch.tensor([True, False])
    torch.testing.assert_close(planner.target(high_palm, cube, active), start)
    torch.testing.assert_close(planner.advance, initial.new_zeros(2))

    cube[0] += cube.new_tensor([.01, .05, 0.])
    observed_cube = cube.clone()
    expected = start.clone()
    expected[0] += start.new_tensor([.01, .0012, 0.])
    torch.testing.assert_close(planner.target(start, cube, active), expected)
    torch.testing.assert_close(cube, observed_cube)
    assert not bool(planner.released.any())
    # Release is relative to the measured palm, not the commanded target:
    # this removes any accumulated pressure even when tracking lags.
    cube[0] = goal[0]-.007*direction[0]
    actual = start+.1*direction+start.new_tensor([.003, 0., .001])
    release = planner.target(actual, cube, active)
    torch.testing.assert_close(release[0], actual[0]-.006*direction[0])
    assert planner.released.tolist() == [True, False]
    previous_advance = planner.advance.clone()
    later = planner.target(actual+.04*direction, initial, active)
    torch.testing.assert_close(later[0], release[0])
    torch.testing.assert_close(planner.advance, previous_advance)


def test_terminal_recorder_keeps_pre_reset_physical_copies_and_restores_hook(script):
    @dataclass
    class State:
        success: torch.Tensor
        failure: torch.Tensor
        settled_time_s: torch.Tensor

    state = State(torch.tensor([False, True]), torch.tensor([False, False]), torch.tensor([0., .2]))
    cube_position = torch.tensor([[-.48, 0., 1.08], [-.48, -.2, 1.08]])
    goal = cube_position.clone()
    initial = cube_position.clone()+torch.tensor([0., .2, 0.])
    calls = []
    original = lambda sample: calls.append(sample)
    command = SimpleNamespace(_record_metrics=original, target_pos_w=initial, goal_pos_w=goal)
    smoke = SimpleNamespace(env=SimpleNamespace(num_envs=2), command=command,
        cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=cube_position, root_lin_vel_w=torch.zeros(2, 3))),
        counters=lambda: {"ik_iterations": torch.tensor([12, 17])})
    with script.TerminalPhysicalRecorder(smoke) as recorder:
        command._record_metrics(state)
        captured = recorder.records[1]
        before = cube_position[1].clone()
        cube_position[:] = 0.
        initial[:] = 5.
        goal[:] = 6.
        state.success[:] = False
        state.settled_time_s[:] = 0.
        command._record_metrics(state)
        assert recorder.records[0] is None
        assert recorder.records[1] is captured
        torch.testing.assert_close(captured["cube_position_w"], before)
        assert bool(captured["state"]["success"])
        assert float(captured["state"]["settled_time_s"]) == pytest.approx(.2)
        assert int(captured["counter"]["ik_iterations"]) == 17
        torch.testing.assert_close(captured["goal_w"], before)
    assert command._record_metrics is original and len(calls) == 2
    with pytest.raises(RuntimeError, match="probe error"):
        with script.TerminalPhysicalRecorder(smoke):
            raise RuntimeError("probe error")
    assert command._record_metrics is original
