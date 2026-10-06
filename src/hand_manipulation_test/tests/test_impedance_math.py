"""Physical response checks for the pure, finite-timestep impedance probe."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


from hand_manipulation_test.impedance_math import (
    cube_reaction_wrench_b,
    implicit_cartesian_impedance_efforts as efforts,
)
from hand_manipulation_test.action_math import quaternion_rotate


def _plant(*, dtype=torch.float64, count=1):
    masses = torch.tensor((1., 1., 1., .001, .001, .001), dtype=dtype)
    jacobian = torch.eye(6, dtype=dtype).expand(count, -1, -1).clone()
    mass = torch.diag(masses).expand(count, -1, -1).clone()
    stiffness = torch.full((count, 6), 200., dtype=dtype)
    damping = torch.ones_like(stiffness)
    return jacobian, mass, stiffness, damping


def _rollout(jacobian, mass, target, stiffness, damping, *, external=None, gravity=None,
             measured_wrench=None, steps=400):
    position = torch.zeros_like(target)
    velocity = torch.zeros_like(target)
    history = []
    external = torch.zeros_like(target) if external is None else external
    for _ in range(steps):
        error = target - (jacobian @ position.unsqueeze(-1)).squeeze(-1)
        twist = (jacobian @ velocity.unsqueeze(-1)).squeeze(-1)
        torque = efforts(jacobian, mass, error, twist, stiffness, damping, .01, gravity,
                         external_wrench=measured_wrench)
        acceleration = torch.linalg.solve(mass, (torque + external).unsqueeze(-1)).squeeze(-1)
        velocity += .01 * acceleration
        position += .01 * velocity
        history.append(position.clone())
    return position, velocity, torque, torch.stack(history)


def test_tiny_rotational_inertia_settles_without_explicit_spring_overshoot():
    jacobian, mass, stiffness, damping = _plant()
    target = torch.zeros(1, 6, dtype=torch.float64)
    target[:, 4] = .10
    position, velocity, _, history = _rollout(jacobian, mass, target, stiffness, damping)
    assert torch.isfinite(history).all()
    assert history[..., 4].abs().max() <= .100001
    torch.testing.assert_close(position, target, atol=1e-8, rtol=0)
    assert velocity.abs().max() < 1e-8
    # An ordinary explicit spring at the same gain/10ms timestep moves 2rad
    # on its first update for this 0.1rad target and 0.001kg m^2 inertia.
    explicit_first_position = .01**2 * 200 * .10 / .001
    assert explicit_first_position > 10 * target[0, 4]
    assert history[0, 0, 4] < target[0, 4]


@pytest.mark.parametrize("moment", (-.02, .02))
def test_constant_external_moment_balances_at_the_softened_static_equilibrium(moment):
    jacobian, mass, stiffness, damping = _plant()
    target = torch.zeros(1, 6, dtype=torch.float64)
    external = torch.zeros_like(target)
    external[:, 5] = moment
    position, velocity, torque, _ = _rollout(jacobian, mass, target, stiffness, damping, external=external)
    # The mechanical equilibrium is measured by integrating the plant,
    # including the external moment. Configured K=200 is not its DC gain.
    nominal_inertia = .001
    nominal_damping = 2 * math.sqrt(nominal_inertia * 200)
    effective_gain = nominal_inertia * 200 / (nominal_inertia + .01*nominal_damping + .01**2*200)
    expected_deflection = moment / effective_gain
    assert position[0, 5] == pytest.approx(expected_deflection, rel=2e-6, abs=1e-10)
    assert abs(position[0, 5]) < .0031
    assert abs(position[0, 5]) > 20 * abs(moment / 200)
    assert velocity.abs().max() < 1e-9
    torch.testing.assert_close(torque, -external, atol=1e-9, rtol=0)


def test_coupled_joint_mass_and_point_jacobian_converge_to_the_requested_pose():
    jacobian, mass, stiffness, damping = _plant()
    # A nonorthogonal point Jacobian and SPD mass couple translation and
    # rotation; independent diagonal-axis checks cannot cover this plant.
    jacobian[0, 0, 4] = .13
    jacobian[0, 1, 3] = -.09
    factor = torch.linalg.cholesky(mass[0])
    factor[1, 0] = .15
    factor[3, 0] = .003
    factor[4, 2] = -.004
    mass[0] = factor @ factor.T
    target = torch.tensor(((.005, -.003, .004, .015, -.010, .012),), dtype=torch.float64)
    position, velocity, _, history = _rollout(jacobian, mass, target, stiffness, damping)
    actual = (jacobian @ position.unsqueeze(-1)).squeeze(-1)
    assert torch.isfinite(history).all()
    torch.testing.assert_close(actual, target, atol=2e-7, rtol=0)
    assert velocity.abs().max() < 1e-6


def test_rotating_an_isotropic_task_frame_preserves_physical_joint_efforts():
    jacobian, mass, stiffness, damping = _plant()
    theta = math.pi / 3
    rotation = torch.tensor(((math.cos(theta), -math.sin(theta), 0.),
                             (math.sin(theta), math.cos(theta), 0.), (0., 0., 1.)), dtype=torch.float64)
    transform = torch.block_diag(rotation, rotation).unsqueeze(0)
    error = torch.tensor(((.03, -.02, .01, .05, -.01, .02),), dtype=torch.float64)
    twist = torch.tensor(((.2, -.1, .3, -.03, .04, .01),), dtype=torch.float64)
    reference = efforts(jacobian, mass, error, twist, stiffness, damping, .01)
    rotated = efforts(transform @ jacobian, mass, (transform @ error.unsqueeze(-1)).squeeze(-1),
                      (transform @ twist.unsqueeze(-1)).squeeze(-1), stiffness, damping, .01)
    torch.testing.assert_close(rotated, reference, atol=1e-10, rtol=1e-10)


def test_selected_batch_rows_are_independent_and_inputs_are_not_mutated():
    jacobian, mass, stiffness, damping = _plant(dtype=torch.float32, count=4)
    mass *= torch.tensor((.5, 1., 2., 3.))[:, None, None]
    error = torch.arange(24, dtype=torch.float32).reshape(4, 6) * .001
    twist = torch.flip(error, dims=(0,))
    before = [v.clone() for v in (jacobian, mass, error, twist, stiffness, damping)]
    full = efforts(jacobian, mass, error, twist, stiffness, damping, .01)
    selected = torch.tensor((3, 1))
    partial = efforts(*(v[selected] for v in before), .01)
    torch.testing.assert_close(partial, full[selected])
    for actual, original in zip((jacobian, mass, error, twist, stiffness, damping), before):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


def test_singular_or_nearly_singular_axes_return_finite_bounded_efforts():
    jacobian, mass, stiffness, damping = _plant(dtype=torch.float32, count=2)
    jacobian[1] = torch.diag(torch.tensor((1., 1., 1., 1e-7, 0., 1e-7)))
    error = torch.full((2, 6), .02)
    torque = efforts(jacobian, mass, error, torch.zeros_like(error), stiffness, damping, .01)
    assert torch.isfinite(torque).all()
    assert torque.abs().max() <= 4.01
    assert torque[1, 4] == 0


def test_gravity_compensation_holds_a_stationary_pose_without_feedback_error():
    jacobian, mass, stiffness, damping = _plant()
    target = torch.zeros(1, 6, dtype=torch.float64)
    gravity = torch.tensor(((.3, -.2, .1, .001, -.002, .001),), dtype=torch.float64)
    position, velocity, torque, _ = _rollout(jacobian, mass, target, stiffness, damping,
                                          gravity=gravity, external=-gravity, steps=50)
    torch.testing.assert_close(position, target, atol=0, rtol=0)
    torch.testing.assert_close(velocity, target, atol=0, rtol=0)
    torch.testing.assert_close(torque, gravity, atol=0, rtol=0)


@pytest.mark.parametrize("moment", (-.02, .02))
def test_measured_external_moment_recovers_configured_static_gain_and_correct_sign(moment):
    jacobian, mass, stiffness, damping = _plant()
    target = torch.zeros(1, 6, dtype=torch.float64)
    external = torch.zeros_like(target)
    external[:, 5] = moment
    position, velocity, torque, history = _rollout(
        jacobian, mass, target, stiffness, damping, external=external, measured_wrench=external)
    # Integrate the actual plant with the same known external moment. The
    # resulting error is -F/K, rather than the softened unknown-load result.
    expected_error = -moment/200
    assert (target-position)[0, 5] == pytest.approx(expected_error, abs=1e-12)
    assert position[0, 5] == pytest.approx(moment/200, abs=1e-12)
    assert torch.isfinite(history).all()
    assert history[..., 5].abs().max() <= 1.01 * abs(moment/200)
    assert velocity.abs().max() < 1e-10
    torch.testing.assert_close(torque, -external, atol=1e-11, rtol=0)


def test_known_contact_wrench_and_gravity_balance_in_a_coupled_point_frame():
    jacobian, mass, stiffness, damping = _plant()
    jacobian[0, 0, 4] = .13
    jacobian[0, 1, 3] = -.09
    contact = torch.tensor(((.3, -.2, .1, .02, -.015, .01),), dtype=torch.float64)
    gravity = torch.tensor(((.1, -.4, .2, .003, -.002, .001),), dtype=torch.float64)
    # q=0 already has the correct deflection from the desired task pose.
    # Contact acts on the robot; compensated joint gravity acts oppositely.
    target = -contact/stiffness
    joint_contact = (jacobian.mT @ contact.unsqueeze(-1)).squeeze(-1)
    position, velocity, torque, _ = _rollout(
        jacobian, mass, target, stiffness, damping,
        external=joint_contact-gravity, gravity=gravity, measured_wrench=contact, steps=50)
    torch.testing.assert_close(position, torch.zeros_like(position), atol=1e-14, rtol=0)
    torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=1e-13, rtol=0)
    torch.testing.assert_close(torque, gravity-joint_contact, atol=1e-13, rtol=0)


def test_rotating_known_contact_wrench_preserves_the_same_physical_command():
    jacobian, mass, stiffness, damping = _plant()
    rotation = torch.tensor(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)), dtype=torch.float64)
    transform = torch.block_diag(rotation, rotation).unsqueeze(0)
    contact = torch.tensor(((.4, -.2, .1, .02, -.01, .03),), dtype=torch.float64)
    error = torch.tensor(((.003, -.002, .001, .01, -.02, .015),), dtype=torch.float64)
    twist = torch.tensor(((.01, -.03, .02, -.04, .02, .01),), dtype=torch.float64)
    reference = efforts(jacobian, mass, error, twist, stiffness, damping, .01, external_wrench=contact)
    def rotate(vector):
        return (transform @ vector.unsqueeze(-1)).squeeze(-1)
    actual = efforts(transform @ jacobian, mass, rotate(error), rotate(twist), stiffness, damping,
                     .01, external_wrench=rotate(contact))
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_known_contact_prediction_preserves_selected_batch_independence():
    jacobian, mass, stiffness, damping = _plant(count=4)
    contact = torch.tensor(((0., 0., 0., 0., 0., .02), (0., 0., 0., 0., 0., -.02),
                            (.2, -.1, 0., .01, 0., 0.), (.1, .2, -.3, 0., -.02, .01)), dtype=torch.float64)
    error = -contact/stiffness
    twist = torch.zeros_like(error)
    full = efforts(jacobian, mass, error, twist, stiffness, damping, .01, external_wrench=contact)
    selected = torch.tensor((3, 1))
    partial = efforts(jacobian[selected], mass[selected], error[selected], twist[selected],
                      stiffness[selected], damping[selected], .01, external_wrench=contact[selected])
    torch.testing.assert_close(full, -contact, atol=1e-12, rtol=0)
    torch.testing.assert_close(partial, full[selected], atol=0, rtol=0)


def _identity_quaternion(count, *, dtype=torch.float64):
    return torch.tensor((1., 0., 0., 0.), dtype=dtype).expand(count, -1).clone()


def test_cube_reaction_obeys_newton_third_law_and_moment_about_the_control_point():
    # Cube experiences +Y; the hand experiences -Y. The pad's positive X/Z
    # lever gives positive roll and negative yaw about C.
    on_cube = torch.tensor([[[0., 2., 0.]]], dtype=torch.float64)
    control = torch.tensor([[1., 3., 5.]], dtype=torch.float64)
    centers = control[:, None] + torch.tensor([[[.3, 0., .2]]], dtype=torch.float64)
    actual = cube_reaction_wrench_b(on_cube, centers, control, _identity_quaternion(1))
    expected = torch.tensor([[0., -2., 0., .4, 0., -.6]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)


def test_cube_reaction_is_independent_of_translated_world_origins():
    forces = torch.tensor([[[1., 2., -3.], [-.4, .5, .6]]], dtype=torch.float64)
    centers = torch.tensor([[[.05, -.02, .1], [-.04, .03, -.08]]], dtype=torch.float64)
    control = torch.tensor([[.01, -.03, .04]], dtype=torch.float64)
    original = cube_reaction_wrench_b(forces, centers, control, _identity_quaternion(1))
    origins = torch.tensor([[0., 0., 0.], [-48.75, 37.5, 2.], [50., -60., 1.]], dtype=torch.float64)
    translated = cube_reaction_wrench_b(forces.expand(3, -1, -1), centers+origins[:, None],
                                        control+origins, _identity_quaternion(3))
    torch.testing.assert_close(translated, original.expand_as(translated), atol=3e-14, rtol=0)


def test_cube_reaction_uses_inverse_robot_base_rotation_with_the_correct_sign():
    base = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]], dtype=torch.float64)
    forces = torch.tensor([[[1., 0., 0.]]], dtype=torch.float64)
    centers = torch.tensor([[[0., 1., 0.]]], dtype=torch.float64)
    actual = cube_reaction_wrench_b(forces, centers, torch.zeros(1, 3, dtype=torch.float64), base)
    # World -X reaction is base +Y for a +90deg yawed robot. The +Z moment
    # is invariant to yaw; rotating with the forward base quaternion fails.
    expected = torch.tensor([[0., 1., 0., 0., 0., 1.]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)


def test_opposing_pad_forces_keep_a_pure_couple_when_the_net_force_is_zero():
    forces = torch.tensor([[[0., 2., 0.], [0., -2., 0.]]], dtype=torch.float64)
    centers = torch.tensor([[[.5, 0., 0.], [-.5, 0., 0.]]], dtype=torch.float64)
    actual = cube_reaction_wrench_b(forces, centers, torch.zeros(1, 3, dtype=torch.float64),
                                    _identity_quaternion(1))
    expected = torch.tensor([[0., 0., 0., 0., 0., -2.]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_cube_force_components_are_ignored_without_mutating_other_batch_rows():
    forces = torch.tensor([[[float("nan"), 2., float("inf")]],
                           [[1., -float("inf"), 3.]], [[-2., 0., 1.]]], dtype=torch.float64)
    centers = torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]], dtype=torch.float64)
    control = torch.zeros(3, 3, dtype=torch.float64)
    before = forces.clone()
    actual = cube_reaction_wrench_b(forces, centers, control, _identity_quaternion(3))
    expected = torch.tensor([[0., -2., 0., 0., 0., -2.],
                              [-1., 0., -3., -3., 0., 1.],
                              [2., 0., -1., 0., 2., 0.]], dtype=torch.float64)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(forces, before, equal_nan=True, atol=0, rtol=0)
    rows = torch.tensor((2, 0))
    selected = cube_reaction_wrench_b(forces[rows], centers[rows], control[rows], _identity_quaternion(2))
    torch.testing.assert_close(selected, actual[rows], atol=0, rtol=0)


@pytest.fixture(scope="module")
def contact_controller_class():
    # Compile the unchanged production class while substituting only the
    # simulator-owned base class/asset constant. All reaction and EMA methods
    # execute their real source without requiring AppLauncher or USD imports.
    class ParentController:
        def __init__(self, cfg, count, device):
            self.cfg = cfg

    path = Path(__file__).resolve().parents[1] / "hand_manipulation_test/mdp/push_v1_controller.py"
    source = ast.parse(path.read_text())
    definition = next(node for node in source.body if isinstance(node, ast.ClassDef)
                      and node.name == "ContactCartesianOscController")
    namespace = dict(OperationalSpaceController=ParentController, torch=torch,
                     quaternion_rotate=quaternion_rotate,
                     PALM_SENSOR_BODY_NAMES=tuple(f"pad_{i}" for i in range(17)),
                     cube_reaction_wrench_b=cube_reaction_wrench_b,
                     implicit_cartesian_impedance_efforts=efforts)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ContactCartesianOscController"]


def _contact_controller_fixture(controller_class, count=3):
    pad_names = tuple(f"pad_{i}" for i in range(17))
    positions = torch.zeros(count, 17, 3)
    positions[:, 0, 0] = .25
    robot = SimpleNamespace(
        find_bodies=lambda names, preserve_order: (list(range(17)), list(pad_names)),
        data=SimpleNamespace(body_link_pos_w=positions,
                             body_link_quat_w=_identity_quaternion(count, dtype=torch.float32)[:, None].expand(-1, 17, -1).clone(),
                             root_link_pos_w=torch.zeros(count, 3),
                             root_link_quat_w=_identity_quaternion(count, dtype=torch.float32)),
    )
    forces = torch.zeros(count, 1, 17, 3)
    safe = SimpleNamespace(collision_bounds=SimpleNamespace(centers=torch.zeros(17, 3)))
    env = SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(dt=.01)),
                          scene={"cube_palm_contacts": SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces))},
                          event_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=safe)))
    action = SimpleNamespace(num_envs=count, device="cpu", _env=env, _asset=robot,
                             _c_pose_b=torch.cat((torch.zeros(count, 3), _identity_quaternion(count, dtype=torch.float32)), -1),
                             cfg=SimpleNamespace(motion_stiffness=(200.,)*6, motion_damping_ratio=(1.,)*6))
    controller = controller_class(SimpleNamespace(cfg=SimpleNamespace()), action)
    return controller, action, forces, safe


def test_production_contact_reaction_uses_real_pad_centers_and_base_axes_independent_of_c_orientation(contact_controller_class):
    controller, action, forces, safe = _contact_controller_fixture(contact_controller_class, count=1)
    root = action._asset.data
    root.root_link_pos_w[:] = torch.tensor([-48.75, 37.5, 0.])
    root.root_link_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    root.body_link_pos_w[:] = root.root_link_pos_w[:, None]
    # The collision center is offset +X in the pad's body axes, which are
    # yawed +90deg. Its world lever is +Y, not the pad link origin at C.
    root.body_link_quat_w[:] = root.root_link_quat_w[:, None]
    safe.collision_bounds.centers[0, 0] = .25
    forces[:, 0, 0, 0] = 2.
    first = controller._contact_reaction().clone()
    expected_ema = torch.tensor([[0., 1., 0., 0., 0., .25]])
    torch.testing.assert_close(first, expected_ema, atol=2e-6, rtol=0)
    controller.reset_contact_history()
    action._c_pose_b[:, 3:] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
    other_c_orientation = controller._contact_reaction().clone()
    torch.testing.assert_close(other_c_orientation, first, atol=0, rtol=0)


def test_production_contact_ema_partial_reset_clears_only_selected_episode_history(contact_controller_class):
    controller, _, forces, _ = _contact_controller_fixture(contact_controller_class)
    forces[:, 0, 0, 1] = torch.tensor([2., 4., 6.])
    measured = torch.tensor([[0., -2., 0., 0., 0., -.5],
                             [0., -4., 0., 0., 0., -1.],
                             [0., -6., 0., 0., 0., -1.5]])
    first = controller._contact_reaction().clone()
    second = controller._contact_reaction().clone()
    torch.testing.assert_close(first, .5*measured, atol=0, rtol=0)
    torch.testing.assert_close(second, .75*measured, atol=0, rtol=0)
    controller.reset_contact_history(torch.tensor([1]))
    torch.testing.assert_close(controller._external_wrench[[0, 2]], second[[0, 2]], atol=0, rtol=0)
    assert torch.count_nonzero(controller._external_wrench[1]) == 0
    forces.zero_()
    released = controller._contact_reaction().clone()
    expected = .5*second
    expected[1] = 0.
    torch.testing.assert_close(released, expected, atol=0, rtol=0)
    # A new episode sees its first sample, not the old episode's EMA tail.
    forces[1, 0, 0, 1] = 4.
    next_step = controller._contact_reaction().clone()
    torch.testing.assert_close(next_step[1], .5*measured[1], atol=0, rtol=0)
    torch.testing.assert_close(next_step[[0, 2]], .5*released[[0, 2]], atol=0, rtol=0)
    controller.reset_contact_history()
    assert torch.count_nonzero(controller._external_wrench) == 0
