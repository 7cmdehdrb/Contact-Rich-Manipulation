#!/usr/bin/env python3
"""Fixed-impedance, absolute-pose OSC executed through ur_rtde ``servoJ``.

Controller configuration (kept intentionally non-configurable):

* impedance mode: fixed
* target type: absolute TCP pose
* control axes: motion on all six axes; no force axes
* full inertial dynamics decoupling: enabled
* partial inertial dynamics decoupling: disabled
* null-space control: disabled

The OSC calculation follows Isaac Lab's ``OperationalSpaceController``.  Unlike
the Isaac Lab controller, this program cannot return joint torque to a simulator.
It therefore converts the modeled torque to joint acceleration with

    qdd = solve(M(q), tau - C(q, qd) qd - G(q))

and integrates over the measured robot-state period twice to obtain joint
velocity and position targets for ``servoJ``.

ur_rtde 1.6.3 exposes M, C*qd, and J, but not G.  UR's position controller
internally compensates gravity.  Accordingly, this controller enables matched
gravity compensation in the modeled OSC torque.  The same G is added to tau and
subtracted by the inverse-dynamics equation, so it cancels exactly when computing
qdd.  ``gravity_torque`` remains an explicit input for model logging/testing; a
zero vector is correct for the acceleration conversion under this matched-
compensation assumption.  It must not be interpreted as a measurement of G.

Example (first verify in URSim, then use a collision-free target):

    python src/test/rtde_osc_velocity.py \
        --robot-ip 192.168.56.101 \
        --target -0.40 -0.20 0.35 0.0 3.14 0.0 \
        --duration 5 --execute

The target and measured TCP poses use the ur_rtde format
``[x, y, z, rx, ry, rz]``; orientation is a rotation vector in radians.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
GravityTorqueModel = Callable[[Vector], Vector]


@dataclass(frozen=True)
class FixedPoseOscConfig:
    """Numerical gains and safety limits for the fixed OSC configuration."""

    impedance_mode: str = "fixed"
    target_type: str = "pose_abs"
    motion_axes: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    force_axes: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    inertial_dynamics_decoupling: bool = True
    partial_inertial_dynamics_decoupling: bool = False
    gravity_compensation: bool = True
    nullspace_control: bool = False

    # Conservative initial gains for a real UR. Units are those of Cartesian
    # acceleration per Cartesian pose error, as in the Isaac Lab controller.
    stiffness: tuple[float, ...] = (
        80.0,
        80.0,
        80.0,
        40.0,
        40.0,
        40.0,
    )
    damping_ratio: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # The ur_rtde dynamics calls are synchronous and can make the effective
    # update rate much slower than the receive rate. Keep speed conservative so
    # the configuration cannot move far while a dynamics query is outstanding.
    max_joint_speed: float = 0.10  # rad/s
    max_joint_acceleration: float = 1.00  # rad/s^2
    max_joint_position_step: float = 0.02  # rad per OSC update
    servoj_speed: float = 0.10  # currently unused by the UR servoJ implementation
    servoj_acceleration: float = (
        1.00  # currently unused by the UR servoJ implementation
    )
    servoj_lookahead_time: float = 0.10  # s; documented range [0.03, 0.2]
    servoj_gain: float = 100.0  # documented range [100, 2000]
    servoj_max_time: float = 0.05  # s; hold command despite slow model queries
    stop_deceleration: float = 2.00  # rad/s^2
    max_position_error: float = 0.10  # m
    max_orientation_error: float = 0.50  # rad
    max_task_inertia_condition: float = 1.0e8
    max_observed_period: float = 0.25  # s; stop if RTDE state/dynamics stall longer
    max_dynamics_joint_skew: float = 0.03  # rad between model query and command state
    dynamics_query_retries: int = 3

    def __post_init__(self) -> None:
        required = (
            self.impedance_mode == "fixed",
            self.target_type == "pose_abs",
            self.motion_axes == (1, 1, 1, 1, 1, 1),
            self.force_axes == (0, 0, 0, 0, 0, 0),
            self.inertial_dynamics_decoupling,
            not self.partial_inertial_dynamics_decoupling,
            self.gravity_compensation,
            not self.nullspace_control,
        )
        if not all(required):
            raise ValueError("The requested OSC mode is fixed and must not be changed.")
        if len(self.stiffness) != 6 or len(self.damping_ratio) != 6:
            raise ValueError(
                "stiffness and damping_ratio must each contain six values."
            )
        if min(self.stiffness) <= 0.0 or min(self.damping_ratio) <= 0.0:
            raise ValueError("OSC stiffness and damping ratios must be positive.")
        if not 0.03 <= self.servoj_lookahead_time <= 0.2:
            raise ValueError("servoj_lookahead_time must be in [0.03, 0.2] seconds.")
        if not 100.0 <= self.servoj_gain <= 2000.0:
            raise ValueError("servoj_gain must be in [100, 2000].")


@dataclass(frozen=True)
class OscStep:
    """Intermediate values from one controller step, useful for inspection."""

    pose_error: Vector
    desired_tcp_acceleration: Vector
    task_wrench: Vector
    motion_torque: Vector
    modeled_total_torque: Vector
    joint_acceleration: Vector
    joint_velocity_target: Vector
    joint_position_target: Vector


def _as_vector(value: object, size: int, name: str) -> Vector:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return vector


def _as_matrix(value: object, rows: int, columns: int, name: str) -> Matrix:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != rows * columns:
        raise ValueError(
            f"{name} must contain {rows * columns} values, got {matrix.size}."
        )
    matrix = matrix.reshape(rows, columns)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return matrix


def _rotation_vector_to_quaternion(rotation_vector: Vector) -> Vector:
    """Convert a rotation vector to a unit quaternion in (w, x, y, z)."""

    rotation_vector = _as_vector(rotation_vector, 3, "rotation_vector")
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1.0e-12:
        # sin(angle / 2) / angle = 1/2 - angle^2/48 + O(angle^4)
        scale = 0.5 - angle * angle / 48.0
    else:
        scale = math.sin(0.5 * angle) / angle
    quaternion = np.concatenate(([math.cos(0.5 * angle)], rotation_vector * scale))
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_multiply(left: Vector, right: Vector) -> Vector:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _orientation_error(
    current_rotation_vector: Vector, target_rotation_vector: Vector
) -> Vector:
    """Return the base-frame target-minus-current error as a shortest rotation vector."""

    current = _rotation_vector_to_quaternion(current_rotation_vector)
    target = _rotation_vector_to_quaternion(target_rotation_vector)
    current_inverse = current.copy()
    current_inverse[1:] *= -1.0
    error = _quaternion_multiply(target, current_inverse)
    error /= np.linalg.norm(error)

    # q and -q represent the same orientation. Choosing w >= 0 gives the
    # shortest rotation and matches Isaac Lab's axis_angle_from_quat behavior.
    if error[0] < 0.0:
        error *= -1.0
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm < 1.0e-12:
        return 2.0 * error[1:]
    angle = 2.0 * math.atan2(vector_norm, float(error[0]))
    return error[1:] * (angle / vector_norm)


def _clip_vector_norm(vector: Vector, maximum: float) -> Vector:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm == 0.0:
        return vector
    return vector * (maximum / norm)


def _scale_to_max_abs(vector: Vector, maximum: float) -> Vector:
    """Limit the largest component while preserving the vector direction."""

    peak = float(np.max(np.abs(vector)))
    if peak <= maximum or peak == 0.0:
        return vector
    return vector * (maximum / peak)


def _mass_matrix_is_valid(mass_matrix: Matrix) -> bool:
    """Check the invariants of a rigid-body joint-space mass matrix."""

    if not np.allclose(mass_matrix, mass_matrix.T, rtol=1.0e-6, atol=1.0e-8):
        return False
    return bool(float(np.min(np.linalg.eigvalsh(mass_matrix))) > 1.0e-9)


class FixedPoseOsc:
    """Full-inertia OSC with no force or null-space controller."""

    def __init__(
        self,
        target_pose: Vector,
        period: float,
        config: FixedPoseOscConfig | None = None,
        gravity_model: GravityTorqueModel | None = None,
    ) -> None:
        self.config = config or FixedPoseOscConfig()
        self.target_pose = _as_vector(target_pose, 6, "target_pose").copy()
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("period must be a positive finite number.")
        self.period = period
        self._gravity_model = gravity_model

        stiffness = np.asarray(self.config.stiffness, dtype=np.float64)
        damping_ratio = np.asarray(self.config.damping_ratio, dtype=np.float64)
        self._kp = np.diag(stiffness)
        self._kd = np.diag(2.0 * np.sqrt(stiffness) * damping_ratio)
        self._motion_selection = np.diag(
            np.asarray(self.config.motion_axes, dtype=np.float64)
        )

    def gravity_torque(self, q: Vector) -> Vector:
        """Return G(q), or zero under RTDE's matched internal gravity compensation."""

        if self._gravity_model is None:
            return np.zeros(6, dtype=np.float64)
        return _as_vector(self._gravity_model(q.copy()), 6, "gravity_model(q)")

    def compute(
        self,
        q: object,
        qd: object,
        tcp_pose: object,
        tcp_speed: object,
        mass_matrix: object,
        coriolis_and_centrifugal_torque: object,
        jacobian: object,
        integration_period: float | None = None,
    ) -> OscStep:
        q = _as_vector(q, 6, "q")
        qd = _as_vector(qd, 6, "qd")
        tcp_pose = _as_vector(tcp_pose, 6, "tcp_pose")
        tcp_speed = _as_vector(tcp_speed, 6, "tcp_speed")
        mass_matrix = _as_matrix(mass_matrix, 6, 6, "mass_matrix")
        coriolis = _as_vector(
            coriolis_and_centrifugal_torque, 6, "coriolis_and_centrifugal_torque"
        )
        jacobian = _as_matrix(jacobian, 6, 6, "jacobian")
        if integration_period is None:
            integration_period = self.period
        if not math.isfinite(integration_period) or integration_period <= 0.0:
            raise ValueError("integration_period must be a positive finite number.")
        if integration_period > self.config.max_observed_period:
            raise RuntimeError(
                f"RTDE state period {integration_period:.4f}s exceeds the "
                f"{self.config.max_observed_period:.4f}s safety limit."
            )

        # Reject corrupt dynamics before they can become a motion command.  In
        # particular, some ur_rtde/PolyScope combinations have returned a row of
        # M in place of C*qd when the output-register recipe is incompatible.
        if not _mass_matrix_is_valid(mass_matrix):
            raise RuntimeError(
                "RTDE returned a mass matrix that is non-symmetric or not positive definite."
            )
        if np.allclose(coriolis, mass_matrix[0, :], rtol=1.0e-10, atol=1.0e-12):
            raise RuntimeError(
                "RTDE dynamics register mismatch: C*qd equals the first mass-matrix row. "
                "Check the ur_rtde/PolyScope versions and RTDE register configuration."
            )

        position_error = _clip_vector_norm(
            self.target_pose[:3] - tcp_pose[:3], self.config.max_position_error
        )
        orientation_error = _clip_vector_norm(
            _orientation_error(tcp_pose[3:], self.target_pose[3:]),
            self.config.max_orientation_error,
        )
        pose_error = np.concatenate((position_error, orientation_error))

        # Fixed impedance, stationary absolute target: xdd_des = Kp*e - Kd*xd.
        desired_tcp_acceleration = self._kp @ pose_error - self._kd @ tcp_speed

        # Full dynamic decoupling (partial decoupling is deliberately absent):
        # Lambda = (J M^-1 J^T)^-1.
        mass_inverse_jacobian_transpose = np.linalg.solve(mass_matrix, jacobian.T)
        task_inertia_inverse = jacobian @ mass_inverse_jacobian_transpose
        condition = float(np.linalg.cond(task_inertia_inverse))
        if (
            not math.isfinite(condition)
            or condition > self.config.max_task_inertia_condition
        ):
            raise RuntimeError(
                "OSC stopped near a singularity: "
                f"cond(J M^-1 J^T)={condition:.3e} exceeds "
                f"{self.config.max_task_inertia_condition:.3e}."
            )
        task_inertia = np.linalg.solve(
            task_inertia_inverse, np.eye(6, dtype=np.float64)
        )
        task_wrench = task_inertia @ desired_tcp_acceleration
        motion_torque = jacobian.T @ self._motion_selection @ task_wrench

        gravity = self.gravity_torque(q)
        # Isaac Lab adds G when gravity compensation is enabled. There is no C*qd
        # feed-forward in that controller, so C*qd remains in the inverse dynamics.
        modeled_total_torque = motion_torque + gravity

        # Required rigid-body equation:
        # tau = M*qdd + C*qd + G  ->  qdd = M^-1*(tau - C*qd - G).
        # With matched gravity compensation, G cancels here even when the external
        # gravity model is omitted; the Coriolis/centrifugal term does not cancel.
        joint_acceleration = np.linalg.solve(
            mass_matrix, modeled_total_torque - coriolis - gravity
        )
        joint_acceleration = _scale_to_max_abs(
            joint_acceleration, self.config.max_joint_acceleration
        )

        # Re-anchor integration at measured qd every cycle so numerical drift does
        # not accumulate before the second integration used by servoJ.
        joint_velocity_target = qd + joint_acceleration * integration_period
        joint_velocity_target = _scale_to_max_abs(
            joint_velocity_target, self.config.max_joint_speed
        )

        # Integrate once more for joint-position control. Trapezoidal integration
        # is less sensitive to the variable RTDE period than forward Euler.
        joint_position_step = 0.5 * (qd + joint_velocity_target) * integration_period
        joint_position_step = _scale_to_max_abs(
            joint_position_step, self.config.max_joint_position_step
        )
        joint_position_target = q + joint_position_step

        return OscStep(
            pose_error=pose_error,
            desired_tcp_acceleration=desired_tcp_acceleration,
            task_wrench=task_wrench,
            motion_torque=motion_torque,
            modeled_total_torque=modeled_total_torque,
            joint_acceleration=joint_acceleration,
            joint_velocity_target=joint_velocity_target,
            joint_position_target=joint_position_target,
        )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-ip", default="192.168.56.101", help="UR controller or URSim IP address"
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        required=True,
        help="absolute TCP target pose in the UR base frame",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=125.0,
        help="RTDE loop frequency in Hz (125 by default because M, C*qd, and J are synchronous queries)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="maximum control duration in seconds",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        metavar="N",
        help="print OSC and servo telemetry every N cycles (0 disables logging)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required safety interlock; without it, no robot connection or motion is attempted",
    )
    return parser.parse_args()


def run_rtde_controller(
    robot_ip: str,
    target_pose: Vector,
    frequency: float,
    duration: float,
    log_every: int = 0,
) -> None:
    """Connect to the robot and run the bounded OSC loop."""

    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be positive and finite.")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be positive and finite.")
    if log_every < 0:
        raise ValueError("log_every must be non-negative.")

    try:
        import rtde_control
        import rtde_receive
    except ImportError as exc:
        raise RuntimeError(
            "ur_rtde is required: install the Python package 'ur_rtde'."
        ) from exc

    period = 1.0 / frequency
    controller = FixedPoseOsc(target_pose=target_pose, period=period)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip, frequency)
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip, frequency)
    except BaseException:
        rtde_c.disconnect()
        raise
    start_time = time.monotonic()
    previous_robot_timestamp: float | None = None
    warned_about_effective_rate = False
    cycle_index = 0

    try:
        while time.monotonic() - start_time < duration:
            cycle_wall_start = time.monotonic()
            cycle_start = rtde_c.initPeriod()
            if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
                raise RuntimeError("Robot entered a protective or emergency stop.")

            dynamics_q = _as_vector(rtde_r.getActualQ(), 6, "getActualQ()")
            dynamics_qd = _as_vector(rtde_r.getActualQd(), 6, "getActualQd()")

            query_start = time.monotonic()
            for mass_attempt in range(controller.config.dynamics_query_retries):
                mass_matrix = _as_matrix(
                    rtde_c.getMassMatrix(dynamics_q.tolist(), False),
                    6,
                    6,
                    "getMassMatrix()",
                ).copy()
                if _mass_matrix_is_valid(mass_matrix):
                    break
            else:
                raise RuntimeError(
                    "RTDE returned an invalid mass matrix on "
                    f"{controller.config.dynamics_query_retries} consecutive reads."
                )
            mass_query_time = time.monotonic() - query_start
            if mass_attempt:
                print(
                    f"Warning: discarded {mass_attempt} corrupt RTDE mass-matrix read(s).",
                    file=sys.stderr,
                    flush=True,
                )

            query_start = time.monotonic()
            for coriolis_attempt in range(controller.config.dynamics_query_retries):
                coriolis = _as_vector(
                    rtde_c.getCoriolisAndCentrifugalTorques(
                        dynamics_q.tolist(), dynamics_qd.tolist()
                    ),
                    6,
                    "getCoriolisAndCentrifugalTorques()",
                ).copy()
                if not np.allclose(
                    coriolis, mass_matrix[0, :], rtol=1.0e-10, atol=1.0e-12
                ):
                    break
            else:
                raise RuntimeError(
                    "RTDE returned a corrupt Coriolis vector on "
                    f"{controller.config.dynamics_query_retries} consecutive reads."
                )
            coriolis_query_time = time.monotonic() - query_start
            if coriolis_attempt:
                print(
                    f"Warning: discarded {coriolis_attempt} corrupt RTDE Coriolis read(s).",
                    file=sys.stderr,
                    flush=True,
                )

            query_start = time.monotonic()
            # ur_rtde 1.6.3 serializes the Jacobian column-by-column. NumPy's
            # default reshape is row-major, so transpose the C-order reshape to
            # recover J[twist_axis, joint]. Without this, OSC uses J^T as J and
            # can move away from the Cartesian target.
            jacobian = _as_matrix(
                rtde_c.getJacobian(dynamics_q.tolist(), []), 6, 6, "getJacobian()"
            ).T
            jacobian_query_time = time.monotonic() - query_start

            # Refresh feedback after the blocking model calls. The model remains
            # evaluated at dynamics_q; reject the command if the arm moved too far
            # for that model snapshot to remain trustworthy.
            q = _as_vector(rtde_r.getActualQ(), 6, "getActualQ() after dynamics")
            qd = _as_vector(rtde_r.getActualQd(), 6, "getActualQd() after dynamics")
            tcp_pose = _as_vector(
                rtde_r.getActualTCPPose(), 6, "getActualTCPPose() after dynamics"
            )
            tcp_speed = _as_vector(
                rtde_r.getActualTCPSpeed(), 6, "getActualTCPSpeed() after dynamics"
            )
            joint_skew = float(np.max(np.abs(q - dynamics_q)))
            if joint_skew > controller.config.max_dynamics_joint_skew:
                raise RuntimeError(
                    "Robot moved too far while RTDE dynamics were queried "
                    f"({joint_skew:.4f} rad > "
                    f"{controller.config.max_dynamics_joint_skew:.4f} rad)."
                )

            robot_timestamp = float(rtde_r.getTimestamp())
            if not math.isfinite(robot_timestamp):
                raise RuntimeError("RTDE returned an invalid robot timestamp.")
            if previous_robot_timestamp is None:
                observed_period = period
            else:
                observed_period = robot_timestamp - previous_robot_timestamp
                if observed_period <= 0.0:
                    raise RuntimeError(
                        "RTDE robot timestamp did not advance; refusing to reuse stale state."
                    )
            previous_robot_timestamp = robot_timestamp

            step = controller.compute(
                q,
                qd,
                tcp_pose,
                tcp_speed,
                mass_matrix,
                coriolis,
                jacobian,
                integration_period=observed_period,
            )
            query_and_compute_time = time.monotonic() - cycle_wall_start
            if query_and_compute_time > controller.config.max_observed_period:
                raise RuntimeError(
                    "RTDE dynamics queries stalled for "
                    f"{query_and_compute_time:.4f}s, exceeding the "
                    f"{controller.config.max_observed_period:.4f}s safety limit "
                    f"(M={mass_query_time:.4f}s, C={coriolis_query_time:.4f}s, "
                    f"J={jacobian_query_time:.4f}s)."
                )
            # servoJ's time is a blocking command-hold interval. It gives the
            # position controller time to act before the next synchronous model
            # query takes over the RTDE control channel.
            servo_time = min(
                max(period, query_and_compute_time), controller.config.servoj_max_time
            )
            if query_and_compute_time > period and not warned_about_effective_rate:
                effective_rate = 1.0 / query_and_compute_time
                print(
                    "Warning: synchronous RTDE dynamics queries are slower than the requested "
                    f"{frequency:.1f} Hz loop; servoJ updates will run at approximately "
                    f"{effective_rate:.1f} Hz using the measured robot timestep "
                    f"(M={mass_query_time:.4f}s, C={coriolis_query_time:.4f}s, "
                    f"J={jacobian_query_time:.4f}s, servo={servo_time:.4f}s).",
                    file=sys.stderr,
                    flush=True,
                )
                warned_about_effective_rate = True

            if log_every and cycle_index % log_every == 0:
                raw_position_error = controller.target_pose[:3] - tcp_pose[:3]
                print(
                    f"cycle={cycle_index:04d} "
                    f"t={time.monotonic() - start_time:.3f}s "
                    f"tcp={np.array2string(tcp_pose, precision=5)} "
                    f"|e_pos_raw|={np.linalg.norm(raw_position_error):.6f} "
                    f"|e_pose_used|={np.linalg.norm(step.pose_error):.6f} "
                    f"|F|={np.linalg.norm(step.task_wrench):.6f} "
                    f"max|qdd|={np.max(np.abs(step.joint_acceleration)):.6f} "
                    f"max|qd_cmd|={np.max(np.abs(step.joint_velocity_target)):.6f} "
                    f"max|dq_cmd|={np.max(np.abs(step.joint_position_target - q)):.6f}",
                    flush=True,
                )

            success = rtde_c.servoJ(
                step.joint_position_target.tolist(),
                controller.config.servoj_speed,
                controller.config.servoj_acceleration,
                servo_time,
                controller.config.servoj_lookahead_time,
                controller.config.servoj_gain,
            )
            if not success:
                raise RuntimeError("RTDE servoJ command was rejected.")

            # initPeriod/waitPeriod is useful only when work completed inside the
            # requested period. Slow synchronous queries plus blocking servoJ
            # already provide synchronization, so do not wait a second time.
            if time.monotonic() - cycle_wall_start < period:
                rtde_c.waitPeriod(cycle_start)
            cycle_index += 1
    finally:
        # servoStop is the correct termination for servoJ. stopScript releases the
        # uploaded RTDE control script after the robot has decelerated.
        try:
            rtde_c.servoStop(controller.config.stop_deceleration)
        finally:
            try:
                rtde_c.stopScript()
            finally:
                try:
                    rtde_r.disconnect()
                finally:
                    rtde_c.disconnect()


def main() -> None:
    args = _parse_arguments()
    target = _as_vector(args.target, 6, "--target")
    if not args.execute:
        raise SystemExit(
            "Safety interlock active: no robot command was sent. "
            "Verify the target in URSim and add --execute to run."
        )
    run_rtde_controller(
        args.robot_ip,
        target,
        args.frequency,
        args.duration,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
