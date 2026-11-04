"""Sensor-reference wrench control shared by simulation and future F/T hardware adapters.

Only torch is required. No Isaac Lab/PhysX or EEF pose is used here.
"""

import torch


def sensor_wrench_efforts(desired_wrench_s, current_wrench_s, jacobian_s, p_gains,
                          gravity=None, feedback_valid=None, commanded_wrench_limit=None):
    """Deployable feedforward + six-axis P feedback, then geometric J transpose.

    All wrenches are [Fx,Fy,Fz,Mx,My,Mz], N/Nm, *about the sensor origin*,
    expressed in its current axes. Positive feedback readings represent the
    wrench the robot exerts on the environment (compensated parent-on-child
    mount reaction in this simulator). An external-on-tool F/T convention must
    be negated by the hardware adapter before calling this function.
    Jacobian rows use the same point/axes; gravity is joint-space Nm.
    """
    if desired_wrench_s.ndim != 2 or desired_wrench_s.shape[-1] != 6:
        raise ValueError('Expected desired wrench [envs, 6]')
    if current_wrench_s.shape != desired_wrench_s.shape:
        raise ValueError('Current wrench must have the same [envs, 6] shape')
    if jacobian_s.ndim != 3 or jacobian_s.shape[:2] != desired_wrench_s.shape:
        raise ValueError('Expected sensor Jacobian [envs, 6, joints]')
    error = desired_wrench_s - current_wrench_s
    if feedback_valid is not None:
        if feedback_valid.shape != desired_wrench_s.shape[:1] or feedback_valid.dtype != torch.bool:
            raise ValueError('Expected feedback_valid [envs] bool')
        error = torch.where(feedback_valid[:, None], error, 0.0)
    commanded_wrench_s = desired_wrench_s + p_gains * error
    # Bound the corrected command at the sensor point before J transpose.
    if commanded_wrench_limit is not None:
        commanded_wrench_s = torch.clamp(
            commanded_wrench_s, min=-commanded_wrench_limit, max=commanded_wrench_limit)
    efforts = (jacobian_s.mT @ commanded_wrench_s.unsqueeze(-1)).squeeze(-1)
    if gravity is not None:
        if gravity.shape != efforts.shape:
            raise ValueError('Gravity must be a joint-space vector [envs, joints]')
        efforts = efforts + gravity
    return efforts

