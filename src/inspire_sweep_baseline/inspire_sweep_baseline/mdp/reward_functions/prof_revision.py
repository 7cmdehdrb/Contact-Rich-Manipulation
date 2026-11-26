"""Signed progress is a distance increment; formation is a bounded rate."""


def reaching(env):
    return env._stabilize_state.reach_delta / env.step_dt


def contact_formation(env):
    return env._formation


def precision_sweep_progress(env):
    return env._stabilize_state.precision_positive / env.step_dt


def precision_sweep_retreat(env):
    return env._stabilize_state.precision_retreat / env.step_dt


def arm_joint_velocity_margin(env, asset_cfg, free_speed=0.8, limit_speed=1.0):
    """Squared maximum arm-speed excess; RewardManager applies weight and dt."""
    import math

    if not (math.isfinite(free_speed) and math.isfinite(limit_speed)
            and 0.0 <= free_speed < limit_speed):
        raise ValueError('Expected finite 0 <= free_speed < limit_speed')
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    speed = velocity.abs().amax(dim=-1)
    return ((speed - free_speed) / (limit_speed - free_speed)).clamp(0.0, 1.0).square()
