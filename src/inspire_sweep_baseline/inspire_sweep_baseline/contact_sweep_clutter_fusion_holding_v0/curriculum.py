"""Absolute schedules restore consistently on checkpoint load and rewind."""
import math


def scheduled_value(iteration, points, interval=250):
    if interval <= 0 or not points:
        raise ValueError('Expected positive interval and nonempty points')
    if any(not math.isfinite(v) for _, v in points) or any(b[0] <= a[0] for a,b in zip(points,points[1:])):
        raise ValueError('Expected finite values at strictly increasing iterations')
    if iteration <= points[0][0]:
        return points[0][1]
    for (start, first), (end, last) in zip(points, points[1:]):
        if iteration < end:
            elapsed = ((iteration-start)//interval)*interval
            return first + (last-first)*elapsed/(end-start)
    return points[-1][1]


def reward_schedule(env, env_ids, term_name, points, parameter=None, interval=250):
    value = scheduled_value(env.learning_iteration, points, interval)
    cfg = env.reward_manager.get_term_cfg(term_name)
    if parameter is None:
        # Zero-weight terms are not called by RewardManager. Invalidate history
        # when enabling/disabling so a stale wrench cannot create an impulse.
        if (cfg.weight == 0.) != (value == 0.) and hasattr(cfg.func, 'reset'):
            cfg.func.reset()
        cfg.weight = value
    else:
        cfg.params[parameter] = value
    env.reward_manager.set_term_cfg(term_name, cfg)
    return value
