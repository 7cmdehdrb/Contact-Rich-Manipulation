"""Single terminal reward, paid before the manager resets the success counter."""


def settled_success_bonus(env):
    return env.termination_manager.get_term('target_goal_success').float() / env.step_dt


def desired_wrench_action_rate_penalty(env):
    """Squared change of normalized desired wrench; preserve the raw Hand cost.

    Desired wrench = clip(raw, -limit, limit) * wrench_scale. Dividing by
    the policy target range (limit * wrench_scale) cancels the physical scale.
    Compare targets before feedback, not measured or corrected wrench. Use the
    manager's previous action so selected resets keep its existing zero history.
    """
    from .goal_rewards import action_rate_l2

    current = env.action_manager.action
    previous = env.action_manager.prev_action
    if current.shape[-1] != 8 or previous.shape != current.shape:
        raise ValueError("Wrench penalties require six Arm and two Hand actions")
    limit = env.action_manager.get_term("arm_action").cfg.normalized_wrench_limit
    current_arm = current[:, :6].clamp(-limit, limit) / limit
    previous_arm = previous[:, :6].clamp(-limit, limit) / limit
    arm_cost = (current_arm - previous_arm).square().sum(dim=-1).clamp_max(100.0)
    return 0.03 * arm_cost + 0.03 * action_rate_l2(env, action_group="hand", max_cost=100.0)
