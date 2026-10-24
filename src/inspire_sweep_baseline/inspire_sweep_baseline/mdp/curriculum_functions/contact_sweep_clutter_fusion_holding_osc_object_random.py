"""Absolute Gray-zone padding schedule affecting future scene resets only."""

from .contact_sweep_clutter_fusion_holding import scheduled_value


def gray_zone_padding_schedule(env, env_ids, points=((6000, 0.0), (12000, 0.15)), interval=250):
    padding = scheduled_value(env.learning_iteration, points, interval)
    if not 0 <= padding <= 0.15:
        raise ValueError("Gray padding fraction must be in [0, 0.15]")
    term = env.event_manager.get_term_cfg("object_spawn")
    term.params["gray_red_padding_fraction"] = padding
    env.event_manager.set_term_cfg("object_spawn", term)
    return padding
