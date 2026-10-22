"""Count-configurable fixed-command layouts with unchanged geometric rules."""

from dataclasses import replace

from ..contact_sweep_clutter.layout import (
    DEFAULT_LAYOUT_CONFIG, ClutterLayoutGenerator, generator_cm_to_shelf_xy,
    shelf_xy_to_generator_cm,
)


def population_layout_generator(env, gray_obstacles, white_obstacles):
    """Keep a seeded RNG per population, without mutating shared source config."""
    if not hasattr(env, "_population_layout_generators"):
        env._population_layout_generators = {}
    key = (gray_obstacles, white_obstacles)
    if key not in env._population_layout_generators:
        config = replace(
            DEFAULT_LAYOUT_CONFIG, obstacles=gray_obstacles + white_obstacles,
            gray_obstacles=gray_obstacles, rear_obstacles=min(1, gray_obstacles),
        )
        seed = getattr(env, "_population_layout_seed", env.cfg.seed)
        env._population_layout_generators[key] = ClutterLayoutGenerator(seed=seed, config=config)
    return env._population_layout_generators[key]
