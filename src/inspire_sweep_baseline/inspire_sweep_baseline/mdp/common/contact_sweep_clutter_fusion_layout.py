"""Count-configurable fixed-command layouts with unchanged geometric rules."""

from ...assets.scenes.contact_sweep_clutter_layout import ClutterLayoutGenerator
from ...assets.scenes.contact_sweep_clutter_layout import DEFAULT_LAYOUT_CONFIG
from dataclasses import replace

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
        env._population_layout_generators[key] = ClutterLayoutGenerator(
            seed=seed, config=config, max_obstacles=getattr(env.cfg, "obstacle_observation_slots", 10),
        )
    return env._population_layout_generators[key]
