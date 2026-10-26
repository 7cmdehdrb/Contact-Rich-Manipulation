"""Finite population fallback and shape-aware Final layout sampling."""

from dataclasses import replace
import math
import numpy as np
from ...assets.scenes.contact_sweep_clutter_layout import (
    DEFAULT_LAYOUT_CONFIG, GenerationError, ClutterLayoutGenerator,
    shelf_xy_to_generator_cm, generator_cm_to_shelf_xy,
)
from .contact_sweep_clutter_fusion_holding_osc_object_random_placement import (
    valid_layout, scene_generator,
)


class FinalLayoutGenerator(ClutterLayoutGenerator):
    """Scale the Gray inner zone independently of physical separation.

    For large props, 15% padding can imply a negative *virtual* red margin.
    This is only a zone-classification radius, never the physical placement
    config; OBB, approach and source object-distance checks remain intact.
    """

    def __init__(self, *args, gray_red_padding_fraction=0., **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= gray_red_padding_fraction <= .15:
            raise ValueError('Gray padding fraction must be in [0, .15]')
        radius = 2 * self.config.radius + self.config.red_margin
        self._gray_zone_config = replace(self.config,
            red_margin=radius * (1 - gray_red_padding_fraction) - 2 * self.config.radius)

    def zone_masks(self, points, target, direction, distance, role):
        cfg = self.config if role == 'white' else self._gray_zone_config
        return scene_generator.zone_masks(points, target, direction, distance, cfg)


def population_attempts(gray, white):
    """Requested counts, then White/Gray alternating reductions down to zero."""
    yield gray, white
    white_turn = True
    while gray or white:
        if (white_turn and white) or not gray:
            white -= 1
        else:
            gray -= 1
        white_turn = not white_turn
        yield gray, white


def sample_target_goal(rng, half, wide, spawn_radius, goal_distance, edge_clearance,
                       excluded_x_half_angle, shelf_half_extents=(.20, .50)):
    """Sample an in-bounds target and a feasible 5--15 cm ray, without clipping.

    Wide targets use the central 80% of the shelf, intersected with footprint,
    edge clearance and the source approach-ring safe domain. The same target
    and goal survive all later population reductions.
    """
    radius_cm = max(4.0, float(np.linalg.norm(half)) * 100)
    contact_clearance = max(.25, (radius_cm + 1.1) / math.cos(math.pi / 16) - radius_cm - 1.0 + 1e-8)
    config = replace(DEFAULT_LAYOUT_CONFIG, target_bounds=(-50., 50., 0., 40.),
                     radius=radius_cm, gray_radius=max(14., 2 * radius_cm + 3.3),
                     contact_clearance=contact_clearance)
    safe = np.asarray(shelf_half_extents) - half - edge_clearance
    # Existing goal rewards require the bottom center in +/- (.16, .46).
    safe = np.minimum(safe, (.16, .46))
    source_goal_bounds = generator_cm_to_shelf_xy(config.object_bounds)
    safe = np.minimum(safe, np.min(np.abs(source_goal_bounds), axis=0)) - 1e-8
    source = generator_cm_to_shelf_xy(config.safe_target_bounds)
    lower = np.maximum(np.min(source, axis=0), -safe)
    upper = np.minimum(np.max(source, axis=0), safe)
    if wide:
        lower = np.maximum(lower, -.8 * np.asarray(shelf_half_extents))
        upper = np.minimum(upper, .8 * np.asarray(shelf_half_extents))
        target = rng.uniform(lower, upper)
    else:
        angle = rng.uniform(-math.pi, math.pi)
        target = spawn_radius * math.sqrt(rng.random()) * np.array([math.cos(angle), math.sin(angle)])
    if np.any(lower >= upper) or np.any(target < lower) or np.any(target > upper):
        raise ValueError('No safe target domain for the configured object size')
    for attempt in range(129):
        if attempt == 128:
            direction = np.array([0., -1. if target[1] > 0 else 1.])
        else:
            angle = rng.uniform(-math.pi, math.pi)
            if abs(math.sin(angle)) < math.sin(excluded_x_half_angle):
                continue
            direction = np.array([math.cos(angle), math.sin(angle)])
        room = np.full(2, np.inf)
        nonzero = np.abs(direction) > 1e-12
        room[nonzero] = (safe[nonzero] - np.sign(direction[nonzero]) * target[nonzero]) / np.abs(direction[nonzero])
        distance = min(goal_distance, float(room.min()))
        if distance >= .05:
            return target, target + distance * direction, config
    raise ValueError('No feasible goal of at least 5 cm for this target domain')


def sample_population(rng, target, goal, half, max_gray, gray, white, config, padding,
                      attempts_per_count=8):
    """Retry only failed rows; zero obstacles is a guaranteed geometric fallback."""
    yaw = np.zeros(len(half))
    for actual_gray, actual_white in population_attempts(gray, white):
        indices = [0, *range(1, 1 + actual_gray),
                   *range(1 + max_gray, 1 + max_gray + actual_white)]
        if not actual_gray + actual_white:
            return np.empty((0, 2)), yaw, actual_gray, actual_white
        cfg = replace(config, obstacles=actual_gray + actual_white,
                      gray_obstacles=actual_gray, rear_obstacles=min(1, actual_gray), scene_attempts=4)
        generator = FinalLayoutGenerator(seed=0, config=cfg, max_obstacles=20,
                                         gray_red_padding_fraction=padding)
        generator.rng = rng
        for _ in range(attempts_per_count):
            try:
                batch = generator.generate_batch(shelf_xy_to_generator_cm(target[None]),
                                                 shelf_xy_to_generator_cm(goal[None]))
            except GenerationError:
                continue
            centers = generator_cm_to_shelf_xy(batch.obstacles)[0]
            yaw = rng.uniform(-math.pi, math.pi, len(half))
            yaw[0] = 0.0
            if not valid_layout(np.concatenate((target[None], centers)), yaw[indices], half[indices]):
                continue
            # Source certificates use circles; enlarge obstacle clearance to
            # their actual OBB circumscribed radii after scaling.
            radius_cm = np.linalg.norm(half[indices[1:]], axis=-1) * 100
            clearance = radius_cm + cfg.hand_radius + cfg.finger_clearance
            d2 = scene_generator.polyline_distance2(batch.obstacles, batch.approach)[0]
            if np.all(d2 >= clearance ** 2):
                return centers, yaw, actual_gray, actual_white
    raise AssertionError('Population fallback must reach the empty scene')
