"""Place clutter around fixed Contact Sweep targets with the scene tool's rules.

The sibling ``src/scene_randomization`` remains the source of geometry, proposal
and approach calculations. Only its scene-construction control is adapted:
target and goal are inputs and never change, including after placement failure.
All layout calculations use the scene tool's centimetres and board frame.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_scene_generator():
    """Load the checkout dependency without changing the launcher's sys.path."""
    name = "_inspire_contact_sweep_scene_generator"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[3] / "scene_randomization" / "generator.py"
        if not path.is_file():
            raise ImportError(f"Contact Sweep Clutter requires the workspace scene generator: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolve the defining module while the file is executing.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


scene_generator = _load_scene_generator()
Config = scene_generator.Config
GenerationError = scene_generator.GenerationError

# Clutter v0 override; the standalone scene tool keeps its original defaults.
# Exclusion radius = 2 * 4 cm + 2.3 cm = 10.3 cm.
DEFAULT_LAYOUT_CONFIG = Config(obstacles=3, gray_obstacles=1, rear_obstacles=1, red_margin=2.3)


def shelf_xy_to_generator_cm(points):
    """Rotate shelf-local metres to board cm, preserving the robot-side front.

    The shelf spans X +/-0.20, Y +/-0.50. The generator spans X +/-50,
    Y [0,40], with its front at Y=0. Its -Y Wrist shadow therefore maps to
    shelf +X, toward the robot. This is a proper rotation, not a reflection.
    """
    points = np.asarray(points, dtype=np.float64)
    return np.stack((100.0 * points[..., 1], 20.0 - 100.0 * points[..., 0]), axis=-1)


def generator_cm_to_shelf_xy(points):
    """Inverse of :func:`shelf_xy_to_generator_cm`."""
    points = np.asarray(points, dtype=np.float64)
    return np.stack((0.20 - points[..., 1] / 100.0, points[..., 0] / 100.0), axis=-1)


class ClutterLayoutGenerator:
    """Persistent geometry RNG; three obstacles by default, commands supplied.

    Placement predicates, candidate probabilities, density and approach
    certificates use the source generator. Counts and RED clearance follow
    DEFAULT_LAYOUT_CONFIG. Retries may replace obstacles and their approach
    certificate, but never the target or goal.
    """

    def __init__(self, seed=0, config=None):
        self.config = config if config is not None else DEFAULT_LAYOUT_CONFIG
        self.config.check()
        if self.config.obstacles > 10:
            raise ValueError("Clutter observation supports at most 10 obstacles")
        self.rng = np.random.default_rng(seed)

    def generate(self, target_cm, goal_cm):
        """Return obstacle centres shaped (batch, obstacles, 2), in board cm."""
        return self.generate_batch(target_cm, goal_cm).obstacles

    def generate_batch(self, target_cm, goal_cm):
        """Return a SceneBatch including the original independent certificates."""
        target = np.asarray(target_cm, dtype=np.float64).copy()
        goal = np.asarray(goal_cm, dtype=np.float64).copy()
        if target.ndim != 2 or target.shape[1:] != (2,) or goal.shape != target.shape:
            raise ValueError("target_cm and goal_cm must have matching (batch, 2) shapes")
        if not np.isfinite(target).all() or not np.isfinite(goal).all():
            raise ValueError("Target and goal positions must be finite")
        c = self.config
        if not np.all((target >= c.safe_target_bounds[0]) & (target <= c.safe_target_bounds[1])):
            raise ValueError("Fixed target lies outside the scene generator's safe target bounds")
        if not np.all((goal >= c.object_bounds[0]) & (goal <= c.object_bounds[1])):
            raise ValueError("Fixed goal lies outside the scene generator's object bounds")
        delta = goal - target
        distance = np.linalg.norm(delta, axis=-1)
        if np.any((distance < c.distance_min) | (distance > c.distance_max)):
            raise ValueError("Fixed goal distance lies outside scene generator's supported range")
        angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi)
        direction = delta / distance[:, None]
        parts = []
        for start in range(0, len(target), c.chunk_size):
            stop = start + c.chunk_size
            try:
                parts.append(self._construct(
                    target[start:stop], goal[start:stop], angle[start:stop],
                    distance[start:stop], direction[start:stop],
                ))
            except GenerationError as exc:
                raise GenerationError(
                    exc.indices + start, exc.angle, exc.distance, exc.reasons,
                ) from exc
        if not parts:
            return scene_generator.SceneBatch(
                target, goal, angle, distance, np.empty((0, c.obstacles, 2)),
                np.empty((0, 11, 2)), np.empty(0, dtype=np.int32), c,
            )
        return scene_generator._join(parts, c)

    def _construct(self, target, goal, angle, distance, direction):
        c, rng, count = self.config, self.rng, len(target)
        obstacles = np.empty((count, c.obstacles, 2))
        approach = np.empty((count, 11, 2))
        attempts = np.zeros(count, dtype=np.int32)
        pending = np.arange(count)
        roles = (["rear"] * c.rear_obstacles
                 + ["gray"] * (c.gray_obstacles - c.rear_obstacles)
                 + ["white"] * (c.obstacles - c.gray_obstacles))
        reasons = {role: 0 for role in set(roles)}
        for _ in range(c.scene_attempts):
            if not len(pending):
                break
            ids = pending
            attempts[ids] += 1
            approach[ids] = scene_generator._approach(target[ids], direction[ids], rng, c)
            alive = np.ones(len(ids), dtype=bool)
            for slot, role in enumerate(roles):
                waiting = ids[alive]
                for _ in range(c.placement_rounds):
                    if not len(waiting):
                        break
                    q = scene_generator._proposal(
                        target[waiting], direction[waiting], distance[waiting], role, rng, c,
                    )
                    red, gray, rear = scene_generator.zone_masks(
                        q, target[waiting], direction[waiting], distance[waiting], c,
                    )
                    valid = rear if role == "rear" else gray if role == "gray" else ~red & ~gray
                    valid &= np.all((q >= c.object_bounds[0]) & (q <= c.object_bounds[1]), axis=-1)
                    existing = np.concatenate((target[waiting, None, :], obstacles[waiting, :slot]), axis=1)
                    d2 = np.sum((q[:, :, None, :] - existing[:, None, :, :]) ** 2, axis=-1)
                    valid &= np.all(d2 >= (2 * c.radius + c.gap_factor * c.finger_thickness) ** 2, axis=-1)
                    old_cells = scene_generator.cells(existing, c)
                    new_cells = scene_generator.cells(q, c)
                    occupancy = np.sum(new_cells[..., None] == old_cells[:, None, :], axis=-1)
                    valid &= occupancy < c.max_per_cell
                    near = d2 < c.density_radius ** 2
                    valid &= np.sum(near, axis=-1) <= c.max_neighbors
                    old_d2 = np.sum((existing[:, :, None, :] - existing[:, None, :, :]) ** 2, axis=-1)
                    old_degree = np.sum(old_d2 < c.density_radius ** 2, axis=-1) - 1
                    valid &= np.all(old_degree[:, None, :] + near <= c.max_neighbors, axis=-1)
                    valid &= scene_generator.polyline_distance2(q, approach[waiting]) >= (
                        c.radius + c.hand_radius + c.finger_clearance
                    ) ** 2
                    score = -np.log(np.maximum(rng.random(valid.shape), 1e-30)) * (1 + occupancy)
                    score[~valid] = np.inf
                    pick = np.argmin(score, axis=1)
                    accepted = np.isfinite(score[np.arange(len(waiting)), pick])
                    obstacles[waiting[accepted], slot] = q[np.flatnonzero(accepted), pick[accepted]]
                    waiting = waiting[~accepted]
                if len(waiting):
                    reasons[role] += len(waiting)
                    alive[np.isin(ids, waiting)] = False
                if not alive.any():
                    break
            pending = ids[~alive]
        if len(pending):
            raise GenerationError(pending, angle[pending], distance[pending], reasons)
        return scene_generator.SceneBatch(
            target, goal, angle, distance, obstacles, approach, attempts, c,
        )
