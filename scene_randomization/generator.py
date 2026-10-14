"""Vectorized rejection within scenes; push commands are never resampled on failure.

RED and the outer GRAY capsule extend in world -y to account for the connected
Tool/Wrist. An additional circular-finger approach certificate replaces grid
search. This is a geometric occupancy rule, not a full robot kinematic model.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
import math

import numpy as np

RED, GRAY, WHITE = 0, 1, 2


@dataclass(frozen=True)
class Config:
    zone_model: str = "wrist_downward_v2"
    board: tuple[float, float, float, float] = (-50.0, 50.0, 0.0, 40.0)
    target_bounds: tuple[float, float, float, float] = (-30.0, 30.0, 12.0, 28.0)
    radius: float = 4.0
    height: float = 10.0  # metadata only; this is a 2D geometric model
    wall_margin: float = 0.5
    distance_min: float = 5.0
    distance_max: float = 20.0
    finger_thickness: float = 2.0
    finger_clearance: float = 0.1
    contact_clearance: float = 0.25
    gap_factor: float = 0.8
    red_margin: float = 0.3
    gray_radius: float = 14.0
    rear_length: float = 16.0
    rear_width: float = 22.0
    obstacles: int = 6
    gray_obstacles: int = 3
    rear_obstacles: int = 1
    cell_columns: int = 5
    cell_rows: int = 2
    max_per_cell: int = 3
    density_radius: float = 18.0
    max_neighbors: int = 3  # excluding the object itself, including target
    angle_bins: int = 24
    distance_bins: int = 8
    candidates: int = 32
    placement_rounds: int = 12
    scene_attempts: int = 40
    chunk_size: int = 512

    @property
    def object_bounds(self) -> np.ndarray:
        x0, x1, y0, y1 = self.board
        m = self.radius + self.wall_margin
        return np.array([[x0 + m, y0 + m], [x1 - m, y1 - m]])

    @property
    def hand_radius(self) -> float:
        return self.finger_thickness / 2

    @property
    def orbit_radius(self) -> float:
        return self.radius + self.hand_radius + self.contact_clearance

    @property
    def safe_target_bounds(self) -> np.ndarray:
        x0, x1, y0, y1 = self.target_bounds
        bx0, bx1, _, by1 = self.board
        m = self.orbit_radius + self.hand_radius + self.finger_clearance
        # A full approach ring must clear side/back walls; front has no wall.
        return np.array([
            [max(x0, bx0 + m, self.object_bounds[0, 0]),
             max(y0, self.object_bounds[0, 1])],
            [min(x1, bx1 - m, self.object_bounds[1, 0]),
             min(y1, by1 - m, self.object_bounds[1, 1])],
        ])

    def check(self) -> None:
        if self.zone_model != "wrist_downward_v2":
            raise ValueError("Unsupported zone_model; regenerate scenes with wrist_downward_v2")
        positive = ("radius", "height", "finger_thickness", "distance_min", "gray_radius",
                    "rear_length", "rear_width", "density_radius")
        nonnegative = ("wall_margin", "finger_clearance", "contact_clearance",
                       "gap_factor", "red_margin")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or (value <= 0 if name in positive else value < 0):
                raise ValueError(f"Invalid {name}: {value}")
        for name in ("cell_columns", "cell_rows", "max_per_cell", "angle_bins",
                     "distance_bins", "candidates", "placement_rounds", "scene_attempts",
                     "chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("obstacles", "gray_obstacles", "rear_obstacles", "max_neighbors"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("board", "target_bounds"):
            b = np.asarray(getattr(self, name))
            if b.shape != (4,) or not np.isfinite(b).all() or b[0] >= b[1] or b[2] >= b[3]:
                raise ValueError(f"Invalid rectangle: {name}")
        if not math.isfinite(self.distance_max) or self.distance_max <= self.distance_min:
            raise ValueError("distance_max must exceed distance_min")
        if not 0 <= self.rear_obstacles <= self.gray_obstacles <= self.obstacles:
            raise ValueError("Require 0 <= rear_obstacles <= gray_obstacles <= obstacles")
        if self.gray_radius <= 2 * self.radius + self.red_margin:
            raise ValueError("GRAY radius must exceed RED radius")
        if self.orbit_radius * math.cos(math.pi / 16) < (
                self.radius + self.hand_radius + self.finger_clearance):
            raise ValueError("contact_clearance too small for the 16-sided approach ring")
        t, g = self.safe_target_bounds, self.object_bounds
        if np.any(t[0] >= t[1]) or np.any(g[0] >= g[1]):
            raise ValueError("Empty target/object domain")
        # Sufficient AND necessary for every displacement in the D_max disk to
        # have a nonempty rectangular target/goal intersection (up to boundaries).
        if np.any(g[0] - t[1] > -self.distance_max) or np.any(g[1] - t[0] < self.distance_max):
            raise ValueError("Full angle/distance support is impossible with these bounds; "
                             "reduce distance_max or widen target/goal bounds")
        if self.obstacles + 1 > self.cell_columns * self.cell_rows * self.max_per_cell:
            raise ValueError("Cell capacity is smaller than the object count")


@dataclass
class SceneBatch:
    target: np.ndarray
    goal: np.ndarray
    angle: np.ndarray
    distance: np.ndarray
    obstacles: np.ndarray
    approach: np.ndarray
    attempts: np.ndarray
    config: Config

    def __len__(self) -> int:
        return len(self.angle)

    @property
    def direction(self) -> np.ndarray:
        return np.stack((np.cos(self.angle), np.sin(self.angle)), axis=-1)

    def save(self, path: str) -> None:
        import json
        np.savez_compressed(path, **{
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "config"
        }, config_json=np.array(json.dumps(asdict(self.config))))

    @classmethod
    def load(cls, path: str) -> "SceneBatch":
        import json
        with np.load(path, allow_pickle=False) as data:
            config_data = json.loads(str(data["config_json"]))
            if "zone_model" not in config_data:
                raise ValueError("Legacy scenes use capsule-only RED; regenerate them for wrist_downward_v2")
            cfg = Config(**config_data)
            cfg.check()
            return cls(**{f.name: data[f.name].copy() for f in fields(cls)
                          if f.name != "config"}, config=cfg)


class GenerationError(RuntimeError):
    def __init__(self, indices: np.ndarray, angle: np.ndarray, distance: np.ndarray,
                 reasons: dict[str, int]):
        self.indices, self.angle, self.distance, self.reasons = indices, angle, distance, reasons
        super().__init__(f"Could not construct {len(indices)} scenes; commands were NOT "
                         f"replaced. Failed indices: {indices[:10].tolist()}; "
                         f"placement failures by role: {reasons}")

    def __reduce__(self):
        return type(self), (self.indices, self.angle, self.distance, self.reasons)


def cells(points: np.ndarray, c: Config) -> np.ndarray:
    x0, x1, y0, y1 = c.board
    ij = np.floor((points - [x0, y0]) / [x1 - x0, y1 - y0]
                  * [c.cell_columns, c.cell_rows]).astype(int)
    ij = np.clip(ij, [0, 0], [c.cell_columns - 1, c.cell_rows - 1])
    return ij[..., 1] * c.cell_columns + ij[..., 0]


def downward_sweep_distance2(points: np.ndarray, target: np.ndarray,
                             direction: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Distance squared to S + {(0,-t): t>=0}, without rasterization.

    Its boundary is S and the two downward endpoint rays. Outside that convex
    set, the closest point lies on this boundary; inside, the distance is zero.
    A downward ray at the closest x on S detects interior points, including
    horizontal/vertical paths. All operations are vectorized over B,C.
    """
    rel = points - target[:, None, :]
    xi = np.sum(rel * direction[:, None, :], axis=-1)
    closest = np.clip(xi, 0, distance[:, None])[..., None] * direction[:, None, :]
    segment_d2 = np.sum((rel - closest) ** 2, axis=-1)
    end_rel = rel - distance[:, None, None] * direction[:, None, :]
    start_ray_d2 = rel[..., 0] ** 2 + np.maximum(rel[..., 1], 0) ** 2
    end_ray_d2 = end_rel[..., 0] ** 2 + np.maximum(end_rel[..., 1], 0) ** 2
    at_x = np.divide(rel[..., 0], direction[:, None, 0], out=np.zeros_like(xi),
                     where=direction[:, None, 0] != 0)
    at_x = np.clip(at_x, 0, distance[:, None])
    vertical_rel = rel - at_x[..., None] * direction[:, None, :]
    vertical_d2 = vertical_rel[..., 0] ** 2 + np.maximum(vertical_rel[..., 1], 0) ** 2
    return np.minimum(np.minimum(segment_d2, vertical_d2), np.minimum(start_ray_d2, end_ray_d2))


def zone_masks(points: np.ndarray, target: np.ndarray, direction: np.ndarray,
               distance: np.ndarray, c: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World -y extrusion of inner/outer capsules; rear quota stays target-relative.

    points=(B,C,2); return RED, GRAY, rear masks for obstacle CENTERS. Callers
    intersect these masks with object_bounds before accepting any candidate.
    """
    rel = points - target[:, None, :]
    xi = np.sum(rel * direction[:, None, :], axis=-1)
    eta = rel[..., 0] * -direction[:, None, 1] + rel[..., 1] * direction[:, None, 0]
    projected_d2 = downward_sweep_distance2(points, target, direction, distance)
    red = projected_d2 < (2 * c.radius + c.red_margin) ** 2
    rear = (xi >= -c.rear_length) & (xi <= 0) & (np.abs(eta) <= c.rear_width / 2)
    gray = ((projected_d2 < c.gray_radius ** 2) | rear) & ~red
    return red, gray, rear & ~red


def polyline_distance2(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Squared point-to-polyline distance; supports padded zero-length segments."""
    start = path[:, :-1, :]
    seg = path[:, 1:, :] - start
    rel = points[:, :, None, :] - start[:, None, :, :]
    denom = np.sum(seg * seg, axis=-1)
    t = np.clip(np.sum(rel * seg[:, None, :, :], axis=-1)
                / np.maximum(denom[:, None, :], 1e-30), 0, 1)
    delta = rel - t[..., None] * seg[:, None, :, :]
    return np.min(np.sum(delta * delta, axis=-1), axis=-1)


def _approach(target: np.ndarray, direction: np.ndarray, rng: np.random.Generator,
              c: Config) -> np.ndarray:
    """Front entry + shorter side of a circumscribed ring to the rear contact.

    Ring chords clear the inflated target, checked by Config.check. A final short
    segment closes contact_clearance to reach exact tangent contact with target.
    """
    contact_angle = np.arctan2(-direction[:, 1], -direction[:, 0])
    angles = contact_angle[:, None] + np.arange(16) * (2 * np.pi / 16)
    ring = target[:, None, :] + c.orbit_radius * np.stack((np.cos(angles), np.sin(angles)), -1)
    front_idx = np.argmin(ring[..., 1], axis=1)
    sign = np.where(front_idx < 8, -1, 1)
    sign = np.where(front_idx == 8, rng.choice([-1, 1], len(target)), sign)
    steps = np.minimum(front_idx, 16 - front_idx)
    idx = (front_idx[:, None] + sign[:, None] * np.minimum(np.arange(9), steps[:, None])) % 16
    arc = ring[np.arange(len(target))[:, None], idx]
    entry = arc[:, 0, :].copy()
    entry[:, 1] = c.board[2] - c.hand_radius - 1
    contact = target - (c.radius + c.hand_radius) * direction
    return np.concatenate((entry[:, None, :], arc, contact[:, None, :]), axis=1)


def _proposal(target: np.ndarray, direction: np.ndarray, distance: np.ndarray,
              role: str, rng: np.random.Generator, c: Config) -> np.ndarray:
    uv = rng.random((len(target), c.candidates, 2))
    if role == "white":
        lo, hi = c.object_bounds
        return lo + uv * (hi - lo)
    if role == "rear":
        xi = -c.rear_length * uv[..., 0]
        eta = c.rear_width * (uv[..., 1] - 0.5)
    else:
        # Sample the full extended GRAY, including the Wrist sides near y=front.
        # Its bounding rectangle covers the projected capsule and original rear.
        goal = target + distance[:, None] * direction
        lo, hi = c.object_bounds
        half_x = c.rear_width / 2 * np.abs(direction[:, 1])
        half_y = c.rear_width / 2 * np.abs(direction[:, 0])
        rear_end = target - c.rear_length * direction
        rear_lo_x = np.minimum(target[:, 0], rear_end[:, 0]) - half_x
        rear_hi_x = np.maximum(target[:, 0], rear_end[:, 0]) + half_x
        rear_hi_y = np.maximum(target[:, 1], rear_end[:, 1]) + half_y
        qlo = np.stack((np.maximum(lo[0], np.minimum(
            np.minimum(target[:, 0], goal[:, 0]) - c.gray_radius, rear_lo_x)),
            np.full(len(target), lo[1])), axis=-1)
        qhi = np.stack((np.minimum(hi[0], np.maximum(
            np.maximum(target[:, 0], goal[:, 0]) + c.gray_radius, rear_hi_x)),
            np.minimum(hi[1], np.maximum(
                np.maximum(target[:, 1], goal[:, 1]) + c.gray_radius, rear_hi_y))), axis=-1)
        return qlo[:, None, :] + uv * (qhi - qlo)[:, None, :]
    perp = np.stack((-direction[:, 1], direction[:, 0]), -1)
    return target[:, None, :] + xi[..., None] * direction[:, None, :] + eta[..., None] * perp[:, None, :]


def _construct(angle: np.ndarray, distance: np.ndarray, rng: np.random.Generator,
               c: Config) -> SceneBatch:
    n = len(angle)
    direction = np.stack((np.cos(angle), np.sin(angle)), -1)
    displacement = direction * distance[:, None]
    low = np.maximum(c.safe_target_bounds[0], c.object_bounds[0] - displacement)
    high = np.minimum(c.safe_target_bounds[1], c.object_bounds[1] - displacement)
    target = np.empty((n, 2))
    obstacles = np.empty((n, c.obstacles, 2))
    approach = np.empty((n, 11, 2))
    attempts = np.zeros(n, dtype=np.int32)
    pending = np.arange(n)
    roles = (["rear"] * c.rear_obstacles + ["gray"] * (c.gray_obstacles - c.rear_obstacles)
             + ["white"] * (c.obstacles - c.gray_obstacles))
    reasons = {role: 0 for role in set(roles)}
    for _ in range(c.scene_attempts):
        if not len(pending):
            break
        ids = pending
        attempts[ids] += 1
        target[ids] = low[ids] + rng.random((len(ids), 2)) * (high[ids] - low[ids])
        approach[ids] = _approach(target[ids], direction[ids], rng, c)
        alive = np.ones(len(ids), bool)
        for slot, role in enumerate(roles):
            waiting = ids[alive]
            for _ in range(c.placement_rounds):
                if not len(waiting):
                    break
                q = _proposal(target[waiting], direction[waiting], distance[waiting], role, rng, c)
                red, gray, rear = zone_masks(q, target[waiting], direction[waiting], distance[waiting], c)
                valid = rear if role == "rear" else gray if role == "gray" else ~red & ~gray
                valid &= np.all((q >= c.object_bounds[0]) & (q <= c.object_bounds[1]), axis=-1)
                existing = np.concatenate((target[waiting, None, :], obstacles[waiting, :slot]), axis=1)
                d2 = np.sum((q[:, :, None, :] - existing[:, None, :, :]) ** 2, axis=-1)
                valid &= np.all(d2 >= (2 * c.radius + c.gap_factor * c.finger_thickness) ** 2, axis=-1)
                old_cells, new_cells = cells(existing, c), cells(q, c)
                occupancy = np.sum(new_cells[..., None] == old_cells[:, None, :], axis=-1)
                valid &= occupancy < c.max_per_cell
                near = d2 < c.density_radius ** 2
                valid &= np.sum(near, axis=-1) <= c.max_neighbors
                old_d2 = np.sum((existing[:, :, None, :] - existing[:, None, :, :]) ** 2, axis=-1)
                old_degree = np.sum(old_d2 < c.density_radius ** 2, axis=-1) - 1
                valid &= np.all(old_degree[:, None, :] + near <= c.max_neighbors, axis=-1)
                valid &= polyline_distance2(q, approach[waiting]) >= (
                    c.radius + c.hand_radius + c.finger_clearance) ** 2
                # Exponential race: valid candidates in less occupied cells
                # receive greater probability, with no forced grid filling.
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
    return SceneBatch(target, target + displacement, angle, distance, obstacles, approach, attempts, c)


def _join(parts: list[SceneBatch], c: Config) -> SceneBatch:
    return SceneBatch(**{f.name: np.concatenate([getattr(p, f.name) for p in parts])
                         for f in fields(SceneBatch) if f.name != "config"}, config=c)


class SceneGenerator:
    """Persistent joint-bin schedule stays balanced across successive generate calls.

    One independently jittered sample per (angle, distance) bin per cycle. This
    gives continuous uniform marginals without a shared histogram or two-scene
    selection. Repeating the same seed creates the same stream.
    """
    def __init__(self, config: Config | None = None, seed: int = 0):
        self.config = config or Config()
        self.config.check()
        streams = np.random.SeedSequence(seed).spawn(2)
        self.command_rng = np.random.default_rng(streams[0])
        self.geometry_rng = np.random.default_rng(streams[1])
        self.schedule = np.empty(0, dtype=int)
        self.cursor = 0

    def commands(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        c = self.config
        remaining, ids = count, []
        while remaining:
            if self.cursor == len(self.schedule):
                self.schedule = self.command_rng.permutation(c.angle_bins * c.distance_bins)
                self.cursor = 0
            size = min(remaining, len(self.schedule) - self.cursor)
            ids.append(self.schedule[self.cursor:self.cursor + size])
            self.cursor += size
            remaining -= size
        bins = np.concatenate(ids)
        angle = (bins // c.distance_bins + self.command_rng.random(count)) * (2 * np.pi / c.angle_bins)
        distance = c.distance_min + (bins % c.distance_bins + self.command_rng.random(count)) * (
            (c.distance_max - c.distance_min) / c.distance_bins)
        return angle, distance

    def generate(self, count: int) -> SceneBatch:
        angle, distance = self.commands(count)
        c, parts = self.config, []
        for start in range(0, count, c.chunk_size):
            try:
                parts.append(_construct(angle[start:start + c.chunk_size], distance[start:start + c.chunk_size],
                                        self.geometry_rng, c))
            except GenerationError as exc:
                exc.indices += start
                raise GenerationError(exc.indices, exc.angle, exc.distance, exc.reasons) from exc
        return _join(parts, c)


def _worker(args: tuple) -> SceneBatch:
    angle, distance, seed, config, start = args
    try:
        return _construct(angle, distance, np.random.default_rng(seed), config)
    except GenerationError as exc:
        raise GenerationError(exc.indices + start, exc.angle, exc.distance, exc.reasons) from exc


def generate_parallel(count: int, config: Config | None = None, seed: int = 0,
                      workers: int = 1) -> SceneBatch:
    """Commands scheduled centrally; workers construct independent vectorized chunks.

    Use workers=1 for small batches: process startup/IPC often exceeds generation.
    worker count changes geometry, but not the globally scheduled push commands.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    generator = SceneGenerator(config, seed)
    if workers == 1:
        return generator.generate(count)
    c = generator.config
    angle, distance = generator.commands(count)
    starts = list(range(0, count, c.chunk_size))
    seeds = np.random.SeedSequence([seed, 1]).spawn(len(starts))
    jobs = [(angle[s:s + c.chunk_size], distance[s:s + c.chunk_size], child, c, s)
            for s, child in zip(starts, seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return _join(list(pool.map(_worker, jobs)), c)
