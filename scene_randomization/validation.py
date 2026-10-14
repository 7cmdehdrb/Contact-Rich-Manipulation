"""Independent checks of returned geometry, rather than trusting placement masks."""

from __future__ import annotations

import numpy as np

from .generator import SceneBatch


def _segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched points B,P,2 and segments B,S,2 -> B,P,S Euclidean distances."""
    v = b - a
    w = points[:, :, None, :] - a[:, None, :, :]
    norm2 = np.einsum("bsi,bsi->bs", v, v)
    dot = np.einsum("bpsi,bsi->bps", w, v)
    fraction = np.clip(dot / np.maximum(norm2[:, None, :], 1e-30), 0, 1)
    residual = w - fraction[..., None] * v[:, None, :, :]
    return np.sqrt(np.einsum("bpsi,bpsi->bps", residual, residual))


def _downward_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Independent boundary + inside test for the segment's downward shadow.

    Uses endpoints and interpolation of y at each candidate x, rather than the
    generator's direction/length-based vertical-ray projection.
    """
    delta = end - start
    rel_start, rel_end = points - start[:, None], points - end[:, None]
    ray_start = np.hypot(rel_start[..., 0], np.maximum(rel_start[..., 1], 0))
    ray_end = np.hypot(rel_end[..., 0], np.maximum(rel_end[..., 1], 0))
    segment = _segment_distances(points, start[:, None], end[:, None])[..., 0]
    fraction = np.divide(rel_start[..., 0], delta[:, None, 0],
                         out=np.zeros(points.shape[:2]), where=delta[:, None, 0] != 0)
    at_y = start[:, None, 1] + fraction * delta[:, None, 1]
    inside = ((delta[:, None, 0] != 0) & (fraction >= 0) & (fraction <= 1)
              & (points[..., 1] <= at_y))
    return np.where(inside, 0, np.minimum(segment, np.minimum(ray_start, ray_end)))


def validate_scenes(batch: SceneBatch, tolerance: float = 1e-8) -> dict:
    """Validate ALL scenes; return failure counts and geometric extrema.

    No raster/grid discretization: every certificate segment is checked against
    every expanded disk. The intended endpoint may touch the target.
    """
    c = batch.config
    failures: dict[str, int] = {}
    minima = {"surface_gap_cm": float("inf"), "target_sweep_clearance_cm": float("inf"),
              "wrist_projection_clearance_cm": float("inf"),
              "approach_obstacle_clearance_cm": float("inf"),
              "approach_target_clearance_cm": float("inf")}
    interference = 0
    x0, x1, y0, y1 = c.board
    for start in range(0, len(batch), c.chunk_size):
        stop = start + c.chunk_size
        p, goal, obs, path = (getattr(batch, key)[start:stop]
                              for key in ("target", "goal", "obstacles", "approach"))
        angle, distance = batch.angle[start:stop], batch.distance[start:stop]
        d = np.stack((np.cos(angle), np.sin(angle)), -1)
        objects = np.concatenate((p[:, None, :], obs), axis=1)
        bad = {}
        bad["nonfinite"] = (~np.isfinite(objects).all(axis=(1, 2))
                            | ~np.isfinite(path).all(axis=(1, 2))
                            | ~np.isfinite(goal).all(axis=1)
                            | ~np.isfinite(angle) | ~np.isfinite(distance))
        bad["command_bounds"] = ((angle < 0) | (angle >= 2 * np.pi)
                                  | (distance < c.distance_min) | (distance > c.distance_max))
        bad["goal_displacement"] = np.linalg.norm(goal - p - distance[:, None] * d, axis=1) > tolerance
        bad["object_bounds"] = ~np.all((objects >= c.object_bounds[0] - tolerance)
                                       & (objects <= c.object_bounds[1] + tolerance), axis=(1, 2))
        bad["target_spawn"] = ~np.all((p >= c.safe_target_bounds[0] - tolerance)
                                      & (p <= c.safe_target_bounds[1] + tolerance), axis=1)
        bad["goal_bounds"] = ~np.all((goal >= c.object_bounds[0] - tolerance)
                                     & (goal <= c.object_bounds[1] + tolerance), axis=1)
        pairwise = np.linalg.norm(objects[:, :, None, :] - objects[:, None, :, :], axis=-1)
        diag = np.eye(c.obstacles + 1, dtype=bool)[None]
        pairs = np.where(diag, np.inf, pairwise)
        bad["minimum_gap"] = np.any(pairs < 2 * c.radius + c.gap_factor * c.finger_thickness - tolerance,
                                     axis=(1, 2))
        bad["local_density"] = np.any(np.sum(pairs < c.density_radius, axis=2) > c.max_neighbors, axis=1)
        ix = np.floor((objects[..., 0] - x0) / (x1 - x0) * c.cell_columns).astype(int)
        iy = np.floor((objects[..., 1] - y0) / (y1 - y0) * c.cell_rows).astype(int)
        cell = iy * c.cell_columns + ix
        bad["cell_capacity"] = np.any(np.sum(cell[:, :, None] == cell[:, None, :], axis=2)
                                      > c.max_per_cell, axis=1)
        bad["front_entry"] = path[:, 0, 1] > y0 - c.hand_radius + tolerance
        bad["approach_walls"] = np.any(
            (path[..., 0] < x0 + c.hand_radius + c.finger_clearance - tolerance)
            | (path[..., 0] > x1 - c.hand_radius - c.finger_clearance + tolerance)
            | (path[..., 1] > y1 - c.hand_radius - c.finger_clearance + tolerance), axis=1)
        bad["contact_pose"] = np.linalg.norm(path[:, -1] - (p - (c.radius + c.hand_radius) * d), axis=1) > tolerance
        target_approach = _segment_distances(p[:, None, :], path[:, :-1], path[:, 1:]).min(axis=(1, 2))
        bad["approach_target"] = target_approach < c.radius + c.hand_radius - tolerance
        minima["approach_target_clearance_cm"] = min(minima["approach_target_clearance_cm"],
                                                     float(target_approach.min() - c.radius - c.hand_radius))
        if c.obstacles:
            sweep = _segment_distances(obs, p[:, None, :], goal[:, None, :])[..., 0]
            rel = obs - p[:, None, :]
            xi = np.einsum("bpi,bi->bp", rel, d)
            eta = rel[..., 1] * d[:, None, 0] - rel[..., 0] * d[:, None, 1]
            rear = (xi >= -c.rear_length) & (xi <= 0) & (np.abs(eta) <= c.rear_width / 2)
            projected = _downward_distances(obs, p, goal)
            red = projected < 2 * c.radius + c.red_margin - tolerance
            gray = ((projected < c.gray_radius) | rear) & ~red
            bad["red_obstacle"] = np.any(sweep < 2 * c.radius + c.red_margin - tolerance, axis=1)
            bad["wrist_projection_obstacle"] = np.any(red, axis=1)
            bad["gray_count"] = ~np.all(gray[:, :c.gray_obstacles], axis=1)
            bad["white_count"] = np.any(gray[:, c.gray_obstacles:], axis=1)
            bad["rear_count"] = ~np.all(rear[:, :c.rear_obstacles], axis=1)
            access = _segment_distances(obs, path[:, :-1], path[:, 1:]).min(axis=2)
            bad["approach_obstacle"] = np.any(access < c.radius + c.hand_radius + c.finger_clearance - tolerance,
                                               axis=1)
            minima["surface_gap_cm"] = min(minima["surface_gap_cm"], float(pairs.min() - 2 * c.radius))
            minima["target_sweep_clearance_cm"] = min(minima["target_sweep_clearance_cm"],
                                                       float(sweep.min() - 2 * c.radius))
            minima["wrist_projection_clearance_cm"] = min(minima["wrist_projection_clearance_cm"],
                                                           float(projected.min() - 2 * c.radius))
            minima["approach_obstacle_clearance_cm"] = min(minima["approach_obstacle_clearance_cm"],
                                                            float(access.min() - c.radius - c.hand_radius))
            hand_goal = goal - (c.radius + c.hand_radius) * d
            hand_sweep = _segment_distances(obs, path[:, -1:, :], hand_goal[:, None, :])[..., 0]
            interference += int(np.any(hand_sweep < c.radius + c.hand_radius, axis=1).sum())
        for name, values in bad.items():
            failures[name] = failures.get(name, 0) + int(values.sum())
    return {"scenes_checked": len(batch), "valid": not any(failures.values()),
            "failures": failures,
            "minima": {key: value if np.isfinite(value) else None for key, value in minima.items()},
            "scenes_with_hand_push_interference": interference}


def distribution_report(batch: SceneBatch) -> dict:
    c = batch.config
    counts, _, _ = np.histogram2d(batch.angle, batch.distance,
                                bins=(c.angle_bins, c.distance_bins),
                                range=((0, 2 * np.pi), (c.distance_min, c.distance_max)))
    counts = counts.astype(int)
    harmonics = [float(abs(np.exp(1j * k * batch.angle).mean())) for k in range(1, 5)]
    r = ((batch.distance - c.distance_min) / (c.distance_max - c.distance_min))
    # Within-bin means also catch a correct count with e.g. all samples at edges.
    angle_jitter = batch.angle / (2 * np.pi / c.angle_bins) % 1
    distance_jitter = r * c.distance_bins % 1
    return {"angle_bins": c.angle_bins, "distance_bins": c.distance_bins,
            "joint_counts": counts.tolist(), "angle_counts": counts.sum(axis=1).tolist(),
            "distance_counts": counts.sum(axis=0).tolist(),
            "joint_count_min": int(counts.min()), "joint_count_max": int(counts.max()),
            "angle_fourier_magnitudes_1_to_4": harmonics,
            "distance_mean_cm": float(batch.distance.mean()),
            "distance_expected_mean_cm": (c.distance_min + c.distance_max) / 2,
            "normalized_distance_variance": float(r.var()),
            "uniform_expected_variance": 1 / 12,
            "angle_within_bin_mean": float(angle_jitter.mean()),
            "distance_within_bin_mean": float(distance_jitter.mean())}
