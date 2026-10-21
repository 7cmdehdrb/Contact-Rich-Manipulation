"""Reset-time front visibility over the existing fixed-command clutter layout."""

import importlib.util
import math
from pathlib import Path
import sys

import torch
from isaaclab.utils.math import quat_apply_inverse

from ..contact_sweep_clutter.events import randomize_clutter_scene


def _load_visibility_classifier():
    """Reuse the workspace geometry tool without changing global import paths."""
    name = "_inspire_contact_sweep_scene_visibility"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[3] / "scene_randomization" / "visibility.py"
        if not path.is_file():
            raise ImportError(f"Contact Sweep Clutter Fusion requires the visibility tool: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].classify_visibility_torch


classify_visibility_torch = _load_visibility_classifier()


def apply_initial_obstacle_visibility(
    env, env_ids, object_radius=0.04, visibility_threshold=0.35,
    visibility_flip_probability=0.01,
):
    """Mask the initial snapshot once, including every physical occluder.

    Visibility is evaluated before observation noise, using the disjoint upright
    cylinders just written by the placement event. The source classifier looks
    along +Y; (shelf Y, -shelf X) therefore makes shelf +X the robot-side front.
    Independent flips affect real obstacle slots only. Observation reads never
    resample this mask or recover hidden coordinates from current physics.
    """
    if not math.isfinite(object_radius) or object_radius <= 0.0:
        raise ValueError("object_radius must be finite and positive")
    if not math.isfinite(visibility_threshold) or not 0.0 < visibility_threshold <= 1.0:
        raise ValueError("visibility_threshold must be in (0, 1]")
    if not math.isfinite(visibility_flip_probability) or not 0.0 <= visibility_flip_probability <= 1.0:
        raise ValueError("visibility_flip_probability must be in [0, 1]")
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
    if ids.numel() == 0:
        return
    objects = env.scene["object_collection"]
    object_count = objects.num_objects
    obstacle_count = object_count - 1
    if not 1 <= obstacle_count <= 10:
        raise ValueError("Initial obstacle visibility requires 1 to 10 real obstacles")
    shelf = env.scene["shelf"].data
    relative_positions = objects.data.object_pos_w[ids] - shelf.root_pos_w[ids, None, :]
    shelf_positions = quat_apply_inverse(
        shelf.root_quat_w[ids, None, :].expand(-1, object_count, -1), relative_positions,
    )
    centers = torch.stack((shelf_positions[..., 1], -shelf_positions[..., 0]), dim=-1)
    # Placement guarantees finite, positive, disjoint geometry. Avoid the
    # optional CUDA-to-host input audit in this reset path.
    visible = classify_visibility_torch(
        centers, radii=object_radius, threshold=visibility_threshold, check_inputs=False,
    ).visible[:, 1:]
    flips = torch.rand(visible.shape, device=env.device) < visibility_flip_probability
    visible = torch.logical_xor(visible, flips)
    slots = env._initial_obstacle_observation[ids].reshape(-1, 10, 4)
    slots[:, :obstacle_count] *= visible[..., None]
    slots[:, obstacle_count:] = 0.0
    env._initial_obstacle_observation[ids] = slots.flatten(1)


def randomize_visible_clutter_scene(
    env, env_ids, spawn_radius=0.02, goal_distance=0.15, object_radius=0.04,
    edge_clearance=0.02, shelf_half_extents=(0.20, 0.50), shelf_surface_height=1.05,
    excluded_x_half_angle=math.pi / 3.0, observation_noise=0.01,
    visibility_threshold=0.35, visibility_flip_probability=0.01,
):
    """Preserve target/goal placement and physical obstacles; filter inputs only."""
    randomize_clutter_scene(
        env, env_ids, spawn_radius=spawn_radius, goal_distance=goal_distance,
        object_radius=object_radius, edge_clearance=edge_clearance,
        shelf_half_extents=shelf_half_extents, shelf_surface_height=shelf_surface_height,
        excluded_x_half_angle=excluded_x_half_angle, observation_noise=observation_noise,
    )
    apply_initial_obstacle_visibility(
        env, env_ids, object_radius=object_radius, visibility_threshold=visibility_threshold,
        visibility_flip_probability=visibility_flip_probability,
    )
