"""Whole-hand normal contact vectors, separate from the 17 tactile channels.

Isaac Lab's ``net_forces_w`` and ``force_matrix_w`` report normal contact
forces, without tangential friction. Sum their world-frame XYZ vectors before
the reward takes a Euclidean norm; summing per-body norms would be a different
quantity. Contact against the shelf/ground is included in whole-hand contact,
while the obstacle view contains only non-target cylinders.
"""

import torch


WHOLE_HAND_SENSOR_NAME = "whole_hand_contact"


def object_hand_sensor_name(object_name):
    return "hand_contact_" + object_name


def whole_hand_contact_force_w(env):
    """Return the resultant normal force on all hand links and pads, [N, 3]."""
    forces = env.scene[WHOLE_HAND_SENSOR_NAME].data.net_forces_w
    if forces is None or forces.ndim != 3 or forces.shape[0] != env.num_envs or forces.shape[-1] != 3:
        raise RuntimeError("Whole-hand sensor must provide [num_envs, hand_bodies, 3] forces")
    return forces.sum(dim=1)


def hand_obstacle_contact_force_w(env):
    """Return only the non-target cylinders' resultant force on the hand.

    Each cylinder is the single reporting body and the hand is its filter.
    This is supported one-to-many filtering, unlike a sensor spanning multiple
    hand bodies with multiple object filters. Negating the cylinder reactions
    gives the force on the hand. Select by collection order and per-env target
    ID, so changing which cylinder is the target does not penalize that target.
    """
    objects = env.scene["object_collection"]
    object_forces = []
    for name in objects.object_names:
        matrix = env.scene[object_hand_sensor_name(name)].data.force_matrix_w
        if (
            matrix is None or matrix.ndim != 4 or matrix.shape[0] != env.num_envs
            or matrix.shape[1] != 1 or matrix.shape[2] < 1 or matrix.shape[-1] != 3
        ):
            raise RuntimeError("Object/hand contact sensor must provide [num_envs, 1, filters, 3] forces")
        object_forces.append(-matrix.sum(dim=(1, 2)))
    forces = torch.stack(object_forces, dim=1)
    object_ids = torch.arange(len(object_forces), device=forces.device).view(1, -1, 1)
    target_ids = env.target_id.reshape(env.num_envs, 1, 1)
    obstacles = object_ids != target_ids
    active = getattr(env, "_active_object_mask", None)
    if active is not None:
        if active.shape != forces.shape[:2] or active.dtype != torch.bool:
            raise ValueError("_active_object_mask must be boolean [num_envs, num_objects]")
        obstacles &= active[..., None]
    return torch.where(obstacles, forces, 0.0).sum(dim=1)
