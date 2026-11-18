"""Reset pose calculations in the shelf frame; no simulation writes."""
import math
import torch
from isaaclab.utils.math import quat_apply, quat_mul


def hand_pose(center, direction, target_radius, default_pos, default_quat,
              hand_vertices_eef, draws, base_pos, mount_offset_eef, elevation=0., *, target_half_height):
    """Keep the palm (EEF +Z) left; use the dorsum for rightward goals.

    The nominal facing-surface gap is 5cm. Place the lowest hand collision
    vertex 5mm above the target bottom (the shelf top), independent of wrist
    roll and default EEF height. Rotate about the palm normal so the wrist/mount
    faces the robot, with the specified elevation in the shelf frame. Fixed roll/pitch can put the
    wrist beyond arm reach even when the EEF position alone is reachable.
    draws supplies three uniform [0,1] variates (shelf X/Y and yaw). Keeping
    random draws outside this calculation makes batch reset reproducible.
    """
    # Right-hand palm tactile outward +Z transforms to EEF +Z; -Z is the dorsum.
    normal = quat_apply(default_quat, default_pos.new_tensor([0., 0., 1.]).expand_as(default_pos))
    palm_direction = torch.where(direction[:, 1:2] > 0, -direction, direction)
    yaw = torch.atan2(palm_direction[:, 1], palm_direction[:, 0]) - torch.atan2(normal[:, 1], normal[:, 0])
    # Surface gap belongs to the nominal pose, before the agreed XY/yaw jitter.
    rotation = default_quat.new_zeros(default_quat.shape)
    rotation[:, 0], rotation[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
    nominal = quat_mul(rotation, default_quat)
    normal = quat_apply(nominal, default_pos.new_tensor([0., 0., 1.]).expand_as(default_pos))
    wrist = quat_apply(nominal, mount_offset_eef.expand_as(default_pos))
    horizontal = base_pos - center
    horizontal[:, 2] = 0.
    horizontal -= normal * (horizontal * normal).sum(-1, keepdim=True)
    length = horizontal.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(horizontal)
    up[:, 2] = 1.
    up -= normal * (up * normal).sum(-1, keepdim=True)
    up /= up.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    horizontal = torch.where(length > 1e-8, horizontal / length.clamp_min(1e-8), up)
    desired = math.cos(elevation) * horizontal + math.sin(elevation) * up
    wrist -= normal * (wrist * normal).sum(-1, keepdim=True)
    angle = torch.atan2((normal * torch.cross(wrist, desired, dim=-1)).sum(-1),
                        (wrist * desired).sum(-1))
    roll = torch.cat(((angle / 2).cos()[:, None], normal * (angle / 2).sin()[:, None]), -1)
    nominal = quat_mul(roll, nominal)
    vertices = hand_vertices_eef.expand(len(center), -1, -1)
    transformed = quat_apply(nominal[:, None].expand(-1, vertices.shape[1], -1), vertices)
    support = (transformed * direction[:, None]).sum(-1).amax(-1)
    pos = default_pos.clone()
    pos[:, 2] = center[:, 2] - target_half_height + .005 - transformed[..., 2].amin(-1)
    pos[:, :2] = center[:, :2] - direction[:, :2] * (target_radius + .05 + support)[:, None]
    pos[:, :2] += (2 * draws[:, :2] - 1) * .01
    yaw = (2 * draws[:, 2] - 1) * math.radians(10)
    rotation[:, 0], rotation[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
    return pos, quat_mul(rotation, nominal)
