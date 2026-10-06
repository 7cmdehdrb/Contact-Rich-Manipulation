"""Horizontal palm poses with fingers facing away from the robot base."""

import torch


def push_v1_palm_rotation(direction_w: torch.Tensor) -> torch.Tensor:
    """Return H +Y along the planar push and H +Z toward projected world -X.

    H +X completes the right-handed frame. The thumb is consequently up on
    the right push side and down on the left, while both finger axes face
    away from the base without tilting the contact normal.
    """

    if direction_w.ndim < 1 or direction_w.shape[-1] != 3:
        raise ValueError("A push direction must have a trailing dimension of three")
    if not direction_w.dtype.is_floating_point:
        raise ValueError("A push direction must use a floating-point dtype")
    horizontal = direction_w.clone()
    horizontal[..., 2] = 0.0
    magnitude = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
    invalid = (~torch.isfinite(direction_w).all(dim=-1)) | (~torch.isfinite(magnitude[..., 0])) | (
        magnitude[..., 0] <= torch.finfo(direction_w.dtype).eps
    )
    if bool(invalid.any()):
        raise ValueError("A push direction must be finite with a nonzero horizontal component")
    normal = horizontal / magnitude
    outward = torch.zeros_like(normal)
    outward[..., 0] = -1
    finger = outward - (outward * normal).sum(dim=-1, keepdim=True) * normal
    finger_magnitude = torch.linalg.vector_norm(finger, dim=-1, keepdim=True)
    if bool((finger_magnitude[..., 0] <= torch.finfo(direction_w.dtype).eps).any()):
        raise ValueError("A push direction must leave a nonzero outward finger projection")
    finger = finger / finger_magnitude
    thumb = torch.linalg.cross(normal, finger, dim=-1)
    return torch.stack((thumb, normal, finger), dim=-1)
