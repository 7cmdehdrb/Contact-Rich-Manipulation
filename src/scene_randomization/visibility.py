"""Exact front visibility of disjoint, equal-height upright cylinders.

Parallel rays travel from -y toward +y. Visibility means visible projected width
divided by full diameter (also area fraction for cylinders of equal height).
Between consecutive silhouette x-edges, disjoint circles cannot exchange depth
order. One midpoint ray therefore assigns the entire interval without sampling
error. Runtime/moved overlapping footprints and perspective cameras need a
different model. The default input audit rejects overlapping active circles.

NumPy has no Torch dependency; Torch is imported only by the tensor entrypoint.
``check_inputs=False`` skips numerical/overlap audits for already-validated hot
paths, avoiding device-to-host synchronization. Shape checks remain enabled.
"""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class VisibilityResult:
    ratio: Any                 # [B, N]
    visible: Any               # [B, N], >= threshold
    not_visible: Any           # [B, N], active but below threshold
    visible_width: Any         # [B, N], input length units
    edges: Any                 # [B, 2N]
    owners: Any                # [B, 2N-1], object index or -1 for empty intervals
    active: Any
    threshold: float


def _check_shape(centers, threshold):
    if len(centers.shape) != 3 or centers.shape[-1] != 2 or min(centers.shape[:2]) < 1:
        raise ValueError("centers must have nonempty shape [batch, objects, 2]")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")


def classify_visibility(centers, radii=4.0, active=None, *, threshold=0.35, check_inputs=True):
    """NumPy implementation; radii and active broadcast to [B,N].

    Float32/64 input precision is preserved; integer centers become float64.
    Inactive padding may contain NaNs and never occludes or receives visibility.
    """
    centers = np.asarray(centers)
    if centers.dtype not in (np.dtype("float32"), np.dtype("float64")):
        centers = centers.astype(np.float64)
    _check_shape(centers, threshold)
    shape = centers.shape[:2]
    radii = np.broadcast_to(np.asarray(radii, dtype=centers.dtype), shape)
    active = np.ones(shape, dtype=bool) if active is None else np.broadcast_to(np.asarray(active, dtype=bool), shape)
    if check_inputs and (not np.isfinite(centers[active]).all()
                         or not np.isfinite(radii[active]).all() or np.any(radii[active] <= 0)):
        raise ValueError("Active centers/radii must be finite and radii positive")
    centers = np.where(active[..., None], centers, 0)
    radii = np.where(active, radii, 1)
    n = shape[1]
    if check_inputs:
        d2 = np.sum((centers[:, :, None] - centers[:, None, :]) ** 2, axis=-1)
        pairs = active[:, :, None] & active[:, None, :] & np.triu(np.ones((n, n), dtype=bool), 1)
        if np.any(pairs & (d2 < (radii[:, :, None] + radii[:, None, :]) ** 2)):
            raise ValueError("Active circles overlap; interval visibility requires disjoint footprints")
    left = np.where(active, centers[..., 0] - radii, 0)
    right = np.where(active, centers[..., 0] + radii, 0)
    edges = np.sort(np.concatenate((left, right), axis=1), axis=1)
    widths = np.diff(edges, axis=1)
    ray_x = edges[:, :-1] + widths / 2
    dx = ray_x[..., None] - centers[:, None, :, 0]
    hit = active[:, None, :] & (np.abs(dx) <= radii[:, None, :])
    depth = centers[:, None, :, 1] - np.sqrt(np.maximum(radii[:, None, :] ** 2 - dx ** 2, 0))
    owner = np.argmin(np.where(hit, depth, np.inf), axis=-1)
    valid = (widths > 0) & hit.any(axis=-1)
    # Reduction over interval axis: small N keeps this [B,2N-1,N] tensor cheap.
    assignment = (owner[..., None] == np.arange(n)) & valid[..., None]
    visible_width = np.sum(assignment * widths[..., None], axis=1)
    ratio = np.where(active, np.clip(visible_width / (2 * radii), 0, 1), 0)
    return VisibilityResult(ratio, active & (ratio >= threshold), active & (ratio < threshold),
                            visible_width, edges, np.where(valid, owner, -1), active, threshold)


def classify_visibility_torch(centers, radii=4.0, active=None, *, threshold=0.35, check_inputs=True):
    """Tensor implementation on the input CPU/CUDA device, no NumPy conversion.

    The optional input audit synchronizes CUDA to raise useful Python errors.
    Disable it only after validating finite, positive, disjoint geometry.
    """
    import torch

    _check_shape(centers, threshold)
    if centers.dtype not in (torch.float32, torch.float64):
        raise ValueError("Tensor centers must be float32 or float64")
    shape = centers.shape[:2]
    radii = torch.broadcast_to(torch.as_tensor(radii, dtype=centers.dtype, device=centers.device), shape)
    active = (torch.ones(shape, dtype=torch.bool, device=centers.device) if active is None else
              torch.broadcast_to(torch.as_tensor(active, dtype=torch.bool, device=centers.device), shape))
    if check_inputs and (not bool(torch.isfinite(centers[active]).all())
                         or not bool(torch.isfinite(radii[active]).all()) or bool((radii[active] <= 0).any())):
        raise ValueError("Active centers/radii must be finite and radii positive")
    centers = torch.where(active[..., None], centers, 0.0)
    radii = torch.where(active, radii, 1.0)
    n = shape[1]
    if check_inputs:
        d2 = (centers[:, :, None] - centers[:, None, :]).square().sum(dim=-1)
        pairs = active[:, :, None] & active[:, None, :] & torch.triu(
            torch.ones((n, n), dtype=torch.bool, device=centers.device), diagonal=1)
        if bool((pairs & (d2 < (radii[:, :, None] + radii[:, None, :]).square())).any()):
            raise ValueError("Active circles overlap; interval visibility requires disjoint footprints")
    left = torch.where(active, centers[..., 0] - radii, 0.0)
    right = torch.where(active, centers[..., 0] + radii, 0.0)
    edges = torch.sort(torch.cat((left, right), dim=1), dim=1).values
    widths = edges[:, 1:] - edges[:, :-1]
    ray_x = edges[:, :-1] + widths / 2
    dx = ray_x[..., None] - centers[:, None, :, 0]
    hit = active[:, None, :] & (dx.abs() <= radii[:, None, :])
    depth = centers[:, None, :, 1] - (radii[:, None, :].square() - dx.square()).clamp_min(0).sqrt()
    owner = torch.where(hit, depth, float("inf")).argmin(dim=-1)
    valid = (widths > 0) & hit.any(dim=-1)
    visible_width = torch.zeros(shape, dtype=centers.dtype, device=centers.device)
    visible_width.scatter_add_(1, owner, torch.where(valid, widths, 0.0))
    ratio = torch.where(active, (visible_width / (2 * radii)).clamp(0, 1), 0.0)
    return VisibilityResult(ratio, active & (ratio >= threshold), active & (ratio < threshold),
                            visible_width, edges, torch.where(valid, owner, -1), active, threshold)
