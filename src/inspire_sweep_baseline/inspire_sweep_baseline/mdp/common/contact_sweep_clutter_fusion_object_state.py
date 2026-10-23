"""Reward and failure reductions over the episode's active object slots."""

import torch as torch

def active_object_mask(env, object_values):
    """Return [env, object] membership, matching the supplied object's tensor.

    Older callers without variable object slots retain their all-active
    behavior. The reset event owns the mask; reward and failure queries never
    change membership during an episode.
    """
    mask = getattr(env, "_active_object_mask", None)
    shape = object_values.shape[:2]
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=object_values.device)
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError("_active_object_mask must be boolean [num_envs, num_objects]")
    return mask
