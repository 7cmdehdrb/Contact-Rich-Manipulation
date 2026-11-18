"""Choose inspection layouts with lateral overlap from the existing sampler."""
import numpy as np
import torch


def sample_inspection_layout(layout, generator, command_params):
    """Condition existing collision-free layouts on possible partial occlusion.

    Nonintersecting placement circles keep boxes spatially separated.
    Overlapping shelf-Y projections create frontal occlusion without moving
    sampled objects. Only the inspection distribution changes, not training.
    """
    for _ in range(256):
        result = layout.sample(torch.tensor([0]), generator, **command_params)
        local, yaw, active = result[:3]
        indices = active[0].nonzero().flatten().numpy()
        if len(indices) < 2:
            continue
        bounds = layout.bounds[0, indices].numpy()
        angle = yaw[0, indices].numpy().astype(float)
        offset = bounds.mean(axis=1)
        half = (bounds[:, 1] - bounds[:, 0]) / 2
        sine, cosine = np.sin(angle), np.cos(angle)
        centers = local[0, indices, 1].numpy() + sine*offset[:, 0] + cosine*offset[:, 1]
        radii = abs(sine)*half[:, 0] + abs(cosine)*half[:, 1]
        widths = 2*radii
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                overlap = min(centers[i]+radii[i], centers[j]+radii[j]) - max(
                    centers[i]-radii[i], centers[j]-radii[j])
                smaller_width = min(widths[i], widths[j])
                remaining = max(.02, .20*smaller_width)
                # Neither interval contains the other. Both have an exposed
                # side before accounting for other objects in the scene.
                if .02 <= overlap <= smaller_width-remaining:
                    return result
    raise RuntimeError("Could not sample a partially overlapping inspection layout in 256 attempts")
