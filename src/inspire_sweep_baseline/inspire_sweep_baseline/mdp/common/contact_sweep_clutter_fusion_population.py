"""Episode-local active obstacle slots and absolute population schedules."""

import torch as torch

PARKING_HEIGHT = 100.0


def validate_capacity(max_gray_obstacles, max_white_obstacles, observation_slots=10):
    for name, count in (("max_gray_obstacles", max_gray_obstacles), ("max_white_obstacles", max_white_obstacles)):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if max_gray_obstacles + max_white_obstacles > observation_slots:
        raise ValueError(f"The fixed {4 * observation_slots}D obstacle observation supports at most {observation_slots} obstacle slots")


def active_population_mask(num_envs, max_gray_obstacles, max_white_obstacles, gray_obstacles, white_obstacles, device, observation_slots=10):
    validate_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)
    for name, count, maximum in (("gray", gray_obstacles, max_gray_obstacles), ("white", white_obstacles, max_white_obstacles)):
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= maximum:
            raise ValueError(f"{name}_obstacles must be an integer between zero and its capacity")
    mask = torch.zeros((num_envs, 1 + max_gray_obstacles + max_white_obstacles), dtype=torch.bool, device=device)
    mask[:, 0] = True
    mask[:, 1:1 + gray_obstacles] = True
    start = 1 + max_gray_obstacles
    mask[:, start:start + white_obstacles] = True
    return mask
