"""Episode-local active obstacle slots and absolute population schedules."""

import torch

PARKING_HEIGHT = 100.0


def validate_capacity(max_gray_obstacles, max_white_obstacles):
    for name, count in (("max_gray_obstacles", max_gray_obstacles), ("max_white_obstacles", max_white_obstacles)):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if max_gray_obstacles + max_white_obstacles > 10:
        raise ValueError("The fixed 40D obstacle observation supports at most 10 obstacle slots")


def active_population_mask(num_envs, max_gray_obstacles, max_white_obstacles, gray_obstacles, white_obstacles, device):
    validate_capacity(max_gray_obstacles, max_white_obstacles)
    for name, count, maximum in (("gray", gray_obstacles, max_gray_obstacles), ("white", white_obstacles, max_white_obstacles)):
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= maximum:
            raise ValueError(f"{name}_obstacles must be an integer between zero and its capacity")
    mask = torch.zeros((num_envs, 1 + max_gray_obstacles + max_white_obstacles), dtype=torch.bool, device=device)
    mask[:, 0] = True
    mask[:, 1:1 + gray_obstacles] = True
    start = 1 + max_gray_obstacles
    mask[:, start:start + white_obstacles] = True
    return mask


def obstacle_population_schedule(
    env, env_ids, initial_gray=1, initial_white=1, final_gray=3, final_white=3,
    gray_start_iteration=4500, white_start_iteration=3000, interval=3000,
):
    """Update future reset counts; never change an ongoing episode's mask."""
    del env_ids
    values = (initial_gray, initial_white, final_gray, final_white, gray_start_iteration, white_start_iteration)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("Population counts and schedule start iterations must be nonnegative integers")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("Population interval must be a positive integer")
    if final_gray < initial_gray or final_white < initial_white:
        raise ValueError("Population final counts must not be smaller than initial counts")
    iteration = env.learning_iteration
    counts = {}
    for role, initial, final, start in (
        ("gray", initial_gray, final_gray, gray_start_iteration),
        ("white", initial_white, final_white, white_start_iteration),
    ):
        increments = 0 if iteration < start else (iteration - start) // interval + 1
        counts[role + "_obstacles"] = min(initial + increments, final, getattr(env.cfg, "max_" + role + "_obstacles"))
    term = env.event_manager.get_term_cfg("object_spawn")
    term.params.update(counts)
    env.event_manager.set_term_cfg("object_spawn", term)
    return counts


def apply_gravity_mask(env, env_ids):
    """Keep inactive bodies parked without changing the PhysX actor topology.

    View rows are object-major, matching RigidObjectCollection's own mapping.
    Inactive bodies have zero velocity at reset and no gravity, 100 m above the
    scene. They stay outside all task contacts. Simulation-disable toggles are
    deliberately avoided: this backend reordered disabled-body pose reads when
    populations changed, including a newly enabled actor's response.
    """
    objects = env.scene["object_collection"]
    flags = (~env._active_object_mask).T.reshape(-1, 1).to(device="cpu", dtype=torch.uint8).contiguous()
    ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
    rows = (torch.arange(objects.num_objects)[:, None] * env.num_envs + ids[None, :]).flatten()
    objects.root_physx_view.set_disable_gravities(flags, rows)
    active_rows = rows[flags[rows, 0] == 0]
    if active_rows.numel():
        objects.root_physx_view.wake_up(active_rows)
