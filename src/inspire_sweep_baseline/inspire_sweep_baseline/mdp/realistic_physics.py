"""Batched episode mass/friction updates without USD edits or view readbacks."""

from dataclasses import dataclass
import math

import torch

MASS_MEAN = 0.365
MASS_STD = 0.0675
MASS_LIMITS = (0.230, 0.500)
STATIC_FRICTION_MEANS = (0.315, 0.315, 0.585)
STATIC_FRICTION_STDS = (0.051, 0.040, 0.145)
DYNAMIC_FRICTION_MEANS = (0.252, 0.252, 0.468)
DYNAMIC_FRICTION_STDS = (0.041, 0.032, 0.116)
FRICTION_BUCKET_COUNT = 4096


def make_friction_buckets(count=FRICTION_BUCKET_COUNT):
    """Finite Gaussian material pairs, avoiding PhysX's 64k material limit.

    Midpoint Gaussian quantiles retain the requested marginal mean/std with
    little finite-pool error. Independent permutations decorrelate dynamic
    friction from static friction. Each role is sampled separately at reset.
    """
    if count < 2:
        raise ValueError("At least two friction buckets are required")
    quantiles = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    normal = torch.special.ndtri(quantiles).float()
    pools = []
    for sm, ss, dm, ds in zip(STATIC_FRICTION_MEANS, STATIC_FRICTION_STDS,
                              DYNAMIC_FRICTION_MEANS, DYNAMIC_FRICTION_STDS):
        static = (sm + ss * normal).clamp_min(0.0)
        dynamic = (dm + ds * normal[torch.randperm(count)]).clamp_min(0.0)
        pools.append(torch.stack((static, dynamic), dim=-1))
    return torch.stack(pools)  # role x bucket x [static, dynamic]


def sample_episode_properties(count, friction_buckets=None):
    mass = (MASS_MEAN + MASS_STD * torch.randn(count)).clamp(*MASS_LIMITS)
    if friction_buckets is None:
        # Continuous reference sampler, used to verify distribution parameters.
        static = (torch.tensor(STATIC_FRICTION_MEANS)
                  + torch.tensor(STATIC_FRICTION_STDS) * torch.randn(count, 3)).clamp_min(0.0)
        dynamic = (torch.tensor(DYNAMIC_FRICTION_MEANS)
                   + torch.tensor(DYNAMIC_FRICTION_STDS) * torch.randn(count, 3)).clamp_min(0.0)
    else:
        ids = torch.randint(friction_buckets.shape[1], (count, 3))
        pairs = friction_buckets[torch.arange(3), ids]
        static, dynamic = pairs[..., 0], pairs[..., 1]
    return mass, static, dynamic


@dataclass
class FrictionViewBuffer:
    view: object
    properties: torch.Tensor
    shape_ids: torch.Tensor

    def apply(self, env_ids, static, dynamic):
        # All selected shapes share the role's sample. Restitution and other
        # shapes (Arm/Axia80) keep their existing values.
        self.properties[env_ids[:, None], self.shape_ids[None, :], 0] = static[:, None]
        self.properties[env_ids[:, None], self.shape_ids[None, :], 1] = dynamic[:, None]
        self.view.set_material_properties(self.properties, env_ids)


@dataclass
class EpisodePhysicsState:
    materials: tuple[FrictionViewBuffer, ...]
    default_mass: torch.Tensor
    default_inertia: torch.Tensor
    mass_buffer: torch.Tensor
    inertia_buffer: torch.Tensor
    friction_buckets: torch.Tensor
    mass: torch.Tensor
    static_friction: torch.Tensor
    dynamic_friction: torch.Tensor


def hand_material_shape_ids(robot):
    """Resolve Hand collider indices once, using the official shape mapping."""
    from ..assets.robot import HAND_CONTACT_BODY_NAMES

    selected = set(HAND_CONTACT_BODY_NAMES)
    missing = selected - set(robot.body_names)
    if missing:
        raise RuntimeError(f"Missing Hand friction bodies: {sorted(missing)}")
    ids, offset = [], 0
    paths = robot.root_physx_view.link_paths[0]
    if len(paths) != len(robot.body_names):
        raise RuntimeError("Articulation link paths/body names disagree")
    for name, path in zip(robot.body_names, paths):
        shapes = robot._physics_sim_view.create_rigid_body_view(path).max_shapes
        if name in selected:
            ids.extend(range(offset, offset + shapes))
        offset += shapes
    if offset != robot.root_physx_view.max_shapes or not ids:
        raise RuntimeError("Unexpected articulation material shape layout")
    return torch.tensor(ids, dtype=torch.long)


def initialize_realistic_materials(env, env_ids=None):
    """Cache mutable CPU buffers once; no USD work is permitted here."""
    del env_ids
    if hasattr(env, "_realistic_physics"):
        return
    objects = env.scene["object_collection"]
    if objects.num_objects != 1:
        raise RuntimeError("Realistic physics requires exactly one cylinder")
    robot = env.scene["robot"]
    views = (env.scene["shelf"].root_physx_view, objects.root_physx_view, robot.root_physx_view)
    shapes = (torch.arange(views[0].max_shapes), torch.arange(views[1].max_shapes),
              hand_material_shape_ids(robot))
    materials = tuple(FrictionViewBuffer(view, view.get_material_properties().cpu().clone(), ids)
                      for view, ids in zip(views, shapes))
    masses = objects.root_physx_view.get_masses().cpu().clone()
    inertias = objects.root_physx_view.get_inertias().cpu().clone()
    env._realistic_physics = EpisodePhysicsState(
        materials=materials, default_mass=masses.clone(), default_inertia=inertias.clone(),
        mass_buffer=masses, inertia_buffer=inertias, friction_buckets=make_friction_buckets(),
        mass=torch.full((env.num_envs,), MASS_MEAN),
        static_friction=torch.tensor(STATIC_FRICTION_MEANS).expand(env.num_envs, 3).clone(),
        dynamic_friction=torch.tensor(DYNAMIC_FRICTION_MEANS).expand(env.num_envs, 3).clone(),
    )


def randomize_episode_physics(env, env_ids=None):
    """Original reset event with the default object-mass distribution."""
    randomize_scaled_episode_physics(env, env_ids, mass_scale=1.0)


def randomize_scaled_episode_physics(env, env_ids, mass_scale=1.0):
    """One batched update per view, independent of the number of reset rows."""
    if not math.isfinite(mass_scale) or mass_scale <= 0.0:
        raise ValueError("mass_scale must be positive and finite")
    if env_ids is None:
        ids = torch.arange(env.num_envs)
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(device="cpu", dtype=torch.long)
    else:
        ids = torch.tensor(env_ids, dtype=torch.long)
    if ids.numel() == 0:
        return
    initialize_realistic_materials(env)
    state = env._realistic_physics
    mass, static, dynamic = sample_episode_properties(ids.numel(), state.friction_buckets)
    mass = mass * mass_scale
    view = env.scene["object_collection"].root_physx_view
    state.mass_buffer[ids] = mass[:, None]
    ratio = mass / state.default_mass[ids, 0]
    state.inertia_buffer[ids] = state.default_inertia[ids] * ratio[:, None]
    view.set_masses(state.mass_buffer, ids)
    view.set_inertias(state.inertia_buffer, ids)
    for role, material in enumerate(state.materials):
        material.apply(ids, static[:, role], dynamic[:, role])
    state.mass[ids] = mass
    state.static_friction[ids] = static
    state.dynamic_friction[ids] = dynamic
