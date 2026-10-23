"""Batched episode mass/friction updates without USD edits or view readbacks."""

from dataclasses import dataclass
import torch as torch

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
    from ...assets.robot import HAND_CONTACT_BODY_NAMES

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
