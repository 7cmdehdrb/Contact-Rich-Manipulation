"""Episode Gaussian mass/friction changes, confined to the inherited variant."""

from dataclasses import dataclass

import torch

MASS_MEAN = 0.365
MASS_STD = 0.0675
MASS_LIMITS = (0.230, 0.500)
# Independent floor/object/Tool draws, with independent static/dynamic noise.
STATIC_FRICTION_MEANS = (0.315, 0.315, 0.585)
STATIC_FRICTION_STDS = (0.051, 0.040, 0.145)
DYNAMIC_FRICTION_MEANS = (0.252, 0.252, 0.468)
DYNAMIC_FRICTION_STDS = (0.041, 0.032, 0.116)


def sample_episode_properties(count):
    mass = (MASS_MEAN + MASS_STD * torch.randn(count)).clamp(*MASS_LIMITS)
    means = torch.tensor(STATIC_FRICTION_MEANS)
    stds = torch.tensor(STATIC_FRICTION_STDS)
    # Friction ±2σ is a confidence range, not a truncation interval. Only
    # negative Gaussian tails are clamped to the physically valid zero value.
    static = (means + stds * torch.randn(count, 3)).clamp_min(0.0)
    dynamic = (torch.tensor(DYNAMIC_FRICTION_MEANS)
               + torch.tensor(DYNAMIC_FRICTION_STDS) * torch.randn(count, 3)).clamp_min(0.0)
    return mass, static, dynamic


@dataclass
class EpisodePhysicsState:
    materials: list
    default_mass: torch.Tensor
    default_inertia: torch.Tensor
    mass: torch.Tensor
    static_friction: torch.Tensor
    dynamic_friction: torch.Tensor


def initialize_realistic_materials(env, env_ids=None):
    """Once at startup, bind private Average materials for each environment."""
    del env_ids
    if hasattr(env, "_realistic_physics"):
        return
    from isaaclab.sim.utils import get_current_stage
    from ..assets.realistic_materials import bind_role_materials, material_defaults
    from ..assets.robot import HAND_CONTACT_BODY_NAMES

    objects = env.scene["object_collection"]
    if objects.num_objects != 1:
        raise RuntimeError("Realistic physics requires exactly one cylinder")
    stage = get_current_stage()
    defaults = material_defaults(env.cfg.sim.physics_material)
    materials = []
    for index, env_path in enumerate(env.scene.env_prim_paths):
        paths = (
            (f"{env_path}/Shelf",),
            (f"{env_path}/mass_cylinder",),
            tuple(f"{env_path}/Robot/{name}" for name in HAND_CONTACT_BODY_NAMES),
        )
        role_materials = []
        for role, body_paths, mean in zip(("floor", "object", "tool"), paths, STATIC_FRICTION_MEANS):
            role_materials.append(bind_role_materials(
                stage, body_paths, f"/World/InspireSweepRealisticMaterials/env_{index}/{role}", mean, defaults,
            ))
        materials.append(role_materials)
    view = objects.root_physx_view
    env._realistic_physics = EpisodePhysicsState(
        materials=materials,
        default_mass=view.get_masses().cpu().clone(),
        default_inertia=view.get_inertias().cpu().clone(),
        mass=torch.full((env.num_envs,), MASS_MEAN),
        static_friction=torch.tensor(STATIC_FRICTION_MEANS).expand(env.num_envs, 3).clone(),
        dynamic_friction=torch.tensor(DYNAMIC_FRICTION_MEANS).expand(env.num_envs, 3).clone(),
    )


def randomize_episode_physics(env, env_ids=None):
    """Resample selected episodes without modifying geometry, COM or controls."""
    from pxr import Sdf

    ids = torch.arange(env.num_envs) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
    if ids.numel() == 0:
        return
    initialize_realistic_materials(env)
    state = env._realistic_physics
    mass, static, dynamic = sample_episode_properties(ids.numel())
    view = env.scene["object_collection"].root_physx_view
    masses = view.get_masses().cpu().clone()
    inertias = view.get_inertias().cpu().clone()
    masses[ids] = mass[:, None]
    # Scale the original bottom-heavy inertia, never the last sampled inertia.
    # The bottom-origin frame and low COM remain unchanged.
    ratio = mass / state.default_mass[ids, 0]
    inertias[ids] = state.default_inertia[ids] * ratio[:, None]
    view.set_masses(masses, ids)
    view.set_inertias(inertias, ids)
    with Sdf.ChangeBlock():
        for row, env_id in enumerate(ids.tolist()):
            for role, bindings in enumerate(state.materials[env_id]):
                for static_attr, dynamic_attr in bindings:
                    static_attr.Set(float(static[row, role]))
                    dynamic_attr.Set(float(dynamic[row, role]))
    state.mass[ids] = mass
    state.static_friction[ids] = static
    state.dynamic_friction[ids] = dynamic
