"""Batched episode mass/friction updates without USD edits or view readbacks."""

from ..common.realistic_physics import DYNAMIC_FRICTION_MEANS
from ..common.realistic_physics import EpisodePhysicsState
from ..common.realistic_physics import FrictionViewBuffer
from ..common.realistic_physics import MASS_MEAN
from ..common.realistic_physics import STATIC_FRICTION_MEANS
from ..common.realistic_physics import hand_material_shape_ids
from ..common.realistic_physics import make_friction_buckets
from ..common.realistic_physics import sample_episode_properties
import math as math
import torch as torch

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
