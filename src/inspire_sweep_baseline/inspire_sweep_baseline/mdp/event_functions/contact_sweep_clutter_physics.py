"""Episode physics with identical properties for target and obstacle cylinders."""

from ..common.realistic_physics import DYNAMIC_FRICTION_MEANS
from ..common.realistic_physics import EpisodePhysicsState
from ..common.realistic_physics import FrictionViewBuffer
from ..common.realistic_physics import STATIC_FRICTION_MEANS
from ..common.realistic_physics import hand_material_shape_ids
from ..common.realistic_physics import make_friction_buckets
from ..common.realistic_physics import sample_episode_properties
import math as math
import torch as torch

def initialize_clutter_materials(env, env_ids=None):
    """Cache all PhysX buffers once; the target is collection object zero."""
    del env_ids
    if hasattr(env, "_clutter_physics"):
        return
    objects = env.scene["object_collection"]
    capacity = getattr(env.cfg, "obstacle_observation_slots", 10)
    if not 1 <= objects.num_objects <= capacity + 1:
        raise ValueError(f"Clutter requires one target and at most {capacity} real obstacles")
    robot = env.scene["robot"]
    views = (env.scene["shelf"].root_physx_view, objects.root_physx_view, robot.root_physx_view)
    shapes = (
        torch.arange(views[0].max_shapes),
        torch.arange(views[1].max_shapes),
        hand_material_shape_ids(robot),
    )
    materials = tuple(
        FrictionViewBuffer(view, view.get_material_properties().cpu().clone(), ids)
        for view, ids in zip(views, shapes)
    )
    masses = objects.root_physx_view.get_masses().cpu().clone()
    inertias = objects.root_physx_view.get_inertias().cpu().clone()
    if masses.shape != (env.num_envs * objects.num_objects, 1):
        raise RuntimeError("Unexpected rigid-object collection mass buffer layout")
    env._clutter_physics = EpisodePhysicsState(
        materials=materials,
        default_mass=masses.clone(),
        default_inertia=inertias.clone(),
        mass_buffer=masses,
        inertia_buffer=inertias,
        friction_buckets=make_friction_buckets(),
        mass=masses[:env.num_envs, 0].clone(),
        static_friction=torch.tensor(STATIC_FRICTION_MEANS).expand(env.num_envs, 3).clone(),
        dynamic_friction=torch.tensor(DYNAMIC_FRICTION_MEANS).expand(env.num_envs, 3).clone(),
    )


def randomize_clutter_episode_physics(env, env_ids, mass_scale=1.5):
    """Draw once per environment, assigning the same object sample to all bodies.

    RigidObjectCollection's view rows are object-major: object_id * num_envs
    + env_id (see its reshape_view_to_data/reshape_data_to_view helpers).
    Shelf and robot views instead have one row per environment.
    """
    if not math.isfinite(mass_scale) or mass_scale <= 0.0:
        raise ValueError("mass_scale must be positive and finite")
    if env_ids is None:
        ids = torch.arange(env.num_envs)
    else:
        ids = torch.as_tensor(env_ids, device="cpu", dtype=torch.long)
    if ids.numel() == 0:
        return
    initialize_clutter_materials(env)
    state = env._clutter_physics
    objects = env.scene["object_collection"]
    object_ids = (
        torch.arange(objects.num_objects)[:, None] * env.num_envs + ids[None, :]
    ).flatten()
    mass, static, dynamic = sample_episode_properties(ids.numel(), state.friction_buckets)
    mass = mass * mass_scale
    # All shapes are identical. Copy the target's scaled inertia exactly, rather
    # than accumulating scale factors from the previous episode's properties.
    target_ratio = mass / state.default_mass[ids, 0]
    target_inertia = state.default_inertia[ids] * target_ratio[:, None]
    state.mass_buffer[object_ids] = mass.repeat(objects.num_objects)[:, None]
    state.inertia_buffer[object_ids] = target_inertia.repeat(objects.num_objects, 1)
    objects.root_physx_view.set_masses(state.mass_buffer, object_ids)
    objects.root_physx_view.set_inertias(state.inertia_buffer, object_ids)
    state.materials[0].apply(ids, static[:, 0], dynamic[:, 0])
    state.materials[1].apply(
        object_ids, static[:, 1].repeat(objects.num_objects), dynamic[:, 1].repeat(objects.num_objects),
    )
    state.materials[2].apply(ids, static[:, 2], dynamic[:, 2])
    state.mass[ids] = mass
    state.static_friction[ids] = static
    state.dynamic_friction[ids] = dynamic
