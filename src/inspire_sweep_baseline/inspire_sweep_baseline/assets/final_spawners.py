"""Bottom-origin shelf objects for Final v1: cube, mug and capped bottle.

The mug has a hollow wall and handle made of convex pieces; the bottle has
separate body, shoulder, neck and cap. Neither uses a box/cylinder collision
proxy for the entire object. Legacy scanned props are reused unchanged.
"""

from collections.abc import Callable
import math
from pxr import Gf, UsdGeom, UsdPhysics
import isaaclab.sim as sim
from isaaclab.utils import configclass
from .contact_sweep_clutter_fusion_holding_osc_object_random_spawners import prop_variants
from .contact_sweep_clutter_fusion_holding_osc_object_random_geometry import flat_collision_mesh
from .realistic_materials import bind_role_materials, material_defaults


def shelf_object_meshes(kind, cube_size=(0.08, 0.08, 0.08)):
    """Return closed convex pieces in metres, all supported at z=0."""
    pieces = []

    def box(size, center):
        points, counts, indices = flat_collision_mesh(size, 'box')
        pieces.append(([(x + center[0], y + center[1], z + center[2]) for x, y, z in points], counts, indices))

    def cylinder(radius, height, bottom, top_radius=None):
        points, counts, indices = flat_collision_mesh((2 * radius, 2 * radius, height), 'can')
        ratio = 1.0 if top_radius is None else top_radius / radius
        pieces.append(([(x * (ratio if z else 1), y * (ratio if z else 1), z + bottom)
                        for x, y, z in points], counts, indices))

    if kind == 'cube':
        if len(cube_size) != 3 or not all(math.isfinite(v) and v > 0 for v in cube_size):
            raise ValueError('cube_size must contain three positive finite dimensions')
        box(cube_size, (0, 0, 0))
    elif kind == 'bottle':
        cylinder(.03, .075, 0)
        cylinder(.03, .025, .075, top_radius=.012)
        cylinder(.012, .025, .100)
        cylinder(.014, .01, .125)
    elif kind == 'mug':
        # Center the complete body+handle envelope on the reporting origin.
        center_x, outer, inner, segments = -.012, .035, .029, 16
        points, counts, indices = flat_collision_mesh((2 * outer, 2 * outer, .006), 'can')
        pieces.append(([(x + center_x, y, z) for x, y, z in points], counts, indices))
        _, face_counts, face_indices = flat_collision_mesh((1, 1, 1), 'box')
        for i in range(segments):
            a, b = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
            ring = [(r * math.cos(t) + center_x, r * math.sin(t))
                    for r, t in ((outer, a), (outer, b), (inner, b), (inner, a))]
            pieces.append(([(x, y, z) for z in (.006, .10) for x, y in ring], face_counts, face_indices))
        box((.03, .012, .012), (.033, 0, .02))
        box((.03, .012, .012), (.033, 0, .073))
        box((.012, .012, .065), (.048, 0, .02))
    else:
        raise ValueError(f'Unknown shelf object: {kind}')
    return pieces


def spawn_shelf_object(prim_path, cfg, translation=None, orientation=None, **kwargs):
    stage = sim.get_current_stage()
    root = sim.create_prim(prim_path, 'Xform', translation=translation, orientation=orientation)
    UsdGeom.Xform.Define(stage, prim_path + '/Geometry')
    for index, (points, counts, indices) in enumerate(shelf_object_meshes(cfg.kind, cfg.cube_size)):
        path = f'{prim_path}/Geometry/Part_{index}'
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        sim.modify_collision_properties(path, cfg.collision_props)
    sim.define_rigid_body_properties(prim_path, cfg.rigid_props)
    sim.define_mass_properties(prim_path, cfg.mass_props)
    if cfg.activate_contact_sensors:
        sim.activate_contact_sensors(prim_path)
    bind_role_materials(stage, (prim_path,), prim_path + '/RealisticObjectMaterial',
                        .315, material_defaults(sim.RigidBodyMaterialCfg()))
    root.SetCustomDataByKey('final_object_kind', cfg.kind)
    return root


@configclass
class FinalShelfObjectCfg(sim.SpawnerCfg):
    func: Callable = spawn_shelf_object
    kind: str = 'mug'
    cube_size: tuple[float, float, float] = (0.08, 0.08, 0.08)
    color: tuple[float, float, float] = (.3, .65, .75)
    mass_props: sim.MassPropertiesCfg = sim.MassPropertiesCfg(mass=1.0)
    rigid_props: sim.RigidBodyPropertiesCfg = sim.RigidBodyPropertiesCfg()
    collision_props: sim.CollisionPropertiesCfg = sim.CollisionPropertiesCfg()
    activate_contact_sensors: bool = True


def final_prop_variants(original, target=False):
    """Target categories cylinder/cube/random have equal creation probability."""
    def procedural(kind):
        return FinalShelfObjectCfg(kind=kind, mass_props=original.mass_props.copy(),
            rigid_props=original.rigid_props.copy(), collision_props=original.collision_props.copy())

    random_props = prop_variants(original) + [procedural('mug'), procedural('bottle')]
    if not target:
        return random_props
    return ([original.copy() for _ in random_props]
            + [procedural('cube') for _ in random_props] + random_props)
