"""Read-only collider evidence for Revision's fixed EEF contact-height gate.

Import after AppLauncher and call ``measure_contact_height_geometry(env)`` after
reset. This does not step physics, move the robot, or establish IK reachability.
It tests whether a frozen hand articulation can geometrically fit above a plane
while a sideways palm overlaps each object's vertical envelope below the gate.
"""

from itertools import product
import math

import numpy as np


def _rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _body_collider_points(stage, body_path):
    """Mesh vertices in the rigid-body frame; primitives use conservative boxes."""
    from pxr import Usd, UsdGeom, UsdPhysics

    body = stage.GetPrimAtPath(body_path)
    if not body:
        raise RuntimeError(f"Missing collider body: {body_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"],
                            useExtentsHint=False, ignoreVisibility=True)
    inverse_body = cache.GetLocalToWorldTransform(body).GetInverse()
    all_points, count, conservative, seen_geometry = [], 0, [], set()
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        if collision.GetCollisionEnabledAttr().Get() is False:
            continue
        ancestor = prim
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor = ancestor.GetParent()
        if ancestor and ancestor.GetPath() != body.GetPath():
            continue
        # Some source colliders carry CollisionAPI on an instance/Xform, with
        # the actual Mesh below it. Query leaf geometry, never an empty Xform
        # range (USD represents empty bounds by finite +/-FLT_MAX sentinels).
        geometry = [leaf for leaf in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                    if leaf.IsA(UsdGeom.Gprim)]
        if not geometry:
            raise RuntimeError(f"Collider has no readable descendant geometry: {prim.GetPath()}")
        for leaf in geometry:
            key = str(leaf.GetPath())
            if key in seen_geometry:
                continue
            seen_geometry.add(key)
            if leaf.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(leaf).GetPointsAttr().Get(), dtype=float)
            else:
                box = bbox.ComputeUntransformedBound(leaf).ComputeAlignedRange()
                if box.IsEmpty():
                    raise RuntimeError(f"Collider primitive has empty bounds: {leaf.GetPath()}")
                points = np.array(list(product(*zip(box.GetMin(), box.GetMax()))), dtype=float)
                conservative.append(key)
            if not points.size or points.ndim != 2 or points.shape[1] != 3:
                raise RuntimeError(f"Collider has no valid 3D vertices: {leaf.GetPath()}")
            matrix = np.asarray(cache.GetLocalToWorldTransform(leaf) * inverse_body, dtype=float)
            local = points @ matrix[:3, :3] + matrix[3, :3]
            if not np.isfinite(local).all() or np.max(np.abs(local)) >= 1.:
                raise RuntimeError(f"Implausible hand collider body-local geometry: {leaf.GetPath()}")
            all_points.append(local)
        count += 1
    return (np.concatenate(all_points) if all_points else np.empty((0, 3))), count, conservative


def side_clearance_intervals(hand_points_eef, palm_points_eef, object_heights,
                             height_limit=.12, clearance=.001, roll_samples=360,
                             palm_outward_normal_eef=(0., 0., 1.)):
    """Find height intervals for horizontal palm normal and a fixed joint shape.

    For every roll phi, the measured palm normal is horizontal. World Z in
    the EEF frame spans the plane perpendicular to that normal. With
    m=min((R p_i).z), non-penetration requires h >= clearance-m and the reward
    gate requires h < height_limit. Palm/object vertical overlap further limits
    the upper bound by object_height - min((R p_palm).z).
    """
    hand = np.asarray(hand_points_eef, dtype=float)
    palm = np.asarray(palm_points_eef, dtype=float)
    heights = np.asarray(object_heights, dtype=float)
    normal = np.asarray(palm_outward_normal_eef, dtype=float)
    normal = normal / np.linalg.norm(normal)
    reference = np.array((1., 0., 0.)) if abs(normal[0]) < .9 else np.array((0., 1., 0.))
    tangent_u = reference - (reference @ normal) * normal
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    if (not hand.size or not palm.size or not np.isfinite(hand).all() or not np.isfinite(palm).all()
            or np.max(np.abs(hand)) >= 1. or np.max(np.abs(palm)) >= 1.):
        raise ValueError("Finite nonempty hand and palm collision geometry is required")
    if height_limit <= 0 or clearance < 0 or roll_samples < 4 or np.any(heights <= 0):
        raise ValueError("Invalid gate, clearance, sampling count or object height")
    best = None
    for index in range(roll_samples):
        angle = 2 * math.pi * index / roll_samples
        vertical = math.sin(angle) * tangent_u + math.cos(angle) * tangent_v
        rotation = np.stack((normal, np.cross(vertical, normal), vertical))
        hand_z, palm_z = hand @ vertical, palm @ vertical
        lowest, palm_lowest = float(hand_z.min()), float(palm_z.min())
        lower = max(0., clearance - lowest)
        # Strictly below the fixed gate and strictly overlapping the side band.
        upper = np.minimum(height_limit, heights - palm_lowest) - 1.e-5
        widths = upper - lower
        score = float(widths.min())
        if best is None or score > best[0]:
            best = (score, {
                "roll_degrees": math.degrees(angle),
                "palm_outward_normal_eef": normal.tolist(),
                "palm_outward_normal_world": [1., 0., 0.],
                "eef_plus_z_world": rotation[:, 2].tolist(),
                "rotation_world_from_eef": rotation.tolist(),
                "hand_lowest_relative_to_eef_m": lowest,
                "palm_front_vertical_band_relative_to_eef_m": [palm_lowest, float(palm_z.max())],
                "minimum_eef_height_for_floor_clearance_m": lower,
                "object_specific_upper_eef_heights_m": upper.tolist(),
                "object_specific_feasible": (widths > 0).tolist(),
                "all_object_minimum_interval_width_m": score,
                "one_common_eef_height_m": float((lower + upper.min()) / 2) if score > 0 else None,
            })
    return best[1]


def measure_contact_height_geometry(env, env_id=0, height_limit=None, clearance=.001):
    """Return JSON-serializable evidence using current tensor body/EEF poses.

    USD is used only for static collider-local geometry; current world body
    poses come from PhysX tensors, so Fabric's unsynchronized USD poses cannot
    invalidate the measurement. Mesh convex-hull extrema equal vertex extrema.
    Primitive bounding boxes are conservative. Convex cooking contact margins,
    other robot links, IK and dynamic contact still need a physical rollout.
    """
    import isaaclab.sim as sim
    from inspire_sweep_baseline.assets.robot import HAND_CONTACT_BODY_NAMES
    from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT

    stage = sim.get_current_stage()
    robot = env.scene["robot"]
    eef = env.scene["ee_frame"].data
    eef_pos = eef.target_pos_w[env_id, 0].detach().cpu().numpy()
    eef_rotation = _rotation(eef.target_quat_w[env_id, 0].detach().cpu().numpy())
    # tactile_regions.json defines each pad's outward axis as sensor-local +Z.
    # Measure that axis, instead of assuming a sign from an EEF-frame comment.
    palm_sensor = robot.body_names.index("inspire_palm_force_sensor")
    palm_sensor_rotation = _rotation(robot.data.body_link_quat_w[env_id, palm_sensor].detach().cpu().numpy())
    palm_normal_eef = eef_rotation.T @ palm_sensor_rotation[:, 2]
    hand_points, body_rows, conservative = [], [], []
    palm_points = None
    for name in ("wrist_3_link", "axia80_link", *HAND_CONTACT_BODY_NAMES):
        if name not in robot.body_names:
            continue
        points, colliders, approximations = _body_collider_points(
            stage, f"/World/envs/env_{env_id}/Robot/{name}")
        if not points.size:
            continue
        index = robot.body_names.index(name)
        pos = robot.data.body_link_pos_w[env_id, index].detach().cpu().numpy()
        rot = _rotation(robot.data.body_link_quat_w[env_id, index].detach().cpu().numpy())
        world = points @ rot.T + pos
        in_eef = (world - eef_pos) @ eef_rotation
        hand_points.append(in_eef)
        body_rows.append({"name": name, "collider_count": colliders, "vertex_count": len(points),
                          "current_lowest_world_z_m": float(world[:, 2].min())})
        conservative.extend(approximations)
        if name == "inspire_base_link":
            # Frontmost means maximum support along the measured outward
            # normal. A measured -EEF Z selects minimum Z; +EEF Z selects max.
            projection = in_eef @ palm_normal_eef
            palm_points = in_eef[projection >= projection.max() - .003]
    if not hand_points or palm_points is None:
        raise RuntimeError("No actual hand/palm collider geometry was found")
    hand = np.concatenate(hand_points)
    initial_surface = float(env._revision_initial_shelf_pos[env_id, 2].item()) + SHELF_SURFACE_HEIGHT
    limit = float(env.cfg.contact_height_limit if height_limit is None else height_limit)
    base_bounds = np.asarray(env._revision_base_object_bounds)
    base_heights = base_bounds[:, 1, 2] - base_bounds[:, 0, 2]
    side = side_clearance_intervals(hand, palm_points, base_heights * .9, limit, clearance,
                                   palm_outward_normal_eef=palm_normal_eef)
    one_height = side["one_common_eef_height_m"]
    return {
        "method": "actual collider vertices, live body poses, hypothetical rigid side-push rotation",
        "scope": "geometric floor/side-height feasibility only; not IK, dynamics, or prevention of every top press",
        "eef_frame_body": env.cfg.scene.ee_frame.target_frames[0].prim_path,
        "eef_local_offset_m": list(env.cfg.scene.ee_frame.target_frames[0].offset.pos),
        "measured_palm_outward_normal_eef": palm_normal_eef.tolist(),
        "eef_current_height_above_initial_surface_m": float(eef_pos[2] - initial_surface),
        "hand_current_minimum_floor_clearance_m": min(row["current_lowest_world_z_m"] for row in body_rows) - initial_surface,
        "contact_height_limit_m": limit,
        "requested_geometric_floor_clearance_m": clearance,
        "base_object_names": list(env._revision_object_kind_names),
        "base_object_top_heights_m": base_heights.tolist(),
        "base_object_mean_top_height_m": float(base_heights.mean()),
        "smallest_scale_object_top_heights_m": (base_heights * .9).tolist(),
        "side_push": side,
        "side_push_proposed_world_eef_z_m": initial_surface + one_height if one_height is not None else None,
        "conservative_primitive_bounds": conservative,
        "body_geometry": body_rows,
    }
