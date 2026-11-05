#!/usr/bin/env python3
"""Build a right-hand USD overlay and virtual tactile regions from the vendor ZIP.

CPU only: requires pxr, numpy and scipy. Existing articulation identifiers are
kept as compatibility slots; every hand mesh, joint frame and inertial property
comes from the RIGHT URDF, not a reflected left hand. No source USD is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

DATA = Path(__file__).resolve().parents[1] / 'inspire_sweep_baseline/assets/data'
HAND_MASS_KG = .790
MASS_SOURCE = 'https://en.inspire-robots.com/product/rh56e2/'


def slot(name):
    return 'inspire_' + name.replace('right_', 'left_')


def origin(element):
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ') if element is not None else np.zeros(3)
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ') if element is not None else np.zeros(3)
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    m[:3, 3] = xyz
    return m


def quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3]))


def stl_points(data):
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError('Expected binary vendor STL')
    triangles = np.ndarray(count, dtype=np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]), buffer=data, offset=84)
    vertices, inverse = np.unique(triangles['vertices'].reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, inverse


def build(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    base = DATA / 'ur5e_inspire_usd/ur5e_inspire.usd'
    source = Usd.Stage.Open(str(base))
    root_path = str(source.GetDefaultPrim().GetPath())
    mount = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(source.GetPrimAtPath(root_path + '/inspire_base_link'))).T
    with zipfile.ZipFile(archive) as z:
        urdf_name = next(n for n in z.namelist() if n.endswith('.urdf'))
        robot = ET.fromstring(z.read(urdf_name))
        links = {x.get('name'): x for x in robot.findall('link') if x.get('name') != 'world'}
        source_mass = sum(float(link.find('inertial/mass').get('value')) for link in links.values())
        mass_scale = HAND_MASS_KG / source_mass
        joints = [x for x in robot.findall('joint') if x.find('parent').get('link') != 'world']
        poses = {'base_link': np.eye(4)}
        pending = joints.copy()
        while pending:
            ready = [j for j in pending if j.find('parent').get('link') in poses]
            if not ready:
                raise ValueError('Disconnected hand URDF')
            for j in ready:
                poses[j.find('child').get('link')] = poses[j.find('parent').get('link')] @ origin(j.find('origin'))
                pending.remove(j)
        geometry = {}
        for name, link in links.items():
            mesh_name = link.find('collision/geometry/mesh').get('filename').split('/')[-1]
            data = z.read(next(n for n in z.namelist() if n.endswith('/meshes/' + mesh_name)))
            points, faces = stl_points(data)
            transform = origin(link.find('collision/origin'))
            points = (np.c_[points, np.ones(len(points))] @ transform.T)[:, :3]
            geometry[name] = points, faces

    path = output / 'ur5e_inspire_right.usda'
    stage = Usd.Stage.CreateNew(str(path)) if not path.exists() else Usd.Stage.Open(str(path))
    stage.GetRootLayer().Clear()
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    root.GetReferences().AddReference('../ur5e_inspire_usd/ur5e_inspire.usd')
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey('handedness', 'right')
    root.SetCustomDataByKey('hand_mass_kg', HAND_MASS_KG)
    root.SetCustomDataByKey('source_urdf_hand_mass_kg', source_mass)
    root.SetCustomDataByKey('hand_mass_scale', mass_scale)
    root.SetCustomDataByKey('hand_mass_source', MASS_SOURCE)
    root.SetCustomDataByKey('source_zip_sha256', hashlib.sha256(Path(archive).read_bytes()).hexdigest())
    for name, link in links.items():
        body_path = root_path + '/' + slot(name)
        body = stage.OverridePrim(body_path)
        for child in ('visuals', 'collisions'):
            stage.OverridePrim(body_path + '/' + child).SetActive(False)
        tf = UsdGeom.Xformable(body)
        tf.ClearXformOpOrder()
        tf.AddTransformOp(opSuffix='right').Set(Gf.Matrix4d((mount @ poses[name]).T))
        body.SetCustomDataByKey('vendor_right_link', name)
        inertial = link.find('inertial')
        inertia = inertial.find('inertia')
        a = {k: float(v) for k, v in inertia.attrib.items()}
        matrix = np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]])
        vals, axes = np.linalg.eigh(matrix)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        inertial_tf = origin(inertial.find('origin'))
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(float(inertial.find('mass').get('value')) * mass_scale)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*inertial_tf[:3, 3]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(vals * mass_scale)))
        mass.CreatePrincipalAxesAttr(quat(inertial_tf[:3, :3] @ axes))
        points, faces = geometry[name]
        mesh = UsdGeom.Mesh.Define(stage, body_path + '/right_geometry')
        mesh.CreatePointsAttr(points.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))
        mesh.CreateFaceVertexIndicesAttr(faces.tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([(.42, .42, .42) if 'sensor' in name else (.85, .85, .85)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    for j in joints:
        path = root_path + '/joints/' + slot(j.get('name'))
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        if not joint:
            raise ValueError('Missing compatibility joint: ' + path)
        tf = origin(j.find('origin'))
        axis_tf = np.eye(3)
        if j.get('type') == 'revolute':
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
            # USD axis is +Z. Rotate both joint frames for a vendor -Z axis.
            if np.allclose(axis, [0, 0, -1]):
                axis_tf = Rotation.from_euler('x', np.pi).as_matrix()
            elif not np.allclose(axis, [0, 0, 1]):
                raise ValueError('Unexpected vendor revolute axis')
            rev = UsdPhysics.RevoluteJoint(joint)
            rev.CreateAxisAttr('Z')
            rev.CreateLowerLimitAttr(np.degrees(float(j.find('limit').get('lower'))))
            rev.CreateUpperLimitAttr(np.degrees(float(j.find('limit').get('upper'))))
        joint.CreateLocalPos0Attr(Gf.Vec3f(*tf[:3, 3]))
        joint.CreateLocalRot0Attr(quat(tf[:3, :3] @ axis_tf))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(quat(axis_tf))
        joint.CreateBody0Rel().SetTargets([root_path + '/' + slot(j.find('parent').get('link'))])
        joint.CreateBody1Rel().SetTargets([root_path + '/' + slot(j.find('child').get('link'))])
    stage.GetRootLayer().Save()

    names = ['palm_force_sensor', *[f'thumb_force_sensor_{i}' for i in range(1, 5)],
             *[f'{finger}_force_sensor_{i}' for finger in ('index', 'middle', 'ring', 'little') for i in range(1, 4)]]
    regions = []
    for name in names:
        joint = next(j for j in joints if j.find('child').get('link') == name)
        parent = joint.find('parent').get('link')
        pad = geometry[name][0]
        parent_tf = np.linalg.inv(poses[name]) @ poses[parent]
        vertices = (np.c_[geometry[parent][0], np.ones(len(geometry[parent][0]))] @ parent_tf.T)[:, :3]
        footprint, hull = ConvexHull(pad[:, :2]), ConvexHull(vertices)
        regions.append({'name': slot(name), 'parent_name': slot(parent),
                        'footprint_halfspaces': np.unique(np.round(footprint.equations, 12), axis=0).tolist(),
                        'parent_halfspaces': np.unique(np.round(hull.equations, 12), axis=0).tolist(),
                        'pad_bounds_m': {'min': pad.min(0).tolist(), 'max': pad.max(0).tolist()}})
    metadata = {'schema_version': 2, 'handedness': 'right', 'source_zip_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                'hand_mass_kg': HAND_MASS_KG, 'source_urdf_hand_mass_kg': source_mass,
                'hand_mass_scale': mass_scale, 'hand_mass_source': MASS_SOURCE,
                'coordinate_frame': 'sensor_body_local', 'regions': regions,
                'compatibility_names': 'inspire_left_* are stable internal slots populated exclusively by vendor right-hand data'}
    (output / 'tactile_regions.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Built {len(links)} right-hand bodies, {len(joints)} joints, {len(regions)} paired regions: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=DATA / 'wrench_right_hand')
    args = parser.parse_args()
    build(args.archive, args.output)
