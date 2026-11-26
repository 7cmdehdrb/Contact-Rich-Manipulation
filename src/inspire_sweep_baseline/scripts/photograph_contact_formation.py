"""Static Prof Revision v0_3_2 pre-grasp studio; no policy or physics stepping.

Run with Isaac Lab Python. Orbit the viewport and capture it yourself. Use
--headless --preview-only to write a preview and a reopenable static USD.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    from cli_args import apply_renderer_compatibility

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/contact_formation_photography'))
    parser.add_argument('--gap-mm', type=float, default=7.)
    parser.add_argument('--pose', choices=('top_near', 'top_far', 'side_near', 'side_far'), default='side_near')
    parser.add_argument('--retreat-cm', type=float, default=6.)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1200)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 2. <= args.gap_mm <= 20. or min(args.width, args.height) < 256:
        parser.error('Use gap 2--20 mm and image dimensions >=256.')
    if not 2. <= args.retreat_cm <= 20.:
        parser.error('Use retreat 2--20 cm.')
    if args.headless and not args.preview_only:
        parser.error('Use --preview-only with --headless.')
    args.enable_cameras = True
    apply_renderer_compatibility(args)
    app = AppLauncher(args).app
    try:
        import imageio.v3 as iio
        import numpy as np
        import torch
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        import isaaclab.sim as sim
        from isaaclab.sensors import Camera, CameraCfg
        from inspire_sweep_baseline.envs.prof_revision_envs import ProfRevisionV0_3_2EnvCfg
        from inspire_sweep_baseline.assets.robot import _set_world_transform, HAND_JOINT_NAMES
        from inspire_sweep_baseline.mdp.common.prof_revision_geometry import FINGER_BODIES, distance_angle_formation

        source = ProfRevisionV0_3_2EnvCfg()
        simulation = sim.SimulationContext(sim.SimulationCfg(
            device=args.device or 'cuda:0', gravity=(0., 0., 0.), use_fabric=False))
        cfg = source.scene.robot.spawn
        cfg.func('/World/Robot', cfg)
        stage = sim.get_current_stage()
        root = '/World/Robot/'

        def matrix(position, quaternion):
            q = np.array([*quaternion.GetImaginary(), quaternion.GetReal()])
            result = np.eye(4)
            result[:3, :3] = Rotation.from_quat(q).as_matrix()
            result[:3, 3] = position
            return result

        # Compute exact authored joint-frame FK on the CPU. No PhysX settling,
        # coupled finger controller, gravity, action, or checkpoint is involved.
        joints = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            p0, p1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not p0 or not p1:
                continue
            parent, child = str(p0[0]), str(p1[0])
            if 'inspire_' not in parent or 'inspire_' not in child:
                continue
            revolute = prim.IsA(UsdPhysics.RevoluteJoint)
            axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() if revolute else 'Z'
            joints.append((prim.GetName(), parent, child,
                matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                np.linalg.inv(matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())),
                np.eye(3)['XYZ'.index(axis)], revolute))
        names = list(HAND_JOINT_NAMES)
        limits = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in names:
                joint = UsdPhysics.RevoluteJoint(prim)
                limits[prim.GetName()] = np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
        if set(limits) != set(names):
            raise RuntimeError('Source hand joint roster changed.')
        regions_path = Path(cfg.usd_path).parent / 'tactile_regions.json'
        regions = {r['name']: r for r in json.loads(regions_path.read_text())['regions']}
        offsets = []
        for name in FINGER_BODIES:
            bounds = regions[name]['pad_bounds_m']
            lo, hi = np.array(bounds['min']), np.array(bounds['max'])
            offsets.append([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, hi[2]])
        offsets = np.array(offsets)

        def fk(values):
            angles = dict(zip(names, values))
            poses = {root + 'inspire_base_link': np.eye(4)}
            pending = joints.copy()
            while pending:
                previous = len(pending)
                for joint in pending.copy():
                    name, parent, child, before, after, axis, revolute = joint
                    if parent not in poses:
                        continue
                    rotation = np.eye(4)
                    if revolute:
                        rotation[:3, :3] = Rotation.from_rotvec(axis * angles[name]).as_matrix()
                    poses[child] = poses[parent] @ before @ rotation @ after
                    pending.remove(joint)
                if len(pending) == previous:
                    raise RuntimeError('Disconnected hand joint graph.')
            fingers = [poses[root + name] for name in FINGER_BODIES]
            points = np.array([p[:3, :3] @ offset + p[:3, 3] for p, offset in zip(fingers, offsets)])
            normals = np.array([p[:3, 2] for p in fingers])
            return poses, points, normals

        radius, height, gap = .03, .08, args.gap_mm / 1000.
        cylinder_axis = 1
        size = np.array([2*radius, height, 2*radius])
        lower = np.array([limits[n][0] for n in names] + [-.025, .025, .115])
        upper = np.array([min(limits[n][1], 1.6) for n in names] + [.025, .085, .195])

        def residual(values):
            _, points, normals = fk(values[:12])
            delta = values[12:] - points
            direction = delta / np.linalg.norm(delta, axis=1)[:, None]
            # Signed +Z pad normals must point INTO the object. This is
            # deliberately stricter than the source reward's two-sided cos².
            local = points - values[12:]
            outside = np.maximum(np.abs(local)-size/2, 0.)
            gaps = np.linalg.norm(outside, axis=1)
            radial = np.linalg.norm(np.delete(local, cylinder_axis, axis=1), axis=1) - radius
            cylinder_gap = np.linalg.norm(np.stack((np.maximum(radial, 0.),
                np.maximum(np.abs(local[:, cylinder_axis])-height/2, 0.)), axis=1), axis=1)
            return np.concatenate(((gaps-gap)*100, (cylinder_gap-gap)*70,
                ((normals-direction)*1.5).ravel(),
                np.minimum(radial, 0.)*150, (values[12:]-[0., .05, .155])*8))

        solutions_by_pose = {}
        for approach, cylinder_axis in (('top', 1), ('side', 0)):
            size = np.full(3, 2*radius)
            size[cylinder_axis] = height
            rng = np.random.default_rng(32)
            solutions = []
            for _ in range(8):
                initial = lower + (.15 + .65*rng.random(15))*(upper-lower)
                solutions.append(least_squares(residual, initial, bounds=(lower, upper), max_nfev=250))
            solutions_by_pose[approach] = min(solutions, key=lambda s: np.dot(s.fun, s.fun))
        solution = solutions_by_pose['top']
        size = np.array([2*radius, height, 2*radius])
        poses, points, normals = fk(solution.x[:12])
        center = solution.x[12:]
        # Palm down, fingers extending toward +Y; thumb and curled fingers
        # surround the upright cylinder from opposite sides and above.
        world = np.eye(4)
        world[:3, :3] = Rotation.from_euler('x', -90, degrees=True).as_matrix()
        world[:3, 3] = [0., 0., .25]
        cache = UsdGeom.XformCache()
        original_base = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root+'inspire_base_link'))
        mount = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root+'axia80_link')) * original_base.GetInverse()

        def author(path, transform):
            _set_world_transform(stage.GetPrimAtPath(path), Gf.Matrix4d(transform.T.tolist()))

        for path, pose in poses.items():
            author(path, world @ pose)
        author(root+'axia80_link', world @ np.array(mount).T)
        for prim in stage.GetPrimAtPath('/World/Robot').GetChildren():
            if UsdGeom.Imageable(prim) and not (prim.GetName().startswith('inspire_') or prim.GetName() == 'axia80_link'):
                UsdGeom.Imageable(prim).MakeInvisible()
        center_w = (world @ np.r_[center, 1.])[:3]
        target = sim.CylinderCfg(radius=radius, height=height,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.12, .42, .8), roughness=.38))
        target.func('/World/Target', target, translation=tuple(center_w))
        floor_z = center_w[2]-height/2-.001
        floor = sim.CuboidCfg(size=(200., 200., .01),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.19, .21, .25), roughness=1.))
        floor.func('/World/Backdrop', floor, translation=(0., 0., floor_z-.005))
        dome = sim.DomeLightCfg(intensity=1600., color=(.96, .97, 1.))
        dome.func('/World/Light', dome)

        guide = UsdGeom.Xform.Define(stage, '/World/Guides')
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*size/2+center
        corners_w = (world[:3, :3] @ corners.T).T + world[:3, 3]
        lines = []
        for i in range(8):
            for j in range(i+1, 8):
                if np.count_nonzero(np.abs(corners[i]-corners[j]) > 1e-8) == 1:
                    lines.extend([corners_w[i], corners_w[j]])

        def curves(path, vertices, color, width):
            curve = UsdGeom.BasisCurves.Define(stage, path)
            curve.CreateTypeAttr('linear')
            curve.CreateCurveVertexCountsAttr([2]*(len(vertices)//2))
            curve.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in vertices])
            curve.CreateWidthsAttr([width])
            curve.SetWidthsInterpolation('constant')
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        curves('/World/Guides/OBB', lines, (1., .7, .12), .00065)
        normal_lines = []
        for point, normal in zip(points, normals):
            for p in (point, point+normal*.022):
                normal_lines.append(world[:3, :3] @ p + world[:3, 3])
        curves('/World/Guides/PadNormals', normal_lines, (.15, 1., .4), .0009)

        def select_pose(name):
            approach, separation = name.split('_')
            solution = solutions_by_pose[approach]
            poses, points, normals = fk(solution.x[:12])
            center = solution.x[12:]
            rotation = Rotation.from_euler('x', -90, degrees=True).as_matrix() if approach == 'top' else (
                Rotation.from_euler('y', 90, degrees=True).as_matrix())
            retreat = args.retreat_cm / 100. if separation == 'far' else 0.
            shift = np.array([0., -retreat, 0.])
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = center_w - rotation @ center + rotation @ shift
            for path, pose in poses.items():
                author(path, transform @ pose)
            author(root+'axia80_link', transform @ np.array(mount).T)
            vertices = []
            for point, normal in zip(points, normals):
                for p in (point, point+normal*.022):
                    vertices.append(rotation @ p + transform[:3, 3])
            UsdGeom.BasisCurves(stage.GetPrimAtPath('/World/Guides/PadNormals')).GetPointsAttr().Set(
                [Gf.Vec3f(*map(float, p)) for p in vertices])
            # Measurements describe the displayed hand, including rigid retreat.
            axis = 1 if approach == 'top' else 0
            size = np.full(3, 2*radius)
            size[axis] = height
            displayed_points = points + shift
            gaps = np.linalg.norm(np.maximum(np.abs(displayed_points-center)-size/2, 0.), axis=1)
            direction = center-displayed_points
            cosines = (normals*direction/np.linalg.norm(direction, axis=1)[:, None]).sum(-1)
            formation = distance_angle_formation(torch.tensor(gaps[None]), torch.tensor(normals[None]),
                torch.tensor(direction[None]), source.formation_distance_scale)[0].item()
            report = {'pose': name, 'source': 'ProfRevisionV0_3_2EnvCfg.scene.robot', 'physics_steps': 0,
                'joint_positions_rad': dict(zip(names, solution.x[:12].tolist())),
                'finger_order': list(FINGER_BODIES), 'obb_gaps_m': gaps.tolist(),
                'signed_pad_center_cosines': cosines.tolist(), 'formation_raw': formation,
                'target_radius_m': radius, 'target_height_m': height, 'requested_gap_m': gap,
                'hand_retreat_m': retreat, 'hand_world_transform': transform.tolist(),
                'purpose': 'Authored illustration, not a trained-policy or collision-valid grasp.'}
            print('[Formation photo] Pose:', name, flush=True)
            return report

        # All authored bodies/joints are static before starting the renderer.
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.HasAttribute('physxArticulation:articulationEnabled'):
                prim.GetAttribute('physxArticulation:articulationEnabled').Set(False)
        camera = Camera(CameraCfg(prim_path='/World/PhotoCamera', width=args.width, height=args.height,
            data_types=['rgb'], spawn=sim.PinholeCameraCfg(focal_length=50., horizontal_aperture=36.,
                clipping_range=(.001, 100.), lock_camera=False)))
        simulation.reset()
        views = {'Three quarter': (.32, .46, .28), 'Opposite side': (-.32, .46, .28),
                 'Front': (0., .52, .17), 'Top': (.01, -.02, .58)}

        def select_view(name):
            look = center_w + np.array([0., .015, .065])
            eye = look + np.array(views[name])*1.25
            simulation.set_camera_view(eye.tolist(), look.tolist())
            camera.set_world_poses_from_view(torch.tensor([eye.tolist()], device=simulation.device),
                torch.tensor([look.tolist()], device=simulation.device))

        select_view('Three quarter')
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        reports = {}
        for name in ('top_near', 'top_far', 'side_near', 'side_far'):
            reports[name] = select_pose(name)
            for _ in range(32):
                simulation.render()
            camera.update(simulation.get_physics_dt(), force_recompute=True)
            rgb = camera.data.output['rgb'][0, :, :, :3].cpu().numpy()
            if rgb.std() < 2:
                raise RuntimeError('Blank preview: '+name)
            iio.imwrite(output/(name+'.png'), rgb)
            stage.Flatten().Export(str(output/(name+'.usdc')))
        report = select_pose(args.pose)
        iio.imwrite(output/'preview.png', iio.imread(output/(args.pose+'.png')))
        (output/'presets.json').write_text(json.dumps(reports, indent=2)+'\n', encoding='utf-8')
        (output/'metadata.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        snapshot_path = output/'contact_formation_scene.usdc'
        stage.Flatten().Export(str(snapshot_path))
        reopened = Usd.Stage.Open(str(snapshot_path))
        if any(UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
               for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)):
            raise RuntimeError('Exported photography scene has dynamic bodies.')
        saved_center = UsdGeom.XformCache().GetLocalToWorldTransform(
            reopened.GetPrimAtPath('/World/Target')).ExtractTranslation()
        if not np.allclose(saved_center, center_w, atol=1e-6):
            raise RuntimeError('Exported target transform changed.')
        print('[Formation photo] Ready:', output, flush=True)
        print(json.dumps(report, indent=2), flush=True)
        if args.preview_only:
            return
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        viewport.camera_path = '/World/PhotoCamera'
        viewport.resolution = (args.width, args.height)
        window = ui.Window('Contact Formation Photography', width=330, height=380)
        with window.frame:
            with ui.VStack(spacing=6):
                ui.Label('Frozen pre-grasp | orbit freely to capture')
                ui.Label('Pose (far = additional retreat from object)')
                for name in reports:
                    ui.Button(name.replace('_', ' ').title(), clicked_fn=lambda n=name: select_pose(n))
                ui.Label('Camera')
                for name in views:
                    ui.Button(name, clicked_fn=lambda n=name: select_view(n))
                guide_model = ui.SimpleBoolModel(True)
                guide_model.add_value_changed_fn(lambda model: UsdGeom.Imageable(guide).GetVisibilityAttr().Set(
                    'inherited' if model.as_bool else 'invisible'))
                ui.CheckBox(model=guide_model)
                ui.Label('Checkbox: OBB + fingerprint normal guides')
        print('[Formation photo] Interactive viewport and controls ready.', flush=True)
        while app.is_running():
            simulation.render()
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == '__main__':
    main()
