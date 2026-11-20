"""Photograph the Sweep Only/Finger UR5e--Axia80--RH56E2 assembly, without RL.

Run with Isaac Lab's Python. --headless --capture-only saves preset PNGs and a
static USD; omit those options for a mouse-controlled viewport and photo panel.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    from cli_args import apply_renderer_compatibility

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robot_photography"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--capture-only", action="store_true", help="Save all presets, then exit.")
    parser.add_argument("--warmup-frames", type=int, default=48)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.width, args.height) < 256 or args.warmup_frames < 8:
        parser.error("Use dimensions >=256 and at least 8 warmup frames.")
    if args.headless and not args.capture_only:
        parser.error("Headless photography requires --capture-only.")
    args.enable_cameras = True
    apply_renderer_compatibility(args)
    app = AppLauncher(args).app
    try:
        import imageio.v3 as iio
        import numpy as np
        import torch
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
        import isaaclab.sim as sim
        from isaaclab.assets import Articulation
        from isaaclab.sensors import Camera, CameraCfg
        from inspire_sweep_baseline.assets.scenes.stabilize_sweep import _robot
        from inspire_sweep_baseline.assets.robot import HAND_CONTACT_BODY_NAMES, _set_world_transform

        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        # Reuse the exact latest scene's robot factory, including its RH56E2
        # right-hand overlay, revision spawner and two Axia80 fixed joints.
        robot_cfg = _robot().replace(prim_path="/World/Robot")
        simulation = sim.SimulationContext(sim.SimulationCfg(
            device=args.device or "cuda:0", dt=1 / 120, gravity=(0., 0., 0.), use_fabric=False,
        ))
        robot = Articulation(robot_cfg)
        stage = sim.get_current_stage()
        dome = sim.DomeLightCfg(intensity=650., color=(.96, .97, 1.))
        dome.func("/World/Studio/Dome", dome)
        for name, position, intensity, size in (
            ("Key", (0., -2., 3.5), 500., 2.),
            ("Fill", (-2., 1.5, 2.5), 300., 2.),
        ):
            light = UsdLux.SphereLight.Define(stage, "/World/Studio/" + name)
            light.CreateIntensityAttr(intensity)
            light.CreateRadiusAttr(size)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))
        # Large matte backdrop: no shelf, props, collision geometry or markers.
        backdrop = sim.CuboidCfg(size=(200., 200., .02),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.19, .21, .25), roughness=1.))
        backdrop.func("/World/Studio/Backdrop", backdrop, translation=(0., 0., -.08))
        camera = Camera(CameraCfg(
            prim_path="/World/PhotoCamera", width=args.width, height=args.height,
            data_types=["rgb"], spawn=sim.PinholeCameraCfg(
                focal_length=60., horizontal_aperture=36., clipping_range=(.005, 100.),
                lock_camera=False,
            ),
        ))
        simulation.reset()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        # One zero-gravity FK/PhysX update, followed only by render calls.
        simulation.step()
        robot.update(simulation.get_physics_dt())

        def bake_pose(destination):
            for index, name in enumerate(robot.body_names):
                state = robot.data.body_link_state_w[0, index].cpu().numpy()
                transform = Gf.Matrix4d(1.)
                transform.SetRotate(Gf.Quatd(float(state[3]), Gf.Vec3d(*map(float, state[4:7]))))
                transform.SetTranslateOnly(Gf.Vec3d(*map(float, state[:3])))
                _set_world_transform(destination.GetPrimAtPath("/World/Robot/" + name), transform)

        # GPU PhysX does not always publish manual joint writes into USD on the
        # first step. Both framing and delivered images must use actual FK.
        bake_pose(stage)
        for _ in range(8):
            simulation.render()

        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

        def corners(names):
            result = []
            for name in names:
                prim = stage.GetPrimAtPath("/World/Robot/" + name)
                bound = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                low, high = np.array(bound.GetMin()), np.array(bound.GetMax())
                if np.isfinite([low, high]).all():
                    result.extend([[x, y, z] for x in (low[0], high[0])
                                   for y in (low[1], high[1]) for z in (low[2], high[2])])
            return np.array(result)

        sensor = stage.GetPrimAtPath("/World/Robot/axia80_link")
        sensor_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(sensor)
        sensor_center = np.array(sensor_matrix.ExtractTranslation())
        sensor_axis = np.array(sensor_matrix.TransformDir(Gf.Vec3d(0, 0, 1)))
        sensor_axis /= np.linalg.norm(sensor_axis)
        side = np.cross(sensor_axis, [0., 0., 1.])
        if np.linalg.norm(side) < .1:
            side = np.cross(sensor_axis, [0., 1., 0.])
        side /= np.linalg.norm(side)
        up = np.cross(side, sensor_axis)
        hand_points = corners(["wrist_2_link", "wrist_3_link", "axia80_link", *HAND_CONTACT_BODY_NAMES])
        all_points = corners(["base_link", "shoulder_link", "upper_arm_link", "forearm_link",
                              "wrist_1_link", "wrist_2_link", "wrist_3_link", "axia80_link",
                              *HAND_CONTACT_BODY_NAMES])
        mount_points = corners(["wrist_3_link", "axia80_link", "inspire_base_link"])

        def frame(points, direction, margin=1.2):
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            direction = direction / np.linalg.norm(direction)
            right = np.cross(-direction, [0., 0., 1.])
            right /= np.linalg.norm(right)
            camera_up = np.cross(right, -direction)
            offsets = points - center
            tan_x = 36. / (2 * 60.)
            tan_y = tan_x * args.height / args.width
            required = np.maximum(np.abs(offsets @ right) / tan_x,
                                  np.abs(offsets @ camera_up) / tan_y) * margin
            distance = float(np.max(required + offsets @ direction))
            return (center + direction * distance).tolist(), center.tolist()

        views = {
            "assembly_closeup": frame(hand_points, side + .35 * up + .22 * sensor_axis),
            "assembly_reverse": frame(hand_points, -side + .3 * up + .22 * sensor_axis),
            "assembly_isolated": frame(hand_points, side + .35 * up + .22 * sensor_axis),
            "mount_detail": frame(mount_points, side + .3 * up),
            "robot_overview": frame(all_points, side + .55 * up + .35 * sensor_axis, 1.12),
        }

        def select_view(name):
            for body_name in ("base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link"):
                imageable = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Robot/" + body_name))
                hidden = name != "robot_overview" and (
                    body_name in ("base_link", "shoulder_link", "upper_arm_link") or name == "assembly_isolated")
                if hidden:
                    imageable.MakeInvisible()
                else:
                    imageable.MakeVisible()
            eye, target = views[name]
            distance = float(np.linalg.norm(np.array(eye) - target))
            camera_prim = stage.GetPrimAtPath("/World/PhotoCamera")
            UsdGeom.Camera(camera_prim).GetFocusDistanceAttr().Set(distance)
            camera_prim.CreateAttribute("omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d,
                                        custom=True).Set(Gf.Vec3d(0., 0., -distance))
            camera.set_world_poses_from_view(torch.tensor([eye], device=simulation.device),
                                             torch.tensor([target], device=simulation.device))
            simulation.set_camera_view(eye, target)
            print(f"[Photo] View: {name}", flush=True)

        def save_preset(name):
            select_view(name)
            for _ in range(args.warmup_frames):
                simulation.render()
            camera.update(simulation.get_physics_dt(), force_recompute=True)
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
            if rgb.shape != (args.height, args.width, 3) or rgb.std() < 2:
                raise RuntimeError(f"Invalid photograph: {name}, {rgb.shape}")
            path = output / f"{name}.png"
            iio.imwrite(path, rgb)
            print(f"[Photo] Saved {path}", flush=True)

        for name in views:
            save_preset(name)
        select_view("assembly_closeup")

        # Bake the actual FK state into a standalone, static stage. Flattening
        # resolves the layered right-hand overlay and relative mesh references.
        snapshot = Usd.Stage.Open(stage.Flatten())
        snapshot.SetDefaultPrim(snapshot.GetPrimAtPath("/World"))
        bake_pose(snapshot)
        for prim in snapshot.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.HasAttribute("physxArticulation:articulationEnabled"):
                prim.GetAttribute("physxArticulation:articulationEnabled").Set(False)
        snapshot.GetRootLayer().Export(str(output / "robot_photo_scene.usdc"))
        reopened = Usd.Stage.Open(str(output / "robot_photo_scene.usdc"))
        reopened_center = UsdGeom.XformCache().GetLocalToWorldTransform(
            reopened.GetPrimAtPath("/World/Robot/axia80_link")).ExtractTranslation()
        if not np.allclose(reopened_center, sensor_center, atol=1e-6):
            raise RuntimeError("Exported scene did not preserve the sensor pose.")
        if any(UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
               for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)):
            raise RuntimeError("Exported photography scene still contains dynamic bodies.")
        report = {
            "source_scene_robot": "StabilizeSweepSceneCfg.robot (shared with Sweep Only)",
            "source_usd": robot_cfg.spawn.usd_path,
            "sensor_radius_m": robot_cfg.spawn.sensor_radius,
            "sensor_height_m": robot_cfg.spawn.sensor_height,
            "sensor_center_world_m": sensor_center.tolist(),
            "joint_names": robot.joint_names,
            "joint_positions_rad": robot.data.joint_pos[0].cpu().tolist(),
            "joint_targets_rad": joint_pos[0].cpu().tolist(),
            "resolution": [args.width, args.height], "views": views,
            "physics": "one zero-gravity initialization step, render-only thereafter; USD snapshot has physics disabled",
            "training": False,
            "static_scene_reopen_verified": True,
        }
        (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[Photo] Ready. Output: {output}", flush=True)
        if args.capture_only:
            return

        import omni.ui as ui
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        # Use the same lens and camera as the saved photographs. The default
        # Perspective viewport has a much wider lens than our photo camera.
        photo_viewport = get_active_viewport()
        photo_viewport.camera_path = "/World/PhotoCamera"
        photo_viewport.resolution = (args.width, args.height)
        settings_window = ui.Workspace.get_window("Simulation Settings")
        if settings_window:
            settings_window.visible = False

        def save_viewport():
            path = output / ("manual_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
            viewport = get_active_viewport()
            capture_viewport_to_file(viewport, str(path))
            print(f"[Photo] Capturing current viewport: {path}", flush=True)

        window = ui.Window("Robot Photography", width=340, height=285)
        with window.frame:
            with ui.VStack(spacing=6):
                ui.Label("UR5e + Axia80 + RH56E2 | frozen pose")
                for name in views:
                    ui.Button(name.replace("_", " ").title(), clicked_fn=lambda n=name: select_view(n))
                ui.Button("Save current viewport PNG", clicked_fn=save_viewport)
                ui.Label("Orbit/pan/zoom in viewport, then save.\nPreset PNGs and static USD are already saved.", word_wrap=True)
        window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        # Retain window and capture callback for the lifetime of the app.
        while app.is_running():
            simulation.render()
    finally:
        # All PNG writes above are synchronous. Use SimulationApp's documented
        # immediate shutdown: live camera/PhysX cleanup can stall on Windows.
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
