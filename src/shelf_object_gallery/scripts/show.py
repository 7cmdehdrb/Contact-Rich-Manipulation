"""Show the Final shelf with base-size procedural and textured YCB objects."""

import argparse
import importlib.metadata
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "inspire_sweep_baseline"))

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2, help="Rows along shelf depth (front to back).")
    parser.add_argument("--cols", type=int, default=4, help="Columns along shelf width (left to right).")
    parser.add_argument("--gap", type=float, default=0.05, help="Gap between largest cell footprints, metres.")
    parser.add_argument("--shelf-usd", help="Explicit shelf USD; otherwise use the Final environment asset path.")
    parser.add_argument("--steps", type=int, default=0, help="Exit after this many physics steps; 0 keeps the viewer open.")
    parser.add_argument("--validate", action="store_true", help="Verify live geometry, placement and static poses.")
    parser.add_argument("--export-usd", type=Path, help="Export the assembled display stage for inspection.")
    parser.add_argument("--screenshot", type=Path, help="Save a PNG from the gallery view.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.rows < 1 or args.cols < 1 or args.steps < 0:
        parser.error("rows/cols must be positive and steps must be nonnegative")
    if args.screenshot:
        args.enable_cameras = True
    # Match the active launcher's Windows Isaac Sim 5.1 renderer compatibility.
    if sys.platform == "win32" and importlib.metadata.version("isaacsim").startswith("5.1."):
        kit_args = args.kit_args or ""
        if not any(v in kit_args for v in ("--vulkan", "--d3d12", "/app/vulkan=", "/app/d3d12=")):
            args.kit_args = (kit_args + " --/app/vulkan=false").strip()
    app = AppLauncher(args).app
    sim = None
    try:
        import isaaclab.sim as sim_utils
        from shelf_object_gallery.scene import ShelfObjectGallery

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
        eye, target = (0.8, 1.35, 1.95), (-0.7, 0.0, 1.10)
        sim.set_camera_view(eye=eye, target=target)
        gallery = ShelfObjectGallery(rows=args.rows, cols=args.cols, gap=args.gap, shelf_usd=args.shelf_usd)
        camera = None
        if args.screenshot:
            from isaaclab.sensors import Camera, CameraCfg
            camera = Camera(CameraCfg(
                prim_path="/World/GalleryCamera", width=1280, height=720, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, horizontal_aperture=30.0, clipping_range=(0.05, 100.0)),
            ))
        sim.reset()
        gallery.scene.reset()
        if camera is not None:
            import torch
            from PIL import Image
            camera.set_world_poses_from_view(
                torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device))
            for _ in range(12):
                sim.render()
            camera.update(sim.get_physics_dt(), force_recompute=True)
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            Image.fromarray(rgb).save(args.screenshot)
            print(f"[Gallery] Screenshot: {args.screenshot}", flush=True)
        print(f"[Gallery] shelf={gallery.shelf_usd}", flush=True)
        for item in gallery.placements:
            print(f"[Gallery] row={item.row} col={item.col}: {item.kind}, scale=1.0", flush=True)
        if args.export_usd:
            args.export_usd.parent.mkdir(parents=True, exist_ok=True)
            sim.stage.Export(str(args.export_usd.resolve()))
        count = 0
        while app.is_running() and (args.steps == 0 or count < args.steps):
            gallery.step(sim)
            count += 1
            if args.validate and count == 1:
                gallery.validate()
        if args.validate:
            gallery.validate()
            print(f"[Gallery] Validation passed: {len(gallery.placements)} objects, {count} steps, no robot.", flush=True)
    finally:
        if sim is not None:
            # Unsubscribe Isaac Lab's STOP handler before stopping the timeline;
            # that handler otherwise enters a render loop during viewer shutdown.
            sim.clear_instance()
            sim.clear_all_callbacks()
            sim.stop()
        # This viewer has no Replicator jobs to wait for on shutdown.
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
