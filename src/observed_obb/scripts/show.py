"""Inspect an initial OBB snapshot without a policy or physics stepping."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inspire_sweep_baseline"))


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true", help="Check live stage, snapshot operations and static poses")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        import omni.usd
        import omni.ui as ui
        from pxr import UsdGeom, UsdLux
        from isaacsim.core.utils.viewports import set_camera_view
        from observed_obb.display import ObservationDisplay
        from observed_obb.inspection import Inspection
        from observed_obb.simple_scene import SimpleScene

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(2500)
        scene = SimpleScene(stage, args.seed)
        inspection = Inspection(scene.new_layout, ObservationDisplay(stage))
        inspection.new_placement()
        set_camera_view(eye=[.65, 1.1, 1.8], target=[-.7, 0, 1.1])
        window = ui.Window("Initial observed OBB", width=390, height=340)
        with window.frame:
            with ui.VStack():
                ui.Label("Fixed frontal observation: shelf +X toward -X", height=25)
                ui.Button("New placement", clicked_fn=lambda: update(inspection.new_placement), height=30)
                ui.Button("Reobserve current placement", clicked_fn=lambda: update(inspection.reobserve), height=30)
                ui.Button("Toggle full-box reference", clicked_fn=inspection.toggle_reference, height=30)
                label = ui.Label("", word_wrap=True)

        def update(action=None):
            if action is not None:
                action()
            label.text = "\n".join(
                f"ID {item.observation_id}: ({item.box.center[0]:.3f}, "
                f"{item.box.center[1]:.3f}, {item.box.center[2]:.3f}) m"
                for item in inspection.snapshot) or "No observed boxes"
            print(f"[ObservedOBB] startup={scene.startup_seconds:.6f}s "
                  f"observation={inspection.observation_seconds:.6f}s "
                  f"display={inspection.display_seconds:.6f}s "
                  f"objects={len(inspection.boxes)} observations={len(inspection.snapshot)}", flush=True)

        update()
        if args.validate:
            from observed_obb.validation import validate_inspection
            validate_inspection(inspection, scene, app, set_camera_view)
            update()
        while app.is_running():
            app.update()
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
