"""Show observed OBBs and single-Gaussian centers in the frozen Simple scene."""
import argparse
import os
from pathlib import Path
import sys
from functools import partial
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
for package in ("centroid_belief", "observed_obb", "inspire_sweep_baseline"):
    sys.path.insert(0, str(ROOT / "src" / package))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048, help="Reference estimator only")
    parser.add_argument("--estimator", choices=("lookup", "reference"), default="lookup")
    parser.add_argument("--benchmark-resets", type=int, default=0, help="Run N complete placements, print timings, then close")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    runner = None
    try:
        import omni.usd
        import omni.ui as ui
        from pxr import UsdGeom, UsdLux
        from isaacsim.core.utils.viewports import set_camera_view
        from observed_obb.display import ObservationDisplay
        from observed_obb.inspection import Inspection
        from observed_obb.simple_scene import SimpleScene
        from centroid_belief.preview import preview_rgba
        from centroid_belief import Config
        from centroid_belief.display import BeliefDisplay
        from centroid_belief.live import SnapshotRunner, SnapshotJob, lookup_inference
        from centroid_belief.lookup import default_estimator

        config = Config(samples_per_shape=args.samples, seed=args.seed)
        if args.estimator == "lookup":
            default_estimator()  # one startup load; never rebuild during a reset
            runner = SnapshotRunner(config, job_factory=partial(SnapshotJob, infer=lookup_inference))
        else:
            runner = SnapshotRunner(config)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(2500)
        scene = SimpleScene(stage, args.seed)
        inspection = Inspection(scene.new_layout, ObservationDisplay(stage))
        display = BeliefDisplay(stage)
        rows: list[str] = []
        preview_items: list = []
        preview_dirty = False
        reset_started = perf_counter()
        completion_logged = False
        preview_index = 0
        preview_provider = ui.ByteImageProvider()
        preview_window = ui.Window("Gaussian density (XY zoom)", width=740, height=540)
        preview_window.position_x, preview_window.position_y = 1750, 650
        with preview_window.frame:
            with ui.VStack():
                preview_status = ui.Label("Waiting for the first estimate...", height=30)
                ui.Button("Next completed observation", clicked_fn=lambda: next_preview(), height=30)
                preview_image = ui.ImageWithProvider(preview_provider, height=462,
                                                     fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)

        def next_preview():
            nonlocal preview_index, preview_dirty
            if preview_items:
                preview_index = (preview_index + 1) % len(preview_items)
                preview_dirty = True
        window = ui.Window("Observed OBB + Gaussian centroid", width=470, height=520)
        window.position_x, window.position_y = 1800, 80
        with window.frame:
            with ui.VStack():
                ui.Label("Green: observed OBB | Red: estimated XY mean", height=25)
                ui.Label("Color: Gaussian density | Cyan: 90% XY boundary", height=25)
                ui.Label("Fixed observation: +X toward -X. No dynamics/control.", height=25)
                ui.Button("New placement", clicked_fn=lambda: refresh(True), height=30)
                ui.Button("Reobserve current placement", clicked_fn=lambda: refresh(False), height=30)
                ui.Button("Toggle full-box reference (orange)", clicked_fn=inspection.toggle_reference, height=30)
                gaussian_button = ui.Button("Gaussian: ON (click to hide)", clicked_fn=lambda: toggle_gaussian(), height=30)
                status = ui.Label("", height=35, word_wrap=True)
                with ui.ScrollingFrame():
                    label = ui.Label("", word_wrap=True)

        def toggle_gaussian():
            display.set_visible(not display.visible)
            gaussian_button.text = "Gaussian: ON (click to hide)" if display.visible else "Gaussian: OFF (click to show)"

        def refresh(new):
            nonlocal preview_index, preview_dirty, reset_started, completion_logged
            reset_started = perf_counter()
            completion_logged = False
            try:
                inspection.new_placement() if new else inspection.reobserve()
            except Exception as error:
                status.text = f"Placement/observation failed: {error}"
                rows.append(status.text)
                label.text = "\n".join(rows)
                print(f"[CentroidViewer] refresh failed: {error}", flush=True)
                return
            preview_items.clear()
            preview_index = 0
            preview_dirty = False
            preview_image.visible = False
            preview_status.text = "Computing new snapshot; no estimate yet"
            runner.replace(inspection.snapshot)
            display.clear()
            rows.clear()
            label.text = ""
            status.text = f"Computing Gaussian 0/{len(inspection.snapshot)}; density appears when each ID finishes"
            print(f"[CentroidViewer] snapshot: {len(inspection.snapshot)} observed OBBs; new={new}; "
                  f"placement+observation+USD={(perf_counter()-reset_started)*1000:.1f}ms; "
                  f"observation={inspection.observation_seconds*1000:.1f}ms; "
                  f"OBB_display={inspection.display_seconds*1000:.1f}ms", flush=True)

        set_camera_view(eye=[.65, 1.1, 1.8], target=[-.7, 0, 1.1])
        refresh(True)
        print("[CentroidViewer] scene ready; background inference started", flush=True)
        completed_resets = 0
        while app.is_running():
            app.update()
            for result in runner.poll():
                item = next(o for o in inspection.snapshot if o.observation_id == result.observation_id)
                if result.parameters is not None:
                    display.show(item, result.parameters)
                    preview_items.append((item, result.parameters))
                    preview_index = len(preview_items)-1
                    preview_dirty = True
                    x, y = result.parameters[:2]
                    row = f"ID {result.observation_id}: mean ({x:.3f}, {y:.3f}) m; {result.seconds*1000:.2f}ms"
                    if result.warnings:
                        row += "\nWarning: " + ", ".join(result.warnings)
                else:
                    row = f"ID {result.observation_id}: {result.error}"
                rows.append(row)
                label.text = "\n".join(rows)
                print("[CentroidViewer] " + row, flush=True)
            if preview_dirty and preview_items:
                preview_started = perf_counter()
                item, parameters = preview_items[preview_index]
                rgba = preview_rgba(item, parameters)
                preview_provider.set_data_array(rgba, [rgba.shape[1], rgba.shape[0]])
                preview_image.visible = True
                preview_status.text = f"Showing ID {item.observation_id}; {len(preview_items)} completed"
                preview_dirty = False
                print(f"[CentroidViewer] preview={(perf_counter()-preview_started)*1000:.1f}ms", flush=True)
            if not runner.busy:
                if not completion_logged:
                    print(f"[CentroidViewer] reset_to_all_displays={(perf_counter()-reset_started)*1000:.1f}ms", flush=True)
                    app.update()  # Include a rendered update after USD/image publication.
                    print(f"[CentroidViewer] reset_to_rendered_frame={(perf_counter()-reset_started)*1000:.1f}ms", flush=True)
                    completion_logged = True
                    completed_resets += 1
                    if args.benchmark_resets:
                        if completed_resets >= args.benchmark_resets:
                            break
                        refresh(True)
                        continue
                status.text = f"Finished {len(rows)}/{len(inspection.snapshot)} (see per-ID warnings/errors)"
            else:
                status.text = f"Computing {len(rows)}/{len(inspection.snapshot)}; density appears per completed ID"
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if runner is not None:
            runner.close()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
