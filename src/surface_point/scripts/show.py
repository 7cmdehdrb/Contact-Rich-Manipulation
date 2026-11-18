"""Actual visible-mesh representative points, legacy OBBs and free inspection view."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'/p) for p in ('surface_point','observed_obb','inspire_sweep_baseline')]


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=('simple','fixtures'),default='simple')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--placement',type=int,default=0,help='Advance initial seed sequence, e.g. 3 for bottle occlusion')
    parser.add_argument('--spacing-mm',type=float,default=1.,help='Metric geometric ray lattice, not RGB resolution')
    parser.add_argument('--representative',choices=('mask-centroid','bbox-center'),default='mask-centroid')
    parser.add_argument('--ray-device',default='auto',help='auto, cpu or cuda:0')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--capture-validation',action='store_true',help='Save first validation viewport front/oblique')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/surface_point_mesh')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.placement<0:parser.error('--placement must be nonnegative')
    app=AppLauncher(args).app
    try:
        import numpy as np
        import warp as wp
        import omni.ui as ui
        import omni.usd
        from PIL import Image
        from pxr import Usd,UsdGeom,UsdLux,UsdPhysics
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport
        from surface_point.mesh import MeshObserver
        from surface_point.usd_mesh import StageGeometry
        from surface_point.mask_preview import preview
        from surface_point.display import ComparisonDisplay,show_proxies
        from surface_point.fixtures import scenes
        stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1800)
        source=None
        if args.scene=='simple':
            from observed_obb.simple_scene import SimpleScene
            source=SimpleScene(stage,args.seed)
        device=('cuda:0' if wp.is_cuda_available() else 'cpu') if args.ray_device=='auto' else args.ray_device
        observer=MeshObserver(device=device,spacing=args.spacing_mm/1000,representative=args.representative)
        geometry=StageGeometry(stage)
        fixtures=list(scenes().items());fixture_index=-1
        boxes: tuple = ()
        records: tuple = ()
        history: list = []
        frame=None
        display=ComparisonDisplay(stage)
        viewport=get_active_viewport()
        camera_path='/World/SurfaceInspectionCamera'
        camera=UsdGeom.Camera.Define(stage,camera_path)
        camera.CreateClippingRangeAttr((.001,100.))
        camera.CreateFocalLengthAttr(24.)
        viewport.camera_path=camera_path
        yaw=ui.SimpleFloatModel(35.);pitch=ui.SimpleFloatModel(25.);distance=ui.SimpleFloatModel(1.4)
        provider=ui.ByteImageProvider()
        mask_window=ui.Window('Observation mask (fixed frontal rays)',width=900,height=320)
        mask_window.position_x,mask_window.position_y=650,70
        with mask_window.frame:
            image_widget=ui.ImageWithProvider(provider,fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        window=ui.Window('Visible mesh point / observed OBB',width=590,height=760)
        window.position_x,window.position_y=30,70
        with window.frame:
            with ui.VStack():
                ui.Label('MESH visibility | observation fixed +X -> -X',height=25)
                ui.Label(f'Representative: {args.representative} | rays: {args.spacing_mm:g} mm',height=25)
                ui.Label('Red: actual surface | Blue: OBB center | Green: old OBB',height=25)
                ui.Label('Inspection camera is FREE; moving it does not change observation.',height=25)
                ui.Button('New placement / Next fixture',clicked_fn=lambda:refresh(True),height=30)
                ui.Button('Reobserve current placement',clicked_fn=lambda:refresh(False),height=30)
                ui.Button('Toggle surface points + center links',clicked_fn=lambda:toggle('show_points'),height=30)
                ui.Button('Toggle observed 3D OBB',clicked_fn=lambda:toggle('show_observed'),height=30)
                ui.Button('Toggle full source OBB (orange)',clicked_fn=lambda:toggle('show_reference'),height=30)
                ui.Button('Front orthographic inspection',clicked_fn=lambda:preset(False),height=30)
                ui.Button('Oblique perspective inspection',clicked_fn=lambda:preset(True),height=30)
                for label,model in [('View azimuth (deg)',yaw),('View elevation (deg)',pitch),('View distance (m)',distance)]:
                    with ui.HStack(height=25):
                        ui.Label(label,width=200);ui.FloatDrag(model=model,step=.5 if model is not distance else .02)
                ui.Label('Viewport: Alt + left drag orbit; Alt + middle drag pan; scroll zoom',height=25)
                status=ui.Label('',height=60,word_wrap=True)
                with ui.ScrollingFrame():detail=ui.Label('',word_wrap=True)

        def toggle(name):
            setattr(display,name,not getattr(display,name));display.visibility()

        def move_view():
            if not boxes:return
            center=np.mean([b.center for b in boxes],axis=0)
            az=np.deg2rad(yaw.as_float);el=np.deg2rad(np.clip(pitch.as_float,-85,85))
            radius=max(.1,distance.as_float)
            offset=radius*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
            set_camera_view(eye=center+offset,target=center,camera_prim_path=camera_path)

        def preset(oblique):
            camera.CreateProjectionAttr('perspective' if oblique else 'orthographic')
            camera.CreateHorizontalApertureAttr(20.955 if oblique else (12. if source else 6.))
            camera.CreateVerticalApertureAttr(15.2908 if oblique else (8. if source else 4.))
            yaw.set_value(35. if oblique else 0.);pitch.set_value(25. if oblique else 0.)
            move_view()
        for model in (yaw,pitch,distance):model.add_value_changed_fn(lambda _:move_view())

        def refresh(new):
            nonlocal boxes,records,fixture_index,frame
            start=perf_counter()
            if new:
                if source:boxes=source.new_layout()
                else:
                    fixture_index=(fixture_index+1)%len(fixtures)
                    boxes=fixtures[fixture_index][1];show_proxies(stage,boxes)
                    geometry.cache.clear()  # fixture prim paths now contain new shapes
            paths=[source.paths[i] for i in source.active_indices] if source else [f'/World/FixtureProxies/Box_{i}' for i in range(len(boxes))]
            geometry_start=perf_counter()
            vertices,faces,owners=geometry.collect(paths,('/World/envs/env_0/Shelf',) if source else ())
            extraction_ms=(perf_counter()-geometry_start)*1000
            frame=observer.observe(vertices,faces,owners,boxes)
            records,legacy=frame.records,frame.legacy
            display.show(boxes,records,legacy)
            rgba=preview(frame);provider.set_data_array(rgba,[rgba.shape[1],rgba.shape[0]])
            name='Simple' if source else fixtures[fixture_index][0]
            lines=[]
            for r in records:
                p=r.patch.observation
                matching='unmatched (visibility/shape differs)' if r.legacy is None else (
                    f'OBB {r.legacy.observation_id}: center delta={r.center_distance_m*1000:.1f} mm; '
                    f'front delta={r.front_depth_error_m*1000:.2f} mm')
                lines.append(f'Point {p.observation_id}: ({p.point[0]:.3f}, {p.point[1]:.3f}, {p.point[2]:.3f}) m\n'
                             f'{matching}\nrepresentative pixel displacement={p.correction_m*1000:.2f} mm')
            detail.text='\n\n'.join(lines) or 'No visible components'
            status.text=f'{name}: {len(records)} mesh surface points / {len(legacy)} old OBBs\n'
            status.text+=f'Geometry={frame.timings["total_ms"]:.2f} ms | {frame.timings["rays"]} rays | {device}'
            report=dict(scene=name,new=new,extraction_ms=extraction_ms,geometry=frame.timings,
                        total_ms=(perf_counter()-start)*1000,sources=[asdict(b) for b in boxes],
                        comparisons=[asdict(r) for r in records],pixels=frame.pixels,legacy=[asdict(o) for o in legacy])
            history.append(report)
            print('[SurfacePointMesh] '+status.text.replace('\n','; '),flush=True)
            if args.validate:
                args.output.mkdir(parents=True,exist_ok=True)
                index=sum(r['new'] for r in history)-1
                Image.fromarray(rgba).save(args.output/f'{args.scene}_mask_{index}.png')
                if new:
                    np.savez_compressed(args.output/f'{args.scene}_mesh_{index}.npz',vertices=vertices,faces=faces,
                                        owners=owners,boxes=json.dumps([asdict(b) for b in boxes]))

        if source:
            for _ in range(args.placement):source.new_layout()
        else:fixture_index=args.placement-1
        refresh(True);preset(True)
        if args.validate:
            count=6 if source else len(fixtures)
            for index in range(count):
                if index:refresh(True)
                before=records
                preset(False);app.update();preset(True);app.update()
                yaw.set_value(42.);pitch.set_value(31.);distance.set_value(1.25)
                free_camera=np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).copy()
                refresh(False)
                np.testing.assert_allclose(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),free_camera)
                assert before==records,'Inspection camera changed fixed observation'
                for flag in ('show_points','show_observed','show_reference'):
                    toggle(flag);app.update();toggle(flag)
                for _ in range(3):app.update()
                if index==0 and args.capture_validation:
                    import asyncio
                    from omni.kit.viewport.utility import capture_viewport_to_file
                    for oblique in (False,True):
                        preset(oblique)
                        destination=args.output/f'{args.scene}_{"oblique" if oblique else "front"}.png'
                        for attempt in range(3):
                            for _ in range(20):app.update()
                            capture=capture_viewport_to_file(viewport,str(destination))
                            future=asyncio.ensure_future(capture.wait_for_result())
                            deadline=perf_counter()+15
                            while not future.done() and perf_counter()<deadline:app.update()
                            if not future.done():raise TimeoutError('Viewport capture did not finish')
                            future.result()
                            with Image.open(destination) as captured:
                                if np.ptp(np.asarray(captured.convert('RGB'))) > 10:break
                        else:raise RuntimeError('Viewport capture remained blank after warmup')
                for root in ('/World/SurfacePointBench','/World/ObservedOBB','/World/FullBoxReference'):
                    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
                        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            (args.output/f'{args.scene}_mesh_gpu.json').write_text(json.dumps(history,indent=2)+'\n')
            print('[SurfacePointMesh] VALIDATED mesh/placement/reobserve/free-camera/toggles/noncollision',flush=True)
        else:
            while app.is_running():app.update()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__=='__main__':main()
