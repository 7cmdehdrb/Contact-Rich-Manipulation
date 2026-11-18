"""Live viewer acceptance checks, separate from the observation contract tests."""
import math

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def validate_inspection(inspection, scene, app, set_camera_view) -> None:
    stage = scene.stage
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

    def check_graphics(path, expected_boxes):
        root = stage.GetPrimAtPath(path)
        assert len(root.GetChildren()) == len(expected_boxes)
        for prim in Usd.PrimRange(root):
            assert not prim.HasAPI(UsdPhysics.CollisionAPI), str(prim.GetPath())
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI), str(prim.GetPath())
        for index, box in enumerate(expected_boxes):
            curves = UsdGeom.BasisCurves(stage.GetPrimAtPath(f"{path}/Box_{index}"))
            points = np.asarray(curves.GetPointsAttr().Get(), dtype=float)
            assert len(points) == 24
            c, s = math.cos(box.yaw), math.sin(box.yaw)
            rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            local = (points-box.center) @ rotation
            np.testing.assert_allclose(abs(local), np.tile(np.asarray(box.size)/2, (24, 1)), atol=2e-6)
            assert len(np.unique(points, axis=0)) == 8

    for _ in range(3):
        snapshot, boxes = inspection.snapshot, inspection.boxes
        check_graphics("/World/ObservedOBB", [item.box for item in snapshot])
        check_graphics("/World/FullBoxReference", boxes)
        for index, item in enumerate(snapshot):
            assert stage.GetPrimAtPath(f"/World/ObservedOBB/Box_{index}").GetCustomDataByKey(
                "observation_id") == item.observation_id
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        for index, box in zip(scene.active_indices, boxes):
            actual = cache.ComputeWorldBound(stage.GetPrimAtPath(scene.paths[index])).ComputeAlignedRange()
            c, s = abs(math.cos(box.yaw)), abs(math.sin(box.yaw))
            extent = np.array((c*box.size[0]+s*box.size[1], s*box.size[0]+c*box.size[1], box.size[2]))
            np.testing.assert_allclose(actual.GetMin(), np.asarray(box.center)-extent/2, atol=2e-5)
            np.testing.assert_allclose(actual.GetMax(), np.asarray(box.center)+extent/2, atol=2e-5)
        before = [UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation() for path in scene.paths]
        visibility = inspection.reference_visible
        inspection.toggle_reference()
        assert inspection.reference_visible != visibility
        expected = "inherited" if inspection.reference_visible else "invisible"
        assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/FullBoxReference")).GetVisibilityAttr().Get() == expected
        set_camera_view(eye=[.5, -.9, 1.7], target=[-.7, 0, 1.1])
        for _ in range(5):
            app.update()
        assert inspection.snapshot is snapshot
        assert inspection.boxes is boxes
        inspection.reobserve()
        assert inspection.snapshot == snapshot
        after = [UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation() for path in scene.paths]
        assert before == after
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                assert not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        inspection.new_placement()
        assert inspection.boxes != boxes
    print("[ObservedOBB] Live validation passed: geometry, graphics, static poses, toggles and snapshots", flush=True)
