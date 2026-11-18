"""USD wire graphics, imported only after the Isaac Sim application is running."""
import math
from collections.abc import Sequence

from pxr import Gf, UsdGeom

from .observation import Box, Observation


def wire_boxes(stage, path: str, boxes: Sequence[Box], color: tuple[float, float, float]) -> None:
    """Reuse graphics-only prims and remove surplus boxes; no physics schemas."""
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    expected = {f"Box_{i}" for i in range(len(boxes))}
    for child in root.GetChildren():
        if child.GetName() not in expected:
            stage.RemovePrim(child.GetPath())
    for index, box in enumerate(boxes):
        c, s = math.cos(box.yaw), math.sin(box.yaw)
        points = []
        for z in (-0.5, 0.5):
            for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
                x, y = x * box.size[0], y * box.size[1]
                points.append(Gf.Vec3f(box.center[0] + c*x - s*y,
                                      box.center[1] + s*x + c*y,
                                      box.center[2] + z*box.size[2]))
        edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
        curves = UsdGeom.BasisCurves.Define(stage, f"{path}/Box_{index}")
        curves.CreateTypeAttr("linear")
        curves.CreateWrapAttr("nonperiodic")
        curves.CreateCurveVertexCountsAttr([2] * len(edges))
        curves.CreatePointsAttr([points[i] for edge in edges for i in edge])
        curves.CreateWidthsAttr([0.0015])
        curves.SetWidthsInterpolation("constant")
        curves.CreateDisplayColorAttr([Gf.Vec3f(*color)])


class ObservationDisplay:
    """Display a snapshot without reading or updating the source scene."""

    def __init__(self, stage):
        self.stage = stage
        self.snapshot: tuple[Observation, ...] = ()

    def show(self, snapshot: tuple[Observation, ...]) -> None:
        self.snapshot = snapshot
        wire_boxes(self.stage, "/World/ObservedOBB", [item.box for item in snapshot], (0.1, 1.0, 0.25))
        for index, item in enumerate(snapshot):
            self.stage.GetPrimAtPath(f"/World/ObservedOBB/Box_{index}").SetCustomDataByKey(
                "observation_id", item.observation_id)

    def reference(self, boxes: Sequence[Box], visible: bool) -> None:
        wire_boxes(self.stage, "/World/FullBoxReference", boxes, (1.0, 0.65, 0.1))
        self.set_reference_visible(visible)

    def set_reference_visible(self, visible: bool) -> None:
        imageable = UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/FullBoxReference"))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
