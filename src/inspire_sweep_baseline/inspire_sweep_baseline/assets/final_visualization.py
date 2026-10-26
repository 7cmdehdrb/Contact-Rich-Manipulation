"""Render-only target wire box, authored only for Final playback."""

import itertools
from pxr import Gf, UsdGeom
import isaaclab.sim as sim


def create_target_bounding_boxes(env):
    stage = sim.get_current_stage()
    for row in range(env.num_envs):
        # Child follows the rigid body's render transform, including Fabric.
        path = f'/World/envs/env_{row}/mass_cylinder/TargetBoundingBox'
        bounds = env._final_object_bounds[row, 0] / env._final_object_scales[row, 0]
        lower, upper = bounds[0] - .003, bounds[1] + .003
        corners = list(itertools.product((0, 1), repeat=3))
        points = []
        for i, a in enumerate(corners):
            for b in corners[i + 1:]:
                if sum(x != y for x, y in zip(a, b)) == 1:
                    for corner in (a, b):
                        points.append(Gf.Vec3f(*(float(upper[k] if corner[k] else lower[k]) for k in range(3))))
        curves = UsdGeom.BasisCurves.Define(stage, path)
        curves.CreateTypeAttr('linear')
        curves.CreateWrapAttr('nonperiodic')
        curves.CreateCurveVertexCountsAttr([2] * 12)
        curves.CreatePointsAttr(points)
        curves.CreateWidthsAttr([.002])
        curves.SetWidthsInterpolation('constant')
        curves.CreateDisplayColorAttr([Gf.Vec3f(1., .9, .05)])
        # No CollisionAPI, rigid body, mass, sensor or observation is added.
