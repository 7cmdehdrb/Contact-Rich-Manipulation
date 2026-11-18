"""Collision-free, double-sided Gaussian density meshes for RTX viewports."""
import numpy as np
from pxr import Gf, UsdGeom, UsdShade, Sdf, Vt

from .gaussian import GaussianBelief


def density_geometry(parameters, z, rings=24, segments=96):
    """True XY Gaussian contours; height is display-only, colors are peak-relative."""
    belief = GaussianBelief.from_parameters(parameters)
    values, vectors = np.linalg.eigh(belief.covariance)
    radii = np.linspace(0., np.sqrt(-2*np.log(.01)), rings+1)
    angles = np.linspace(0., 2*np.pi, segments, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    points, colors = [], []
    for radius in radii:
        xy = belief.mean + (circle*np.sqrt(values)*radius) @ vectors.T
        points.extend(np.column_stack((xy, np.full(segments, z))))
        intensity = np.exp(-radius**2/2)
        # Saturated blue outer region to yellow peak; independent of scene lighting.
        color = (intensity, .15+.8*intensity, 1.-.9*intensity)
        colors.extend([color]*segments)
    indices: list[int] = []
    for ring in range(rings):
        for i in range(segments):
            j = (i+1) % segments
            indices.extend((ring*segments+i, ring*segments+j,
                            (ring+1)*segments+j, (ring+1)*segments+i))
    return np.asarray(points), np.asarray(colors), indices


class BeliefDisplay:
    path = "/World/CentroidBelief"

    def __init__(self, stage):
        self.stage = stage
        self.visible = True
        UsdGeom.Xform.Define(self.stage, self.path)
        self.set_visible(self.visible)
        # Vertex colors drive emission so high dome illumination cannot wash them out.
        material = UsdShade.Material.Define(self.stage, self.path + "/Material")
        reader = UsdShade.Shader.Define(self.stage, self.path + "/Material/Color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader = UsdShade.Shader.Define(self.stage, self.path + "/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.material = material

    def clear(self):
        for child in self.stage.GetPrimAtPath(self.path).GetChildren():
            if child.GetName().startswith("Observation_"):
                UsdGeom.Imageable(child).MakeInvisible()

    def set_visible(self, visible):
        self.visible = visible
        root = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.path))
        root.MakeVisible() if visible else root.MakeInvisible()

    def _mesh(self, path, points, colors, indices):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr([4]*(len(indices)//4))
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.asarray(colors, dtype=np.float32)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.material)

    def show(self, observation, parameters):
        belief = GaussianBelief.from_parameters(parameters)
        path = f"{self.path}/Observation_{observation.observation_id}"
        root = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        UsdGeom.Imageable(root).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        root.SetCustomDataByKey("observation_id", observation.observation_id)
        # XY only, lifted 25 mm above the top, never scale the covariance for visibility.
        z = observation.box.center[2] + observation.box.size[2]/2 + .025
        points, colors, indices = density_geometry(parameters, z)
        self._mesh(path + "/Density99", points, colors, indices)
        ellipse = belief.ellipse(.9, count=97)[:-1]
        tangent = np.roll(ellipse, -1, axis=0)-np.roll(ellipse, 1, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        # A real 3 mm mesh strip is reliable in RTX; BasisCurves can be subpixel.
        points = np.concatenate([np.column_stack((ellipse+side*.0015*normal,
                                 np.full(len(ellipse), z+.001))) for side in (-1, 1)])
        n = len(ellipse)
        indices = [v for i in range(n) for v in (i, (i+1)%n, (i+1)%n+n, i+n)]
        self._mesh(path + "/Ellipse90", points, np.tile((0., 1., 1.), (len(points), 1)), indices)
        marker = UsdGeom.Sphere.Define(self.stage, path + "/Mean")
        marker.CreateRadiusAttr(.006)
        translate = marker.GetPrim().GetAttribute("xformOp:translate")
        if not translate:
            translate = marker.AddTranslateOp().GetAttr()
        translate.Set(Gf.Vec3d(*belief.mean, z+.004))
        marker.CreateDisplayColorAttr([Gf.Vec3f(1., .02, .1)])
        UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(self.material)
