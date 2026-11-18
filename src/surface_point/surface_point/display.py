"""USD-only visualization. No marker or proxy has collision/rigid-body schemas."""
import math
import numpy as np
from pxr import Gf, UsdGeom, UsdShade, Sdf
from observed_obb.display import ObservationDisplay


class ComparisonDisplay:
    def __init__(self,stage):
        self.stage=stage
        self.obb=ObservationDisplay(stage)
        self.show_reference=True
        self.show_points=True
        self.show_observed=True
        self.path='/World/SurfacePointBench'
        UsdGeom.Xform.Define(stage,self.path)
        self.materials={}
        for name,color in [('Point',(1.,.02,.04)),('Center',(.05,.5,1.))]:
            path=f'{self.path}/{name}Material'
            material=UsdShade.Material.Define(stage,path)
            shader=UsdShade.Shader.Define(stage,path+'/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            self.materials[name]=material

    def _sphere(self,path,point,radius,material):
        sphere=UsdGeom.Sphere.Define(self.stage,path)
        sphere.CreateRadiusAttr(radius)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*point))
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)

    def show(self,boxes,records,legacy):
        self.obb.show(legacy)
        self.obb.reference(boxes,self.show_reference)
        path=self.path+'/Markers'
        self.stage.RemovePrim(path)
        UsdGeom.Xform.Define(self.stage,path)
        for record in records:
            item=record.patch.observation
            self._sphere(f'{path}/Point_{item.observation_id}',item.point,.004,self.materials['Point'])
            if record.legacy is not None:
                center=record.legacy.box.center
                self._sphere(f'{path}/Center_{item.observation_id}',center,.003,self.materials['Center'])
                line=UsdGeom.BasisCurves.Define(self.stage,f'{path}/Link_{item.observation_id}')
                line.CreateTypeAttr('linear');line.CreateCurveVertexCountsAttr([2])
                line.CreatePointsAttr([Gf.Vec3f(*item.point),Gf.Vec3f(*center)])
                line.CreateWidthsAttr([.001]);line.SetWidthsInterpolation('constant')
                line.CreateDisplayColorAttr([Gf.Vec3f(.05,.5,1.)])
        self.visibility()

    def visibility(self):
        for path,visible in [(self.path+'/Markers',self.show_points),('/World/ObservedOBB',self.show_observed),
                             ('/World/FullBoxReference',self.show_reference)]:
            prim=self.stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set('inherited' if visible else 'invisible')


def show_proxies(stage,boxes):
    stage.RemovePrim('/World/FixtureProxies')
    for index,box in enumerate(boxes):
        cube=UsdGeom.Cube.Define(stage,f'/World/FixtureProxies/Box_{index}')
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*box.center))
        cube.AddRotateZOp().Set(math.degrees(box.yaw))
        cube.AddScaleOp().Set(Gf.Vec3f(*box.size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*((.35,.48,.65) if index==0 else (.65,.4,.25)))])
