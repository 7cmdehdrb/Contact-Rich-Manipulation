import pytest
pytest.importorskip('pxr.Usd')
import numpy as np
from pxr import Usd,UsdGeom,Gf
from surface_point.usd_mesh import StageGeometry


def test_visible_geometry_cache_preserves_world_transforms_and_ignores_hidden_collision():
    stage=Usd.Stage.CreateInMemory()
    root=UsdGeom.Xform.Define(stage,'/Object')
    translate=root.AddTranslateOp();translate.Set(Gf.Vec3d(1,2,3))
    root.AddScaleOp().Set(Gf.Vec3f(2,3,4))
    cube=UsdGeom.Cube.Define(stage,'/Object/Visible');cube.CreateSizeAttr(.1)
    cube.AddTranslateOp().Set(Gf.Vec3d(.1,0,0))
    hidden=UsdGeom.Cube.Define(stage,'/Object/CollisionOnly');hidden.CreateSizeAttr(10)
    hidden.MakeInvisible()
    cache=StageGeometry(stage)
    v,f,ids=cache.collect(['/Object'])
    assert len(f)==12 and set(ids)=={0}
    np.testing.assert_allclose(v.mean(0),(1.2,2,3))
    np.testing.assert_allclose(np.ptp(v,axis=0),(.2,.3,.4))
    translate.Set(Gf.Vec3d(2,2,3))
    moved,_,_=cache.collect(['/Object'])
    np.testing.assert_allclose(moved-v,np.tile((1,0,0),(len(v),1)))
