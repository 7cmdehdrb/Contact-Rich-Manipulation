"""Run with Isaac Sim USD libraries on PYTHONPATH; no renderer required."""
import pytest
pytest.importorskip('pxr.Usd')
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from centroid_belief.display import BeliefDisplay, density_geometry
from centroid_belief import GaussianBelief
from observed_obb import Box, Observation


def test_density_is_visible_geometry_without_changing_gaussian_extent():
    b = GaussianBelief(np.array([-.7,.1]), np.array([[1e-4,3e-5],[3e-5,3e-4]]))
    points, colors, indices = density_geometry(b.parameters, 1.2)
    np.testing.assert_allclose(b.mahalanobis_squared(points[-96:,:2]), -2*np.log(.01))
    assert len(indices) == 24*96*4
    assert np.isfinite(points).all() and colors.max() <= 1
    stage = Usd.Stage.CreateInMemory()
    display = BeliefDisplay(stage)
    display.show(Observation(7, Box((-.7,.1,1.1),(.08,.08,.1))), b.parameters)
    for name in ('Density99', 'Ellipse90'):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f'/World/CentroidBelief/Observation_7/{name}'))
        assert mesh.GetDoubleSidedAttr().Get()
        assert len(mesh.GetFaceVertexIndicesAttr().Get()) > 0
    for prim in stage.Traverse():
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    display.set_visible(False)
    display.clear()
    assert UsdGeom.Imageable(stage.GetPrimAtPath(display.path)).GetVisibilityAttr().Get() == 'invisible'


def test_reused_geometry_keeps_toggle_and_removes_stale_visibility():
    stage = Usd.Stage.CreateInMemory()
    display = BeliefDisplay(stage)
    b = GaussianBelief(np.array([0.,0.]),np.eye(2)*1e-4)
    observation = Observation(0,Box((0,0,.1),(.08,.08,.1)))
    display.show(observation,b.parameters)
    display.set_visible(False)
    display.clear()
    display.show(observation,b.parameters)
    assert UsdGeom.Imageable(stage.GetPrimAtPath(display.path)).GetVisibilityAttr().Get() == 'invisible'
    display.set_visible(True)
    display.clear()
    assert UsdGeom.Imageable(stage.GetPrimAtPath(display.path+'/Observation_0')).GetVisibilityAttr().Get() == 'invisible'
