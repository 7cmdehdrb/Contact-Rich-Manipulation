"""Analytic occlusion cases and independent interval-union verification."""

import unittest

import numpy as np

from .generator import SceneGenerator
from .visibility import classify_visibility, classify_visibility_torch


def union_oracle(centers, radius):
    """Equal-radius disjoint circles: union the projections of closer circles.

    Independent from production midpoint ray intersections, no square roots.
    """
    ratios=[]
    for x,y in centers:
        spans=[]
        for ox,oy in centers:
            left,right=max(x-radius,ox-radius),min(x+radius,ox+radius)
            if oy<y and right>left:spans.append((left,right))
        covered=0.;end=-np.inf
        for left,right in sorted(spans):
            covered+=max(0,right-max(left,end));end=max(end,right)
        ratios.append(1-covered/(2*radius))
    return ratios


class VisibilityTests(unittest.TestCase):
    def test_front_direction_and_35_percent_boundary(self):
        offsets=np.array([0.,2.,2.8,4.,8.])
        centers=np.zeros((5,2,2));centers[:,0]=[0.,5.]
        centers[:,1,0]=offsets;centers[:,1,1]=20.
        result=classify_visibility(centers)
        np.testing.assert_allclose(result.ratio[:,1],[0,.25,.35,.5,1],atol=1e-14)
        np.testing.assert_array_equal(result.visible[:,1],[False,False,True,True,True])
        np.testing.assert_allclose(result.ratio[:,0],1)

    def test_two_occluders_union_and_no_double_count(self):
        centers=np.array([[[0.,30.],[-4.,10.],[4.,10.]]])
        result=classify_visibility(centers)
        np.testing.assert_allclose(result.ratio,[[0.,1.,1.]])
        self.assertAlmostEqual(result.visible_width.sum(),16.)

    def test_below_threshold_objects_still_occlude(self):
        result=classify_visibility(np.array([[[0.,5.],[2.,15.],[4.,25.]]]))
        np.testing.assert_allclose(result.ratio,[[1.,.25,.25]])
        np.testing.assert_array_equal(result.visible,[[True,False,False]])

    def test_invalid_padding_is_not_an_object_or_occluder(self):
        result=classify_visibility(np.array([[[0.,10.],[np.nan,np.nan]]]),
                                   radii=np.array([4.,np.nan]),active=[True,False])
        np.testing.assert_allclose(result.ratio,[[1.,0.]])
        np.testing.assert_array_equal(result.not_visible,[[False,False]])
        empty=classify_visibility(np.full((2,3,2),np.nan),active=np.zeros((2,3),dtype=bool))
        self.assertFalse(empty.visible.any());self.assertFalse(empty.not_visible.any())
        self.assertTrue(np.all(empty.owners==-1))

    def test_disjoint_unequal_radii_and_object_permutation(self):
        centers=np.array([[[0.,5.],[2.,20.],[-12.,15.]]])
        radii=np.array([[5.,3.,2.]])
        result=classify_visibility(centers,radii)
        np.testing.assert_allclose(result.ratio,[[1,0,1]])
        order=[2,0,1]
        permuted=classify_visibility(centers[:,order],radii[:,order])
        np.testing.assert_allclose(permuted.ratio,result.ratio[:,order])

    def test_generated_scenes_match_independent_interval_union(self):
        batch=SceneGenerator(seed=271).generate(128)
        centers=np.concatenate((batch.target[:,None],batch.obstacles),axis=1)
        result=classify_visibility(centers,batch.config.radius)
        expected=np.array([union_oracle(scene,batch.config.radius) for scene in centers])
        np.testing.assert_allclose(result.ratio,expected,atol=2e-14)
        np.testing.assert_array_equal(result.visible,expected>=.35)
        moved=classify_visibility(centers*2+[100.,-20.],batch.config.radius*2)
        np.testing.assert_allclose(result.ratio,moved.ratio,atol=1e-14)

    def test_invalid_geometry_and_threshold_are_rejected(self):
        for kwargs in ({"radii":0},{"threshold":0},{"threshold":1.1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                classify_visibility(np.array([[[0.,5.]]]),**kwargs)
        with self.assertRaisesRegex(ValueError,"overlap"):
            classify_visibility(np.array([[[0.,5.],[0.,6.]]]))
        with self.assertRaises(ValueError):classify_visibility(np.zeros((0,2,2)))

    def test_torch_matches_numpy_without_changing_input(self):
        try:import torch
        except ImportError:self.skipTest("Optional torch dependency is not installed")
        batch=SceneGenerator(seed=45).generate(64)
        data=np.concatenate((batch.target[:,None],batch.obstacles),axis=1)
        for dtype in (np.float32,np.float64):
            with self.subTest(dtype=dtype):
                centers=torch.from_numpy(data.astype(dtype))
                original=centers.clone()
                result=classify_visibility_torch(centers)
                expected=classify_visibility(centers.numpy())
                np.testing.assert_allclose(result.ratio.numpy(),expected.ratio,atol=1e-6)
                np.testing.assert_array_equal(result.visible.numpy(),expected.visible)
                self.assertTrue(torch.equal(centers,original))
                fast=classify_visibility_torch(centers,check_inputs=False)
                self.assertTrue(torch.equal(fast.ratio,result.ratio))


if __name__=="__main__":
    unittest.main()
