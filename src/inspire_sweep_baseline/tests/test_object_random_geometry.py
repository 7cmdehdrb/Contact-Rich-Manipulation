"""Conservative geometry envelope protects the inherited cylinder layout."""
import math
import runpy
from pathlib import Path
import pytest

fit_scale=runpy.run_path(str(Path(__file__).parents[1]/'inspire_sweep_baseline/contact_sweep_clutter_fusion_holding_osc_object_random_v0/geometry.py'))['fit_scale']

@pytest.mark.parametrize('size',[(.164,.072,.214),(.093,.045,.176),(.068,.068,.102),(.096,.058,.191)])
@pytest.mark.parametrize('fraction',[.8,.9,1.])
def test_all_yaws_fit_parent_layout(size,fraction):
    lower=tuple(-x/2 for x in size);upper=tuple(x/2 for x in size)
    scale,offset=fit_scale(lower,upper,fraction)
    assert offset[2]+scale*lower[2]==pytest.approx(0)
    assert size[2]*scale<=.18+1e-9
    for yaw in range(360):
        c,s=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
        for x in (-size[0]/2,size[0]/2):
            for y in (-size[1]/2,size[1]/2):
                assert math.hypot(scale*(c*x-s*y),scale*(s*x+c*y))<=.06+1e-9
    # Same factor for all axes preserves aspect ratios exactly.
    assert (size[0]*scale)/(size[2]*scale)==pytest.approx(size[0]/size[2])


def test_bad_bounds_fail_instead_of_overlapping():
    with pytest.raises(ValueError):fit_scale((0,0,0),(0,1,1))
    with pytest.raises(ValueError):fit_scale((0,0,0),(1,1,1),1.5)


def test_obb_rejects_large_boxes_and_shelf_overhang():
    import ast
    import numpy as np
    path=Path(__file__).parents[1]/'inspire_sweep_baseline/contact_sweep_clutter_fusion_holding_osc_object_random_v0/placement.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='valid_layout')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    valid=ns['valid_layout'];half=np.array([[.05,.02],[.05,.02]])
    centers=np.array([[0,0],[.09,0]])
    assert not valid(centers,np.zeros(2),half)
    assert valid(centers,np.full(2,np.pi/2),half)
    assert not valid(np.array([[.18,0],[0,0]]),np.zeros(2),half)


@pytest.mark.parametrize('kind', ['box', 'can'])
def test_collision_proxy_is_closed_and_has_level_support_inside_visual_bounds(kind):
    import collections
    import numpy as np
    mesh=runpy.run_path(str(Path(__file__).parents[1]/'inspire_sweep_baseline/contact_sweep_clutter_fusion_holding_osc_object_random_v0/geometry.py'))['flat_collision_mesh']
    dimensions=np.array([.098,.043,.129])
    points,counts,indices=mesh(dimensions,kind)
    points=np.asarray(points)
    np.testing.assert_allclose(points.min(0),[-dimensions[0]/2,-dimensions[1]/2,0.],atol=1e-10)
    np.testing.assert_allclose(points.max(0),[dimensions[0]/2,dimensions[1]/2,dimensions[2]],atol=1e-10)
    # Both caps are planar; every edge has two opposing uses (closed manifold).
    offset=0;edges=collections.Counter();volume=0.
    for count in counts:
        face=indices[offset:offset+count];offset+=count
        for a,b in zip(face,face[1:]+face[:1]):edges[a,b]+=1
        for j in range(1,count-1):
            volume+=np.dot(points[face[0]],np.cross(points[face[j]],points[face[j+1]]))/6
    assert all(edges[b,a]==n==1 for (a,b),n in edges.items())
    assert volume>0
    if kind=='box':assert volume==pytest.approx(np.prod(dimensions))
    else:assert .99<volume/(math.pi*dimensions[0]*dimensions[1]*dimensions[2]/4)<1.
    bottom=points[indices[:counts[0]]]
    assert np.all(bottom[:,2]==0.) and len(bottom)>=4
