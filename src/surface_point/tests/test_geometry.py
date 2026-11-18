import numpy as np
from observed_obb import Box
from surface_point import benchmark
from surface_point.fixtures import scenes


def test_clear_is_front_surface_not_box_center():
    records,legacy=benchmark(scenes()['clear'])
    point=records[0].patch.observation.point
    np.testing.assert_allclose(point,(.05,0,.1))
    assert records[0].center_distance_m == .05
    assert records[0].front_depth_error_m == 0
    assert legacy[0].box.center==(0.,0.,.1)


def test_split_produces_independent_points():
    records,_=benchmark(scenes()['split'])
    back=[r for r in records if r.patch.source_index==0]
    assert len(back)==2
    assert back[0].patch.observation.point[1]<0<back[1].patch.observation.point[1]
    assert all(r.legacy is not None for r in back)


def test_finite_height_and_center_hole_correction():
    records,old=benchmark(scenes()['hole_center'])
    back=[r for r in records if r.patch.source_index==0]
    assert len(back)==1  # connected around the hole, legacy has two strips
    assert len(old)==3
    item=back[0].patch.observation
    assert item.corrected and item.correction_m > .029
    y,z=item.point[1:]
    assert not (-.03 <= y <= .03 and .06 <= z <= .14)
    assert item.point[0]==.05


def test_occluded_objects_not_returned_and_empty_validated():
    records,_=benchmark(scenes()['fully_hidden'])
    assert len(records)==1 and records[0].patch.source_index==1
    assert benchmark(())==((),())


def test_translation_and_surface_membership():
    boxes=scenes()['rotated']; delta=np.array([1.2,-.8,2.])
    a,_=benchmark(boxes)
    b,_=benchmark([Box(tuple(np.array(v.center)+delta),v.size,v.yaw) for v in boxes])
    assert len(a)==len(b)
    for left,right in zip(a,b):
        np.testing.assert_allclose(np.array(right.patch.observation.point)-left.patch.observation.point,delta,atol=1e-10)
        source=boxes[left.patch.source_index]
        p=np.array(left.patch.observation.point)-source.center
        c,s=np.cos(source.yaw),np.sin(source.yaw)
        local=np.array([[c,s],[-s,c]]) @ p[:2]
        half=np.array(source.size[:2])/2
        assert np.all(abs(local)<=half+1e-10)
        assert np.min(abs(abs(local)-half))<1e-10


def test_points_have_no_foreground_box_on_the_same_ray():
    from observed_obb.observation import _footprint, _front_depth
    from surface_point.geometry import FRAME
    for boxes in scenes().values():
        records,_=benchmark(boxes)
        for r in records:
            x,y,z=r.patch.observation.point
            for i,b in enumerate(boxes):
                if i==r.patch.source_index:continue
                if b.center[2]-b.size[2]/2 <= z <= b.center[2]+b.size[2]/2:
                    front=-_front_depth(_footprint(b) @ FRAME.T,y)
                    assert front<=x+1e-12


def test_short_occluder_exposes_surface_above_legacy_infinite_occlusion():
    records,legacy=benchmark(scenes()['short_occluder_U'])
    back=[r for r in records if r.patch.source_index==0]
    assert len(back)==1 and back[0].legacy is None
    np.testing.assert_allclose(back[0].patch.observation.point,(.05,0,.10))
    assert len(legacy)==3
