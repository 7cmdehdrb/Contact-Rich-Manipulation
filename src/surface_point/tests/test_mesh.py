"""Real visibility seam: bottle silhouette differs from its enclosing box."""
import numpy as np
from observed_obb import Box
from surface_point.mesh import MeshObserver, box_triangles


def combine(parts):
    vertices=[];faces=[];owners=[];offset=0
    for owner,box in parts:
        v,f=box_triangles(box)
        vertices.extend(v);faces.extend(f+offset);owners.extend([owner]*len(f));offset+=len(v)
    return np.array(vertices),np.array(faces),np.array(owners)


def test_visible_bottle_body_midpoint_not_enclosing_box_edge():
    # Bottle body + narrow neck, front box hides left body and lower neck.
    body=Box((0,0,.04),(.04,.10,.08))
    neck=Box((0,0,.10),(.025,.03,.04))
    blocker=Box((.1,-.0225,.05),(.04,.095,.10))
    vertices,faces,owners=combine([(0,body),(0,neck),(1,blocker)])
    observer=MeshObserver(device='cpu',spacing=.001)
    result=observer.observe(vertices,faces,owners,(Box((0,0,.06),(.04,.10,.12)),blocker))
    bottle=[r for r in result.records if r.patch.source_index==0]
    body_points=[r.patch.observation.point for r in bottle if r.patch.observation.point[2]<.08]
    assert len(body_points)==1
    np.testing.assert_allclose(body_points[0],(.02,.0375,.04),atol=.0011)
    assert all(result.owner[r,c]==item['source_index'] for item in result.pixels for r,c in [item['pixel_rc']])


def test_raycast_is_actual_surface_and_no_foreground_leak():
    boxes=(Box((0,0,.05),(.10,.12,.10)),Box((.2,.03,.05),(.04,.08,.10)))
    v,f,ids=combine(list(enumerate(boxes)))
    model=MeshObserver(device='cpu',spacing=.002)
    result=model.observe(v,f,ids,boxes)
    for record in result.records:
        p=record.patch.observation.point
        assert abs(p[0]-(.05 if record.patch.source_index==0 else .22))<1e-5
        if record.patch.source_index==0: assert p[1]<-.01
    again=model.observe(v,f,ids,boxes)
    assert result.records==again.records


def test_representative_rules_and_static_occluder_are_explicit():
    back=Box((0,0,.1),(.1,.16,.2))
    obstacle=Box((.2,-.03,.06),(.04,.1,.12))
    v,f,ids=combine([(0,back),(-1,obstacle)])  # environment masks, never emits a point
    centroid=MeshObserver(device='cpu',spacing=.001).observe(v,f,ids,(back,))
    bbox=MeshObserver(device='cpu',spacing=.001,representative='bbox-center').observe(v,f,ids,(back,))
    assert len(centroid.records)==len(bbox.records)==1
    a=centroid.records[0].patch.observation
    b=bbox.records[0].patch.observation
    assert np.linalg.norm(np.array(a.point)-b.point)>.005
    assert b.corrected  # bbox center is behind the foreground environment geometry
    for frame in (centroid,bbox):
        r,c=frame.pixels[0]['pixel_rc']
        assert frame.owner[r,c]==0
        np.testing.assert_allclose(frame.records[0].patch.observation.point,frame.hits[r,c])
