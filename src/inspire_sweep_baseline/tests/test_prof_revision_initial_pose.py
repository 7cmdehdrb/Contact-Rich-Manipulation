"""Independent geometry checks for the bounded initial-pose selector."""
import importlib
import math

import pytest
import torch


@pytest.fixture
def initial_pose(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.prof_revision_initial_pose')


def test_red_samples_exclude_whole_sweep_lane_and_reproduce(initial_pose):
    start = torch.tensor([[0., 0.], [.01, -.01], [0., 0.]])
    goal = torch.tensor([[0., .12], [-.1, -.02], [.1, 0.]])
    radius = torch.tensor([.08, .08, .08])
    points, valid = initial_pose.red_candidates(start, goal, radius, torch.Generator().manual_seed(42), 2048)
    again = initial_pose.red_candidates(start, goal, radius, torch.Generator().manual_seed(42), 2048)
    assert torch.equal(points, again[0]) and torch.equal(valid, again[1])
    # Independent closest-point scalar oracle, including segment endpoints.
    for row in range(3):
        a, b = start[row].tolist(), goal[row].tolist()
        for x, y in points[row, valid[row]].tolist():
            dx, dy = b[0]-a[0], b[1]-a[1]
            t = min(1., max(0., ((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)))
            assert math.hypot(x-a[0]-t*dx, y-a[1]-t*dy) >= .08-1e-7
            assert abs(x)<.195 and abs(y)<.495
    assert valid[:2].any(-1).all()


def test_segment_endpoints_and_zero_length(initial_pose):
    points = torch.tensor([[[-1., 0.], [.5, .2], [2., 0.]]])
    actual = initial_pose.segment_distance(points, torch.tensor([[0., 0.]]), torch.tensor([[1., 0.]]))
    torch.testing.assert_close(actual, torch.tensor([[1., .2, 1.]]))
    actual = initial_pose.segment_distance(points, torch.zeros(1, 2), torch.zeros(1, 2))
    torch.testing.assert_close(actual, points.norm(dim=-1))


def test_continuous_front_domain_covers_old_regions_and_full_width(initial_pose):
    start = torch.tensor([[0., 0.], [.01, -.01]])
    goal = torch.tensor([[0., .12], [-.1, -.02]])
    base = torch.tensor([.7, .7])
    radius = torch.tensor([.1, .1])
    points, valid = initial_pose.front_candidates(start, goal, base, radius, torch.Generator().manual_seed(7), 10000)
    assert (points[..., 0] >= torch.minimum(start[:, 0], goal[:, 0])[:, None]).all()
    assert (points[..., 0] < .7).all() and (points[..., 1].abs() < .5).all()
    selected = points[valid]
    assert selected[:, 0].max() > .69 and selected[:, 1].min() < -.49 and selected[:, 1].max() > .49
    # Samples now include both sides, the gap between old bands, and far front.
    for xlow, xhigh in ((.15,.20),(.21,.27),(.28,.30),(.40,.65)):
        for ylow, yhigh in ((-.45,-.2),(.2,.45)):
            assert ((selected[:,0]>xlow)&(selected[:,0]<xhigh)&(selected[:,1]>ylow)&(selected[:,1]<yhigh)).any()
    again=initial_pose.front_candidates(start, goal, base, radius, torch.Generator().manual_seed(7), 10000)
    assert torch.equal(points,again[0]) and torch.equal(valid,again[1])
    distance=initial_pose.segment_distance(points,start,goal)
    assert (distance[valid]>=.1-1e-7).all()


def test_red_diagnostic_is_geometry_not_candidate_category(initial_pose):
    start=torch.tensor([[0.,0.]]).expand(4,-1)
    goal=torch.tensor([[0.,.1]]).expand(4,-1)
    points=torch.tensor([[.18,.05],[.4,.05],[.18,-.2],[.1,.15]])
    assert initial_pose.inside_red_polygon(points,start,goal).tolist()==[True,False,False,False]
    # Degenerate collinear region still has bounded X extent.
    goal=torch.tensor([[.1,0.]]).expand(4,-1)
    points=torch.tensor([[.18,0.],[.4,0.],[.18,.1],[-.1,0.]])
    assert initial_pose.inside_red_polygon(points,start,goal).tolist()==[True,False,False,False]


def test_whole_hand_lane_clear_checks_path_and_both_ends(initial_pose):
    m = model(initial_pose)
    m.hand_boxes = torch.tensor([True])
    center = torch.tensor([[[0., .05, 1.2]], [[0., .12, 1.2]], [[.3, .05, 1.2]]])
    low, high = center-.01, center+.01
    start, goal = torch.zeros(3, 2), torch.tensor([[0., .1]]).expand(3, -1)
    assert m.lane_clear(low, high, start, goal, torch.full((3,), .04)).tolist() == [False,False,True]


def model(initial_pose):
    m = initial_pose.RobotCollisionModel.__new__(initial_pose.RobotCollisionModel)
    m.frames = []
    m.links = torch.tensor([0])
    m.center = torch.zeros(1, 3)
    m.half = torch.tensor([[.1, .02, .03]])
    m.shelf_low = torch.tensor([[-10., -10., -1.]])
    m.shelf_high = torch.tensor([[10., 10., -.5]])
    return m


def test_rotated_link_bounds_and_collision_margin_active_mask(initial_pose):
    m = model(initial_pose)
    q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
    pos, joints = torch.zeros(1, 3), torch.zeros(1, 6)
    low, high = m.boxes(joints, pos, q)
    torch.testing.assert_close(high, torch.tensor([[[.02, .1, .03]]]), atol=1e-7, rtol=0.)
    torch.testing.assert_close(low, -high)
    obj_low, obj_high = torch.tensor([[[.021, -.01, -.01]]]), torch.tensor([[[.04, .01, .01]]])
    assert not m.clear(joints, pos, q, obj_low, obj_high, torch.ones(1, 1, dtype=torch.bool)).item()
    assert m.clear(joints, pos, q, obj_low, obj_high, torch.zeros(1, 1, dtype=torch.bool)).item()
    pos[:, 2] = -.55
    assert not m.clear(joints, pos, q, obj_low, obj_high, torch.zeros(1, 1, dtype=torch.bool)).item()


def test_articulated_child_uses_joint_frame_and_reset_finger_angle(initial_pose):
    m = model(initial_pose)
    identity, origin = torch.tensor([1., 0., 0., 0.]), torch.zeros(3)
    m.frames = [(0, torch.tensor([.2, 0., 0.]), identity, origin, identity, 2, 0)]
    m.links = torch.tensor([1])
    m.center = torch.tensor([[.1, 0., 0.]])
    joint = torch.tensor([[math.pi/2]])
    low, high = m.boxes(joint, torch.zeros(1, 3), identity[None])
    torch.testing.assert_close((low+high)/2, torch.tensor([[[.2, .1, 0.]]]), atol=1e-7, rtol=0.)
