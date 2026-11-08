"""Reset geometry and numerical IK without Isaac Sim."""
import importlib
import pytest
import torch


def module(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_ik')


def test_start_side_ties_randomization_and_outside_shelf(load_mdp):
    m = module(load_mdp)
    target = torch.zeros(4096, 3)
    target[:, 1] = torch.linspace(-.02, .02, 4096)
    direction = torch.zeros_like(target)
    direction[:, 1] = torch.tensor([-1., 1., 0., 0.]).repeat(1024)
    direction[2::4, 0], direction[3::4, 0] = 1., -1.
    default = torch.tensor([.303, -.133, 1.201]).expand_as(target)
    a = m.sample_initial_eef(target, direction, default, torch.Generator().manual_seed(42))
    b = m.sample_initial_eef(target, direction, default, torch.Generator().manual_seed(42))
    assert torch.equal(a, b)
    assert ((a[:, 0] >= .30) & (a[:, 0] <= .33)).all()
    side = torch.where(direction[:, 1] <= 0, 1., -1.)
    offset = (a[:, 1] - target[:, 1]) * side
    assert ((offset >= .10) & (offset <= .14)).all()
    assert (a[:, 2] - default[:, 2]).abs().max() <= .010001
    assert a.std(dim=0).min() > .005
    assert (a[2::4, 1] > target[2::4, 1]).all()
    assert (a[3::4, 1] > target[3::4, 1]).all()


def chain(m):
    identity = (1., 0., 0., 0.)
    joints = [((0., 0., .15), identity, (0., 0., 0.), identity, 2, 0),
              ((.1, 0., .1), identity, (0., 0., 0.), identity, 1, 1),
              ((.3, 0., 0.), identity, (0., 0., 0.), identity, 1, 2),
              ((.25, 0., 0.), identity, (0., 0., 0.), identity, 0, 3),
              ((.1, 0., 0.), identity, (0., 0., 0.), identity, 1, 4),
              ((.1, 0., 0.), identity, (0., 0., 0.), identity, 2, 5)]
    return m.ResetArmIK(joints, (.05, .03, .02), identity, 'cpu')


def test_reset_quality_rejects_singularity_and_joint_limit(load_mdp):
    m = module(load_mdp)
    class KnownJacobian:
        def forward(self, angles, jacobian=False):
            j = torch.eye(6).repeat(3, 1, 1)
            j[:, :3] *= .3
            j[1, -1, -1] = 0.
            return None, None, j
    angles = torch.zeros(3, 6)
    angles[2, 0] = .98
    valid, condition, margin = m.initial_pose_quality(
        KnownJacobian(), angles, torch.tensor([-1., 1.]).expand(3, 6, 2))
    assert valid.tolist() == [True, False, False]
    assert condition[0] == 1 and condition[1] > 80
    assert margin[2] < .05


def test_geometric_jacobian_matches_finite_difference(load_mdp):
    m = module(load_mdp)
    ik = chain(m)
    q = torch.tensor([[.3, -.5, 1., .2, -.6, -.2]])
    p, r, j = ik.forward(q, jacobian=True)
    for i in range(6):
        plus, minus = q.clone(), q.clone()
        plus[:, i] += .005
        minus[:, i] -= .005
        pp, rp = ik.forward(plus)
        pm, rm = ik.forward(minus)
        linear = (pp - pm) / .01
        angular = m.axis_angle_from_quat(m.quat_mul(rp, m.quat_conjugate(rm))) / .01
        torch.testing.assert_close(torch.cat((linear, angular), -1), j[:, :, i], atol=1e-4, rtol=1e-3)


def test_ik_solves_pose_and_reports_unreachable_target(load_mdp):
    m = module(load_mdp)
    ik = chain(m)
    seed = torch.tensor([[.3, -.5, 1., .2, -.6, -.2]])
    known = seed + torch.tensor([[.05, -.08, .1, -.03, .05, .01]])
    pos, quat = ik.forward(known)
    limits = torch.tensor([-6., 6.]).expand(1, 6, 2)
    answer, valid, pe, re = ik.solve(pos, quat, seed, limits)
    assert valid.all() and pe.max() < .0005 and re.max() < .003
    assert torch.isfinite(answer).all()
    _, valid, _, _ = ik.solve(pos + 10, quat, seed, limits, iterations=4)
    assert not valid.any()


@pytest.mark.parametrize('params', [dict(front_clearance=(-.01, .1)), dict(lateral_offset=(.2, .1)),
                                   dict(height_jitter=-.01)])
def test_invalid_start_region_rejected(load_mdp, params):
    m = module(load_mdp)
    with pytest.raises(ValueError):
        m.sample_initial_eef(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3),
                             torch.Generator(), **params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer path')
def test_small_batch_cpu_ik_matches_cuda_reference(load_mdp):
    m = module(load_mdp)
    ik = chain(m)
    ik.device = 'cuda'
    ik.joints = [(*(v.cuda() for v in joint[:4]), *joint[4:]) for joint in ik.joints]
    ik.offset_pos, ik.offset_quat = ik.offset_pos.cuda(), ik.offset_quat.cuda()
    seed = torch.tensor([[.3, -.5, 1., .2, -.6, -.2]], device='cuda').expand(32, -1)
    rng = torch.Generator(device='cuda').manual_seed(42)
    desired = seed + .15 * (torch.rand((32, 6), device='cuda', generator=rng) - .5)
    pos, quat = ik.forward(desired)
    limits = torch.tensor([-6., 6.], device='cuda').expand(32, 6, 2)
    fast = ik.solve(pos, quat, seed, limits)
    reference = ik.solve(pos, quat, seed, limits, _allow_cpu=False)
    assert torch.equal(fast[1], reference[1]) and fast[1].all()
    torch.testing.assert_close(fast[0], reference[0], atol=1e-5, rtol=1e-5)
    _, invalid, _, _ = ik.solve(pos + 10., quat, seed, limits, iterations=4)
    assert not invalid.any()
