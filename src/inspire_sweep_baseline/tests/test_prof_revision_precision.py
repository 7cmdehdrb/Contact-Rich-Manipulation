"""Precision shaping boundaries, lifecycle and independent numerical contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'inspire_sweep_baseline'


def load(path):
    spec = importlib.util.spec_from_file_location('_precision_test_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state_module = load(ROOT / 'mdp/common/prof_revision_state.py')
rewards = load(ROOT / 'mdp/reward_functions/prof_revision.py')


@pytest.mark.parametrize('inner,outer,radius,multiplier', [
    (-.01,.06,.06,2.), (.03,.03,.06,2.), (.06,.03,.06,2.),
    (.03,.06,0.,2.), (.03,.06,.06,.9),
    (float('nan'),.06,.06,2.), (.03,float('inf'),.06,2.),
    (.03,.06,float('nan'),2.), (.03,.06,.06,float('inf')),
])
def test_invalid_parameters(inner, outer, radius, multiplier):
    with pytest.raises(ValueError):
        state_module.ProfRevisionPrecisionState(1, 'cpu', inner, outer, radius, multiplier)


def test_boundaries_derivative_and_independent_quadrature():
    d = torch.tensor([0., .015, .03, .045, .06, .12], dtype=torch.float64)
    alpha = state_module.sweep_force_influence(d, .03, .06)
    torch.testing.assert_close(alpha, torch.tensor([0.,0.,0.,.5,1.,1.], dtype=d.dtype))
    w = state_module.sweep_distance_sensitivity(d, .06, 2.)
    torch.testing.assert_close(w, torch.tensor([2.,1.84375,1.5,1.15625,1.,1.], dtype=d.dtype))
    for boundary in (.03, .06):
        side = torch.tensor([boundary-1e-9,boundary,boundary+1e-9],dtype=d.dtype)
        a = state_module.sweep_force_influence(side,.03,.06)
        assert (a[1:]-a[:-1]).abs().max() < 1e-12
    grid = torch.linspace(0., .15, 20001, dtype=d.dtype, requires_grad=True)
    h = state_module.sweep_weighted_distance(grid, .06, 2.)
    derivative = torch.autograd.grad(h.sum(),grid)[0]
    torch.testing.assert_close(derivative, state_module.sweep_distance_sensitivity(grid,.06,2.))
    # Independent integration of the defining cubic (not of the helper output).
    for distance in d.tolist():
        x = torch.linspace(0.,distance,10001,dtype=d.dtype)
        u = (x/.06).clamp(0.,1.)
        integral = torch.trapz(2.-3.*u.square()+2.*u.pow(3),x)
        actual = state_module.sweep_weighted_distance(torch.tensor(distance,dtype=d.dtype),.06,2.)
        torch.testing.assert_close(actual,integral,atol=1e-9,rtol=0.)
    torch.testing.assert_close(state_module.sweep_weighted_distance(d,.06,1.),d)


def make_state(count=1):
    return state_module.ProfRevisionPrecisionState(count,'cpu',.03,.06,.06,2.)


def positions(distance):
    return torch.stack((torch.as_tensor(distance),torch.zeros(len(distance)),torch.zeros(len(distance))),-1)


@pytest.mark.parametrize('distance,force,contact,valid,positive', [
    (.08, 2., True, True, True), (.08,-2.,True,True,False),
    (.08,.5,True,True,False), (.045,-2.,True,True,True),
    (.02,-2.,True,True,True), (.02,0.,True,True,True),
    (.02,2.,False,True,False), (.02,2.,True,False,False),
])
def test_force_fade_contact_gate_and_parent_buffer_equivalence(distance,force,contact,valid,positive):
    child = make_state()
    parent = state_module.ProfRevisionState(1,'cpu')
    goal = torch.zeros(1,3)
    center = positions([distance+.002])
    for state in (child,parent):
        state.reset(torch.tensor([0]),center,goal,center,.1)
        state.finish(center,goal,positions([-force]),torch.tensor([True]),torch.tensor([True]),center,.1)
        state.finish(positions([distance]),goal,positions([-force]),torch.tensor([contact]),torch.tensor([valid]),center,.1)
    for name,value in vars(parent).items():
        torch.testing.assert_close(getattr(child,name),value)
    assert (child.precision_positive.item()>0.) is positive
    alpha = max(0.,min(1.,(distance-.03)/.03))
    alpha = alpha*alpha*(3.-2.*alpha)
    assert child.effective_alignment.item() == pytest.approx(1.-alpha+alpha*child.alignment.item(),abs=2e-6)


def test_first_step_stationary_crossing_return_and_partial_reset():
    state = make_state(2)
    goal = torch.zeros(2,3)
    center = positions([.04,.08])
    contact = torch.ones(2,dtype=torch.bool)
    state.reset(torch.arange(2),center,goal,center,.1)
    state.finish(positions([.03,.07]),goal,-torch.ones(2,3),contact,contact,center,.1)
    assert state.precision_positive.eq(0.).all()  # Only one recent contact.
    state.finish(positions([.03,.07]),goal,-torch.ones(2,3),contact,contact,center,.1)
    assert state.precision_positive.eq(0.).all() and state.precision_retreat.eq(0.).all()
    # Cross the goal: .03 -> -.04 increases endpoint distance and costs retreat even without contact.
    state.finish(positions([-.04,.07]),goal,torch.zeros(2,3),~contact,contact,center,.1)
    assert state.precision_retreat[0]>0. and state.precision_positive.eq(0.).all()
    state.finish(positions([-.02,.07]),goal,torch.zeros(2,3),contact,contact,center,.1)
    assert state.precision_positive[0]>0.  # History 2/3, current contact and near-goal braking.
    untouched = {name:value[1:].clone() for name,value in vars(state).items() if isinstance(value,torch.Tensor)}
    state.reset(torch.tensor([0]),positions([.09]),goal[:1],center[:1],.1)
    for name,value in untouched.items():
        torch.testing.assert_close(getattr(state,name)[1:],value)
    assert state.precision_positive[0] == state.precision_retreat[0] == 0.
    assert not state.contact_history[0].any()
    state.finish(positions([.10,.07]),goal,torch.zeros(2,3),~contact,contact,center,.1)
    assert state.precision_positive[0] == 0. and state.precision_retreat[0] == pytest.approx(.01,abs=1e-7)


@pytest.mark.parametrize('dt',[.01,.02,.05])
def test_reward_rates_are_order_independent_and_dt_applied_once(dt):
    state = SimpleNamespace(precision_positive=torch.tensor([.01,0.]), precision_retreat=torch.tensor([0.,.01]))
    env = SimpleNamespace(_stabilize_state=state,step_dt=dt)
    for _ in range(2):
        torch.testing.assert_close(-40.*dt*rewards.precision_sweep_retreat(env),torch.tensor([0.,-.4]))
        torch.testing.assert_close(100.*dt*rewards.precision_sweep_progress(env),torch.tensor([1.,0.]))


def test_telescope_finite_and_asymmetric_round_trip():
    d = torch.rand(1000,dtype=torch.float64)*.2
    h = state_module.sweep_weighted_distance(d,.06,2.)
    delta = h[:-1]-h[1:]
    torch.testing.assert_close(delta.sum(),h[0]-h[-1])
    assert torch.isfinite(h).all()
    assert 100.*.01-40.*.01 == pytest.approx(.6)
