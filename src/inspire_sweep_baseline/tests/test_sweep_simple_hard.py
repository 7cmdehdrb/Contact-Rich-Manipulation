"""Hard goals: cached feasibility, unchanged direction, no rejection sampling."""
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from test_wrench_refactor import pool_bounds
from test_final_configuration import configs
from test_omni_configuration import omni_config

PARAMS = dict(spawn_radius=.02, edge_clearance=.02, excluded_x_half_angle=0.,
              min_goal_distance=.07, max_goal_distance=.15)


def make_layout(bounds):
    cls = importlib.import_module('_baseline_cpu_pkg.mdp.common.sweep_simple').HardCircleResetLayout
    layout = cls(bounds, 1.05-bounds[...,0,2])
    layout.configure_commands(**PARAMS)
    return layout


def test_exact_boundary_fallbacks_and_cardinal_directions(load_mdp, monkeypatch):
    bounds = pool_bounds(6)
    half = torch.tensor([[.02,.02],[.05,.05],[.14,.14],[.05,.05],[.4,.02],[.02,.02]])
    bounds[:,0,0,:2], bounds[:,0,1,:2] = -half, half
    layout = make_layout(bounds)
    # zero spawn radius draw, then +X,+X,+X,-X,+Y,-Y; every random distance is 11cm.
    draws = torch.tensor([[0.,0.,a,.5] for a in (0.,0.,0.,.5,.25,.75)])
    monkeypatch.setattr(torch,'rand',lambda *args, **kw: draws)
    start, goal = layout.sample_commands(torch.arange(6),torch.Generator(),**PARAMS)
    delta = goal-start
    torch.testing.assert_close(delta, torch.tensor([
        [.11,0.],[.13,0.],[.04,0.],[-.13,0.],[0.,.08],[0.,-.11]]), atol=1e-7,rtol=1e-6)


def test_random_direction_and_geometry_are_safe_without_device_readback(load_mdp, monkeypatch):
    n=4096
    bounds=pool_bounds(n)
    rng=torch.Generator().manual_seed(529)
    half=.02+.12*torch.rand(n,2,generator=rng)
    offset=(torch.rand(n,2,generator=rng)-.5)*.005
    bounds[:,0,0,:2],bounds[:,0,1,:2]=offset-half,offset+half
    layout=make_layout(bounds)
    ids=torch.randperm(n,generator=rng)[:3072]
    draws=torch.rand(len(ids),4,generator=torch.Generator().manual_seed(42))
    # Disallow Tensor truth tests, which would synchronize a GPU to the host.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,'__bool__',lambda self: (_ for _ in ()).throw(AssertionError('device readback')))
        start,goal=layout.sample_commands(ids,torch.Generator().manual_seed(42),**PARAMS)
    delta=goal-start
    distance=delta.norm(dim=-1)
    side=(draws[:,2]>=.5).float()
    theta=(2*draws[:,2]%1)*math.pi+side*math.pi
    direction=torch.stack((theta.cos(),theta.sin()),-1)
    torch.testing.assert_close(delta/distance[:,None],direction,atol=2e-6,rtol=2e-6)
    assert (start.norm(dim=-1)<=.020001).all()
    # Independently rotate all four corners; every translated corner stays inside.
    b=bounds[ids,0]
    corners=torch.stack([b[:,i,:2].clone() for i in (0,1,0,1)],1)
    corners[:,2,1]=b[:,1,1]
    corners[:,3,1]=b[:,0,1]
    rotated=torch.stack((corners[...,1],-corners[...,0]),-1)
    assert ((goal[:,None]+rotated).abs()<=torch.tensor([.18,.48])+1e-6).all()
    # Every shorter-than-15cm-feasible ray ends at a footprint boundary.
    # Full 15cm feasibility can be determined independently from translated corners.
    full=start+.15*direction
    full_fits=((full[:,None]+rotated).abs()<=torch.tensor([.18,.48])).all(dim=-1).all(dim=-1)
    sampled=.07+.08*draws[:,3]
    torch.testing.assert_close(distance[full_fits],sampled[full_fits])
    margin=torch.tensor([.18,.48])-(goal[:,None]+rotated).abs()
    torch.testing.assert_close(margin[~full_fits].amin(dim=(1,2)),torch.zeros_like(distance[~full_fits]),atol=1e-6,rtol=0)
    assert (distance<.07).any() and ((distance>=.07)&(distance<.15)&~full_fits).any()
    # Full directions include both narrow Pull/Push sectors immediately.
    assert (direction[:,0]>.999).any() and (direction[:,0]<-.999).any()


def test_hard_layout_preserves_fast_population_red_and_seed(load_mdp):
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bounds=pool_bounds(128)
        layout=make_layout(bounds)
        ids=torch.arange(128)
        values=layout.sample(ids,torch.Generator().manual_seed(42),**PARAMS)
        again=layout.sample(ids,torch.Generator().manual_seed(42),**PARAMS)
        for a,b in zip(values,again): assert torch.equal(a,b)
        local,yaw,active,delta,requested=values
        assert (active[:,1:].sum(-1)==requested).all()
        assert requested.unique().tolist()==list(range(7))
        off=layout.offset
        centers=local[...,:2]+torch.stack((yaw.cos()*off[...,0]-yaw.sin()*off[...,1],
                                          yaw.sin()*off[...,0]+yaw.cos()*off[...,1]),-1)
        mask=active.clone();mask[:,0]=False
        assert ((centers[...,0].abs()+layout.expanded<=.200001)|~mask).all()
        assert ((centers[...,1].abs()+layout.expanded<=.500001)|~mask).all()
        distances=torch.cdist(centers,centers)
        pairs=mask[:,:,None]&mask[:,None,:]&~torch.eye(17,dtype=torch.bool)[None]
        assert ((distances>=layout.expanded[:,:,None]+layout.expanded[:,None,:]-1e-6)|~pairs).all()
        # Match actual RED sweep against each environment's obstacle centers.
        for i in range(128):
            red=layout.red_distance(centers[i],centers[i:i+1,0],centers[i:i+1,0]+delta[i:i+1])[0]
            assert ((red>=layout.expanded[i,0]+layout.expanded[i]-1e-6)|~mask[i]).all()
    finally:
        torch.set_num_threads(old_threads)


def test_startup_wiring_cache_and_later_parameter_change(configs, monkeypatch):
    module=configs('event_functions','sweep_simple')
    params=configs('events','sweep_simple').SweepSimpleHardEventsCfg().object_spawn.params
    env=NS(cfg=NS(events=NS(object_spawn=NS(params=params))))
    calls=[]
    def initialize(env,ids,layout_class):
        calls.append(ids)
        bounds=pool_bounds(4)
        env._fast_layout=layout_class(bounds,1.05-bounds[...,0,2])
    monkeypatch.setattr(module,'initialize_fast_geometry',initialize)
    module.initialize_hard_geometry(env,None)
    assert calls==[None]
    cached=env._fast_layout._goal_upper
    env._fast_layout.sample_commands(torch.tensor([3,1]),torch.Generator(),**PARAMS)
    assert env._fast_layout._goal_upper is cached
    with pytest.raises(RuntimeError,match='changed after startup'):
        env._fast_layout.sample_commands(torch.tensor([0]),torch.Generator(),**(PARAMS|{'max_goal_distance':.2}))


def test_parent_goal_hook_is_bitwise_identical(load_mdp):
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    bounds=pool_bounds(32)
    layout=module.CircleResetLayout(bounds,1.05-bounds[...,0,2])
    ids=torch.tensor([17,2,0,5])
    a=torch.Generator().manual_seed(42);b=torch.Generator().manual_seed(42)
    direct=module.sample_commands(bounds[ids,0],a)
    hooked=layout.sample_commands(ids,b)
    for x,y in zip(direct,hooked): assert torch.equal(x,y)
    assert torch.equal(a.get_state(),b.get_state())
