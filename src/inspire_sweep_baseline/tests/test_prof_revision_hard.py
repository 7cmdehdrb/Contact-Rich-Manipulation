"""Shape-safe uncapped commands and eleven-source frontal visibility."""
import importlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]/'inspire_sweep_baseline'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def layout_module(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.prof_revision_layout')


def test_uncapped_asymmetric_rotated_target_goal_domain(layout_module):
    # A shifted, non-square target catches both centre-offset and yaw mistakes.
    bounds = torch.tensor([[-.02,-.03,0.],[.06,.08,.12]]).expand(4096,17,2,3).clone()
    layout = layout_module.ProfRevisionHardLayout(bounds,torch.zeros(4096,17),cell_size=.01)
    ids = torch.arange(4096)
    start, goal = layout.sample_commands(ids,torch.Generator().manual_seed(11))
    distance = (goal-start).norm(dim=-1)
    assert distance.min() >= .05-1e-7
    assert distance.max() > .40  # Old 12cm cap is absent.
    assert bool((start.norm(dim=-1) <= .02+1e-7).all())
    rotated_min = torch.tensor([-.03,-.06])
    rotated_max = torch.tensor([.08,.02])
    assert bool((goal+rotated_min >= torch.tensor([-.18,-.48])-1e-7).all())
    assert bool((goal+rotated_max <= torch.tensor([.18,.48])+1e-7).all())
    # Partial IDs consume the same independent RNG stream and use the right bounds.
    a = layout.sample_commands(ids[::4],torch.Generator().manual_seed(7))
    b = layout.sample_commands(ids[::4],torch.Generator().manual_seed(7))
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    for minimum in (.049, float('nan')):
        with pytest.raises(ValueError):
            layout.sample_commands(ids[:1],torch.Generator(),min_goal_distance=minimum)
    with pytest.raises(ValueError):
        layout.sample_commands(ids[:1],torch.Generator(),max_goal_distance=.049)


def test_dense_packing_counts_clearance_and_reproducibility(layout_module):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    bounds=torch.tensor([[-.04,-.04,0.],[.04,.04,.10]],device='cuda').expand(256,17,2,3).clone()
    layout=layout_module.ProfRevisionHardLayout(bounds,torch.ones(256,17,device='cuda'),cell_size=.01)
    ids=torch.arange(256,device='cuda')
    generator=torch.Generator(device='cuda').manual_seed(42)
    local,yaw,active,delta,requested=layout.sample(ids,generator)
    assert requested.max()==10 and requested.min()==0
    assert torch.equal(active[:,1:].sum(-1),requested)
    centers=local[...,:2]
    assert bool(((centers.abs()+.04<=bounds.new_tensor([.2,.5])+1e-6)|~active[...,None]).all())
    pairs=active[:,:,None]&active[:,None,:]&~torch.eye(17,device='cuda',dtype=torch.bool)[None]
    assert bool(((centers[:,:,None]-centers[:,None,:]).abs()>=.09-1e-6).any(-1)[pairs].all())
    again=layout.sample(ids,torch.Generator(device='cuda').manual_seed(42))
    for a,b in zip((local,yaw,active,delta,requested),again):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Independent convex-hull SAT checks the entire parent's extruded sweep,
    # including the approach side, rather than just target and goal endpoints.
    from scipy.spatial import ConvexHull
    xy=centers.cpu().numpy()
    goals=(centers[:,0]+delta).cpu().numpy()
    square=np.array([[-.04,-.04],[.04,-.04],[.04,.04],[-.04,.04]])
    for row in range(len(ids)):
        start,goal=xy[row,0],goals[row]
        extension=np.array([.2-min(start[0],goal[0]),0.])
        vertices=np.concatenate([square+p for p in (start,goal,start+extension,goal+extension)])
        polygon=vertices[ConvexHull(vertices).vertices]
        edges=np.roll(polygon,-1,axis=0)-polygon
        normals=np.stack((-edges[:,1],edges[:,0]),-1)
        normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
        axes=np.concatenate((np.eye(2),normals))
        p=polygon@axes.T
        for slot in active[row,1:].nonzero().flatten().cpu().tolist():
            box=(xy[row,slot+1]+square)@axes.T
            gaps=np.maximum(p.min(0)-box.max(0),box.min(0)-p.max(0))
            assert gaps.max()>=.01-1e-6


def test_sparse_layouts_use_free_area_instead_of_packing_into_corners(layout_module):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    # A single obstacle has ample legal interior area. Across independent
    # scenes it must explore that area, rather than always touching two walls.
    bounds=torch.tensor([[-.02,-.02,0.],[.02,.02,.10]],device='cuda').expand(2048,17,2,3).clone()
    layout=layout_module.ProfRevisionHardLayout(bounds,torch.ones(2048,17,device='cuda'),cell_size=.01)
    local,_,active,_,requested=layout.sample(torch.arange(2048,device='cuda'),
        torch.Generator(device='cuda').manual_seed(71))
    singles=requested==1
    centers=local[singles,1:,:2][active[singles,1:]]
    assert len(centers)>100
    wall_gap=(centers.new_tensor([.2,.5])-centers.abs()-.02).amin(-1)
    interior_fraction=float((wall_gap>.04).float().mean())
    assert interior_fraction>.40, f'Only {interior_fraction:.1%} of sparse scenes use the free interior'
    # With several objects, nearest neighbors should not predominantly share
    # the same shelf X/Y line. Ignore the target and inactive parking bodies.
    group=requested>=5
    obstacles=active[group].clone();obstacles[:,0]=False
    difference=(local[group,:,None,:2]-local[group,None,:,:2]).abs()
    distance=difference.norm(dim=-1)
    pairs=obstacles[:,:,None]&obstacles[:,None,:]&~torch.eye(17,device='cuda',dtype=torch.bool)[None]
    distance.masked_fill_(~pairs,float('inf'))
    near=difference.gather(2,distance.argmin(-1)[...,None,None].expand(-1,-1,1,2)).squeeze(2)
    aligned_fraction=float((near.amin(-1)[obstacles]<.01).float().mean())
    assert aligned_fraction<.25, f'{aligned_fraction:.1%} of neighbors still form axis-aligned rows'


@pytest.mark.parametrize('device',['cuda'])
def test_eleven_source_observer_matches_independent_polygon_reference(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    gpu = load('_hard_obb_'+device,ROOT/'mdp/common/prof_revision_obb_gpu.py')
    cpu = load('_hard_obb_reference_'+device,ROOT/'mdp/common/prof_revision_obb.py')
    rng = np.random.default_rng(17)
    boxes = np.zeros((40,17,7),dtype=np.float32)
    boxes[...,:3] = rng.uniform(-.2,.2,(40,17,3))
    boxes[...,3:6] = rng.uniform(.025,.13,(40,17,3))
    boxes[...,6] = rng.uniform(-3.,3.,(40,17))
    active = np.zeros((40,17),dtype=bool)
    active[:,[0,1,2,4,5,7,8,10,12,14,16]] = True
    out = [v.cpu().numpy() for v in gpu.observe_hard_batch(torch.tensor(boxes,device=device),torch.tensor(active,device=device))]
    for row in range(len(boxes)):
        sources = np.flatnonzero(active[row])
        parts = cpu.observe([cpu.Box(tuple(b[:3]),tuple(b[3:6]),float(b[6])) for b in boxes[row,sources]])
        target = [o for o in parts if o.source==0]
        if not target:
            assert out[-1][row]==1
            continue
        assert out[-1][row] in (0,2)
        chosen = min(target,key=lambda o:(-o.width,o.lower))
        np.testing.assert_allclose(out[0][row],(*chosen.box.center,*chosen.box.size,chosen.box.yaw),atol=2e-6)
        other = [o for o in parts if o.source!=0]
        assert out[2][row].sum()==min(len(other),16)
        for i in range(min(len(other),16)):
            observed = out[1][row,i]
            candidates = [p for p in other if sources[p.source]==out[3][row,i]]
            assert any(np.allclose(observed,(*p.box.center,*p.box.size,p.box.yaw),atol=2e-6) for p in candidates)


def test_hard_observer_overflow_keeps_widest_sixteen_fragments():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    gpu=load('_hard_overflow_gpu',ROOT/'mdp/common/prof_revision_obb_gpu.py')
    cpu=load('_hard_overflow_cpu_reference',ROOT/'mdp/common/prof_revision_obb.py')
    boxes=torch.zeros((1,17,7),device='cuda')
    boxes[0,0]=torch.tensor([.3,-.49,0.,.06,.04,.1,0.],device='cuda')
    boxes[0,1]=torch.tensor([-.1,0.,0.,.08,.9,.1,0.],device='cuda')
    boxes[0,2:11,0]=.1
    boxes[0,2:11,1]=torch.linspace(-.4,.4,9,device='cuda')
    boxes[0,2:11,3:6]=torch.tensor([.04,.02,.1],device='cuda')
    active=torch.zeros((1,17),device='cuda',dtype=torch.bool)
    active[:,:11]=True
    target,obstacles,valid,sources,status=gpu.observe_hard_batch(boxes,active,.001,0.)
    assert status.item()==2 and valid.sum().item()==16
    assert active.sum().item()==11
    values=boxes[0,:11].cpu().numpy()
    parts=cpu.observe([cpu.Box(tuple(b[:3]),tuple(b[3:6]),float(b[6])) for b in values],minimum_width=.001,minimum_ratio=0.)
    widths=sorted([p.width for p in parts if p.source!=0],reverse=True)[:16]
    np.testing.assert_allclose(sorted(obstacles[0,:,4].cpu().tolist(),reverse=True),widths,atol=1e-6)
    assert gpu.observe_batch(boxes,active)[-1].item()==3  # Parent still caps seven.
