"""Behavioral contracts for the independent OBB environment, without Sim startup."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]/'inspire_sweep_baseline'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def obb():
    return load_file('_prof_obb', ROOT/'mdp/common/prof_revision_obb.py')


@pytest.fixture(scope='module')
def state_class():
    return load_file('_prof_state', ROOT/'mdp/common/prof_revision_state.py').ProfRevisionState


def test_reference_boxes_and_partial_occlusion(obb):
    reference = load_file('_prof_obb_reference', ROOT.parents[1]/'observed_obb/observed_obb/observation.py')
    rng = np.random.default_rng(42)
    for _ in range(40):
        values = [(tuple(rng.uniform(-.2, .2, 3)), tuple(rng.uniform(.03, .15, 3)), rng.uniform(-math.pi, math.pi)) for i in range(7)]
        actual = obb.observe([obb.Box(*v) for v in values])
        expected = reference.observe([reference.Box(*v) for v in values])
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a.box.center[2] == e.box.center[2]
            assert a.box.size[2] == e.box.size[2]
            assert a.box.size[0]*a.box.size[1] == pytest.approx(e.box.size[0]*e.box.size[1], abs=1e-12)
            # Multiple distinct rectangles can enclose the same triangle with
            # equal minimum area. The new CPU/GPU rule stabilizes these ties.
            if abs(a.box.yaw-e.box.yaw) < 1e-10:
                np.testing.assert_allclose(a.box.center, e.box.center, atol=1e-12)
                np.testing.assert_allclose(a.box.size, e.box.size, atol=1e-12)


def test_filter_boundary_and_source_ownership(obb):
    # Centered front occluder splits the rear target into left and right pieces.
    boxes = [obb.Box((0., 0., .1), (.1, .1, .2)), obb.Box((.2, 0., .1), (.1, .06, .2))]
    parts = obb.observe(boxes)
    assert [p.source for p in parts] == [0, 0, 1]
    assert parts[0].width == pytest.approx(.02)
    assert parts[1].width == pytest.approx(.02)
    assert len(obb.observe(boxes, minimum_width=.02001)) == 1
    assert len(obb.observe(boxes, minimum_ratio=.201)) == 1
    # Filtered-out occluders still hide a source.
    assert not any(p.source == 0 for p in obb.observe([boxes[0], obb.Box((.2, 0., .1), (.1, .1, .2))]))


@pytest.mark.parametrize('a,b,expected',[(.2,.19,.049),(.19,.2,-.049),(.105,.095,.0245),(.095,.105,-.0245),(.09,.08,0.)])
def test_signed_reach_and_no_double_consumption(state_class, a, b, expected):
    state = state_class(1, 'cpu')
    center, goal = torch.zeros(1,3), torch.tensor([[0.,1.,0.]])
    state.reset(torch.tensor([0]), center, goal, torch.tensor([[a,0.,0.]]), .1)
    state.finish(center, goal, torch.zeros(1,3), torch.tensor([False]), torch.tensor([True]), torch.tensor([[b,0.,0.]]), .1)
    assert 4.9*state.reach_delta.item() == pytest.approx(expected, abs=1e-7)
    first = state.reach_delta.clone()
    assert torch.equal(state.reach_delta, first)


def test_contact_two_of_three_and_sweep_weights(state_class):
    state = state_class(1,'cpu')
    ids, center, goal = torch.tensor([0]), torch.zeros(1,3), torch.tensor([[1.,0.,0.]])
    state.reset(ids, center, goal, center, .1)
    for contact, x, expected in [(True,.01,0.),(True,.02,.01),(False,.03,0.),(True,.04,.01)]:
        center[:,0] = x
        state.finish(center, goal, torch.tensor([[2.,0.,0.]]), torch.tensor([contact]), torch.tensor([True]), center, .1)
        assert state.positive.item() == pytest.approx(expected, abs=1e-7)
    center[:,0] = .03
    state.finish(center, goal, torch.zeros(1,3), torch.tensor([False]), torch.tensor([True]), center, .1)
    assert -20*state.retreat.item() == pytest.approx(-.2, abs=2e-6)
    assert not hasattr(state,'contact_count')


def test_real_geometry_surface_distance_and_frames(load_mdp):
    import importlib
    geometry = importlib.import_module('_baseline_cpu_pkg.mdp.common.prof_revision_geometry')
    identity = torch.tensor([1.,0.,0.,0.])
    finger = torch.tensor([[[0.,0.,0.],[.5,0.,0.],[.6,0.,0.],[.6,.6,0.],[.6,.6,.6]]])
    data = SimpleNamespace(body_link_pos_w=finger.clone(),body_link_quat_w=identity.expand(1,5,4).clone())
    env = SimpleNamespace(cfg=SimpleNamespace(formation_distance_scale=.05),scene={
        'robot':SimpleNamespace(data=data),
        'ee_frame':SimpleNamespace(data=SimpleNamespace(target_pos_w=torch.zeros(1,1,3),target_quat_w=identity.expand(1,1,4).clone())),
        'object_collection':SimpleNamespace(data=SimpleNamespace(object_link_pos_w=torch.zeros(1,1,3),object_link_quat_w=identity.expand(1,1,4).clone()))},
        _finger_body_ids=torch.arange(5),_finger_offsets=torch.zeros(5,3),
        _finger_pos_w=torch.zeros(1,5,3),_finger_pose_e=torch.zeros(1,5,7),
        _obb_target_local_pos=torch.zeros(1,3),_obb_target_local_quat=identity.expand(1,4).clone(),
        _obb_target_size=torch.ones(1,3),_obb_live_pos_w=torch.zeros(1,3),_obb_live_quat_w=identity.expand(1,4).clone(),
        _finger_gaps=torch.zeros(1,5),_formation=torch.zeros(1))
    geometry.update_geometry(env)
    torch.testing.assert_close(env._finger_gaps,torch.tensor([[0.,0.,.1,math.sqrt(.02),math.sqrt(.03)]]))
    torch.testing.assert_close(env._finger_pose_e[...,:3],finger)
    assert env._finger_pose_e[...,6].eq(1.).all()
    assert env._formation.item() == pytest.approx(.05/(.05+.1+math.sqrt(.02)+math.sqrt(.03)), abs=1e-7)
    # A non-cubic rotated OBB must use its own axes, not a world AABB.
    rotation = torch.tensor([[math.cos(.35),0.,0.,math.sin(.35)]])
    local = torch.tensor([[[1.1,0.,0.],[1.1,.6,0.],[0.,0.,.35],[0.,0.,0.],[1.,.5,.25]]])
    env._obb_target_size[:] = torch.tensor([2.,1.,.5])
    env.scene['object_collection'].data.object_link_quat_w[:] = rotation[:,None]
    data.body_link_pos_w[:] = geometry.quat_apply(rotation[:,None].expand(-1,5,-1),local)
    env.scene['ee_frame'].data.target_quat_w[:] = torch.tensor([math.sqrt(.5),math.sqrt(.5),0.,0.])
    geometry.update_geometry(env)
    torch.testing.assert_close(env._finger_gaps,torch.tensor([[.1,math.sqrt(.02),.1,0.,0.]]),atol=1e-6,rtol=1e-5)
    assert torch.allclose(env._finger_pose_e[...,3:].norm(dim=-1),torch.ones(1,5))
    half_turn = torch.tensor([[0.,-1.,0.,0.],[0.,0.,-1.,0.]])
    assert geometry.canonical_quaternion(half_turn)[:,1:].sum().item() == 2.



def test_obb_quaternions_reconstruct_full_rotation_and_mask_slots(load_mdp):
    import importlib
    geometry = importlib.import_module('_baseline_cpu_pkg.mdp.common.prof_revision_geometry')
    observations = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.prof_revision')
    a = math.sqrt(.5)
    # EEF +90deg pitch and frozen OBB +90deg yaw have a non-planar relative rotation.
    epos = torch.tensor([[1., 2., 3.]])
    equat = torch.tensor([[a, 0., a, 0.]])
    center, size = torch.tensor([[2., 4., 6.]]), torch.tensor([[2., 4., 6.]])
    box_quat = torch.tensor([[a, 0., 0., a]])
    env = SimpleNamespace(scene={'ee_frame': SimpleNamespace(data=SimpleNamespace(
        target_pos_w=epos[:, None], target_quat_w=equat[:, None]))},
        _obb_target_pos_w=center, _obb_target_quat_w=box_quat, _obb_target_size=size,
        _obb_obstacle_pos_w=center[:, None].expand(-1, 16, -1).clone(),
        _obb_obstacle_quat_w=box_quat[:, None].expand(-1, 16, -1).clone(),
        _obb_obstacle_size=size[:, None].expand(-1, 16, -1).clone(),
        _obb_valid=torch.tensor([[True]+[False]*15]))
    actual = observations.target_obb_in_eef(env)
    torch.testing.assert_close(actual, torch.tensor([[-3., 2., 1., 2., 4., 6., -.5, -.5, .5, .5]]))
    local_corners = torch.cartesian_prod(*[torch.tensor([-1., 1.]) for _ in range(3)])[None] * size[:, None] / 2.
    relative = torch.cat((actual[:, 9:10], actual[:, 6:9]), -1)
    eef_corners = actual[:, None, :3] + geometry.quat_apply(relative[:, None].expand(-1, 8, -1), local_corners)
    world_corners = epos[:, None] + geometry.quat_apply(equat[:, None].expand(-1, 8, -1), eef_corners)
    expected = center[:, None] + torch.stack((-local_corners[..., 1], local_corners[..., 0], local_corners[..., 2]), -1)
    torch.testing.assert_close(world_corners, expected, atol=2e-6, rtol=1e-6)
    # Equivalent quaternion signs yield exactly the same canonical observation.
    env._obb_target_quat_w = -box_quat
    env.scene['ee_frame'].data.target_quat_w = -equat[:, None]
    torch.testing.assert_close(observations.target_obb_in_eef(env), actual)
    env._obb_obstacle_pos_w[:, 1:] = float('nan')
    env._obb_obstacle_quat_w[:, 1:] = float('nan')
    rows = observations.obstacle_obbs_in_eef(env).reshape(1, 16, 11)
    torch.testing.assert_close(rows[:, 0, 1:], actual)
    assert rows[:, 0, 0].eq(1.).all() and rows[:, 1:].eq(0.).all()
    assert rows[:, 0, 7:].norm(dim=-1).allclose(torch.ones(1))


def test_gpu_obb_reference_and_sparse_sources(obb):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    gpu = load_file('_prof_obb_gpu_test', ROOT/'mdp/common/prof_revision_obb_gpu.py')
    rng = np.random.default_rng(10)
    boxes = np.zeros((80,17,7),dtype=np.float32)
    boxes[...,:3] = rng.uniform(-.2,.2,(80,17,3))
    boxes[...,3:6] = rng.uniform(.03,.15,(80,17,3))
    boxes[...,6] = rng.uniform(-3.,3.,(80,17))
    active = np.zeros((80,17),dtype=bool)
    active[:,[0,2,4,7,10,12,16]] = True
    out = [v.cpu().numpy() for v in gpu.observe_batch(torch.tensor(boxes,device='cuda'),torch.tensor(active,device='cuda'))]
    for row in range(80):
        source = np.flatnonzero(active[row])
        parts = obb.observe([obb.Box(tuple(b[:3]),tuple(b[3:6]),float(b[6])) for b in boxes[row,source]])
        target = [o for o in parts if o.source == 0]
        others = [o for o in parts if o.source != 0]
        if not target:
            assert out[-1][row] == 1
            continue
        assert out[-1][row] == 0
        assert out[2][row].sum() == len(others)
        chosen = min(target,key=lambda o:(-o.width,o.lower))
        pairs = [(out[0][row],chosen)] + [(out[1][row,i],o) for i,o in enumerate(others)]
        for value, expected in pairs:
            np.testing.assert_allclose(value,(*expected.box.center,*expected.box.size,expected.box.yaw),atol=2e-6)
        assert out[3][row,:len(others)].tolist() == [int(source[o.source]) for o in others]


def test_obb_models_ppo_gradients_checkpoint_and_export(tmp_path):
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict
    module = load_file('_prof_model_test', ROOT/'agent/prof_revision_models.py')
    torch.manual_seed(42)
    obs = TensorDict({'policy': torch.randn(4,108), 'obstacles': torch.zeros(4,176)}, batch_size=[4])
    slots = obs['obstacles'].unflatten(-1,(16,11))
    slots[:,:2,0] = 1.
    slots[:,:2,1:] = torch.randn(4,2,10)*.1
    groups = {'actor':['policy','obstacles'],'critic':['policy','obstacles']}
    actor = module.ProfRevisionModel(obs,groups,'actor',12, distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'scalar'})
    critic = module.ProfRevisionModel(obs,groups,'critic',1)
    assert actor.obstacle_feature_dim == 11
    linears = [layer for layer in actor.obstacle_encoder.mlp if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [(176,128),(128,64),(64,32),(32,16)]
    assert actor.as_jit().input_size == 284
    alg = PPO(actor,critic,RolloutStorage('rl',4,2,obs,[12]),num_learning_epochs=2,num_mini_batches=1)
    initial = [m.obstacle_encoder.mlp[0].weight.detach().clone() for m in (actor,critic)]
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs,torch.ones(4),torch.zeros(4,dtype=torch.long),{})
        alg.compute_returns(obs)
    losses = alg.update()
    assert all(math.isfinite(v) for v in losses.values())
    for model, before in zip((actor,critic),initial):
        assert not torch.equal(model.obstacle_encoder.mlp[0].weight,before)
        assert model.get_latent(obs).shape == (4,124)
    # Missing rows are entirely masked, including dimensions and quaternions.
    before = actor.get_latent(obs).detach().clone()
    slots[:,2:,1:] = float('nan')
    torch.testing.assert_close(actor.get_latent(obs),before)
    slots[:,2:,1:] = 0.
    raw = torch.cat((obs['policy'],obs['obstacles']),-1)
    export = actor.as_jit()
    scripted = torch.jit.script(export)
    torch.testing.assert_close(scripted(raw),export(raw))
    checkpoint = tmp_path/'model.pt'
    torch.save(alg.save(),checkpoint)
    saved = torch.load(checkpoint,weights_only=False)
    alg.load(saved,None,True)
    torch.testing.assert_close(actor.as_jit()(raw),export(raw))
    # Verify ONNX export with the installed exporter, without requiring ORT.
    onnx = actor.as_onnx()
    torch.onnx.export(onnx,(raw[:1],),str(tmp_path/'model.onnx'),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
    assert (tmp_path/'model.onnx').stat().st_size > 0
    import onnx
    from onnx.reference import ReferenceEvaluator
    graph = onnx.load(str(tmp_path/'model.onnx'))
    onnx.checker.check_model(graph)
    evaluated = ReferenceEvaluator(graph).run(None, {'obs':raw[:1].numpy()})[0]
    np.testing.assert_allclose(evaluated,export(raw[:1]).detach().numpy(),rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('distance,timeout,failed,expected',[(.029,True,False,True),(.03,True,False,True),(.031,True,False,False),(.02,False,False,False),(.02,True,True,False)])
def test_success_uses_visible_center_and_inclusive_boundary(distance,timeout,failed,expected):
    module = load_file('_prof_success',ROOT/'mdp/termination_functions/prof_revision.py')
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _:torch.zeros(1,3)),
        _obb_live_pos_w=torch.tensor([[distance,0.,0.]]),
        cfg=SimpleNamespace(goal_success_radius=.03),episode_length_buf=torch.tensor([500 if timeout else 499]),
        max_episode_length=500,termination_manager=SimpleNamespace(get_term=lambda _:torch.tensor([failed])))
    assert module.success_at_timeout(env).item() is expected
