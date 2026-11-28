"""Role changes must select actual bodies without rewriting scene poses."""
import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(load_mdp):
    prefix = '_baseline_cpu_pkg.mdp.common.'
    return (importlib.import_module(prefix+'prof_revision_sequence'),
            importlib.import_module(prefix+'prof_revision_geometry'))


def test_fixed_central_layout_actual_sizes_and_goals(modules):
    sequence,_ = modules
    bounds = torch.tensor([[-.044,-.044,-.055],[.044,.044,.055]]).expand(2,17,2,3).clone()
    layout = sequence.TwoCylinderLayout(bounds,torch.full((2,17),1.105))
    local,yaw,active,delta,count = layout.sample(torch.tensor([0,1]),None)
    torch.testing.assert_close(local[:,:2,:2],torch.tensor(sequence.START_XY).expand(2,-1,-1))
    assert active.sum(-1).eq(2).all() and active[:,:2].all() and not active[:,2:].any()
    assert count.eq(1).all()
    torch.testing.assert_close(delta,torch.tensor([[0.,-.06],[0.,-.06]]))
    goals = torch.tensor(sequence.GOAL_XY)
    distance = (goals-torch.tensor(sequence.START_XY)).norm(dim=-1)
    torch.testing.assert_close(distance,torch.full((2,),.06))
    assert goals[:,1].tolist() == pytest.approx([-.11,.11])
    assert local[:,:2,0].abs().max() <= .03501
    assert yaw[:,:2].eq(-torch.pi/2).all()


def test_layout_rejects_overlap_instead_of_parking_required_objects(modules):
    sequence,_ = modules
    bounds = torch.tensor([[-.07,-.07,-.05],[.07,.07,.05]]).expand(1,17,2,3).clone()
    layout = sequence.TwoCylinderLayout(bounds,torch.full((1,17),1.1))
    with pytest.raises(ValueError,match='overlaps'):
        layout.sample(torch.tensor([0]),None)


def test_snapshot_changes_target_role_and_preserves_every_physical_pose(modules,monkeypatch):
    sequence,_ = modules
    # Test role/order wiring independently of the existing CUDA-only observer kernel.
    # The production geometric observer is exercised by the live sequence test.
    def observer(boxes,active,*args):
        n = len(boxes)
        return (boxes[:,0].clone(),boxes[:,1:].clone(),active[:,1:].clone(),
                torch.arange(1,17).expand(n,-1),torch.zeros(n,dtype=torch.int32))
    monkeypatch.setattr(sequence,'observe_batch',observer)
    bounds = torch.tensor([[-.04,-.04,-.05],[.04,.04,.05]]).expand(1,17,2,3).clone()
    local = torch.zeros(1,17,3)
    local[0,:2,:2] = torch.tensor(sequence.START_XY)
    local[:,:,2] = 1.1
    local[0,0,1] = -.12  # First cylinder has already been swept aside.
    shelf_pos = torch.tensor([[-.7,0.,0.]])
    position = local+shelf_pos[:,None]
    quat = torch.tensor([1.,0.,0.,0.]).expand(1,17,4).clone()
    active = torch.zeros(1,17,dtype=torch.bool)
    active[:,:2] = True
    env = SimpleNamespace(num_envs=1,device='cpu',
        cfg=SimpleNamespace(obb_minimum_width=.02,obb_minimum_ratio=.2),
        _revision_bounds_t=bounds,_active_object_mask=active,
        scene={'object_collection':SimpleNamespace(data=SimpleNamespace(object_link_pos_w=position,object_link_quat_w=quat)),
               'shelf':SimpleNamespace(data=SimpleNamespace(root_pos_w=shelf_pos,root_quat_w=torch.tensor([[1.,0.,0.,0.]])))},
        _obb_target_pos_w=torch.zeros(1,3),_obb_target_quat_w=torch.zeros(1,4),_obb_target_size=torch.zeros(1,3),
        _obb_obstacle_pos_w=torch.zeros(1,16,3),_obb_obstacle_quat_w=torch.zeros(1,16,4),
        _obb_obstacle_size=torch.zeros(1,16,3),_obb_valid=torch.zeros(1,16,dtype=torch.bool),
        _obb_sources=torch.full((1,16),-1,dtype=torch.long),
        _obb_target_local_pos=torch.zeros(1,3),_obb_target_local_quat=torch.zeros(1,4))
    original = position.clone()
    sequence.refresh_sequence_snapshot(env,1)
    assert env._obb_target_pos_w[0,0] == pytest.approx(-.7+sequence.START_XY[1][0],abs=1e-6)
    assert env._obb_target_pos_w[0,1] > 0.
    assert not (env._obb_sources[env._obb_valid] == 1).any()
    torch.testing.assert_close(position,original)
    torch.testing.assert_close(active,env._active_object_mask)


def test_contact_capture_passes_physical_target_index(modules,monkeypatch):
    sequence,_ = modules
    selected = []
    monkeypatch.setattr(sequence.ContactSamples,'capture',lambda self,env,target_index=0:selected.append(target_index))
    contact = object.__new__(sequence.SequenceContactSamples)
    env = SimpleNamespace(sequence_stage=1)
    contact.capture(env)
    assert selected == [1]
