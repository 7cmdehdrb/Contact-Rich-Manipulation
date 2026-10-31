"""v0_2_1 reward boundaries and logical/physical selection regression checks."""
import importlib.util
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from conftest import MODULE_ROOT
from test_final_configuration import configs, assert_ordered_equal
from test_final_versions import final_rewards
from test_omni_configuration import omni_config


def pool_module():
    spec = importlib.util.spec_from_file_location('pool_unit', MODULE_ROOT/'mdp/common/final_target_pool.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_requested_reward_changes(configs):
    m = configs('rewards', 'final')
    parent = m.FinalGoalAccuracyV0_2RewardsCfg()
    before = parent.to_dict()
    child = m.FinalGoalAccuracyV0_2_1RewardsCfg()
    expected = parent.to_dict()
    expected['target_contact']['func'] = child.target_contact.to_dict()['func']
    expected['target_goal_timeout_bonus'] = child.target_goal_timeout_bonus.to_dict()
    assert_ordered_equal(expected, child.to_dict())
    assert child.target_contact.weight == 1.45
    assert child.target_contact.params == {'threshold_n': .01}
    assert child.target_goal_timeout_bonus.weight == 75
    assert child.target_goal_timeout_bonus.params == {'termination_name': 'target_goal_success_at_timeout'}
    assert_ordered_equal(before, m.FinalGoalAccuracyV0_2RewardsCfg().to_dict())


def test_contact_has_no_goal_dependency(final_rewards, monkeypatch):
    f = torch.tensor([[0.,0,0],[.0099,0,0],[.01,0,0],[.0101,0,0],
                      [float('nan'),0,0],[float('inf'),0,0],[1.,0,0]])
    monkeypatch.setattr(final_rewards, 'target_hand_contact_force_w', lambda env:f)
    env = NS(sensor_data_fresh=torch.tensor([1,1,1,1,1,1,0], dtype=torch.bool))
    # No command or reward manager: goal occupancy cannot be consulted.
    torch.testing.assert_close(final_rewards.target_contact_only(env), torch.tensor([0.,0,1,1,0,0,0]))


@pytest.mark.parametrize('dt', [.01,.02,.04])
def test_bonus_is_exactly_75(configs, dt):
    term = configs('rewards','final').FinalGoalAccuracyV0_2_1RewardsCfg().target_goal_timeout_bonus
    env = NS(step_dt=dt, termination_manager=NS(get_term=lambda key:torch.tensor([True,False])))
    torch.testing.assert_close(term.func(env, **term.params)*term.weight*dt, torch.tensor([75.,0.]))


def test_timeout_boundary_uses_inclusive_parent_success(configs, monkeypatch):
    rewards = configs('rewards','final').FinalGoalAccuracyV0_2_1RewardsCfg()
    endings = configs('termination_functions','final')
    positions = torch.tensor([[.02-1e-6,0,0],[.02,0,0],[.02+1e-6,0,0]])
    monkeypatch.setattr(endings,'_object_position',lambda env:positions)
    monkeypatch.setattr(endings,'time_out',lambda env:env.expired)
    env = NS(step_dt=.02,expired=torch.zeros(3,dtype=torch.bool),
             command_manager=NS(get_command=lambda name:torch.zeros(3,3)),
             reward_manager=NS(get_term_cfg=lambda name:rewards.target_goal_hold))
    env.termination_manager=NS(get_term=lambda name:endings.success_at_timeout(env,'target_goal_hold'))
    term = rewards.target_goal_timeout_bonus
    assert not (term.func(env,**term.params)*term.weight*env.step_dt).any()
    env.expired[:]=True
    torch.testing.assert_close(term.func(env,**term.params)*term.weight*env.step_dt,torch.tensor([75.,75.,0.]))


def test_choice_distribution_reseed_and_empty():
    m = pool_module()
    chooser = m.TargetChoice(42,16)
    first = chooser.sample(300000)
    category = np.minimum(first,2)
    np.testing.assert_allclose(np.bincount(category)/len(first), np.ones(3)/3, atol=.004)
    np.testing.assert_allclose(np.bincount(first,minlength=16)[2:]/sum(category==2), np.ones(14)/14, atol=.004)
    assert np.any(first[1:]==first[:-1])
    chooser.reseed(42)
    assert chooser.sample(0).size == 0
    np.testing.assert_array_equal(first,chooser.sample(len(first)))


def test_collection_reads_writes_and_physics_follow_partial_selection():
    m = pool_module()
    n,p = 2,5  # target, two obstacles, two spare targets
    state = torch.arange(n*p*13,dtype=torch.float32).reshape(n,p,13)
    storage = {'masses':torch.arange(n*p,dtype=torch.float32)[:,None]+1,
               'inertias':torch.ones(n*p,9), 'material_properties':torch.ones(n*p,3,3),
               'disable_gravities':torch.zeros(n*p,1,dtype=torch.uint8)}
    class View:
        max_shapes=3
        def __getattr__(self,name):
            if name.startswith('get_'):return lambda:storage[name[4:]].clone()
            if name.startswith('set_'):
                def write(data,ids):storage[name[4:]][ids]=data[ids]
                return write
            raise AttributeError(name)
    class Physical:
        num_objects=p
        num_instances=n
        device='cpu'
        object_names=['mass_cylinder','obstacle_0','obstacle_1','candidate_1','candidate_2']
        data=NS(object_link_state_w=state, object_pos_w=state[...,:3])
        root_physx_view=View()
        def write_object_link_state_to_sim(self, values, env_ids, object_ids):
            self.data.object_link_state_w[env_ids[:,None],object_ids]=values
    physical=Physical()
    logical=m.LogicalTargetCollection(physical,3)
    logical.select(torch.tensor([1]),np.array([2]))
    torch.testing.assert_close(logical.data.object_pos_w[1,0],state[1,4,:3])
    torch.testing.assert_close(logical.data.object_pos_w[0,0],state[0,0,:3])
    assert logical.object_names==physical.object_names[:3]
    before=state.clone()
    values=torch.full((1,3,13),123.)
    logical.write_object_link_state_to_sim(values,env_ids=torch.tensor([1]))
    torch.testing.assert_close(state[0],before[0])
    torch.testing.assert_close(state[1,[4,1,2]],values[0])
    torch.testing.assert_close(state[1,0],before[1,0])
    masses=logical.root_physx_view.get_masses()
    assert masses[:,0].tolist()==[1,10,3,4,5,6]
    masses[1]=99
    logical.root_physx_view.set_masses(masses,torch.tensor([1]))
    assert storage['masses'][9]==99 and storage['masses'][1]==2
    sensors=[NS(data=NS(force_matrix_w=torch.full((2,1,2,3),float(i)))) for i in range(3)]
    sensor=m.SelectedTargetSensor(logical,sensors)
    assert sensor.data.force_matrix_w[:,0,0,0].tolist()==[0.,2.]
