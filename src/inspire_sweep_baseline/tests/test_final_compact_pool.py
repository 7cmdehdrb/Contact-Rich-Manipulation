import importlib
from types import SimpleNamespace as NS
import torch
import pytest


@pytest.fixture
def compact(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.final_compact_pool')


def physical(n):
    state = torch.arange(n*12*13, dtype=torch.float32).reshape(n,12,13)
    storage = dict(masses=torch.ones(n*12,1), inertias=torch.ones(n*12,9),
                   material_properties=torch.ones(n*12,2,3), disable_gravities=torch.zeros(n*12,1))
    class View:
        max_shapes = 2
        def __getattr__(self, name):
            if name.startswith('get_'):return lambda:storage[name[4:]].clone()
            if name.startswith('set_'):
                def put(data, rows):storage[name[4:]][rows]=data[rows]
                return put
            raise AttributeError(name)
    class Physical:
        num_instances, num_objects, device = n, 12, 'cpu'
        object_names = ['mass_cylinder']+[f'obstacle_{i}' for i in range(11)]
        root_physx_view=View()
        data=NS(object_link_state_w=state,object_pos_w=state[...,:3])
        def write_object_link_state_to_sim(self, value, env_ids):
            self.data.object_link_state_w[env_ids]=value
    return Physical()


def test_roles_distribution_and_uniqueness(compact):
    pool=compact.CompactRoleCollection(physical(30000))
    pool.select_roles(pool.env_rows, torch.Generator().manual_seed(32))
    category=pool.mapping[:,0].clamp_max(2)
    torch.testing.assert_close(category.bincount()/len(category), torch.full((3,),1/3), atol=.01,rtol=0)
    assert (pool.mapping[:,1:]>=2).all()
    assert (pool.mapping.sort(-1).values.diff(dim=-1)>0).all()
    assert pool.mapping.shape==(30000,10)


def test_partial_read_write_material_mapping(compact):
    raw=physical(3)
    pool=compact.CompactRoleCollection(raw)
    pool.select_roles(torch.tensor([1]),torch.Generator().manual_seed(49))
    before=raw.data.object_link_state_w.clone()
    mapping=pool.mapping[1]
    values=torch.full((1,10,13),-99.)
    pool.write_object_link_state_to_sim(values,env_ids=torch.tensor([1]))
    torch.testing.assert_close(raw.data.object_link_state_w[1,mapping],values[0])
    torch.testing.assert_close(raw.data.object_link_state_w[[0,2]],before[[0,2]])
    unused=torch.tensor([i for i in range(12) if i not in mapping])
    torch.testing.assert_close(raw.data.object_link_state_w[1,unused],before[1,unused])
    masses=pool.root_physx_view.get_masses()
    masses[1]=13.
    pool.root_physx_view.set_masses(masses,torch.tensor([1]))
    assert raw.root_physx_view.get_masses()[mapping[0]*3+1]==13


def test_sensor_roles_cache_and_reset(compact):
    pool=compact.CompactRoleCollection(physical(2))
    pool.select_roles(pool.env_rows,torch.Generator().manual_seed(8))
    sensors=[NS(data=NS(force_matrix_w=torch.full((2,1,2,3),float(i)))) for i in range(12)]
    bank=compact.RoleSensorBank(pool,sensors)
    sensor=compact.RoleSensor(bank,0)
    expected=pool.mapping[:,0].float()
    torch.testing.assert_close(sensor.data.force_matrix_w[:,0,0,0],expected)
    assert bank.forces() is bank.forces()
    pool.select_roles(pool.env_rows,torch.Generator().manual_seed(19))
    bank.invalidate()
    torch.testing.assert_close(sensor.data.force_matrix_w[:,0,0,0],pool.mapping[:,0].float())
