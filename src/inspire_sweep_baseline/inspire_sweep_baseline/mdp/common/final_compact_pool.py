"""v0_2_2: twelve physical bodies, ten logical roles, no geometry churn on reset."""
import torch
from .final_target_pool import LogicalData, LogicalPhysicsView


class CompactPhysicsView(LogicalPhysicsView):
    def rows(self):
        return self.owner.physics_rows


class CompactRoleCollection:
    def __init__(self, physical):
        if physical.num_objects != 12:
            raise ValueError('Compact scene requires cylinder, cube and ten random props')
        self.physical = physical
        self.num_instances, self.num_objects = physical.num_instances, 10
        self.device = physical.device
        self.env_rows = torch.arange(self.num_instances, device=self.device)
        self.object_names = physical.object_names[:10]
        self.mapping = torch.arange(10, device=self.device).repeat(self.num_instances, 1)
        self.selected = torch.zeros(self.num_instances, dtype=torch.long, device=self.device)
        self.physics_rows = (self.mapping.cpu().T * self.num_instances + torch.arange(self.num_instances)).flatten()
        self.data = LogicalData(self)
        self.root_physx_view = CompactPhysicsView(self)
        self.original_mass = self.root_physx_view.buffers['masses'].clone()
        self.original_inertia = self.root_physx_view.buffers['inertias'].clone()

    def select_roles(self, ids, generator):
        n = len(ids)
        category = torch.randint(3, (n,), device=self.device, generator=generator)
        prop = torch.randint(2, 12, (n,), device=self.device, generator=generator)
        target = torch.where(category == 2, prop, category)
        self.selected[ids] = target
        props = torch.arange(2, 12, device=self.device)
        scores = torch.rand(n, 10, device=self.device, generator=generator)
        scores = scores.masked_fill(props[None] == target[:, None], 2.)
        obstacles = props[scores.argsort(-1)[:, :9]]
        self.mapping[ids] = torch.cat((target[:, None], obstacles), -1)
        self.physics_rows = (self.mapping.cpu().T * self.num_instances + torch.arange(self.num_instances)).flatten()

    def park(self, ids, origins):
        state = self.physical.data.default_object_state[ids].clone()
        state[..., :3] += origins[ids, None]
        self.physical.write_object_link_state_to_sim(state, env_ids=ids)
        flags = self.root_physx_view.buffers['disable_gravities']
        rows = (torch.arange(12)[:, None] * self.num_instances + ids.cpu()[None]).flatten()
        flags[rows] = 1
        self.physical.root_physx_view.set_disable_gravities(flags, rows)

    def _write(self, method, values, env_ids=None, object_ids=None):
        ids = self.env_rows if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        columns = torch.arange(10, device=self.device)
        columns = columns if object_ids is None else columns[object_ids]
        # Compose a full physical buffer, preserving the two unused parked rows.
        field = {'write_object_state_to_sim': 'object_state_w',
                 'write_object_link_state_to_sim': 'object_link_state_w',
                 'write_object_com_state_to_sim': 'object_com_state_w'}
        if method in field:
            result = getattr(self.physical.data, field[method])[ids].clone()
        elif 'pose' in method:
            field_name = 'object_com_state_w' if '_com_' in method else 'object_link_state_w'
            result = getattr(self.physical.data, field_name)[ids, :, :7].clone()
        elif 'velocity' in method:
            field_name = 'object_link_state_w' if '_link_' in method else 'object_com_state_w'
            result = getattr(self.physical.data, field_name)[ids, :, 7:].clone()
        else:
            raise AttributeError(method)
        result[torch.arange(len(ids), device=self.device)[:, None], self.mapping[ids][:, columns]] = values
        getattr(self.physical, method)(result, env_ids=ids)

    def __getattr__(self, name):
        if name.startswith('write_object_') and name.endswith('_to_sim'):
            return lambda values, env_ids=None, object_ids=None: self._write(name, values, env_ids, object_ids)
        return getattr(self.physical, name)


class RoleSensorBank:
    def __init__(self, collection, sensors):
        self.collection, self.sensors, self.cache = collection, sensors, None

    def invalidate(self):
        self.cache = None

    def forces(self):
        if self.cache is None:
            physical = torch.stack([s.data.force_matrix_w for s in self.sensors], dim=1)
            self.cache = physical[self.collection.env_rows[:, None], self.collection.mapping]
        return self.cache


class RoleSensor:
    def __init__(self, bank, column):
        self.bank, self.column = bank, column

    @property
    def data(self):
        return self

    @property
    def force_matrix_w(self):
        return self.bank.forces()[:, self.column]

    def update(self, *args, **kwargs):
        self.bank.invalidate()
        return self.bank.sensors[self.column].update(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.bank.invalidate()
        return self.bank.sensors[self.column].reset(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.bank.sensors[self.column], name)
