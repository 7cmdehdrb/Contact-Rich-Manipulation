"""Logical target slot over a fixed PhysX candidate pool, used only by v0_2_1.

Physical rows never change topology. MDP consumers see one selected target and
the original obstacle slots. Writes map through the same indices as reads.
"""
import numpy as np
import torch


class TargetChoice:
    """Independent reproducible category draw; repeats deliberately remain legal."""

    def __init__(self, seed, count):
        if count < 3:
            raise ValueError('Expected cylinder, cube and random-prop candidates')
        self.count = count
        self.reseed(seed)

    def reseed(self, seed):
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), 2121]))

    def sample(self, count):
        category = self.rng.integers(0, 3, size=count)
        props = self.rng.integers(2, self.count, size=count)
        return np.where(category == 2, props, category)


class LogicalData:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        if name == 'object_names':
            return self.owner.object_names
        value = getattr(self.owner.physical.data, name)
        if isinstance(value, torch.Tensor) and value.shape[:2] == (
                self.owner.num_instances, self.owner.physical.num_objects):
            return value[self.owner.env_rows[:, None], self.owner.mapping]
        return value


class LogicalPhysicsView:
    """Object-major material/mass/gravity interface for the existing event code."""

    _properties = ('masses', 'inertias', 'material_properties', 'disable_gravities')

    def __init__(self, owner):
        self.owner = owner
        self.raw = owner.physical.root_physx_view
        self.buffers = {key: getattr(self.raw, 'get_' + key)().cpu().clone()
                        for key in self._properties}

    @property
    def max_shapes(self):
        return self.raw.max_shapes

    @property
    def count(self):
        return self.owner.num_instances * self.owner.num_objects

    def rows(self):
        return (self.owner.mapping.cpu().T * self.owner.num_instances
                + torch.arange(self.owner.num_instances)).flatten()

    def __getattr__(self, name):
        if name.startswith('get_') and name[4:] in self._properties:
            return lambda: self.buffers[name[4:]][self.rows()].clone()
        if name.startswith('set_') and name[4:] in self._properties:
            def write(data, indices):
                indices = torch.as_tensor(indices, device='cpu', dtype=torch.long)
                rows = self.rows()[indices]
                buffer = self.buffers[name[4:]]
                buffer[rows] = data.cpu()[indices]
                getattr(self.raw, name)(buffer, rows)
            return write
        raise AttributeError(name)

    def wake_up(self, indices):
        self.raw.wake_up(self.rows()[torch.as_tensor(indices, device='cpu', dtype=torch.long)])


class LogicalTargetCollection:
    """Scene lifecycle delegates to physical storage; task state stays logical."""

    def __init__(self, physical, logical_count):
        self.physical = physical
        self.num_objects = logical_count
        self.num_instances = physical.num_instances
        self.object_names = physical.object_names[:logical_count]
        self.device = physical.device
        self.env_rows = torch.arange(self.num_instances, device=self.device)
        self.mapping = torch.arange(logical_count, device=self.device).repeat(self.num_instances, 1)
        self.pool_ids = torch.tensor([0, *range(logical_count, physical.num_objects)], device=self.device)
        self.selected = torch.zeros(self.num_instances, device=self.device, dtype=torch.long)
        self.data = LogicalData(self)
        self.root_physx_view = LogicalPhysicsView(self)
        self.default_pool_mass = self.root_physx_view.buffers['masses'].clone()
        self.default_pool_inertia = self.root_physx_view.buffers['inertias'].clone()

    def __getattr__(self, name):
        if name.startswith('write_object_') and name.endswith('_to_sim'):
            return lambda values, env_ids=None, object_ids=None: self._write(name, values, env_ids, object_ids)
        return getattr(self.physical, name)

    def _write(self, method, values, env_ids, object_ids):
        ids = self.env_rows if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        columns = torch.arange(self.num_objects, device=self.device)
        columns = columns if object_ids is None else columns[object_ids]
        obstacle_columns = columns[columns != 0]
        if obstacle_columns.numel():
            getattr(self.physical, method)(values[:, columns != 0], env_ids=ids, object_ids=obstacle_columns)
        if (columns == 0).any():
            target = values[:, columns == 0]
            for pool_index, physical_id in enumerate(self.pool_ids):
                mask = self.selected[ids] == pool_index
                if mask.any():
                    getattr(self.physical, method)(target[mask], env_ids=ids[mask], object_ids=physical_id.reshape(1))

    def select(self, env_ids, choices):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.selected[ids] = torch.as_tensor(choices, device=self.device, dtype=torch.long)
        self.mapping[ids, 0] = self.pool_ids[self.selected[ids]]

    def switch(self, ids, choices, origins):
        """Park unused bodies with zero velocity/no gravity and exclude them logically.

        GPU probes with actor-disable/live collider changes failed locally.
        Keep all actor and shape indices stable throughout training.
        """
        ids_cpu = ids.cpu().tolist()
        state = self.physical.data.default_object_state[ids[:, None], self.pool_ids].clone()
        state[..., :3] += origins[ids, None]
        self.physical.write_object_link_state_to_sim(state, env_ids=ids, object_ids=self.pool_ids)
        flags = self.root_physx_view.buffers['disable_gravities']
        physical_rows = (self.pool_ids.cpu()[:, None] * self.num_instances
                         + torch.as_tensor(ids_cpu)[None, :]).flatten()
        flags[physical_rows] = 1
        self.physical.root_physx_view.set_disable_gravities(flags, physical_rows)
        self.select(ids, choices)


class SelectedTargetSensor:
    """Keep the public mass_cylinder sensor name attached to the selected body."""

    def __init__(self, collection, sensors):
        self.collection = collection
        self.sensors = sensors

    @property
    def data(self):
        return self

    @property
    def force_matrix_w(self):
        matrices = torch.stack([s.data.force_matrix_w for s in self.sensors], dim=1)
        return matrices[self.collection.env_rows, self.collection.selected]

    def __getattr__(self, name):
        # The remaining pool sensors are updated/reset by the scene itself.
        return getattr(self.sensors[0], name)
