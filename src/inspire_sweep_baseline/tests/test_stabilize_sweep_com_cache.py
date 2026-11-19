"""Exercise fixed local COM caching through the installed Isaac data properties."""
import ast
import importlib
from collections.abc import Sequence
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import PROJECT_ROOT


def _data_class(relative_path, class_name, members, tensor_math):
    """Use the real lazy properties without constructing a PhysX simulation."""
    path = PROJECT_ROOT / 'IsaacLab/source/isaaclab/isaaclab/assets' / relative_path
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition.bases = []
    definition.body = [node for node in definition.body
                       if isinstance(node, ast.FunctionDef) and node.name in members]
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    scope = {'torch': torch, 'math_utils': tensor_math, 'Sequence': Sequence}
    exec(compile(module, str(path), 'exec'), scope)
    return scope[class_name]


@pytest.fixture
def com_environment(load_mdp, tensor_math):
    count, links, objects = 3, 2, 4
    robot_type = _data_class('articulation/articulation_data.py', 'ArticulationData',
                            {'body_com_pose_b'}, tensor_math)
    object_type = _data_class('rigid_object_collection/rigid_object_collection_data.py',
                             'RigidObjectCollectionData',
                             {'object_com_pose_b', '_reshape_view_to_data', 'object_com_pose_w',
                              'object_link_pose_w', 'object_link_pos_w', 'object_link_quat_w',
                              'object_com_pos_b', 'object_com_quat_b', 'object_com_pos_w'}, tensor_math)

    class View:
        def __init__(self, shape):
            self.values = torch.zeros(shape)
            self.values[..., :3] = torch.arange(self.values[..., :3].numel()).reshape(
                self.values[..., :3].shape) / 1000.
            self.values[..., 6] = 1.  # PhysX uses XYZW.
            self.transforms = self.values.clone()
            self.reads = 0

        def get_coms(self):
            self.reads += 1
            # PhysX refreshes the CPU output tensor on every getter call.
            return self.values.clone()

        def get_transforms(self):
            return self.transforms.clone()

        def set_transforms(self, values, indices):
            self.transforms[indices] = values[indices]

    robot, collection = robot_type(), object_type()
    robot_view, object_view = View((count, links, 7)), View((count * objects, 7))
    for data, view, field in ((robot, robot_view, '_body_com_pose_b'),
                              (collection, object_view, '_object_com_pose_b')):
        data.device = 'cpu'
        data._sim_timestamp = 1.
        data._root_physx_view = view
        setattr(data, field, NS(timestamp=-1., data=None))
    collection.num_instances, collection.num_objects = count, objects
    env = NS(scene={'robot': NS(data=robot), 'object_collection': NS(data=collection)},
             _refactor_physics_initialized=True)
    helper = importlib.import_module(
        '_baseline_cpu_pkg.mdp.common.stabilize_sweep_contact').cache_static_local_com
    return env, helper, robot_view, object_view


def test_cache_uses_final_startup_com_and_stays_valid_across_physics_steps(com_environment):
    env, cache, robot_view, object_view = com_environment
    robot, objects = env.scene['robot'].data, env.scene['object_collection'].data
    # A manager can have fetched COMs before startup updated inertia axes.
    robot.body_com_pose_b
    objects.object_com_pose_b
    robot_view.values[..., 0] += .01
    object_view.values[..., 1] -= .02
    cache(env)
    expected_robot = robot_view.values[..., [0, 1, 2, 6, 3, 4, 5]]
    expected_objects = objects._reshape_view_to_data(object_view.values[:, [0, 1, 2, 6, 3, 4, 5]])
    for timestamp in (2., 100., 10000.):
        robot._sim_timestamp = objects._sim_timestamp = timestamp
        torch.testing.assert_close(robot.body_com_pose_b, expected_robot, rtol=0., atol=0.)
        torch.testing.assert_close(objects.object_com_pose_b, expected_objects, rtol=0., atol=0.)
    assert (robot_view.reads, object_view.reads) == (2, 2)


def test_partial_reset_rearms_local_cache_without_freezing_world_motion(com_environment, tensor_math):
    env, cache, robot_view, object_view = com_environment
    robot, objects = env.scene['robot'].data, env.scene['object_collection'].data
    cache(env)
    saved = tuple(pose.clone() for pose in env._stabilize_static_local_com)
    # Isaac's joint-position writer invalidates this buffer even for one row.
    robot._body_com_pose_b.timestamp = -1.
    robot._sim_timestamp = objects._sim_timestamp = 2.
    cache(env)
    assert (robot_view.reads, object_view.reads) == (1, 1)
    assert robot._body_com_pose_b.timestamp == objects._object_com_pose_b.timestamp == float('inf')
    torch.testing.assert_close(robot.body_com_pose_b, saved[0], rtol=0., atol=0.)
    torch.testing.assert_close(objects.object_com_pose_b, saved[1], rtol=0., atol=0.)
    link_pos = torch.zeros_like(saved[1][..., :3])
    link_quat = torch.zeros_like(saved[1][..., 3:])
    link_quat[..., 0] = 1.
    before = link_pos + tensor_math.quat_apply(link_quat, objects.object_com_pose_b[..., :3])
    link_pos[1] += torch.tensor([.1, -.2, .3])
    after = link_pos + tensor_math.quat_apply(link_quat, objects.object_com_pose_b[..., :3])
    torch.testing.assert_close(after[0], before[0], rtol=0., atol=0.)
    torch.testing.assert_close(after[2], before[2], rtol=0., atol=0.)
    torch.testing.assert_close(after[1] - before[1], torch.tensor([.1, -.2, .3]).expand(4, -1))


def test_cache_requires_completed_startup_and_unopted_data_keeps_live_refresh(com_environment):
    env, cache, robot_view, object_view = com_environment
    env._refactor_physics_initialized = False
    with pytest.raises(RuntimeError, match='completed startup physics'):
        cache(env)
    assert (robot_view.reads, object_view.reads) == (0, 0)
    robot, objects = env.scene['robot'].data, env.scene['object_collection'].data
    for timestamp in (1., 2., 3.):
        robot._sim_timestamp = objects._sim_timestamp = timestamp
        robot.body_com_pose_b
        objects.object_com_pose_b
    assert (robot_view.reads, object_view.reads) == (3, 3)
    assert not hasattr(env, '_stabilize_static_local_com')


@pytest.mark.parametrize('cached', [False, True])
def test_same_timestamp_reset_world_com_refresh_matches_isaac_with_and_without_cache(
        com_environment, tensor_math, cached):
    """The native writer leaves world COM stale until the next physics update."""
    env, cache, _, view = com_environment
    data = env.scene['object_collection'].data
    if cached:
        cache(env)
    data._object_link_pose_w = NS(timestamp=-1., data=None)
    data._object_com_pose_w = NS(timestamp=-1., data=None)
    for name in ('_object_link_state_w', '_object_state_w', '_object_com_state_w'):
        setattr(data, name, NS(data=None))
    writer_type = _data_class('rigid_object_collection/rigid_object_collection.py',
                             'RigidObjectCollection',
                             {'write_object_link_pose_to_sim', 'reshape_data_to_view',
                              '_env_obj_ids_to_view_ids'}, tensor_math)
    writer = writer_type()
    writer._data = writer.data = data
    writer.root_physx_view = view
    writer.num_instances, writer.num_objects = data.num_instances, data.num_objects
    writer._ALL_ENV_INDICES = torch.arange(data.num_instances)
    writer._ALL_OBJ_INDICES = torch.arange(data.num_objects)
    before = data.object_com_pos_w.clone()
    changed = data.object_link_pose_w[1:2].clone()
    changed[..., :3] += torch.tensor([.1, -.2, .3])
    writer.write_object_link_pose_to_sim(changed, env_ids=torch.tensor([1]))
    if cached:
        cache(env)
    # Both paths preserve the installed Isaac writer's same-timestamp cache.
    torch.testing.assert_close(data.object_com_pos_w, before, rtol=0., atol=0.)
    expected = data.object_link_pos_w + tensor_math.quat_apply(
        data.object_link_quat_w, data.object_com_pos_b)
    assert not torch.allclose(expected[1], before[1])
    reads = view.reads
    data._sim_timestamp += .01
    torch.testing.assert_close(data.object_com_pos_w, expected, rtol=0., atol=0.)
    assert view.reads == reads + (0 if cached else 1)


def _reset_writer(name):
    path = PROJECT_ROOT / 'src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/event_functions/stabilize_sweep.py'
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope[name]


@pytest.mark.parametrize('warm_combined', [False, True])
@pytest.mark.parametrize('first_reset', [False, True])
def test_sparse_test_object_writer_matches_real_isaac_writer(
        load_mdp, tensor_math, warm_combined, first_reset):
    """Native equality, arbitrary pairs, every cache, partial rows, and RNG."""
    count, total = 3, 4
    relative = 'rigid_object_collection/rigid_object_collection_data.py'
    source = ast.parse((PROJECT_ROOT / 'IsaacLab/source/isaaclab/isaaclab/assets' / relative).read_text())
    members = {node.name for cls in source.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef) and node.name != '__init__'}
    data_type = _data_class(relative, 'RigidObjectCollectionData', members, tensor_math)
    writer_type = _data_class('rigid_object_collection/rigid_object_collection.py', 'RigidObjectCollection', {
        'write_object_link_state_to_sim', 'write_object_link_pose_to_sim', 'write_object_link_velocity_to_sim',
        'write_object_com_velocity_to_sim', 'reshape_data_to_view', 'reshape_view_to_data',
        '_env_obj_ids_to_view_ids'}, tensor_math)

    class View:
        def __init__(self):
            self.pose = torch.zeros(count * total, 7)
            self.pose[:, 0] = torch.arange(count * total) / 10.
            self.pose[:, 6] = 1.
            self.velocity = torch.zeros(count * total, 6)
            self.com = self.pose.clone()
            self.com[:, :3] = .001
            self.calls = []

        def get_transforms(self): return self.pose.clone()
        def get_velocities(self): return self.velocity.clone()
        def get_coms(self): return self.com.clone()
        def get_accelerations(self): return torch.ones(count * total, 6)
        def set_transforms(self, data, indices):
            self.calls.append(('pose', indices.clone()))
            self.pose[indices] = data[indices]
        def set_velocities(self, data, indices):
            self.calls.append(('velocity', indices.clone()))
            self.velocity[indices] = data[indices]

    def make():
        data, view, writer = data_type(), View(), writer_type()
        data.device, data._sim_timestamp = 'cpu', 1.
        data.num_instances, data.num_objects, data._root_physx_view = count, total, view
        fields = ('object_link_pose_w', 'object_link_vel_w', 'object_com_pose_b', 'object_com_pose_w',
                  'object_com_vel_w', 'object_com_acc_w', 'object_state_w', 'object_link_state_w',
                  'object_com_state_w')
        for name in fields:
            setattr(data, '_' + name, NS(timestamp=-1., data=None))
        writer.data = writer._data = data
        writer.root_physx_view = view
        writer.num_instances, writer.num_objects = count, total
        writer._ALL_ENV_INDICES, writer._ALL_OBJ_INDICES = torch.arange(count), torch.arange(total)
        if warm_combined:
            data.object_state_w
            data.object_link_state_w
            data.object_com_state_w
        active = torch.zeros(count, total, dtype=torch.bool)
        active[:, 0] = True
        active[0, 1] = True
        return NS(scene={'object_collection': writer}, num_envs=count,
                  _active_object_mask=active,
                  _revision_layout_complete=torch.full((count,), not first_reset)), fields

    baseline, fields = make()
    optimized, _ = make()
    ids = torch.tensor([2, 0])
    old = baseline.scene['object_collection']
    new = optimized.scene['object_collection']
    state = old.data.object_link_state_w[ids].clone()
    # Also initialize the exact same lazy cache on the optimized instance.
    new.data.object_link_state_w
    active = optimized._active_object_mask[ids].clone()
    active[1, 1] = False              # disappearing body always written
    active[0, 2] = True               # reactivated body always written
    state[:, 0, 1] += .02            # live target always written
    state[0, 3, 0] += 1e-7           # any native mismatch must write
    new.root_physx_view.velocity[1 * count + 2, 0] = 1e-7
    old.root_physx_view.velocity[1 * count + 2, 0] = 1e-7
    before_pose = new.root_physx_view.pose.clone()
    before_rng = torch.random.get_rng_state().clone()
    old.write_object_link_state_to_sim(state, env_ids=ids)
    written = _reset_writer('write_test_reset_object_state')(optimized, ids, state, active)
    torch.testing.assert_close(torch.random.get_rng_state(), before_rng, rtol=0., atol=0.)
    expected = torch.ones(2, total, dtype=torch.bool)
    if not first_reset:
        expected[1, 2:] = False
    torch.testing.assert_close(written, expected)
    expected_rows = (torch.arange(total)[:, None] * count + ids).flatten()[expected.T.flatten()]
    for _, rows in new.root_physx_view.calls:
        torch.testing.assert_close(rows, expected_rows)
    torch.testing.assert_close(new.root_physx_view.pose, old.root_physx_view.pose, rtol=0., atol=0.)
    torch.testing.assert_close(new.root_physx_view.velocity, old.root_physx_view.velocity, rtol=0., atol=0.)
    torch.testing.assert_close(new.root_physx_view.pose[torch.arange(total) * count + 1],
                               before_pose[torch.arange(total) * count + 1], rtol=0., atol=0.)
    for name in fields:
        a, b = getattr(old.data, '_' + name), getattr(new.data, '_' + name)
        assert a.timestamp == b.timestamp, name
        assert (a.data is None) == (b.data is None), name
        if a.data is not None:
            torch.testing.assert_close(b.data, a.data, rtol=0., atol=0., msg=name)
    # Later physics invalidates normal world caches, never the native roster.
    old.data._sim_timestamp += .01
    new.data._sim_timestamp += .01
    for name in ('object_state_w', 'object_link_state_w', 'object_com_state_w'):
        torch.testing.assert_close(getattr(new.data, name), getattr(old.data, name), rtol=0., atol=0.)


@pytest.mark.parametrize('first_reset', [False, True])
def test_sparse_fixed_root_writer_preserves_real_isaac_caches(load_mdp, tensor_math, first_reset):
    writer_type = _data_class('articulation/articulation.py', 'Articulation', {
        'write_root_state_to_sim', 'write_root_link_pose_to_sim', 'write_root_com_velocity_to_sim'}, tensor_math)

    def make():
        writer = writer_type()
        pose = torch.zeros(3, 7)
        pose[:, 0] = torch.arange(3)
        pose[:, 3] = 1.
        velocity = torch.zeros(3, 6)
        data = NS(root_link_pose_w=pose, root_com_vel_w=velocity, body_acc_w=torch.ones(3, 2, 6))
        for name in ('root_link_state_w', 'root_state_w', 'root_com_state_w'):
            value = torch.cat((pose, velocity), -1)
            setattr(data, name, value)
            setattr(data, '_' + name, NS(data=value, timestamp=1.))
        for name in ('_body_link_pose_w', '_body_com_pose_w', '_body_state_w',
                     '_body_link_state_w', '_body_com_state_w'):
            setattr(data, name, NS(timestamp=1.))
        native_pose = pose[:, [0, 1, 2, 4, 5, 6, 3]].clone()
        native_velocity = velocity.clone()
        calls = []
        def set_pose(values, indices):
            calls.append(('pose', indices.clone()))
            native_pose[indices] = values[indices]
        def set_velocity(values, indices):
            calls.append(('velocity', indices.clone()))
            native_velocity[indices] = values[indices]
        writer.data = writer._data = data
        writer.is_fixed_base = True
        writer.root_physx_view = NS(get_root_transforms=lambda: native_pose.clone(),
            get_root_velocities=lambda: native_velocity.clone(), set_root_transforms=set_pose,
            set_root_velocities=set_velocity)
        env = NS(scene={'robot': writer}, _revision_layout_complete=torch.full((3,), not first_reset))
        return env, calls, native_pose, native_velocity

    baseline, _, _, _ = make()
    optimized, calls, native_pose, native_velocity = make()
    old, new = baseline.scene['robot'], optimized.scene['robot']
    ids = torch.tensor([2, 0])
    state = old.data.root_state_w[ids].clone()
    state[1, 1] += .01
    old.write_root_state_to_sim(state, env_ids=ids)
    _reset_writer('write_test_reset_root_state')(optimized, ids, state)
    expected = ids if first_reset else ids[1:]
    assert len(calls) == 2
    for _, written in calls:
        torch.testing.assert_close(written, expected)
    for key, value in vars(old.data).items():
        actual = getattr(new.data, key)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value, rtol=0., atol=0., msg=key)
        else:
            assert actual.timestamp == value.timestamp
    torch.testing.assert_close(native_pose[:, [0, 1, 2, 6, 3, 4, 5]], new.data.root_link_pose_w, rtol=0., atol=0.)
    torch.testing.assert_close(native_velocity, new.data.root_com_vel_w, rtol=0., atol=0.)
