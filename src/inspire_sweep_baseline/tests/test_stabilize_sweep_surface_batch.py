"""Differential tests against the retained visible-mesh reference observer."""
import importlib
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.fixture
def surface(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_surface')


@pytest.fixture(params=['cpu', 'gpu'])
def components_backend(request):
    # Both algorithms execute on Warp CPU here; CUDA is an integration check.
    return request.param


@pytest.fixture(params=['bvh', 'raster'])
def ray_backend(request):
    return request.param


def make_backend(surface, monkeypatch, prototypes, kinds, scales, chunk_size=3, device='cpu', components_backend='cpu', ray_backend='bvh', bounds_backend='numpy'):
    names = ('mass_cylinder', *(f'obstacle_{i}' for i in range(kinds.shape[1] - 1)))
    paths = [f'/env_{row}' for row in range(len(kinds))]
    shelf = surface.box_mesh((0., 0., -.08), (1., .8, .1))
    reads = []

    def local(self, path):
        reads.append(path)
        prefix, name = path.rsplit('/', 1)
        if name == 'Shelf':
            return shelf
        row, source = paths.index(prefix), names.index(name)
        vertices, faces = prototypes[kinds[row, source]]
        return vertices * scales[row, source], faces

    monkeypatch.setitem(sys.modules, 'isaaclab.sim.utils', NS(get_current_stage=lambda: object()))
    monkeypatch.setattr(surface.SurfaceGeometry, 'local', local)
    env = NS(device=device, num_envs=len(kinds), _revision_object_kinds=kinds,
             _final_object_scales=scales, scene=NS(env_prim_paths=paths),
             cfg=NS(scene=NS(object_collection=NS(rigid_objects=dict.fromkeys(names)))))
    backend = surface.BatchedSurfaceGeometry(env, chunk_size=chunk_size, components_backend=components_backend, ray_backend=ray_backend, bounds_backend=bounds_backend)
    return backend, shelf, reads


def reference(surface, prototypes, shelf, kinds, scales, local, yaw, active, device='cpu'):
    vertices, faces, owners = [], [], []
    offset = 0
    for source in [*np.flatnonzero(active), -1]:
        if source == -1:
            v, f = shelf
        else:
            v, f = prototypes[kinds[source]]
            v = v * scales[source]
            c, s = np.cos(yaw[source]), np.sin(yaw[source])
            rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            v = v @ rotation.T + local[source]
        vertices.append(v)
        faces.append(f + offset)
        owners.append(np.full(len(f), source, dtype=np.int32))
        offset += len(v)
    return surface.observe_mesh(np.concatenate(vertices), np.concatenate(faces),
                                np.concatenate(owners), device=device)


def assert_observation_equal(actual, expected):
    target, obstacles, timing = actual
    expected_target, expected_obstacles, expected_timing = expected
    assert (target is None) == (expected_target is None)
    if target is not None:
        np.testing.assert_allclose(target, expected_target, atol=2e-6, rtol=0.)
    np.testing.assert_allclose(obstacles, expected_obstacles, atol=2e-6, rtol=0.)
    assert timing['rays'] == expected_timing['rays']
    assert all(timing[key] >= 0. for key in ('total_ms', 'upload_ms', 'readback_ms', 'ray_ms', 'components_ms'))


@pytest.mark.parametrize('bounds_backend', ['numpy', 'cpu_kernel'])
def test_batch_preserves_random_scales_yaws_row_order_and_shared_geometry(surface, monkeypatch, components_backend, ray_backend, bounds_backend):
    rng = np.random.default_rng(42)
    prototypes = {}
    for kind in range(8):
        vertices, faces = surface.box_mesh((0., 0., .05),
                                          (.045 + kind * .005, .045 + kind * .007, .08 + kind * .004))
        # Sloped visible faces exercise the actual mesh instead of an OBB hit.
        vertices[:, 0] += vertices[:, 2] * .17
        prototypes[kind] = vertices, faces
    kinds = np.tile(np.arange(17) % 8, (12, 1))
    scales = rng.uniform(.9, 1.1, size=kinds.shape)
    backend, shelf, reads = make_backend(surface, monkeypatch, prototypes, kinds, scales, components_backend=components_backend, ray_backend=ray_backend, bounds_backend=bounds_backend)
    ids = np.array([9, 1, 5, 0, 11, 3, 8])
    local = np.zeros((len(ids), 17, 3), dtype=np.float32)
    local[:, :, :2] = rng.uniform(-.16, .16, size=(*local.shape[:2], 2))
    yaw = rng.uniform(-np.pi, np.pi, size=local.shape[:2]).astype(np.float32)
    active = np.zeros(local.shape[:2], dtype=bool)
    for row in range(len(ids)):
        active[row, :row + 1] = True
    actual = backend.observe_batch(ids, local, yaw, active)
    assert len(reads) == 9  # Eight reference shapes, one shelf; never per env.
    for row, env_id in enumerate(ids):
        expected = reference(surface, prototypes, shelf, kinds[env_id], scales[env_id],
                             local[row], yaw[row], active[row])
        assert_observation_equal(actual[row], expected)
    again = backend.observe_batch(ids[::-1], local[::-1], yaw[::-1], active[::-1])
    assert len(reads) == 9
    for original, reordered in zip(actual, again[::-1]):
        assert_observation_equal(reordered, original)


def test_batch_keeps_disconnected_obstacles_and_largest_target_fragment(surface, monkeypatch, components_backend, ray_backend):
    prototypes = {
        0: surface.box_mesh((0., 0., 0.), (.1, .12, .1)),
        1: surface.box_mesh((0., 0., 0.), (.02, .02, .12)),
        2: surface.box_mesh((0., 0., 0.), (.1, .06, .1)),
    }
    kinds = np.tile(np.array([0, 1, 2, 0, 1]), (2, 1))
    scales = np.ones_like(kinds, dtype=float)
    backend, shelf, _ = make_backend(surface, monkeypatch, prototypes, kinds, scales, components_backend=components_backend, ray_backend=ray_backend)
    # Keep the shelf below the fixture's negative-Z geometry.
    backend.shelf_mesh.points.assign(np.asarray(shelf[0], dtype=np.float32) - np.array([0., 0., .1], dtype=np.float32))
    backend.shelf_mesh.refit()
    shelf = shelf[0] - np.array([0., 0., .1]), shelf[1]
    local = np.zeros((2, 5, 3), dtype=np.float32)
    local[:, 1] = (.12, .01, 0.)
    local[:, 2, 1] = .2
    local[1, 0, 1] = .2
    local[1, 2, 1] = .31
    local[1, 3, 1] = 0.
    local[1, 4] = (.12, 0., 0.)
    yaw = np.zeros((2, 5), dtype=np.float32)
    active = np.array([[True, True, True, False, False], [True, False, True, True, True]])
    actual = backend.observe_batch(np.arange(2), local, yaw, active)
    for row in range(2):
        expected = reference(surface, prototypes, shelf, kinds[row], scales[row], local[row], yaw[row], active[row])
        assert_observation_equal(actual[row], expected)
    assert actual[0][0][1] < 0.  # Largest of the two target fragments.
    rear = actual[1][1][np.isclose(actual[1][1][:, 0], .05)]
    assert (rear[:, 1] < -.01).any() and (rear[:, 1] > .01).any()


@pytest.mark.parametrize('target_width, occluder_y, occluder_width, present', [
    (.018, None, None, False),
    (.022, None, None, True),
    (.2, .015, .17, False),
    (.2, .025, .15, True),
])
def test_batch_preserves_width_exposure_and_target_missing(surface, monkeypatch,
                                                          target_width, occluder_y, occluder_width, present, components_backend, ray_backend):
    prototypes = {0: surface.box_mesh((0., 0., .05), (.1, target_width, .1)),
                  1: surface.box_mesh((0., 0., .06), (.02, occluder_width or .02, .12))}
    kinds = np.array([[0, 1]], dtype=np.int32)
    scales = np.ones((1, 2))
    backend, shelf, _ = make_backend(surface, monkeypatch, prototypes, kinds, scales, components_backend=components_backend, ray_backend=ray_backend)
    local = np.zeros((1, 2, 3), dtype=np.float32)
    local[0, 1] = (.12, occluder_y or 0., 0.)
    yaw = np.zeros((1, 2), dtype=np.float32)
    active = np.array([[True, occluder_y is not None]])
    actual = backend.observe_batch([0], local, yaw, active)[0]
    expected = reference(surface, prototypes, shelf, kinds[0], scales[0], local[0], yaw[0], active[0])
    assert_observation_equal(actual, expected)
    assert (actual[0] is not None) == present


def test_batch_retains_row_major_pixel_tie_and_empty_batch(surface, monkeypatch, components_backend, ray_backend):
    prototypes = {0: surface.box_mesh((0., 0., .05), (.1, .06, .1))}
    kinds = np.zeros((1, 1), dtype=np.int32)
    scales = np.ones((1, 1))
    backend, shelf, _ = make_backend(surface, monkeypatch, prototypes, kinds, scales, components_backend=components_backend, ray_backend=ray_backend)
    local = np.zeros((1, 1, 3), dtype=np.float32)
    yaw = np.zeros((1, 1), dtype=np.float32)
    active = np.ones((1, 1), dtype=bool)
    actual = backend.observe_batch([0], local, yaw, active)[0]
    expected = reference(surface, prototypes, shelf, kinds[0], scales[0], local[0], yaw[0], active[0])
    assert_observation_equal(actual, expected)
    assert backend.observe_batch([], local[:0], yaw[:0], active[:0]) == []


def test_batch_reports_all_overflow_fragments_for_target_only_retry(surface, monkeypatch, components_backend, ray_backend):
    prototypes = {0: surface.box_mesh((0., 0., .05), (.1, .06, .1)),
                  1: surface.box_mesh((0., 0., .05), (.1, .12, .1)),
                  2: surface.box_mesh((0., 0., .06), (.02, .022, .12)),
                  3: surface.box_mesh((0., 0., .05), (.1, .04, .1))}
    kinds = np.array([[0, 1, 2, *([3] * 14)]], dtype=np.int32)
    scales = np.ones((1, 17))
    backend, shelf, _ = make_backend(surface, monkeypatch, prototypes, kinds, scales,
                                    components_backend=components_backend, ray_backend=ray_backend)
    local = np.zeros((1, 17, 3), dtype=np.float32)
    local[0, 0, 1] = .2
    local[0, 2, 0] = .12
    local[0, 3:, 1] = .3 + .07 * np.arange(14)
    yaw = np.zeros((1, 17), dtype=np.float32)
    active = np.ones((1, 17), dtype=bool)
    actual = backend.observe_batch([0], local, yaw, active)[0]
    expected = reference(surface, prototypes, shelf, kinds[0], scales[0], local[0], yaw[0], active[0])
    assert len(actual[1]) == 17
    assert_observation_equal(actual, expected)


def test_device_centroid_selection_preserves_numpy_rounding_before_row_major_tie(surface):
    """A real 4-connected mask has two exact ties but unequal NumPy scores."""
    wp = surface.wp
    points = np.array([[0, 5], [0, 6], [0, 7], [0, 8], [1, 6], [1, 7], [2, 7], [2, 9], [2, 11],
                       [3, 7], [3, 8], [3, 9], [3, 11], [4, 6], [4, 7], [4, 9], [4, 10], [4, 11],
                       [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [6, 6], [6, 7], [6, 8],
                       [7, 6], [8, 6], [9, 6]])
    rr, cc = points.T
    score = (rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2
    rational = (rr * len(rr) - rr.sum()) ** 2 + (cc * len(cc) - cc.sum()) ** 2
    assert points[score.argmin()].tolist() == [4, 7]
    assert points[rational.argmin()].tolist() == [3, 8]
    parent = np.full(144, -1, dtype=np.int32)
    parent[rr * 12 + cc] = 0
    area = np.zeros(144, dtype=np.int32)
    area[0] = len(rr)
    row_sum, col_sum = np.zeros(144, dtype=np.int64), np.zeros(144, dtype=np.int64)
    row_sum[0], col_sum[0] = rr.sum(), cc.sum()
    arrays = [wp.array([0, 144], dtype=int, device='cpu'), wp.array([12], dtype=int, device='cpu'),
              wp.array(parent, dtype=int, device='cpu'), wp.array(area, dtype=int, device='cpu'),
              wp.array(row_sum, dtype=wp.uint64, device='cpu'), wp.array(col_sum, dtype=wp.uint64, device='cpu')]
    distance = wp.array(np.full(144, np.iinfo(np.int64).max), dtype=wp.int64, device='cpu')
    legacy_distance = wp.array(np.full(144, 1e300), dtype=wp.float64, device='cpu')
    winner = wp.array(np.full(144, np.iinfo(np.int32).max, dtype=np.int32), dtype=int, device='cpu')
    wp.launch(surface._minimum_component_distance, dim=144, inputs=[*arrays, distance], device='cpu')
    wp.launch(surface._minimum_legacy_distance, dim=144, inputs=[*arrays, distance, legacy_distance], device='cpu')
    wp.launch(surface._pick_component_pixel, dim=144, inputs=[*arrays, distance, legacy_distance, winner], device='cpu')
    assert int(winner.numpy()[0]) == 4 * 12 + 7


def test_gpu_component_width_uses_reference_float64_boundary(surface):
    """Warp 1.8 must not narrow spacing to float32 before its float64 cast."""
    wp = surface.wp
    full_width = np.nextafter(np.float32(.1), np.float32(np.inf))
    threshold = max(.02, .2 * float(full_width))
    assert 20 * .001 + 1e-9 < threshold
    area = wp.array([20], dtype=int, device='cpu')
    wp.launch(surface._filter_components, dim=1, inputs=[
        wp.array([[0., .1]], dtype=wp.vec2, device='cpu'),
        wp.array([0, 1], dtype=int, device='cpu'),
        wp.array([[threshold]], dtype=wp.float64, device='cpu'), area,
        wp.array([0], dtype=int, device='cpu'),
        wp.array([19], dtype=int, device='cpu')], device='cpu')
    np.testing.assert_array_equal(area.numpy(), [0])


def test_device_workspace_reuse_clears_previous_components(surface, monkeypatch, ray_backend):
    prototypes = {0: surface.box_mesh((0., 0., .05), (.1, .06, .1))}
    kinds = np.zeros((4, 2), dtype=np.int32)
    scales = np.ones((4, 2))
    backend, shelf, _ = make_backend(surface, monkeypatch, prototypes, kinds, scales,
                                    components_backend='gpu', ray_backend=ray_backend)
    for ids, separate in (([3, 1, 0, 2], True), ([1], False), ([2, 0], True), ([0], False)):
        local = np.zeros((len(ids), 2, 3), dtype=np.float32)
        local[:, 1, 1] = .17
        yaw = np.zeros((len(ids), 2), dtype=np.float32)
        active = np.ones((len(ids), 2), dtype=bool)
        active[:, 1] = separate
        actual = backend.observe_batch(ids, local, yaw, active)
        for row, env_id in enumerate(ids):
            expected = reference(surface, prototypes, shelf, kinds[env_id], scales[env_id],
                                 local[row], yaw[row], active[row])
            assert_observation_equal(actual[row], expected)


def test_raster_preserves_pixel_owners_depth_shared_edges_and_nearest_source(surface, monkeypatch):
    """Compare every pixel, including equal-depth objects and degenerate faces."""
    vertices = np.array([[.08, -.0305, .0005], [.08, .0305, .0005],
                         [.08, .0305, .0405], [.08, -.0305, .0405]])
    # The diagonal is shared by the two valid triangles; the third has no area.
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 0, 1]], dtype=np.int32)
    prototypes = {0: (vertices, faces)}
    kinds = np.zeros((4, 2), dtype=np.int32)
    scales = np.ones((4, 2))
    scales[2] = [.91, 1.07]
    local = np.zeros((4, 2, 3), dtype=np.float32)
    local[1, 1] = (.01, .02, .02)  # The second source wins only overlapping rays.
    local[2, 1] = (.015, .01, .005)
    local[3, :, 2] = -.05  # Shelf occlusion must still override an object hit.
    yaw = np.zeros((4, 2), dtype=np.float32)
    yaw[2] = [.37, -1.21]
    active = np.ones((4, 2), dtype=bool)
    captures = []
    observations = []
    for ray_backend in ('bvh', 'raster'):
        backend, _, _ = make_backend(surface, monkeypatch, prototypes, kinds, scales,
                                     chunk_size=4, components_backend='gpu', ray_backend=ray_backend)
        component_method = backend._components_gpu
        captured = []

        def capture(pixels, offsets, dimensions, *args, **kwargs):
            captured.append((pixels.numpy().copy(), offsets.copy(), dimensions.copy()))
            return component_method(pixels, offsets, dimensions, *args, **kwargs)

        monkeypatch.setattr(backend, '_components_gpu', capture)
        observations.append(backend.observe_batch(np.arange(4), local, yaw, active))
        assert len(captured) == 1
        captures.append(captured[0])
    expected, actual = captures
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[0][:, 0], expected[0][:, 0])
    np.testing.assert_allclose(actual[0][:, 1], expected[0][:, 1], rtol=0., atol=2e-6)
    first = actual[0][:actual[1][1], 0]
    assert (first == 0).any() and not (first == 1).any()
    second = actual[0][actual[1][1]:actual[1][2], 0]
    assert (second == 0).any() and (second == 1).any()
    fourth = actual[0][actual[1][3]:actual[1][4], 0]
    assert (fourth == -1).any()
    for one, two in zip(observations[0], observations[1]):
        assert_observation_equal(one, two)


def test_yaw_support_vertices_preserve_full_mesh_bounds_bitwise(surface):
    """XY-interior Z extremes and unreferenced X extrema must remain exact."""
    rng = np.random.default_rng(2026)
    vertices = rng.uniform(-.15, .15, (500, 3))
    vertices = np.vstack((vertices, [[0., 0., -.4], [0., 0., .5]]))
    all_vertices = np.vstack((vertices, [[.8, 0., 0.]]))
    selected = surface._support_vertices(vertices)
    assert len(selected) < len(vertices) // 2
    assert selected[:, 2].min() == -.4 and selected[:, 2].max() == .5
    full = NS(support={0: vertices}, all_support={0: all_vertices}, shelf_max_x=np.float32(.2))
    reduced = NS(support={0: selected}, all_support={0: surface._support_vertices(all_vertices)},
                 shelf_max_x=full.shelf_max_x)
    kinds = np.zeros((128, 3), dtype=np.int32)
    scales = rng.uniform(.9, 1.1, kinds.shape)
    local = rng.uniform(-.2, .2, (*kinds.shape, 3)).astype(np.float32)
    yaw = rng.uniform(-np.pi, np.pi, kinds.shape).astype(np.float32)
    yaw[:4, :] = np.array([0., np.pi / 2, np.pi, -np.pi / 2], dtype=np.float32)[:, None]
    rotation = np.stack((np.cos(yaw), np.sin(yaw)), -1)
    active = rng.random(kinds.shape) > .3
    active[:, 0] = True
    expected = surface.BatchedSurfaceGeometry._bounds(full, kinds, scales, local, rotation, active)
    actual = surface.BatchedSurfaceGeometry._bounds(reduced, kinds, scales, local, rotation, active)
    for one, two in zip(actual, expected):
        np.testing.assert_array_equal(one.view(np.uint32), two.view(np.uint32))
    cache = surface._CPUSurfaceBounds(reduced.support, reduced.all_support, reduced.shelf_max_x)
    for one, two in zip(cache.bounds(kinds, scales, local, rotation, active), expected):
        np.testing.assert_array_equal(one.view(np.uint32), two.view(np.uint32))
    # A line in XY cannot construct a 2D hull; its complete geometry is retained.
    line = np.array([[0., 0., -1.], [1., 0., 0.], [2., 0., 1.]])
    np.testing.assert_array_equal(surface._support_vertices(line), line)


def test_cpu_bounds_preserve_20480_scenes_bitwise(surface):
    """Eight shapes, ten yaw batches, arbitrary inactive rows and partial views."""
    rng = np.random.default_rng(120)
    support = {kind: surface._support_vertices(rng.uniform(-.2, .2, (80 + kind * 3, 3)))
               for kind in range(8)}
    all_support = {kind: surface._support_vertices(np.vstack((vertices, [[.7, .05, .1]])))
                   for kind, vertices in support.items()}
    geometry = NS(support=support, all_support=all_support, shelf_max_x=np.float32(.2))
    cache = surface._CPUSurfaceBounds(support, all_support, geometry.shelf_max_x)
    kinds = rng.integers(0, 8, (2048, 17), dtype=np.int32)
    scales = rng.uniform(.9, 1.1, kinds.shape)
    local = rng.uniform(-.2, .2, (*kinds.shape, 3)).astype(np.float32)
    active = rng.random(kinds.shape) > .6
    active[0] = False
    for _ in range(10):
        yaw = rng.uniform(-np.pi, np.pi, kinds.shape).astype(np.float32)
        rotation = np.stack((np.cos(yaw), np.sin(yaw)), -1)
        arguments = kinds, scales, local, rotation, active
        expected = surface.BatchedSurfaceGeometry._bounds(geometry, *arguments)
        for one, two in zip(cache.bounds(*arguments), expected):
            np.testing.assert_array_equal(one.view(np.uint32), two.view(np.uint32))
    # Non-contiguous subsets must not alias or change row order in CPU arrays.
    for subset in (slice(None, None, 3), slice(0, 0)):
        arguments = tuple(value[subset] for value in arguments)
        expected = surface.BatchedSurfaceGeometry._bounds(geometry, *arguments)
        for one, two in zip(cache.bounds(*arguments), expected):
            np.testing.assert_array_equal(one.view(np.uint32), two.view(np.uint32))


def test_component_integer_statistics_match_reference_for_fragmented_rows(surface):
    """Interleaved roots, empty pixels and a partial warp preserve integer sums."""
    wp = surface.wp
    offsets = np.array([0, 959, 1675], dtype=np.int32)
    widths = np.array([137, 179], dtype=np.int32)
    count = int(offsets[-1])
    rng = np.random.default_rng(1201)
    parent = np.arange(count, dtype=np.int32)
    for index in range(count):
        if index and index not in offsets and rng.random() < .87:
            parent[index] = parent[index - 1]
    parent[rng.random(count) < .15] = -1
    for root in np.unique(parent[parent >= 0]):
        parent[root] = root
    area = np.zeros(count, dtype=np.int32)
    sum_row = np.zeros(count, dtype=np.uint64)
    sum_col = np.zeros(count, dtype=np.uint64)
    min_col = np.full(count, 2147483647, dtype=np.int32)
    max_col = np.full(count, -1, dtype=np.int32)
    for index, root in enumerate(parent):
        if root < 0:
            continue
        image = np.searchsorted(offsets, index, side='right') - 1
        row, col = divmod(index - offsets[image], widths[image])
        area[root] += 1
        sum_row[root] += np.uint64(row)
        sum_col[root] += np.uint64(col)
        min_col[root] = min(min_col[root], col)
        max_col[root] = max(max_col[root], col)
    arguments = [wp.array(offsets, dtype=int, device='cpu'), wp.array(widths, dtype=int, device='cpu'),
                 wp.array(parent, dtype=int, device='cpu'), wp.zeros(count, dtype=int, device='cpu'),
                 wp.zeros(count, dtype=wp.uint64, device='cpu'), wp.zeros(count, dtype=wp.uint64, device='cpu'),
                 wp.array(np.full(count, 2147483647), dtype=int, device='cpu'),
                 wp.array(np.full(count, -1), dtype=int, device='cpu')]
    wp.launch(surface._reduce_components, dim=count, inputs=arguments, device='cpu')
    for actual, expected in zip(arguments[3:], (area, sum_row, sum_col, min_col, max_col)):
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize('side, alphabet', [(4, 2), (3, 3)])
def test_all_small_masks_keep_four_connectivity_roots_and_integer_statistics(surface, side, alphabet, device='cpu'):
    """All 65,536 binary 4x4 and 19,683 background/two-source 3x3 masks.

    SciPy labels a vertically stacked image with empty separator rows; this
    independently checks every small topology, including holes and diagonals.
    The optional device permits the same fixture in the standalone CUDA check.
    """
    wp = surface.wp
    size = side * side
    batch = alphabet ** size
    count = batch * size
    pattern = np.arange(batch, dtype=np.int64)
    owners = ((pattern[:, None] // (alphabet ** np.arange(size))) % alphabet - 1).astype(np.float32)
    pixels = np.column_stack((owners.ravel(), np.zeros(count, dtype=np.float32)))
    expected_parent = np.full(count, -1, dtype=np.int32)
    padded = np.full((batch, side + 1, side), -1, dtype=np.float32)
    padded[:, :side] = owners.reshape(batch, side, side)
    for source in range(alphabet - 1):
        labels, number = surface.ndimage.label(padded.reshape(-1, side) == source)
        labels = labels.reshape(batch, side + 1, side)[:, :side].reshape(-1)
        selected = np.flatnonzero(labels)
        first = np.full(number + 1, count, dtype=np.int32)
        np.minimum.at(first, labels[selected], selected)
        expected_parent[selected] = first[labels[selected]]
    indices = np.flatnonzero(expected_parent >= 0)
    roots = expected_parent[indices]
    rows, cols = (indices % size) // side, indices % side
    expected_area = np.bincount(roots, minlength=count).astype(np.int32)
    expected_row, expected_col = np.zeros(count, dtype=np.uint64), np.zeros(count, dtype=np.uint64)
    np.add.at(expected_row, roots, rows.astype(np.uint64))
    np.add.at(expected_col, roots, cols.astype(np.uint64))
    expected_min = np.full(count, 2147483647, dtype=np.int32)
    expected_max = np.full(count, -1, dtype=np.int32)
    np.minimum.at(expected_min, roots, cols)
    np.maximum.at(expected_max, roots, cols)
    root_indices = np.flatnonzero(expected_area)
    offsets = wp.array(np.arange(batch + 1, dtype=np.int32) * size, dtype=int, device=device)
    widths = wp.array(np.full(batch, side, dtype=np.int32), dtype=int, device=device)
    packed = wp.array(pixels, dtype=wp.vec2, device=device)
    parent = wp.empty(count, dtype=int, device=device)
    # Nonzero values catch stale data if an active pixel was not initialized.
    area = wp.array(np.full(count, 123, dtype=np.int32), dtype=int, device=device)
    row_sum = wp.array(np.full(count, 456, dtype=np.uint64), dtype=wp.uint64, device=device)
    col_sum = wp.array(np.full(count, 789, dtype=np.uint64), dtype=wp.uint64, device=device)
    minimum = wp.empty(count, dtype=int, device=device)
    maximum = wp.empty(count, dtype=int, device=device)
    distance = wp.empty(count, dtype=wp.int64, device=device)
    legacy = wp.empty(count, dtype=wp.float64, device=device)
    winner = wp.empty(count, dtype=int, device=device)
    wp.launch(surface._initialize_components, dim=count,
              inputs=[packed, parent, area, row_sum, col_sum, minimum, maximum, distance, legacy, winner], device=device)
    wp.launch(surface._union_components, dim=count, inputs=[packed, offsets, widths, parent], device=device)
    wp.launch(surface._reduce_components, dim=count,
              inputs=[offsets, widths, parent, area, row_sum, col_sum, minimum, maximum], device=device)
    np.testing.assert_array_equal(parent.numpy(), expected_parent)
    np.testing.assert_array_equal(area.numpy(), expected_area)
    np.testing.assert_array_equal(row_sum.numpy()[indices], expected_row[indices])
    np.testing.assert_array_equal(col_sum.numpy()[indices], expected_col[indices])
    np.testing.assert_array_equal(minimum.numpy()[root_indices], expected_min[root_indices])
    np.testing.assert_array_equal(maximum.numpy()[root_indices], expected_max[root_indices])
