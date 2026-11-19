"""One layout pass: preserve successful placements and skip only blocked candidates."""
import math
from types import SimpleNamespace
import torch
import warp as wp
from .stabilize_sweep_geometry import corners


@wp.kernel(enable_backward=False, module='unique')
def _red_distance_kernel(points: wp.array2d(dtype=float), start: wp.array2d(dtype=float),  # type: ignore[valid-type]
                         goal: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):  # type: ignore[valid-type]
    row, point = wp.tid()
    sx, sy = float(start[row, 0]), float(start[row, 1])
    gx, gy = float(goal[row, 0]), float(goal[row, 1])
    extension = wp.max(0.2 - wp.min(sx, gx), 0.0)
    px, py = points[point, 0], points[point, 1]
    minimum = float(wp.inf)
    positive = bool(True)
    negative = bool(True)
    for edge_index in range(4):
        ax, ay, bx, by = sx, sy, gx, gy
        if edge_index == 1:
            ax, ay, bx, by = gx, gy, gx + extension, gy + 0.0
        elif edge_index == 2:
            ax, ay, bx, by = gx + extension, gy + 0.0, sx + extension, sy + 0.0
        elif edge_index == 3:
            ax, ay, bx, by = sx + extension, sy + 0.0, sx, sy
        ex, ey = bx - ax, by - ay
        dx, dy = px - ax, py - ay
        # Explicit two-term reduction, division, clamp, multiply and subtraction
        # mirror individual eager Torch operations; FMA is disabled below.
        numerator = dx * ex + dy * ey
        denominator = wp.max(ex * ex + ey * ey, 1.0e-20)
        fraction = numerator / denominator
        fraction = wp.clamp(fraction, 0.0, 1.0)
        rx, ry = dx - fraction * ex, dy - fraction * ey
        minimum = wp.min(minimum, rx * rx + ry * ry)
        cross = ex * dy - ey * dx
        positive = positive and cross >= 0.0
        negative = negative and cross <= 0.0
    area = wp.abs((gy - sy) * extension)
    result = wp.sqrt(minimum)
    if (positive or negative) and area > 1.0e-12:
        result = 0.0
    output[row, point] = result


# Separate module options preserve the Torch operation boundaries without
# changing floating-point compilation flags of existing Warp kernels.
_red_distance_kernel.module.options['fuse_fp'] = False
_red_distance_kernel.module.mark_modified()


def _launch_red_distance(points, start, goal):
    """Execute the exact scalar kernel; CPU execution also supports regression tests."""
    output = torch.empty((len(start), len(points)), device=start.device, dtype=start.dtype)
    if not output.numel():
        return output
    device = wp.get_device(str(start.device))
    args = [wp.from_torch(points), wp.from_torch(start), wp.from_torch(goal), wp.from_torch(output)]
    if start.is_cuda:
        # Torch owns input/output storage. Using its current stream preserves
        # dependencies and allocator lifetimes without a device-wide wait.
        stream = wp.stream_from_torch(torch.cuda.current_stream(start.device))
        wp.launch(_red_distance_kernel, dim=output.shape, inputs=args, device=device, stream=stream)
    else:
        wp.launch(_red_distance_kernel, dim=output.shape, inputs=args, device=device)
    return output


def fused_test_red_distance(points, start, goal):
    """Stabilize-family CUDA acceleration; other inputs retain the Torch reference."""
    if (points.is_cuda and points.device == start.device == goal.device
            and points.dtype == start.dtype == goal.dtype == torch.float32):
        return _launch_red_distance(points, start, goal)
    from .wrench_refactor_layout import CircleResetLayout
    return CircleResetLayout.red_distance(points, start, goal)


@wp.func_native(r'''
const int row = task / 128;
#if defined(__CUDA_ARCH__)
const int lane = task % 128;
const int lanes = 128;
__shared__ bool valid[16000];
__shared__ int counts[128];
__shared__ float center_x[6], center_y[6];
__shared__ bool enabled[6];
__shared__ int chosen_lane, local_rank, chosen_cell, first_failed;
__shared__ bool complete;
#else
if (task % 128 != 0) return;
const int lane = 0;
const int lanes = 1;
bool valid[16000];
int counts[128];
float center_x[6], center_y[6];
bool enabled[6];
int chosen_lane, local_rank, chosen_cell, first_failed;
bool complete;
#endif
const int cells = points.shape[0];
const int span = (cells + lanes - 1) / lanes;
const int begin = lane * span;
const int end = wp::min(cells, begin + span);
if (lane == 0) { complete = true; first_failed = -1; }
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
for (int slot = 0; slot < 6; ++slot) {
    const float radius = wp::index(radii, row, slot);
    const float border = radius + cell_margin;
    const float red_limit = wp::index(target_radius, row) + radius + cell_margin;
    int number = 0;
    for (int cell = begin; cell < end; ++cell) {
        bool ok = wp::index(boundary, cell) >= border;
        ok = ok && wp::index(red, row, cell) >= red_limit;
        const float x = wp::index(points, cell, 0);
        const float y = wp::index(points, cell, 1);
        for (int previous = 0; previous < slot; ++previous) {
            if (enabled[previous]) {
                const float dx = x - center_x[previous];
                const float dy = y - center_y[previous];
                const float distance = dx * dx + dy * dy;
                const float limit = radius + wp::index(radii, row, previous) + cell_margin;
                ok = ok && distance >= limit * limit;
            }
        }
        valid[cell] = ok;
        number += static_cast<int>(ok);
    }
    counts[lane] = number;
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if (lane == 0) {
        int total = 0;
        for (int i = 0; i < lanes; ++i) total += counts[i];
        const bool needed = wp::index(requested, row) > slot;
        const bool ok = total > 0 || !needed;
        if (complete && !ok) first_failed = slot;
        complete = complete && ok;
        enabled[slot] = needed && complete;
        const int rank = static_cast<int>(wp::index(draws, row, slot, 0) * static_cast<float>(total));
        // searchsorted(right=True) returns cells when rank reaches total;
        // the original sampler clamps that result to the final cell.
        chosen_cell = cells - 1;
        chosen_lane = -1;
        local_rank = -1;
        int prefix = 0;
        for (int i = 0; i < lanes; ++i) {
            if (rank >= prefix && rank < prefix + counts[i]) {
                chosen_lane = i;
                local_rank = rank - prefix;
                break;
            }
            prefix += counts[i];
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if (lane == chosen_lane) {
        int rank = 0;
        for (int cell = begin; cell < end; ++cell) {
            if (valid[cell]) {
                if (rank == local_rank) { chosen_cell = cell; break; }
                ++rank;
            }
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if (lane == 0) {
        center_x[slot] = wp::index(points, chosen_cell, 0)
                        + (wp::index(draws, row, slot, 1) - 0.5f) * cell_size;
        center_y[slot] = wp::index(points, chosen_cell, 1)
                        + (wp::index(draws, row, slot, 2) - 0.5f) * cell_size;
        wp::index(indices, row, slot) = chosen_cell;
        wp::index(output_enabled, row, slot) = enabled[slot];
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
}
if (lane == 0) {
    wp::index(output_complete, row) = complete;
    wp::index(output_failed, row) = first_failed;
}
''')
def _select_circle_cells_native(task: int, points: wp.array2d(dtype=float), boundary: wp.array(dtype=float),  # type: ignore[valid-type]
                   red: wp.array2d(dtype=float), target_radius: wp.array(dtype=float),  # type: ignore[valid-type]
                   radii: wp.array2d(dtype=float), requested: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
                   draws: wp.array3d(dtype=float), cell_margin: float, cell_size: float,  # type: ignore[valid-type]
                   indices: wp.array2d(dtype=wp.int64), output_enabled: wp.array2d(dtype=bool),  # type: ignore[valid-type]
                   output_complete: wp.array(dtype=bool), output_failed: wp.array(dtype=wp.int64)):  # type: ignore[valid-type]
    ...


@wp.kernel(enable_backward=False, module='unique')
def _select_circle_cells_kernel(points: wp.array2d(dtype=float), boundary: wp.array(dtype=float),  # type: ignore[valid-type]
                   red: wp.array2d(dtype=float), target_radius: wp.array(dtype=float),  # type: ignore[valid-type]
                   radii: wp.array2d(dtype=float), requested: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
                   draws: wp.array3d(dtype=float), cell_margin: float, cell_size: float,  # type: ignore[valid-type]
                   indices: wp.array2d(dtype=wp.int64), output_enabled: wp.array2d(dtype=bool),  # type: ignore[valid-type]
                   output_complete: wp.array(dtype=bool), output_failed: wp.array(dtype=wp.int64)):  # type: ignore[valid-type]
    _select_circle_cells_native(wp.tid(), points, boundary, red, target_radius, radii, requested, draws,
                   cell_margin, cell_size, indices, output_enabled, output_complete, output_failed)


_select_circle_cells_kernel.module.options['fuse_fp'] = False
_select_circle_cells_kernel.module.mark_modified()


def _select_test_cells(points, boundary, red, target_radius, radii, requested, draws, cell_margin, cell_size):
    """Choose six slots with one CUDA block per row; CPU is for exact regression checks.

    Initial geometry, selected radii and all random draws come from the existing
    Torch sampler. The 16,000-cell limit bounds shared memory to about 17KB.
    """
    if points.shape[1] != 2 or not 0 < len(points) <= 16000:
        raise ValueError('Cooperative Test placement requires 1..16000 XY cells')
    assert red.shape == (len(radii), len(points))
    assert radii.shape == (len(requested),6) and draws.shape == (len(requested),6,4)
    assert target_radius.shape == requested.shape and requested.dtype == torch.int64
    assert all(value.dtype == torch.float32 for value in (points,boundary,red,target_radius,radii,draws))
    device = wp.get_device(str(points.device))
    indices = torch.empty((len(radii),6),device=points.device,dtype=torch.int64)
    enabled = torch.empty_like(indices,dtype=torch.bool)
    complete = torch.empty_like(requested,dtype=torch.bool)
    failed = torch.empty_like(requested)
    if len(radii):
        args = [wp.from_torch(value) for value in (points,boundary,red,target_radius,radii,requested,draws)]
        args += [cell_margin,cell_size]
        args += [wp.from_torch(value) for value in (indices,enabled,complete,failed)]
        # Every launched block is complete: all 128 lanes must reach barriers.
        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
        wp.launch(_select_circle_cells_kernel,dim=len(radii)*128,inputs=args,device=device,stream=stream,block_dim=128)
    return SimpleNamespace(indices=indices,enabled=enabled,complete=complete,failed_slot=failed)


def configure_test_layout_batch(env, batch_size=512):
    """Reduce launch and RED-distance overhead only for the Test circle sampler.

    Its random draws precede the batching loop, so row grouping changes neither
    the sampled layouts nor RNG consumption. Other environments retain the
    CircleResetLayout default of 64 rows and its original Torch distance.
    The instance overrides run only for float32 CUDA geometry. Cooperative
    cell selection additionally requires at most 16,000 cells; other inputs
    keep the original Torch selection and scratch buffers.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Layout batch size must be a positive integer')
    layout = env._fast_layout
    layout.sample_batch_size = batch_size
    if layout.bounds.is_cuda and layout.bounds.dtype == torch.float32:
        layout.red_distance = fused_test_red_distance
        if 0 < len(layout.cells) <= 16000:
            layout.cell_selector = _select_test_cells


def test_inactive_positions(env) -> torch.Tensor:
    """Cached [N,17,3] shelf-frame origins shared by IK and Test inactive bodies.

    A 45cm grid behind the fixed robot retains each slot's exact initial
    shelf-frame Z, reducing GPU broadphase reordering during activity changes.
    Active poses and random draws are unaffected. Existing inter-environment
    collision filtering remains required. The current robot's joint-path plus
    collision-box bound, including its F/T mount, is below 1.479m.
    """
    if not hasattr(env, '_test_inactive_positions'):
        from isaaclab.utils.math import quat_apply_inverse
        bounds = env._revision_bounds_t
        radius = bounds.abs().amax(-2).norm(dim=-1).amax()
        if bounds.shape[1:] != (17, 2, 3) or float(radius) * 2 + .002 >= .45:
            raise ValueError('Test parking requires seventeen bodies with radius below 22.4cm')
        if not env.cfg.scene.filter_collisions:
            raise ValueError('Test parking requires inter-environment collision filtering')
        root = env.scene['robot'].data.default_root_state
        if bool((3. - root[:, 0] - radius <= 1.479 + .1).any()):
            raise ValueError('Test parking overlaps the conservative robot reach bound')
        slot = torch.arange(16, device=bounds.device)
        relative = bounds.new_zeros(17, 3)
        relative[1:, 0] = 3. + .45 * (slot // 4)
        relative[1:, 1] = .45 * (slot % 4 - 1.5)
        relative[1:, 2] = 1.1
        shelf = env.scene['shelf'].data.default_root_state
        env._test_inactive_positions = quat_apply_inverse(
            shelf[:, None, 3:7].expand(-1, 17, -1), relative[None] - shelf[:, None, :3])
        # The shelf is horizontal and reset rotations are yaw-only: matching
        # the original slot Z also preserves its vertical collision bounds.
        env._test_inactive_positions[..., 2] = env._refactor_spawn_z
    return env._test_inactive_positions


@wp.func_native(r'''
const int row = task / 128;
#if defined(__CUDA_ARCH__)
const int lane = task % 128;
const int lanes = 128;
__shared__ bool valid[16000];
__shared__ int counts[128];
__shared__ float center_x[6], center_y[6];
__shared__ bool enabled[6];
__shared__ int chosen_lane, local_rank, chosen_cell;
#else
if (task % 128 != 0) return;
const int lane = 0;
const int lanes = 1;
bool valid[16000];
int counts[128];
float center_x[6], center_y[6];
bool enabled[6];
int chosen_lane, local_rank, chosen_cell;
#endif
const int cells = points.shape[0];
const int span = (cells + lanes - 1) / lanes;
const int begin = lane * span;
const int end = wp::min(cells, begin + span);
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
for (int slot = 0; slot < 6; ++slot) {
    const float radius = wp::index(radii, row, slot);
    const float border = radius + cell_margin;
    const float red_limit = wp::index(target_radius, row) + radius + cell_margin;
    int number = 0;
    for (int cell = begin; cell < end; ++cell) {
        bool ok = wp::index(boundary, cell) >= border;
        ok = ok && wp::index(red, row, cell) >= red_limit;
        ok = ok && wp::index(robot_clear, row, slot, cell);
        const float x = wp::index(points, cell, 0);
        const float y = wp::index(points, cell, 1);
        for (int previous = 0; previous < slot; ++previous) {
            if (enabled[previous]) {
                const float dx = x - center_x[previous];
                const float dy = y - center_y[previous];
                const float distance = dx * dx + dy * dy;
                const float limit = radius + wp::index(radii, row, previous) + cell_margin;
                ok = ok && distance >= limit * limit;
            }
        }
        valid[cell] = ok;
        number += static_cast<int>(ok);
    }
    counts[lane] = number;
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if (lane == 0) {
        int total = 0;
        for (int i = 0; i < lanes; ++i) total += counts[i];
        const bool needed = wp::index(requested, row) > slot;
        // Unlike the Test sampler, failure of one large candidate must not
        // suppress later, smaller candidates.
        enabled[slot] = needed && total > 0;
        const int rank = static_cast<int>(wp::index(draws, row, slot, 0) * static_cast<float>(total));
        // searchsorted(right=True) returns cells when rank reaches total;
        // the original sampler clamps that result to the final cell.
        chosen_cell = cells - 1;
        chosen_lane = -1;
        local_rank = -1;
        int prefix = 0;
        for (int i = 0; i < lanes; ++i) {
            if (rank >= prefix && rank < prefix + counts[i]) {
                chosen_lane = i;
                local_rank = rank - prefix;
                break;
            }
            prefix += counts[i];
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if (lane == chosen_lane) {
        int rank = 0;
        for (int cell = begin; cell < end; ++cell) {
            if (valid[cell]) {
                if (rank == local_rank) { chosen_cell = cell; break; }
                ++rank;
            }
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if (lane == 0) {
        center_x[slot] = wp::index(points, chosen_cell, 0)
                        + (wp::index(draws, row, slot, 1) - 0.5f) * cell_size;
        center_y[slot] = wp::index(points, chosen_cell, 1)
                        + (wp::index(draws, row, slot, 2) - 0.5f) * cell_size;
        wp::index(indices, row, slot) = chosen_cell;
        wp::index(output_enabled, row, slot) = enabled[slot];
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
}
''')
def _select_robot_cells_native(task: int, points: wp.array2d(dtype=float), boundary: wp.array(dtype=float),  # type: ignore[valid-type]
                   red: wp.array2d(dtype=float), target_radius: wp.array(dtype=float),  # type: ignore[valid-type]
                   radii: wp.array2d(dtype=float), requested: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
                   draws: wp.array3d(dtype=float), cell_margin: float, cell_size: float,  # type: ignore[valid-type]
                   robot_clear: wp.array3d(dtype=bool), indices: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
                   output_enabled: wp.array2d(dtype=bool)):  # type: ignore[valid-type]
    ...


@wp.kernel(enable_backward=False, module='unique')
def _select_robot_cells_kernel(points: wp.array2d(dtype=float), boundary: wp.array(dtype=float),  # type: ignore[valid-type]
                   red: wp.array2d(dtype=float), target_radius: wp.array(dtype=float),  # type: ignore[valid-type]
                   radii: wp.array2d(dtype=float), requested: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
                   draws: wp.array3d(dtype=float), cell_margin: float, cell_size: float,  # type: ignore[valid-type]
                   robot_clear: wp.array3d(dtype=bool), indices: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
                   output_enabled: wp.array2d(dtype=bool)):  # type: ignore[valid-type]
    _select_robot_cells_native(wp.tid(), points, boundary, red, target_radius, radii, requested, draws,
                              cell_margin, cell_size, robot_clear, indices, output_enabled)


_select_robot_cells_kernel.module.options['fuse_fp'] = False
_select_robot_cells_kernel.module.mark_modified()


@wp.func_native(r'''
const int row = task / points.shape[0];
const int cell = task % points.shape[0];
const float x = wp::index(points, cell, 0);
const float y = wp::index(points, cell, 1);
bool clear[6] = {true, true, true, true, true, true};
for (int body = 0; body < centers.shape[1]; ++body) {
    bool relevant = false;
    for (int slot = 0; slot < 6; ++slot)
        relevant = relevant || (clear[slot] && wp::index(height_overlap, row, slot, body));
    if (!relevant) continue;
    const float dx = wp::max(wp::abs(x - wp::index(centers, row, body, 0))
                             - wp::index(half, row, body, 0), 0.0f);
    const float dy = wp::max(wp::abs(y - wp::index(centers, row, body, 1))
                             - wp::index(half, row, body, 1), 0.0f);
    const float distance = wp::sqrt(dx * dx + dy * dy);
    for (int slot = 0; slot < 6; ++slot)
        if (wp::index(height_overlap, row, slot, body))
            clear[slot] = clear[slot] && distance >= wp::index(limits, row, slot);
}
for (int slot = 0; slot < 6; ++slot)
    wp::index(output, row, slot, cell) = clear[slot];
''')
def _robot_cell_clearance_native(task: int, points: wp.array2d(dtype=float),  # type: ignore[valid-type]
                centers: wp.array3d(dtype=float), half: wp.array3d(dtype=float),  # type: ignore[valid-type]
                height_overlap: wp.array3d(dtype=bool), limits: wp.array2d(dtype=float),  # type: ignore[valid-type]
                output: wp.array3d(dtype=bool)):  # type: ignore[valid-type]
    ...


@wp.kernel(enable_backward=False, module='unique')
def _robot_cell_clearance_kernel(points: wp.array2d(dtype=float),  # type: ignore[valid-type]
                centers: wp.array3d(dtype=float), half: wp.array3d(dtype=float),  # type: ignore[valid-type]
                height_overlap: wp.array3d(dtype=bool), limits: wp.array2d(dtype=float),  # type: ignore[valid-type]
                output: wp.array3d(dtype=bool)):  # type: ignore[valid-type]
    _robot_cell_clearance_native(wp.tid(), points, centers, half, height_overlap, limits, output)


_robot_cell_clearance_kernel.module.options['fuse_fp'] = False
_robot_cell_clearance_kernel.module.mark_modified()


def _robot_clear_cells(points, centers, half, height_overlap, limits):
    """Evaluate the existing XY envelope and height rules without a [B,K,R,2] tensor."""
    device = wp.get_device(str(points.device))
    output = torch.empty((len(centers),6,len(points)),device=points.device,dtype=torch.bool)
    if output.numel():
        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
        args = [wp.from_torch(value) for value in (points,centers,half,height_overlap,limits,output)]
        wp.launch(_robot_cell_clearance_kernel,dim=len(centers)*len(points),inputs=args,
                  device=device,stream=stream)
    return output


def _select_robot_cells(points, boundary, red, target_radius, radii, requested, draws,
                        cell_margin, cell_size, robot_clear):
    """The six IK-layout candidates retain independent failure and their original draws."""
    if points.shape[1] != 2 or not 0 < len(points) <= 16000:
        raise ValueError('Cooperative IK placement requires 1..16000 XY cells')
    assert radii.shape == (len(requested),6) and draws.shape == (len(requested),6,4)
    assert red.shape == (len(radii),len(points))
    assert robot_clear.shape == (len(radii),6,len(points))
    assert target_radius.shape == requested.shape and requested.dtype == torch.int64
    assert all(value.dtype == torch.float32 for value in (points,boundary,red,target_radius,radii,draws))
    device = wp.get_device(str(points.device))
    indices = torch.empty((len(radii),6),device=points.device,dtype=torch.int64)
    enabled = torch.empty_like(indices,dtype=torch.bool)
    if len(radii):
        args = [wp.from_torch(value) for value in (points,boundary,red,target_radius,radii,requested,draws)]
        args += [cell_margin,cell_size,wp.from_torch(robot_clear),wp.from_torch(indices),wp.from_torch(enabled)]
        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
        wp.launch(_select_robot_cells_kernel,dim=len(radii)*128,inputs=args,
                  device=device,stream=stream,block_dim=128)
    return indices,enabled


def _place_robot_cells(layout,ids,local,yaw,active,delta,selected,radii,requested,draws,low,high,batch_size):
    """GPU path of place_obstacles; CPU invocation is reserved for reference regression."""
    off=layout.offset[ids,0];off=torch.stack((off[:,1],-off[:,0]),-1)
    for begin in range(0,len(ids),batch_size):
        end=min(len(ids),begin+batch_size)
        rows=torch.arange(begin,end,device=local.device)
        objects=selected[rows]
        target=local[rows,0,:2]+off[rows]
        red=layout.red_distance(layout.cells,target,target+delta[rows])
        bounds=layout.bounds[ids[rows,None],objects]
        z=layout.spawn_z[ids[rows,None],objects]
        height_overlap=((z[...,None]+bounds[:,:,None,1,2])>low[rows,None,:,2]) & (
                        (z[...,None]+bounds[:,:,None,0,2])<high[rows,None,:,2])
        limits=layout.radius[ids[rows,None],objects]+layout.cell_margin
        robot_clear=_robot_clear_cells(layout.cells,(low[rows]+high[rows])/2,
                                     (high[rows]-low[rows])/2,height_overlap,limits)
        indices,enabled=_select_robot_cells(layout.cells,layout.boundary,red,layout.expanded[ids[rows],0],
                            radii[rows],requested[rows],draws[rows],layout.cell_margin,layout.cell_size,robot_clear)
        centers=layout.cells[indices]+(draws[rows,:,1:3]-.5)*layout.cell_size
        angle=draws[rows,:,3]*(2*math.pi)-math.pi
        offset=layout.offset[ids[rows,None],objects]
        rotated=torch.stack((angle.cos()*offset[...,0]-angle.sin()*offset[...,1],
                             angle.sin()*offset[...,0]+angle.cos()*offset[...,1]),-1)
        pose=torch.cat((centers-rotated,z[...,None]),-1)
        local[rows[:,None],objects]=torch.where(enabled[...,None],pose,local[rows[:,None],objects])
        yaw[rows[:,None],objects]=torch.where(enabled,angle,0.)
        active[rows[:,None],objects]=enabled


def place_obstacles(layout,ids,local,yaw,active,delta,robot,selected,requested,generator,*,batch_size=8):
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('Obstacle batch size must be a positive integer')
    n=len(ids);device=local.device
    radii=layout.expanded[ids[:,None],selected]
    eligible=torch.arange(6,device=device)[None]<requested[:,None]
    order=torch.where(eligible,radii,-1.).argsort(dim=-1,descending=True,stable=True)
    selected,radii=selected.gather(1,order),radii.gather(1,order)
    draws=torch.rand((n,6,4),device=device,generator=generator)
    off=layout.offset[ids,0];off=torch.stack((off[:,1],-off[:,0]),-1)
    cloud=corners(robot)
    low,high=cloud.amin(-2)-.01,cloud.amax(-2)+.01
    if (local.is_cuda and local.dtype == torch.float32 and layout.cells.dtype == torch.float32
            and 0 < len(layout.cells) <= 16000):
        # Cell decisions are independent across rows. Draws were generated
        # above, so batching cannot change random consumption or row order.
        _place_robot_cells(layout,ids,local,yaw,active,delta,selected,radii,requested,draws,
                           low,high,max(batch_size,256))
        return
    # Conservative XY envelope of each collision OBB, retaining its Z interval.
    # Every valid jittered cell is clear for the entire candidate footprint.
    for begin in range(0,n,batch_size):
        rows=torch.arange(begin,min(n,begin+batch_size),device=device)
        target=local[rows,0,:2]+off[rows]
        red=layout.red_distance(layout.cells,target,target+delta[rows])
        robot_center=(low[rows]+high[rows])/2;robot_half=(high[rows]-low[rows])/2
        gap=(layout.cells[None,:,None]-robot_center[:,None,:,:2]).abs()-robot_half[:,None,:,:2]
        robot_distance=gap.clamp_min(0.).norm(dim=-1)
        distances=[]
        for slot in range(6):
            radius=radii[rows,slot];object_ids=selected[rows,slot]
            valid=layout.boundary[None]>=radius[:,None]+layout.cell_margin
            valid &= red>=layout.expanded[ids[rows],0,None]+radius[:,None]+layout.cell_margin
            bounds=layout.bounds[ids[rows],object_ids]
            z=layout.spawn_z[ids[rows],object_ids]
            height_overlap=((z[:,None]+bounds[:,None,1,2])>low[rows,:,2]) & ((z[:,None]+bounds[:,None,0,2])<high[rows,:,2])
            separation=torch.where(height_overlap[:,None,:],robot_distance,float('inf')).amin(-1)
            valid &= separation>=layout.radius[ids[rows],object_ids,None]+layout.cell_margin
            for previous,distance in enumerate(distances):
                valid &= distance>=(radius+radii[rows,previous]+layout.cell_margin)[:,None].square()
            prefix=valid.cumsum(-1,dtype=torch.int32);totals=prefix[:,-1]
            enabled=(requested[rows]>slot)&(totals>0)
            rank=(draws[rows,slot,0]*totals).to(torch.int32)
            index=torch.searchsorted(prefix,rank[:,None],right=True).squeeze(-1).clamp_max(len(layout.cells)-1)
            centers=layout.cells[index]+(draws[rows,slot,1:3]-.5)*layout.cell_size
            angle=draws[rows,slot,3]*(2*math.pi)-math.pi
            offset=layout.offset[ids[rows],object_ids]
            rotated=torch.stack((angle.cos()*offset[:,0]-angle.sin()*offset[:,1],
                                  angle.sin()*offset[:,0]+angle.cos()*offset[:,1]),-1)
            pose=torch.cat((centers-rotated,z[:,None]),-1)
            local[rows,object_ids]=torch.where(enabled[:,None],pose,local[rows,object_ids])
            yaw[rows,object_ids]=torch.where(enabled,angle,0.)
            active[rows,object_ids]=enabled
            distance=(layout.cells[None]-centers[:,None]).square().sum(-1)
            distances.append(distance.masked_fill(~enabled[:,None],float('inf')))
