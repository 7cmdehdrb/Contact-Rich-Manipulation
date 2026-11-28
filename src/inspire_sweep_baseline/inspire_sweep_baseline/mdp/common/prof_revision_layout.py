"""Hard Revision: uncapped, shape-safe goals and randomized box placement."""
import math
import torch
from .wrench_refactor_layout import CircleResetLayout
import warp as wp
from types import SimpleNamespace


def sample_wide_goals(target, lower, upper, minimum, angle_draw, distance_draw):
    """Uniform feasible angle and ray distance, without rejecting start points.

    Split the circle at its (at most eight) rectangle intersections. Duplicate
    endpoints give zero-length intervals, including tangencies and corners.
    """
    tau = 2 * math.pi
    edges = torch.cat(((lower-target)/minimum, (upper-target)/minimum), -1)
    x, y = edges[:, (0, 2)], edges[:, (1, 3)]
    ax, ay = x.clamp(-1, 1).acos(), y.clamp(-1, 1).asin()
    angles = torch.cat((ax, tau-ax, ay.remainder(tau), (math.pi-ay).remainder(tau)), -1)
    exists = torch.cat((x.abs() <= 1, x.abs() <= 1, y.abs() <= 1, y.abs() <= 1), -1)
    angles = torch.where(exists, angles, 0.)
    cuts = torch.cat((target.new_zeros(len(target), 1), angles,
                      target.new_full((len(target), 1), tau)), -1).sort(-1).values
    mid = (cuts[:, 1:]+cuts[:, :-1])/2
    endpoint = target[:, None]+minimum*torch.stack((mid.cos(), mid.sin()), -1)
    allowed = ((endpoint >= lower[:, None]) & (endpoint <= upper[:, None])).all(-1)
    lengths = (cuts[:, 1:]-cuts[:, :-1])*allowed
    cumulative = lengths.cumsum(-1)
    total = cumulative[:, -1]
    if bool((total <= 0).any()):
        raise ValueError('Wide target domain has no feasible minimum-distance direction')
    rank = angle_draw*total
    index = (cumulative <= rank[:, None]).sum(-1).clamp_max(lengths.shape[1]-1)
    previous = torch.cat((total.new_zeros(len(total), 1), cumulative[:, :-1]), -1)
    theta = cuts.gather(1, index[:, None]).squeeze(-1)+rank-previous.gather(1, index[:, None]).squeeze(-1)
    direction = torch.stack((theta.cos(), theta.sin()), -1)
    boundary = torch.where(direction >= 0, upper, lower)
    ray = torch.where(direction.abs() > 1e-10, (boundary-target)/direction,
                      torch.full_like(direction, float('inf'))).amin(-1)
    if bool((ray < minimum-1e-6).any()):
        raise RuntimeError('Feasible angle produced a ray shorter than the minimum')
    distance = minimum+(ray-minimum)*distance_draw
    return target+distance[:, None]*direction


class ProfRevisionWideLayout(CircleResetLayout):
    """v0_3_4's six-obstacle packing with broad, footprint-safe start/goal XY."""

    def __init__(self, bounds, spawn_z, clearance=.01, cell_size=.005, epsilon=1e-6, *,
                 workspace_inset=.02, spawn_extra_inset=.06):
        super().__init__(bounds, spawn_z, clearance, cell_size, epsilon)
        if not math.isfinite(workspace_inset) or workspace_inset < 0:
            raise ValueError('workspace_inset must be finite and nonnegative')
        self.workspace_inset = workspace_inset
        if not math.isfinite(spawn_extra_inset) or spawn_extra_inset < 0:
            raise ValueError('spawn_extra_inset must be finite and nonnegative')
        self.spawn_extra_inset = spawn_extra_inset
        # This child's target is the parent's known upright cylinder. Its XY
        # footprint is a disk, not the diagonal circle around its bounding box.
        # Use its true conservative radius while retaining all clearance terms.
        self.radius[:, 0] = ((bounds[:, 0, 1, :2]-bounds[:, 0, 0, :2])/2).amax(-1)
        self.expanded[:, 0] = self.radius[:, 0]+clearance

    @staticmethod
    def approach_and_lane_distance(points, start, goal):
        """Reserve the full sweep and the initial front approach, with the
        SAME radius/margin as the parent. Avoid filling their convex hull:
        long cross-shelf goals otherwise consume nearly the whole shelf.
        """
        approach = start.clone()
        approach[:, 0] = .2
        def distance(end):
            delta = end-start
            offset = points[None]-start[:, None]
            t = ((offset*delta[:, None]).sum(-1)/delta.square().sum(-1).clamp_min(1e-20)[:, None]).clamp(0., 1.)
            return (offset-t[..., None]*delta[:, None]).norm(dim=-1)
        return torch.minimum(distance(goal), distance(approach))

    def _retry_circle_cells(self, points, boundary, red, target_radius, radii, requested, draws,
                            cell_margin, cell_size):
        from .stabilize_sweep_layout import _select_test_cells
        result = _select_test_cells(points, boundary, red, target_radius, radii, requested,
                                    draws, cell_margin, cell_size)
        for attempt in range(1, 9):
            rows = (~result.complete).nonzero().flatten()
            if not len(rows):
                break
            retry_draws = draws[rows].clone()
            retry_draws[..., 0] = (retry_draws[..., 0]+attempt*.61803398875).remainder(1.)
            retry = _select_test_cells(points, boundary, red[rows], target_radius[rows], radii[rows],
                                       requested[rows], retry_draws, cell_margin, cell_size)
            for name in ('indices', 'enabled', 'complete', 'failed_slot'):
                getattr(result, name)[rows] = getattr(retry, name)
            self.last_retries = max(self.last_retries, attempt)
        return result

    def sample(self, ids, generator, **command_params):
        # configure_test_layout_batch is shared with the parent. Restore this
        # child's reservation rule on its own instance before running packing.
        self.red_distance = self.approach_and_lane_distance
        self.cell_selector = (self._retry_circle_cells if self.bounds.dtype == torch.float32
                              and len(self.cells) <= 16000 else None)
        self.last_retries = 0
        return super().sample(ids, generator, **command_params)

    def workspace_bounds(self, ids, edge_clearance=.02):
        # Root-relative footprint after the unchanged target yaw (-pi/2).
        bounds = self.bounds[ids, 0]
        low = torch.stack((bounds[:, 0, 1], -bounds[:, 1, 0]), -1)
        high = torch.stack((bounds[:, 1, 1], -bounds[:, 0, 0]), -1)
        shelf = bounds.new_tensor([.2, .5])
        return (-shelf+edge_clearance+self.workspace_inset-low,
                shelf-edge_clearance-self.workspace_inset-high)

    def sample_commands(self, ids, generator, spawn_radius=None, edge_clearance=.02,
                        excluded_x_half_angle=0., min_goal_distance=.06, max_goal_distance=None):
        if spawn_radius is not None or max_goal_distance is not None or excluded_x_half_angle != 0:
            raise ValueError('Wide layout requires rectangular starts and uncapped, full feasible directions')
        if (not math.isfinite(edge_clearance) or edge_clearance < .005
                or not math.isfinite(min_goal_distance) or min_goal_distance <= 0):
            raise ValueError('Invalid wide-layout clearance or minimum distance')
        lower, upper = self.workspace_bounds(ids, edge_clearance)
        if bool(((upper <= lower).any(-1) | ((upper-lower)/2).norm(dim=-1).lt(min_goal_distance)).any()):
            raise ValueError('Wide domain must support the minimum distance from every start')
        spawn_lower, spawn_upper = lower+self.spawn_extra_inset, upper-self.spawn_extra_inset
        if bool((spawn_upper <= spawn_lower).any()):
            raise ValueError('Extra spawn inset leaves no target start area')
        draw = torch.rand((len(ids), 4), device=lower.device, dtype=lower.dtype, generator=generator)
        target = spawn_lower+draw[:, :2]*(spawn_upper-spawn_lower)
        goal = sample_wide_goals(target, lower, upper, min_goal_distance, draw[:, 2], draw[:, 3])
        return target, goal

    def spawn_bounds(self, ids, edge_clearance=.02):
        """Leave hand room around initial targets; retain the wider goal domain."""
        lower, upper = self.workspace_bounds(ids, edge_clearance)
        return lower+self.spawn_extra_inset, upper-self.spawn_extra_inset


class ProfRevisionHardLayout(CircleResetLayout):
    """Ten-slot box exclusion on a cached 4,000-cell shelf grid.

    Independent cell priorities scatter boxes throughout the legal free area.
    Largest-first insertion and bounded retries retain goal/count, physical
    geometry and exclusion distances. No scene traversal or recooking.
    """
    obstacle_capacity = 10
    sample_batch_size = 512

    def __init__(self, bounds, spawn_z, clearance=.01, cell_size=.01, epsilon=1e-6):
        super().__init__(bounds, spawn_z, clearance, cell_size, epsilon)
        self.clearance = clearance
        self.last_retries = 0

    @torch.no_grad()
    def sample(self, ids, generator, **command_params):
        n, device = len(ids), self.bounds.device
        target, goal = self.sample_commands(ids, generator, **command_params)
        requested = torch.randint(0, 11, (n,), device=device, generator=generator)
        selected = torch.rand((n,16),device=device,generator=generator).argsort(-1)[:,:10]+1
        half = (self.bounds[ids[:,None],selected,1,:2]-self.bounds[ids[:,None],selected,0,:2])/2
        order = torch.where(torch.arange(10,device=device)[None]<requested[:,None],
                            half.prod(-1),-1.).argsort(dim=-1,descending=True,stable=True)
        selected=selected.gather(1,order)
        half=half.gather(1,order[...,None].expand(-1,-1,2))
        draws=torch.rand((n,10,4),device=device,generator=generator)
        # Quarter turns retain eight distinct physical geometries while exact
        # box checks avoid the severe wasted area of circumscribed circles.
        turns=(draws[...,3]*4).long()
        angle=turns*(math.pi/2)
        half=torch.where((turns%2)[...,None].bool(),half.flip(-1),half)
        off=self.offset[ids,0]
        target_off=torch.stack((off[:,1],-off[:,0]),-1)
        target_half=(self.bounds[ids,0,1,:2]-self.bounds[ids,0,0,:2]).flip(-1)/2
        segment=torch.cat((target+target_off,goal+target_off,target_half),-1)
        result=select_box_cells(self.cells,half,segment,requested,draws,
                                self.clearance,self.cell_margin,self.cell_size)
        self.last_retries=0
        for attempt in range(1,9):
            missing=(~result.complete).nonzero().flatten()
            if not len(missing):
                break
            retry_draws=draws[missing].clone()
            retry_draws[...,0]=(retry_draws[...,0]+attempt*.61803398875)%1.
            retry=select_box_cells(self.cells,half[missing],segment[missing],requested[missing],
                                   retry_draws,self.clearance,self.cell_margin,self.cell_size)
            for name in ('indices','enabled','complete','failed_slot'):
                getattr(result,name)[missing]=getattr(retry,name)
            self.last_retries=attempt
        if not bool(result.complete.all()):
            failed=(~result.complete).nonzero().flatten()
            raise RuntimeError(f'Hard box packing exhausted; goals/counts preserved: envs={ids[failed].tolist()}, '
                               f'count={requested[failed].tolist()}, target={target[failed].tolist()}, '
                               f'goal={goal[failed].tolist()}, half={half[failed].tolist()}')
        local=self.bounds.new_zeros((n,17,3))
        local[...,2]=50.+2.*torch.arange(17,device=device)
        local[:,0,:2],local[:,0,2]=target,self.spawn_z[ids,0]
        yaw=self.bounds.new_zeros((n,17))
        yaw[:,0]=-math.pi/2
        active=torch.zeros((n,17),device=device,dtype=torch.bool)
        active[:,0]=True
        centers=self.cells[result.indices]+(draws[...,1:3]-.5)*self.cell_size
        off=self.offset[ids[:,None],selected]
        rotated=torch.stack((angle.cos()*off[...,0]-angle.sin()*off[...,1],
                             angle.sin()*off[...,0]+angle.cos()*off[...,1]),-1)
        poses=torch.cat((centers-rotated,self.spawn_z[ids[:,None],selected,None]),-1)
        rows=torch.arange(n,device=device)[:,None]
        local[rows,selected]=torch.where(result.enabled[...,None],poses,local[rows,selected])
        yaw[rows,selected]=torch.where(result.enabled,angle,0.)
        active[rows,selected]=result.enabled
        return local,yaw,active,goal-target,requested

    def sample_commands(self, ids, generator, spawn_radius=.02, edge_clearance=.02,
                        excluded_x_half_angle=0., min_goal_distance=.05,
                        max_goal_distance=None):
        """Uniform full-circle angle, uniform [5cm, shape-safe ray boundary].

        The target's fixed reset yaw is -pi/2. Off-centre assets therefore need
        asymmetric root-position limits, rather than unrotated half-extents.
        None means no artificial distance cap; the shelf remains the boundary.
        """
        if excluded_x_half_angle != 0.:
            raise ValueError('Hard goals use the full circle')
        if (not math.isfinite(spawn_radius) or spawn_radius < 0.
                or not math.isfinite(edge_clearance) or edge_clearance < .005
                or not math.isfinite(min_goal_distance) or min_goal_distance < .05):
            raise ValueError('Invalid Hard spawn/goal clearance or minimum below 5cm')
        if max_goal_distance is not None and (not math.isfinite(max_goal_distance)
                                              or max_goal_distance < min_goal_distance):
            raise ValueError('Maximum distance must be None or finite and >= minimum')
        bounds = self.bounds[ids, 0]
        lo = torch.stack((bounds[:, 0, 1], -bounds[:, 1, 0]), -1)
        hi = torch.stack((bounds[:, 1, 1], -bounds[:, 0, 0]), -1)
        shelf = bounds.new_tensor([.2, .5])
        lower, upper = -shelf + edge_clearance - lo, shelf - edge_clearance - hi
        if bool(((lower > -spawn_radius-min_goal_distance)
                 | (upper < spawn_radius+min_goal_distance)).any()):
            raise ValueError('Target geometry leaves no full-angle 5cm goal domain')
        draws = torch.rand((len(ids), 4), device=bounds.device, dtype=bounds.dtype, generator=generator)
        angle = draws[:, 0] * (2*math.pi)
        target = spawn_radius * draws[:, 1:2].sqrt() * torch.stack((angle.cos(), angle.sin()), -1)
        theta = draws[:, 2] * (2*math.pi)
        direction = torch.stack((theta.cos(), theta.sin()), -1)
        maximum = torch.where(direction > 0., (upper-target)/direction.clamp_min(1e-12),
                              (lower-target)/direction.clamp_max(-1e-12)).amin(-1)
        if max_goal_distance is not None:
            maximum = maximum.clamp_max(max_goal_distance)
        distance = min_goal_distance + draws[:, 3] * (maximum-min_goal_distance)
        return target, target + direction*distance[:, None]


@wp.func_native(r'''
const int row = task / 128;
#if defined(__CUDA_ARCH__)
const int lane=task%128, lanes=128;
__shared__ float scores[128], cx[10], cy[10];
__shared__ int choices[128], chosen;
__shared__ bool complete;
#else
if(task%128!=0)return;
const int lane=0, lanes=1;
float scores[128], cx[10], cy[10];
int choices[128], chosen;
bool complete;
#endif
const int count=wp::index(requested,row);
if(lane==0){complete=true;
    for(int k=0;k<10;++k){wp::index(indices,row,k)=0;wp::index(output_enabled,row,k)=false;}
    wp::index(output_failed,row)=-1;}
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
for(int slot=0;slot<count;++slot){
    const float hx=wp::index(half,row,slot,0), hy=wp::index(half,row,slot,1);
    const unsigned int seed=(unsigned int)(wp::index(draws,row,slot,0)*16777216.0f);
    float best=1.0e30f;int best_cell=-1;
    for(int cell=lane;cell<points.shape[0];cell+=lanes){
        const float x=wp::index(points,cell,0),y=wp::index(points,cell,1);
        const float sx=wp::index(segment,row,0),sy=wp::index(segment,row,1);
        const float gx=wp::index(segment,row,2),gy=wp::index(segment,row,3);
        const float ex=hx+wp::index(segment,row,4)+clearance+cell_margin;
        const float ey=hy+wp::index(segment,row,5)+clearance+cell_margin;
        const float extension=.2f-wp::min(sx,gx),nx=sy-gy,ny=gx-sx;
        const float projection=nx*(x-sx)+ny*(y-sy), edge=nx*extension;
        const float reach=wp::abs(nx)*ex+wp::abs(ny)*ey;
        bool outside=x<wp::min(sx,gx)-ex||x>wp::max(sx,gx)+extension+ex
            ||y<wp::min(sy,gy)-ey||y>wp::max(sy,gy)+ey
            ||projection<wp::min(0.0f,edge)-reach||projection>wp::max(0.0f,edge)+reach;
        bool ok=outside&&wp::abs(x)+hx+clearance+cell_margin<=.2f
            &&wp::abs(y)+hy+clearance+cell_margin<=.5f;
        for(int previous=0;previous<slot;++previous){
            const float dx=wp::abs(x-cx[previous]),dy=wp::abs(y-cy[previous]);
            ok=ok&&(dx>=hx+wp::index(half,row,previous,0)+clearance+cell_margin
                 ||dy>=hy+wp::index(half,row,previous,1)+clearance+cell_margin);
        }
        // Hash ranks are independent of XY direction and change per box.
        // The minimum over legal cells samples the area, not an extreme corner.
        unsigned int rank=(unsigned int)cell+0x9e3779b9u*(seed+1u);
        rank=(rank^(rank>>16))*0x7feb352du;
        rank=(rank^(rank>>15))*0x846ca68bu;
        rank=rank^(rank>>16);
        const float score=(float)rank;
        if(ok&&(score<best||(score==best&&cell<best_cell))){best=score;best_cell=cell;}
    }
    scores[lane]=best;choices[lane]=best_cell;
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if(lane==0){
        chosen=-1;float best=1.0e30f;
        for(int k=0;k<lanes;++k)if(choices[k]>=0&&(scores[k]<best||(scores[k]==best&&choices[k]<chosen))){best=scores[k];chosen=choices[k];}
        if(chosen<0){complete=false;wp::index(output_failed,row)=slot;}
        else{
            cx[slot]=wp::index(points,chosen,0)+(wp::index(draws,row,slot,1)-.5f)*cell_size;
            cy[slot]=wp::index(points,chosen,1)+(wp::index(draws,row,slot,2)-.5f)*cell_size;
            wp::index(indices,row,slot)=chosen;wp::index(output_enabled,row,slot)=true;
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if(!complete)break;
}
if(lane==0)wp::index(output_complete,row)=complete;
''')
def _select_boxes_native(task: int, points: wp.array2d(dtype=float),
                         half: wp.array3d(dtype=float), segment: wp.array2d(dtype=float),
                         requested: wp.array(dtype=wp.int64), draws: wp.array3d(dtype=float),
                         clearance: float, cell_margin: float, cell_size: float,
                         indices: wp.array2d(dtype=wp.int64), output_enabled: wp.array2d(dtype=bool),
                         output_complete: wp.array(dtype=bool), output_failed: wp.array(dtype=wp.int64)):
    ...


@wp.kernel(enable_backward=False, module='unique')
def _select_boxes(points: wp.array2d(dtype=float), half: wp.array3d(dtype=float),
                  segment: wp.array2d(dtype=float), requested: wp.array(dtype=wp.int64),
                  draws: wp.array3d(dtype=float), clearance: float, cell_margin: float, cell_size: float,
                  indices: wp.array2d(dtype=wp.int64), enabled: wp.array2d(dtype=bool),
                  complete: wp.array(dtype=bool), failed: wp.array(dtype=wp.int64)):
    _select_boxes_native(wp.tid(), points, half, segment, requested, draws, clearance,
                         cell_margin, cell_size, indices, enabled, complete, failed)


def select_box_cells(points, half, segment, requested, draws, clearance, cell_margin, cell_size):
    indices=torch.zeros((len(requested),10),device=points.device,dtype=torch.long)
    enabled=torch.zeros_like(indices,dtype=torch.bool)
    complete=torch.zeros_like(requested,dtype=torch.bool)
    failed=torch.full_like(requested,-1)
    wp.init()
    values=(points.contiguous(),half.contiguous(),segment.contiguous(),requested.contiguous(),draws.contiguous())
    args=[wp.from_torch(v) for v in values]+[clearance,cell_margin,cell_size]
    args += [wp.from_torch(v) for v in (indices,enabled,complete,failed)]
    stream=wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
    wp.launch(_select_boxes,dim=len(requested)*128,inputs=args,device=str(points.device),stream=stream,block_dim=128)
    return SimpleNamespace(indices=indices,enabled=enabled,complete=complete,failed_slot=failed)
