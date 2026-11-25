"""Hard Revision: uncapped, shape-safe goals and randomized box placement."""
import math
import torch
from .wrench_refactor_layout import CircleResetLayout
import warp as wp
from types import SimpleNamespace


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
