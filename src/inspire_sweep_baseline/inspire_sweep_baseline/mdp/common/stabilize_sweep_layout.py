"""One layout pass: preserve successful placements and skip only blocked candidates."""
import math
import torch
from .stabilize_sweep_geometry import corners


def place_obstacles(layout,ids,local,yaw,active,delta,robot,selected,requested,generator):
    n=len(ids);device=local.device
    radii=layout.expanded[ids[:,None],selected]
    eligible=torch.arange(6,device=device)[None]<requested[:,None]
    order=torch.where(eligible,radii,-1.).argsort(dim=-1,descending=True,stable=True)
    selected,radii=selected.gather(1,order),radii.gather(1,order)
    draws=torch.rand((n,6,4),device=device,generator=generator)
    off=layout.offset[ids,0];off=torch.stack((off[:,1],-off[:,0]),-1)
    cloud=corners(robot)
    low,high=cloud.amin(-2)-.01,cloud.amax(-2)+.01
    # Conservative XY envelope of each collision OBB, retaining its Z interval.
    # Every valid jittered cell is clear for the entire candidate footprint.
    for begin in range(0,n,8):
        rows=torch.arange(begin,min(n,begin+8),device=device)
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
