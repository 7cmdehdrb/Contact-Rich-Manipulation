"""Target-only contact point reductions; normal load is never a net-force norm."""
import torch
import warp as wp
from isaaclab.utils.math import quat_apply
from .contact_sweep_clutter_fusion_contacts import object_hand_sensor_name


@wp.kernel(enable_backward=False)
def _reduce(force: wp.array(dtype=wp.vec3), point: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
            counts: wp.array2d(dtype=wp.int64), starts: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
            com: wp.array(dtype=wp.vec3), total: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
            torque: wp.array2d(dtype=wp.vec3), load: wp.array2d(dtype=float),  # type: ignore[valid-type]
            valid: wp.array2d(dtype=wp.bool)):  # type: ignore[valid-type]
    row,pair=wp.tid()
    count,start=counts[row,pair],starts[row,pair]
    ok=count>=0 and (count==0 or (start>=0 and start+count<=force.shape[0]))
    f,t,s=wp.vec3(0.0),wp.vec3(0.0),float(0.0)
    if ok:
        for i in range(int(start),int(start+count)):
            value,position=force[i],point[i]
            for j in range(3):ok=ok and wp.isfinite(value[j]) and wp.isfinite(position[j]) and wp.isfinite(com[row][j])
            f+=value;t+=wp.cross(position-com[row],value);s+=wp.length(value)
    for j in range(3):ok=ok and wp.isfinite(f[j]) and wp.isfinite(t[j])
    ok=ok and wp.isfinite(s)
    if not ok:f,t,s=wp.vec3(0.0),wp.vec3(0.0),float(0.0)
    total[row,pair]=f
    torque[row,pair]=t
    load[row,pair]=s
    valid[row,pair]=ok


def reduce_contacts(force,point,counts,starts,com):
    if force.ndim!=2 or force.shape[-1]!=3 or point.shape!=force.shape:
        raise ValueError('Contact vectors and points must be [capacity,3]')
    if counts.ndim!=2 or counts.shape!=starts.shape or com.shape!=(len(counts),3):
        raise ValueError('Contact ranges must be [sensor,filter] with one COM per sensor')
    wp.init()
    counts,starts=counts.long().contiguous(),starts.long().contiguous()
    total=force.new_zeros((*counts.shape,3));torque=torch.zeros_like(total)
    load=force.new_zeros(counts.shape);valid=torch.empty_like(counts,dtype=torch.bool)
    stream=wp.stream_from_torch(torch.cuda.current_stream(force.device)) if force.is_cuda else None
    wp.launch(_reduce,dim=tuple(counts.shape),inputs=[
        wp.from_torch(force.contiguous(),dtype=wp.vec3),wp.from_torch(point.contiguous(),dtype=wp.vec3),
        wp.from_torch(counts),wp.from_torch(starts),wp.from_torch(com.contiguous(),dtype=wp.vec3),
        wp.from_torch(total,dtype=wp.vec3),wp.from_torch(torque,dtype=wp.vec3),wp.from_torch(load),wp.from_torch(valid)],
        device=str(force.device),stream=stream)
    # A full shared buffer may have truncated contacts; never admit it to Gate.
    complete=(counts.clamp_min(0).sum()<len(force)) | (counts.sum()==0)
    return total.sum(1),torque.sum(1),load.sum(1),valid.all(1)&complete


def centroid(env):
    data=env.scene['object_collection'].data
    return data.object_link_pos_w[:,0]+quat_apply(data.object_link_quat_w[:,0],env._revision_bounds_t[:,0].mean(1))


def cache_static_local_com(env):
    """Pin body-local COM poses for the fixed-physics IK and Test resets.

    Call after committing either family reset. Startup has
    finished assigning masses/inertias; subsequent family resets only change
    poses, velocities, gravity flags and observations. Body-local COM poses
    therefore stay constant, while world COM poses and velocities remain live.

    Isaac Lab otherwise queries these CPU-backed PhysX properties on every
    physics timestamp. Its joint-position writer also invalidates the local
    buffer on reset, despite leaving the local COM unchanged. Keep the first
    post-startup sample and re-arm the existing timestamp buffers after each
    reset. No PhysX API or shared data class is replaced.
    """
    if not getattr(env, '_refactor_physics_initialized', False):
        raise RuntimeError('Static local COM caching requires completed startup physics')
    robot = env.scene['robot'].data
    objects = env.scene['object_collection'].data
    buffers = (robot._body_com_pose_b, objects._object_com_pose_b)
    cached = getattr(env, '_stabilize_static_local_com', None)
    if cached is None:
        # Managers may have read these properties before startup set_inertias.
        # Explicitly fetch the final physical values, not a stale startup cache.
        for buffer in buffers:
            buffer.timestamp = -float('inf')
        cached = (robot.body_com_pose_b.clone(), objects.object_com_pose_b.clone())
        env._stabilize_static_local_com = cached
    for buffer, pose in zip(buffers, cached):
        buffer.data = pose
        buffer.timestamp = float('inf')


class ContactSamples:
    def __init__(self,env):
        name=env.scene['object_collection'].object_names[0]
        self.view=env.scene[object_hand_sensor_name(name)].contact_physx_view
        if self.view.sensor_count!=env.num_envs or self.view.filter_count<1:
            raise RuntimeError('Expected one target sensor per environment with hand-only filters')
        n=env.num_envs;device=env.device
        self.force=torch.zeros((n,3),device=device);self.moment=torch.zeros_like(self.force)
        self.load=torch.zeros(n,device=device);self.valid=torch.zeros(n,device=device,dtype=torch.bool)
        self.force_sum=torch.zeros_like(self.force);self.squeeze_sum=torch.zeros_like(self.load)
        self.contact=torch.zeros_like(self.valid);self.all_valid=torch.zeros_like(self.valid)
        self.samples=0

    def begin(self):
        self.samples=0;self.force_sum.zero_();self.squeeze_sum.zero_();self.contact.zero_();self.all_valid.fill_(True)

    def capture(self,env,target_index=0):
        com=env.scene['object_collection'].data.object_com_pos_w[:,target_index]
        scalar,points,normals,_,counts,starts=self.view.get_contact_data(dt=env.physics_dt)
        normal,torque,load,valid=reduce_contacts(scalar.reshape(-1,1)*normals,points,counts,starts,com)
        # Reduction precedes the second API call: counts/starts are shared by PhysX.
        friction,anchors,counts,starts=self.view.get_friction_data(dt=env.physics_dt)
        tangent,moment,_,friction_valid=reduce_contacts(friction,anchors,counts,starts,com)
        self.valid=valid&friction_valid
        self.force=torch.where(self.valid[:,None],normal+tangent,0.)
        self.moment=torch.where(self.valid[:,None],torque+moment,0.)
        self.load=torch.where(self.valid,load,0.)
        self.force_sum+=self.force
        self.squeeze_sum+=(self.load/5.-1.).clamp_min(0.).square()
        self.contact|=self.valid&(self.load>=.01)
        self.all_valid&=self.valid;self.samples+=1

    def reset(self,ids):
        for value in (self.force,self.moment,self.load,self.force_sum,self.squeeze_sum):value[ids]=0.
        for value in (self.valid,self.contact,self.all_valid):value[ids]=False
