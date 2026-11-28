"""Two central cylinders and fresh initial-view snapshots without physical resets."""
import math
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate

from .wrench_refactor_layout import CircleResetLayout
from .stabilize_sweep_contact import ContactSamples
from .contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
from .prof_revision_obb_gpu import observe_batch


START_XY = ((.035, -.05), (-.035, .05))
GOAL_XY = ((.035, -.11), (-.035, .11))
DIRECTIONS = ('left', 'right')
SEQUENCE_COUNT = len(START_XY)


class TwoCylinderLayout(CircleResetLayout):
    """Same 17-body pool, two active cylinders in the shelf's central patch."""

    def sample(self, ids, generator, **command_params):
        n = len(ids)
        local = self.bounds.new_zeros(n, 17, 3)
        local[..., 2] = 50.+2.*torch.arange(17, device=local.device)
        local[:, :SEQUENCE_COUNT, :2] = local.new_tensor(START_XY)
        local[:, :SEQUENCE_COUNT, 2] = self.spawn_z[ids, :SEQUENCE_COUNT]
        yaw = self.bounds.new_zeros(n, 17)
        yaw[:, :SEQUENCE_COUNT] = -math.pi/2
        active = torch.zeros(n, 17, device=local.device, dtype=torch.bool)
        active[:, :SEQUENCE_COUNT] = True
        # Bound-box separation and wall reserve, including each cooked scale.
        half = (self.bounds[ids, :SEQUENCE_COUNT, 1, :2]-self.bounds[ids, :SEQUENCE_COUNT, 0, :2]).flip(-1)/2
        centers = local[:, :SEQUENCE_COUNT, :2]
        for a,b in ((0,1),):
            separated = ((centers[:,a]-centers[:,b]).abs() > half[:,a]+half[:,b]+.01).any(-1)
            if not bool(separated.all()):
                raise ValueError('Two-cylinder central layout overlaps actual cooked bounds')
        goals = local.new_tensor(GOAL_XY)
        if not bool((goals.abs()+half+.02 < goals.new_tensor([.2,.5])).all()):
            raise ValueError('Sequence goal lacks shelf clearance')
        delta = local.new_tensor(GOAL_XY[0])-local.new_tensor(START_XY[0])
        return local, yaw, active, delta.expand(n,-1).clone(), torch.full((n,),SEQUENCE_COUNT-1,device=local.device,dtype=torch.long)


class SequenceContactSamples(ContactSamples):
    """Switch the reporting sensor and COM to the same physical target body."""

    def select_target(self, env, index):
        name = env.scene['object_collection'].object_names[index]
        self.view = env.scene[object_hand_sensor_name(name)].contact_physx_view
        if self.view.sensor_count != env.num_envs or self.view.filter_count < 1:
            raise RuntimeError('Invalid sequence target contact view')

    def capture(self, env):
        super().capture(env, target_index=env.sequence_stage)


def refresh_sequence_snapshot(env, index):
    """Re-observe current poses, in unchanged frontal-visibility coordinates.

    Only boxes are reordered for the observer's slot-zero API. Physical bodies,
    active masks and simulation poses are never reordered or written here.
    """
    if env.num_envs != 1:
        raise ValueError('Continuous demonstration supports exactly one scene')
    data = env.scene['object_collection'].data
    shelf = env.scene['shelf'].data
    order = torch.tensor([index]+[i for i in range(17) if i != index],device=env.device)
    q = shelf.root_quat_w[:,None].expand(-1,17,-1)
    local = quat_apply_inverse(q, data.object_link_pos_w-shelf.root_pos_w[:,None])
    relative = quat_mul(quat_conjugate(q),data.object_link_quat_w)
    w,x,y,z = relative.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    bounds = env._revision_bounds_t
    centers = local+quat_apply(relative,bounds.mean(-2))
    boxes = torch.cat((centers,bounds[...,1,:]-bounds[...,0,:],yaw[...,None]),-1)
    target,obstacles,valid,sources,status = observe_batch(
        boxes[:,order],env._active_object_mask[:,order],env.cfg.obb_minimum_width,env.cfg.obb_minimum_ratio)
    if bool((status != 0).any()):
        raise RuntimeError(f'Current sequence target is not observable: stage={index+1}, status={status.tolist()}')
    def world(boxes):
        count = boxes.shape[1] if boxes.ndim == 3 else None
        quat = shelf.root_quat_w[:,None].expand(-1,count,-1) if count else shelf.root_quat_w
        pos = shelf.root_pos_w[:,None] if count else shelf.root_pos_w
        rotation = torch.zeros_like(boxes[...,:4])
        rotation[...,0],rotation[...,3] = (boxes[...,6]/2).cos(),(boxes[...,6]/2).sin()
        return pos+quat_apply(quat,boxes[...,:3]),quat_mul(quat,rotation)
    tp,tq = world(target)
    op,oq = world(obstacles)
    env._obb_target_pos_w.copy_(tp)
    env._obb_target_quat_w.copy_(tq)
    env._obb_target_size.copy_(target[:,3:6])
    env._obb_obstacle_pos_w.copy_(op)
    env._obb_obstacle_quat_w.copy_(oq)
    env._obb_obstacle_size.copy_(obstacles[...,3:6])
    env._obb_valid.copy_(valid)
    env._obb_sources.copy_(torch.where(valid,order[sources.clamp_min(0)],-1))
    inverse = quat_conjugate(data.object_link_quat_w[:,index])
    env._obb_target_local_pos.copy_(quat_apply(inverse,tp-data.object_link_pos_w[:,index]))
    env._obb_target_local_quat.copy_(quat_mul(inverse,tq))


def configure_sequence(cfg):
    """Demonstration-only overrides on the loaded IK-v1 config; no Gym registration."""
    if cfg.scene.num_envs != 1:
        raise ValueError('Sequence requires one environment')
    cfg.events.object_geometry.params['layout_class'] = TwoCylinderLayout
    objects = cfg.scene.object_collection.rigid_objects
    for name in list(objects)[:SEQUENCE_COUNT]:
        objects[name].spawn.kind_override = 'cylinder'
    return cfg
