"""Dense settling reward and independent strict, sustained success."""
import math
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse, quat_apply


def target_measurements(env, command_name='target_goal_pos'):
    data = env.scene['object_collection'].data
    rows = torch.arange(env.num_envs, device=env.device)
    ids = env.target_id[:, 0].long()
    pos, vel = data.object_pos_w[rows,ids], data.object_lin_vel_w[rows,ids]
    goal = env.command_manager.get_command(command_name)
    shelf = env.scene['shelf'].data
    local = quat_apply_inverse(shelf.root_quat_w, pos-shelf.root_pos_w)
    up = torch.zeros_like(pos); up[:,2] = 1.
    target_up = quat_apply(data.object_quat_w[rows,ids], up)
    alignment = (target_up*quat_apply(shelf.root_quat_w,up)).sum(-1)
    valid = (torch.isfinite(pos).all(-1) & torch.isfinite(vel).all(-1)
             & torch.isfinite(goal[:,:2]).all(-1) & torch.isfinite(alignment)
             & (local[:,0].abs() <= .16) & (local[:,1].abs() <= .46)
             & (local[:,2] >= 1.04) & (local[:,2] <= 1.07)
             & (alignment >= math.cos(.9)))
    distance = (pos[:,:2]-goal[:,:2]).norm(dim=-1)
    speed = vel.norm(dim=-1)
    return distance, speed, valid


def goal_settling(env, position_scale=.01, speed_scale=.05, cutoff_multiple=3., command_name='target_goal_pos'):
    if not all(math.isfinite(v) and v>0 for v in (position_scale,speed_scale,cutoff_multiple)):
        raise ValueError('Settling scales and cutoff must be finite and positive')
    distance,speed,valid = target_measurements(env,command_name)
    score = torch.exp(-math.log(2)*((distance/position_scale)**2+(speed/speed_scale)**2))
    return torch.where(valid & (distance < cutoff_multiple*position_scale),score,0.)


class HoldingSuccess(ManagerTermBase):
    """Never infer success from a nonzero dense reward."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self._steps = torch.zeros(env.num_envs,dtype=torch.long,device=env.device)

    def reset(self,env_ids=None):
        self._steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self,env,position_tolerance=.005,speed_tolerance=.01,hold_time=3.,
                 command_name='target_goal_pos',failure_terms=()):
        if not all(math.isfinite(v) and v>0 for v in (position_tolerance,speed_tolerance,hold_time)):
            raise ValueError('Success thresholds must be finite and positive')
        distance,speed,valid = target_measurements(env,command_name)
        eligible = valid & (distance<=position_tolerance) & (speed<=speed_tolerance)
        for name in failure_terms:
            eligible &= ~env.termination_manager.get_term(name)
        self._steps[:] = torch.where(eligible,self._steps+1,0)
        return self._steps >= math.ceil(hold_time/env.step_dt-1e-9)
