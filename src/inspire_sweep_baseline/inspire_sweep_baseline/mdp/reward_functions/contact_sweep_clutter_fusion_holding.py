"""Dense settling reward and independent strict, sustained success."""

from ..common.contact_sweep_clutter_fusion_holding import target_measurements
import math as math
import torch as torch

def goal_settling(env, position_scale=.01, speed_scale=.05, cutoff_multiple=3., command_name='target_goal_pos'):
    if not all(math.isfinite(v) and v>0 for v in (position_scale,speed_scale,cutoff_multiple)):
        raise ValueError('Settling scales and cutoff must be finite and positive')
    distance,speed,valid = target_measurements(env,command_name)
    score = torch.exp(-math.log(2)*((distance/position_scale)**2+(speed/speed_scale)**2))
    return torch.where(valid & (distance < cutoff_multiple*position_scale),score,0.)
