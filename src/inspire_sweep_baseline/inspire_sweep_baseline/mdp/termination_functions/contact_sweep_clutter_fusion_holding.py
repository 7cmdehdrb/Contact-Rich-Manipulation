"""Dense settling reward and independent strict, sustained success.

One live goal radius for settling, success and contact credit."""

from ..common.contact_sweep_clutter_fusion_holding import goal_radius
from ..common.contact_sweep_clutter_fusion_holding import target_measurements
from isaaclab.managers import ManagerTermBase
import math as math
import torch as torch

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


class ScheduledHoldingSuccess(HoldingSuccess):
    """Keep Holding's speed/failure/timer checks, using the live reward radius."""

    def __call__(self, env, speed_tolerance=0.01, hold_time=3.0, failure_terms=()):
        goal = env.reward_manager.get_term_cfg("goal_hold")
        return super().__call__(
            env, position_tolerance=goal_radius(env), speed_tolerance=speed_tolerance,
            hold_time=hold_time, failure_terms=failure_terms,
            command_name=goal.params.get("command_name", "target_goal_pos"),
        )
