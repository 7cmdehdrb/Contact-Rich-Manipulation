"""Physics-sample Gate and policy-boundary phase state, independently per row."""
import math
import torch
from isaaclab.utils.math import quat_mul,quat_conjugate


class StabilizeState:
    def __init__(self,count,device):
        self.achieved=torch.zeros(count,device=device,dtype=torch.bool)
        self.reward_phase=torch.zeros_like(self.achieved)
        self.consecutive=torch.zeros(count,device=device,dtype=torch.long)
        self.initial_center=torch.zeros((count,3),device=device)
        self.initial_quat=torch.zeros((count,4),device=device);self.initial_quat[:,0]=1.
        self.setup_sum=torch.zeros(count,device=device)
        self.move_cost=torch.zeros_like(self.setup_sum);self.rotate_cost=torch.zeros_like(self.setup_sum)
        self.previous_distance=torch.zeros_like(self.setup_sum)
        self.contact_history=torch.zeros((count,3),device=device,dtype=torch.bool)
        self._init_contact_factor(count,device)
        self.positive=torch.zeros_like(self.setup_sum);self.retreat=torch.zeros_like(self.setup_sum)
        self.samples=0

    def reset(self,ids,center,quat,goal):
        self.achieved[ids]=False;self.reward_phase[ids]=False;self.consecutive[ids]=0
        self.initial_center[ids]=center;self.initial_quat[ids]=quat
        self.setup_sum[ids]=0.;self.move_cost[ids]=0.;self.rotate_cost[ids]=0.
        self.previous_distance[ids]=(goal[:,:2]-center[:,:2]).norm(dim=-1)
        self.contact_history[ids]=False;self._reset_contact_factor(ids)
        self.positive[ids]=0.;self.retreat[ids]=0.

    def begin(self):
        self.reward_phase.copy_(self.achieved)
        self.setup_sum.zero_();self.samples=0

    def sample(self,center,quat,goal,force,moment,load,valid,linear_speed,angular_speed):
        goal=goal[:,:3]
        delta=goal-center;delta=delta.clone();delta[:,2]=0.
        distance=delta.norm(dim=-1);direction=delta/distance[:,None].clamp_min(1e-12)
        parallel=(force*direction).sum(-1)
        perpendicular=(force-parallel[:,None]*direction).norm(dim=-1)
        translation=(center-self.initial_center).norm(dim=-1)
        relative=quat_mul(quat,quat_conjugate(self.initial_quat))
        angle=2*torch.atan2(relative[:,1:].norm(dim=-1),relative[:,0].abs())
        finite=torch.isfinite(torch.cat((force,moment,center,quat,goal,load[:,None],linear_speed[:,None],angular_speed[:,None]),-1)).all(-1)
        contact=valid&finite&(load>=.01)&(distance>1e-12)
        exponent=-((parallel-.5)/.2).square()-(perpendicular/.2).square()-(moment[:,2]/.005).square()-(moment[:,:2]/.01).square().sum(-1)
        score=(parallel/.5).clamp(0.,1.)*exponent.exp()
        self.setup_sum+=torch.where(contact,score,0.)
        good=(contact&((parallel-.5).abs()<=.2)&(perpendicular<=.2)&(moment[:,2].abs()<=.005)
              &(moment[:,:2].norm(dim=-1)<=.01)&(translation<=.003)&(angle<=math.radians(3.))
              &(linear_speed<=.005)&(angular_speed<=.05)&(load<=5.))
        self.consecutive.copy_(torch.where(self.achieved,self.consecutive,
                              torch.where(good,self.consecutive+1,0)))
        self.achieved|=self.consecutive>=30
        self.move_cost.copy_(torch.where(self.reward_phase,0.,(translation/.003-1.).clamp_min(0.).square()))
        self.rotate_cost.copy_(torch.where(self.reward_phase,0.,(angle/math.radians(3.)-1.).clamp_min(0.).square()))
        self.samples+=1

    @property
    def stage_score(self):
        return torch.where(self.reward_phase,1.,self.setup_sum/max(self.samples,1))

    def finish(self,center,goal,mean_force,contact,valid,eef_z,shelf_z,*,height_limit=.12,
               normalize_contact_count=True):
        delta=goal[:,:2]-center[:,:2];distance=delta.norm(dim=-1)
        progress=self.previous_distance-distance
        self.contact_history[:,1:]=self.contact_history[:,:-1].clone()
        self.contact_history[:,0]=contact&valid&self.reward_phase
        recognized=(self.contact_history.sum(-1)>=2)&self.reward_phase&(eef_z<shelf_z+height_limit)&torch.isfinite(eef_z)&torch.isfinite(shelf_z)
        h=self._contact_factor(recognized,normalize_contact_count)
        direction=delta/distance[:,None].clamp_min(1e-12)
        magnitude=mean_force.norm(dim=-1)
        alignment=((mean_force[:,:2]*direction).sum(-1)/magnitude.clamp_min(1e-12)).clamp(0.,1.)
        usable=contact&valid&(mean_force[:,:2].norm(dim=-1)>=1.)&(distance>1e-12)
        alignment=torch.where(usable,alignment,0.)
        self.positive.copy_(torch.where(self.reward_phase,progress.clamp_min(0.)*h*alignment,0.))
        self.retreat.copy_(torch.where(self.reward_phase,(-progress).clamp_min(0.),0.))
        self.previous_distance.copy_(distance)

    def _init_contact_factor(self,count,device):
        self.contact_count=torch.zeros(count,device=device,dtype=torch.long)

    def _reset_contact_factor(self,ids):
        self.contact_count[ids]=0

    def _contact_factor(self,recognized,normalize_contact_count):
        self.contact_count.copy_(torch.where(self.reward_phase,
            (self.contact_count+torch.where(recognized,1,-2)).clamp(0,20),0))
        contact_factor=self.contact_count/20. if normalize_contact_count else self.contact_count.float()
        return torch.where(recognized,contact_factor,0.)


class SweepOnlyState(StabilizeState):
    """Use the existing Sweep accounting from reset, without sampling a Gate."""

    def __init__(self,count,device):
        super().__init__(count,device)
        self.achieved.fill_(True)
        self.reward_phase.fill_(True)

    def reset(self,ids,center,quat,goal):
        super().reset(ids,center,quat,goal)
        self.achieved[ids]=True
        self.reward_phase[ids]=True

    def sample(self,center,quat,goal,force,moment,load,valid,linear_speed,angular_speed):
        # Contact force/squeeze accumulation remains in ContactSamples. Only the
        # stabilization score, initial-pose costs and Gate calculation are absent.
        self.samples+=1
