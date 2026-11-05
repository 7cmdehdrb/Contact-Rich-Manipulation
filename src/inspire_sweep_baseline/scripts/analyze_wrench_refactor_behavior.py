"""Summarize recorded GUI push traces, including preterminal reward accounting."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
a=np.load(args.directory/'push_trace.npz');meta=json.loads((args.directory/'channels.json').read_text())
phases=list(dict.fromkeys(a['phase'].tolist()));rewards=meta['rewards'];terms=meta['terminations']
report={'phase_summary':{},'termination_first':{},'reward_totals_preterminal':{},'checks':{}}
for side,sl in [('left',slice(0,16)),('right',slice(16,32))]:
    table=[]
    for phase in phases:
        m=a['phase']==phase;force=np.linalg.norm(a['wrench'][m,sl,:3],axis=-1);moment=np.linalg.norm(a['wrench'][m,sl,3:],axis=-1)
        bits=a['tactile'][m,sl];dist=np.linalg.norm(a['position'][m,sl,:2]-a['goal'][m,sl,:2],axis=-1)
        table.append(dict(phase=phase,force_mean=float(force.mean()),force_p95=float(np.quantile(force,.95)),
            moment_mean=float(moment.mean()),moment_p95=float(np.quantile(moment,.95)),
            tactile_any_fraction=float(bits.any(-1).mean()),front_fraction=float(bits[...,:17].any(-1).mean()),
            back_fraction=float(bits[...,17:].any(-1).mean()),channel_fraction=bits.mean((0,1)).tolist(),
            final_distance_mean=float(dist[-1].mean()),within_goal_final=int((dist[-1]<=.03).sum()),
            palm_normal_mean=a['palm_normal'][m,sl].mean((0,1)).tolist() if 'palm_normal' in a else None))
    report['phase_summary'][side]=table
    valid=a['preterminal'][:,sl]
    report['reward_totals_preterminal'][side]={name:float((a['rewards'][:,sl,j]*valid).sum(0).mean()) for j,name in enumerate(rewards)}
for e in range(32):
    hits=np.flatnonzero(a['terms'][:,e].any(-1))
    report['termination_first'][str(e)]=None if len(hits)==0 else dict(time=float(a['time_s'][hits[0]]),phase=str(a['phase'][hits[0]]),terms=[n for j,n in enumerate(terms) if a['terms'][hits[0],e,j]])
dist=np.linalg.norm(a['position'][...,:2]-a['goal'][...,:2],axis=-1)
height=a['eef'][...,2]<1.19
expected=5*.02*np.exp(-dist/.05)*height
report['checks']['distance_reward_max_abs_error']=float(np.abs(a['rewards'][...,rewards.index('target_goal_distance')]-expected).max())
report['checks']['sum_terms_total_max_abs_error']=float(np.abs(a['rewards'].sum(-1)-a['total']).max())
report['checks']['finite_all']=all(np.isfinite(a[k]).all() for k in a.files if a[k].dtype.kind in 'fc')
report['checks']['obstacle_penalties_max_abs']=float(np.abs(a['rewards'][...,[rewards.index('obstacle_contact_penalty'),rewards.index('obstacle_displacement_penalty')]]).max())
# Independently recompute terms from saved physical signals, not reward helpers.
def error(name,expected):
    report['checks'][name+'_max_abs_error']=float(np.abs(a['rewards'][...,rewards.index(name)]-expected).max())
u=np.clip((dist-.025)/.01,0,1)
error('target_goal_hold',.2*(1-3*u*u+2*u*u*u)*height)
history=np.zeros((32,3),bool);count=np.zeros(32,int);miss=np.zeros(32,int);expected=[]
force=np.linalg.norm(a['contact_force'],axis=-1)
for t in range(len(dist)):
    inside=(dist[t]<=.03)&height[t]
    history[:,:2]=history[:,1:].copy();history[:,2]=((force[t]>=.01)&height[t])|inside
    accepted=(inside|(history.sum(-1)>=2))&height[t]
    miss=np.where(accepted,0,miss+1);count=np.minimum(count+accepted,10);count[miss>=5]=0
    expected.append(np.where(accepted,count,0)*.002)
error('target_contact',np.array(expected))
v=a['joint_velocity']
# Historical v1-v7 traces used all 12 hand joints; new traces declare the six drives.
hand_ids=([meta['robot_joint_names'].index(n) for n in meta['hand_velocity_joint_names']]
          if 'hand_velocity_joint_names' in meta else list(range(6,v.shape[-1])))
error('robot_joint_velocity_penalty',-.02*(.03*np.minimum((v[...,:6]**2).sum(-1),100)+.01*np.minimum((v[...,hand_ids]**2).sum(-1),100)))
arm=np.clip(a['applied_torque']/np.array([150,150,150,28,28,28]),-1,1)
arm_delta=np.diff(arm,axis=0,prepend=np.zeros_like(arm[:1]))
hand=np.clip(a['action'][...,-2:],0,1);hand_delta=np.diff(hand,axis=0,prepend=np.asarray(meta.get('initial_previous_action',np.ones((32,8))))[None,...,-2:])
error('robot_action_rate_penalty',-.02*.03*((arm_delta**2).sum(-1)+(hand_delta**2).sum(-1)))
# Reach baseline is reset-state distance; first recorded sample differs by <1e-4m.
rd=np.maximum(np.linalg.norm(a['eef']-a['position'],axis=-1)-.10,0)
initial=np.asarray(meta.get('initial_reach_best',rd[0]))[None]
best=np.minimum.accumulate(np.concatenate([initial,rd]),axis=0)
expected_reach=30*np.maximum(0,-np.diff(best,axis=0))
if 'initial_reach_best' in meta:
    error('target_reaching',expected_reach)
else:
    report['checks']['target_reaching_after_first_step_max_abs_error']=float(np.abs(a['rewards'][1:,:,rewards.index('target_reaching')]-expected_reach[1:]).max())
report['checks']['terminal_success_reward_preterminal_per_env']=(a['rewards'][...,rewards.index('target_goal_success_bonus')]*a['preterminal']).sum(0).tolist()
report['checks']['unsafe_termination_seen']=bool(a['terms'][...,:3].any())
report['checks']['arm_speed_peak_rad_s']=float(np.abs(a['joint_velocity'][...,:6]).max())
report['checks']['command_frame_max_error_m']=meta.get('command_frame_max_error_m')
if 'hand_bounds' in a:
    inside=np.isin(a['phase'],['enter_horizontal','approach_contact','push','hold'])
    report['checks']['hand_height_during_entry_and_push_m']=[float(a['hand_bounds'][inside,:,0,2].min()),float(a['hand_bounds'][inside,:,1,2].max())]
entry=a['rewards'][...,rewards.index('target_goal_first_entry_bonus')]
report['checks']['entry_totals_per_env']=entry.sum(0).tolist()
report['checks']['success_seen_per_env']=a['terms'][...,terms.index('target_goal_success')].any(0).tolist()
(args.directory/'summary.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True)
for col,sl in enumerate((slice(0,16),slice(16,32))):
    t=a['time_s']; axes[0,col].plot(t,dist[:,sl]);axes[0,col].axhline(.03,color='black',ls='--');axes[0,col].set_title(('Left/palm','Right/back')[col]);axes[0,col].set_ylabel('Goal distance (m)')
    axes[1,col].plot(t,np.linalg.norm(a['wrench'][:,sl,:3],axis=-1));axes[1,col].set_ylabel('Pure force (N)')
    axes[2,col].plot(t,np.linalg.norm(a['wrench'][:,sl,3:],axis=-1));axes[2,col].set_ylabel('Pure moment (Nm)')
    axes[3,col].imshow(a['tactile'][:,sl].mean(1).T,aspect='auto',origin='lower',extent=[t[0],t[-1],0,34],vmin=0,vmax=1);axes[3,col].axhline(17,color='white');axes[3,col].set_ylabel('Tactile channel');axes[3,col].set_xlabel('Time (s)')
    for ax in axes[:,col]:
        for phase in phases[1:]:ax.axvline(t[np.flatnonzero(a['phase']==phase)[0]],color='gray',lw=.5)
fig.tight_layout();fig.savefig(args.directory/'signals.png',dpi=150)
print(json.dumps(report['checks'],indent=2))
