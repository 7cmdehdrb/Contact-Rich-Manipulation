"""Audit exact reward traces and compare first episodes on a common scene bank."""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def norm(x):
    return np.linalg.norm(x,axis=-1)


def wilson(k,n):
    z=1.95996398454;p=k/n
    c=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(c-h),float(c+h)]


def components(a,dt):
    palm,dorsal=a['palm_load'],a['dorsal_load'];total=palm+dorsal
    preferred=np.where(a['region']<=2,dorsal,palm)
    side=np.where(total>1e-12,(1.5*preferred+.7*(total-preferred))/np.maximum(total,1e-12),1.)
    reach_gap=norm(a['eef']-a['target'])
    out=dict(sweep_progress=a['positive']/dt*side,sweep_retreat=a['retreat']/dt,
        reaching=np.clip(1-np.maximum(reach_gap-.1,0)/.35,0,1),
        arm_action_rate=((a['action'][...,:6]-a['previous_action'][...,:6])**2).sum(-1),
        hand_action_rate=((a['action'][...,6:]-a['previous_action'][...,6:])**2).sum(-1),
        arm_joint_velocity=np.minimum((a['arm_joint_velocities']**2).sum(-1),100),
        hand_joint_velocity=np.minimum((a['hand_joint_velocities']**2).sum(-1),100),
        shelf_collision=a['shelf_moved_proxy'].astype(float)+a['shelf_near_eef']*(np.clip(1-a['finger_height']/.02,0,1).sum(-1)+np.clip(1-a['shelf_penalty_wrist_height']/.08,0,1)),
        shelf_failure=((a['shelf_velocity']>.1)|(a['wrist_clearance']<.03)).astype(float)/dt,
        object_tilt=np.where(a['active_objects'],np.clip((a['object_tilt']-np.pi/36)/(.9-np.pi/36),0,1)**2,0).sum(-1),
        obstacle_displacement=np.minimum(np.where(a['active_objects'][...,1:],a['obstacle_displacement_squared'][...,1:],0).sum(-1),.04))
    excess=np.maximum(norm(a['obstacle_force_w'])-50,0)
    out['obstacle_contact']=np.where(a['sensor_fresh'],.5*np.expm1(2*np.minimum(excess,.5*np.log(201))),0)
    force_change=np.abs(norm(a['wrench_current'][...,:3])-norm(a['wrench_previous'][...,:3]))
    moment_change=np.abs(norm(a['wrench_current'][...,3:])-norm(a['wrench_previous'][...,3:]))
    severity=np.clip(np.maximum((force_change-30)/45,(moment_change-3)/4.5),0,1)
    out['wrench_change']=np.where(a['wrench_previous_valid']&a['sensor_fresh'],np.expm1(3*severity)/np.expm1(3),0)
    arm=a['action'][...,:6];hand=a['action'][...,6:]
    widths=np.array([.5864,1.4381,1.4381,1.4381,1.4381,1.1641])
    out['arm_action_limit']=np.mean(((arm-np.clip(arm,-np.pi/2,np.pi/2))/np.pi)**2,-1)
    out['hand_action_limit']=np.mean(((hand-np.clip(hand,0,widths))/widths)**2,-1)
    align=(a['force'][...,:2]*(a['goal'][...,:2]-a['target'][...,:2])).sum(-1)/np.maximum(norm(a['force'])*a['distance'],1e-12)
    gates=dict(reach_plateau=reach_gap<=.1,reach_zero=reach_gap>=.45,
        contact=a['contact'],recognized=a['contact_recognized'],height_pass=a['eef'][...,2]<1.17,
        force_xy_pass=norm(a['force'][...,:2])>=1,force_alignment_positive=align>0,
        surface_classified=total>=.01,preferred_surface_dominant=preferred>total-preferred,
        obstacle_force_over50=norm(a['obstacle_force_w'])>50,
        obstacle_force_saturated=out['obstacle_contact']>=99.999,
        wrench_over_free=(force_change>30)|(moment_change>3),wrench_saturated=severity>=1,
        tilt_over5deg=(np.where(a['active_objects'],a['object_tilt'],0)>np.pi/36).any(-1),
        displacement_saturated=out['obstacle_displacement']>=.04,
        arm_action_clipped=(np.abs(arm)>np.pi/2).any(-1),hand_action_clipped=((hand<0)|(hand>widths)).any(-1))
    return out,gates


def classify(a):
    goal=a['distance']<.03;contact=a['contact']&a['valid']
    moving=norm(a['velocity'][...,:2])>=.005
    eef_moving=norm(a['eef_velocity_w'])>=.01
    phase=np.full(goal.shape,4,dtype=np.int8)
    phase[~goal&~contact&~moving&~eef_moving]=3
    phase[~goal&contact&~moving]=2
    phase[~goal&contact&moving]=1
    phase[goal]=0
    return phase


def function_figure(output):
    dt=.02;fig,axes=plt.subplots(3,4,figsize=(17,11),constrained_layout=True)
    axes=axes.flat
    x=np.linspace(0,.5,501);axes[0].plot(x*100,.8*dt*np.clip(1-np.maximum(x-.1,0)/.35,0,1));axes[0].set(xlabel='EEF-target gap (cm)',title='Reach: continuous proximity reward')
    x=np.linspace(-.02,.02,401)
    for c in [1.5,1,.7]:axes[1].plot(x*100,50*c*np.maximum(x,0)-10*np.maximum(-x,0),label=f'surface {c}')
    axes[1].legend();axes[1].set(xlabel='Distance reduction per step (cm)',title='Sweep: contact+alignment=1')
    x=np.linspace(0,3,301);axes[2].plot(x,np.where(x>=1,.5,0));axes[2].set(xlabel='XY resultant force (N)',title='1cm progress, side=1: 1N gate')
    x=np.linspace(0,180,361)
    for vertical in [0,1,2]:axes[3].plot(x,np.maximum(np.cos(np.deg2rad(x)),0)/np.sqrt(1+vertical**2),label=f'Fz={vertical}N')
    axes[3].legend();axes[3].set(xlabel='XY force-goal angle (deg)',title='Alignment, horizontal force=1N')
    x=np.linspace(45,55,501);axes[4].plot(x,-.04*dt*.5*np.expm1(2*np.clip(x-50,0,.5*np.log(201))));axes[4].set(xlabel='Obstacle resultant force (N)',title='Obstacle force: 50N free, 52.65N cap')
    x=np.linspace(0,90,501);axes[5].plot(x,-.1*dt*np.expm1(3*np.clip((x-30)/45,0,1))/np.expm1(3));axes[5].set(xlabel='Force magnitude change / step (N)',title='Wrench change: 30N free, 75N cap')
    x=np.linspace(0,.95,501);axes[6].plot(np.rad2deg(x),-30*dt*np.clip((x-np.pi/36)/(.9-np.pi/36),0,1)**2);axes[6].set(xlabel='One active object tilt (deg)',title='Tilt: 5deg free, 51.57deg cap')
    x=np.linspace(0,.25,501);axes[7].plot(x*100,-150*dt*np.minimum(x*x,.04));axes[7].set(xlabel='One obstacle 3D displacement (cm)',title='Displacement: no free band, cap sum=.04m²')
    x=np.linspace(0,3,301);axes[8].plot(x,-.03*dt*6*x*x,label='arm');axes[8].plot(x,-.03*dt*6*x*x,ls='--',label='hand');axes[8].legend();axes[8].set(xlabel='Action change in all 6 channels',title='Action change: raw squared sum')
    x=np.linspace(0,3,301);axes[9].plot(x,-.03*dt*np.minimum(6*x*x,100),label='arm');axes[9].plot(x,-.01*dt*np.minimum(12*x*x,100),label='hand 12 joints');axes[9].axvline(1,color='red',ls=':');axes[9].legend();axes[9].set(xlabel='Equal joint velocity (rad/s)',title='Velocity cost; arm termination >1rad/s')
    x=np.linspace(0,2,301);axes[10].plot(x,-.03*dt*x*x);axes[10].set(xlabel='Excess / allowed width in all 6 channels',title='Action limit: normalized excess mean²')
    x=np.linspace(-.02,.12,301);axes[11].plot(x*100,-.5*dt*np.clip(1-x/.08,0,1),label='wrist');axes[11].plot(x*100,-.5*dt*np.clip(1-x/.02,0,1),label='one finger');axes[11].legend();axes[11].set(xlabel='Height above shelf+1.06 (cm)',title='Shelf proximity, if EEF near shelf center')
    for ax in axes:
        ax.grid(alpha=.2);ax.set_ylabel('Applied reward / policy step',fontsize=9)
        ax.set_title('\n'.join(textwrap.wrap(ax.get_title(),40)),fontsize=10)
        ax.tick_params(labelsize=9)
    axes[3].set_ylabel('Dimensionless alignment factor')
    fig.suptitle('Current reward functions | policy dt=0.02s | conditions shown per panel',fontsize=16)
    fig.savefig(output/'reward_function_shapes.png',dpi=150);plt.close(fig)


def timeline(a,ids,e,meta,path,title):
    names=meta['reward_terms'];dt=meta['dt'];t=a['step'][ids,e]*dt
    reward=a['reward_contribution'][ids,e];cumulative=np.cumsum(reward,axis=0)
    fig,axes=plt.subplots(5,1,figsize=(14,15),sharex=True,constrained_layout=True)
    ax=axes[0];ax.plot(t,a['distance'][ids,e]*100,label='Target-goal');ax.plot(t,norm(a['eef'][ids,e]-a['target'][ids,e])*100,label='EEF-target');ax.axhline(3,color='green',ls='--');ax.axhline(10,color='gray',ls=':');ax.set(ylabel='Distance (cm)');ax.legend(ncol=2)
    ax=axes[1];ax.plot(t,norm(a['velocity'][ids,e,:2])*100,label='Target XY cm/s');ax.plot(t,norm(a['eef_velocity_w'][ids,e])*100,label='EEF cm/s');ax2=ax.twinx();ax2.plot(t,norm(a['force'][ids,e]),color='red',alpha=.5,label='Target force N');ax.set(ylabel='Speed (cm/s)');ax2.set_ylabel('Force (N)');ax.legend(loc='upper left')
    ax=axes[2]
    for name in ['sweep_progress','sweep_retreat','reaching','arm_action_rate','shelf_collision','obstacle_displacement']:
        ax.plot(t,reward[:,names.index(name)],label=name)
    ax.plot(t,a['reward_total'][ids,e],color='black',lw=1,alpha=.5,label='TOTAL');ax.set(ylabel='Applied reward / step');ax.legend(ncol=4,fontsize=8)
    ax=axes[3]
    for i,name in enumerate(names):
        if np.max(abs(cumulative[:,i]))>1e-5:ax.plot(t,cumulative[:,i],label=name)
    ax.plot(t,cumulative.sum(-1),color='black',lw=2,label='TOTAL');ax.set(ylabel='Cumulative reward');ax.legend(ncol=4,fontsize=8)
    ax=axes[4]
    gates=[a['contact'][ids,e],a['contact_recognized'][ids,e],norm(a['force'][ids,e,:2])>=1,a['palm_load'][ids,e]+a['dorsal_load'][ids,e]>=.01,a['distance'][ids,e]<.03]
    ax.imshow(np.stack(gates),aspect='auto',origin='lower',extent=[t[0],t[-1],-.5,4.5],interpolation='nearest',cmap='Greys',vmin=0,vmax=1)
    ax.set(yticks=range(5),yticklabels=['Contact','History+height','XY force >=1N','Surface classified','Within goal'],xlabel='Episode time (s)')
    for ax in axes[:4]:ax.grid(alpha=.2)
    fig.suptitle(title,fontsize=14);fig.savefig(path,dpi=145);plt.close(fig)
    # Preserve all 15 raw and applied channels, including ones omitted from the overview.
    fig,axes=plt.subplots(5,3,figsize=(15,13),sharex=True,constrained_layout=True)
    for i,(ax,name) in enumerate(zip(axes.flat,names)):
        ax.plot(t,reward[:,i],label='Applied / step');ax2=ax.twinx();ax2.plot(t,a['reward_raw'][ids,e,i],color='#b26231',alpha=.45,label='Raw');ax.set(title=name,ylabel='Applied');ax2.set_ylabel('Raw',color='#b26231');ax.grid(alpha=.2)
    fig.suptitle(title+' | all raw/applied terms');fig.savefig(path.with_name(path.stem+'_all_terms.png'),dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    episodes=[];term_stats=[];phase_stats=[];gate_stats=[];summaries=[];data={};baseline_hash=None
    phase_names=['within_goal','moving_contact','stationary_contact_outside','stationary_no_contact_outside','approach_or_free_motion']
    for run in sorted(args.root.glob('iteration_*')):
        if not run.is_dir() or not (run/'metadata.json').exists():continue
        meta=json.loads((run/'metadata.json').read_text());a=dict(np.load(run/'trace.npz'))
        if meta['status']!='complete':raise ValueError(f'Incomplete {run}')
        for name,v in a.items():
            if np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all():raise ValueError(f'Nonfinite {name}')
        bank=json.loads((run/'initial_scene_hashes.json').read_text())
        if baseline_hash is None:baseline_hash=bank
        if bank!=baseline_hash:raise ValueError(f'Initial scene/observation bank differs: {run}')
        iteration=meta['checkpoint_iteration'];names=meta['reward_terms'];dt=meta['dt'];mask=a['episode']==0
        raw_check,gates=components(a,dt)
        raw_errors={name:float(np.max(abs(a['reward_raw'][...,i]-raw_check[name])[mask])) for i,name in enumerate(names)}
        applied_errors={name:float(np.max(abs((a['reward_raw'][...,i]-raw_check[name])*a['reward_weights'][...,i]*dt)[mask])) for i,name in enumerate(names)}
        sum_error=float(np.max(abs(a['reward_contribution'].sum(-1)-a['reward_total'])))
        manager_error=float(np.max(abs(a['reward_weighted_rate']*dt-a['reward_contribution'])))
        # The captured shelf proxy uses runtime's initial shelf pose; the real
        # class uses default_root_state+env_origins. Their binary 5mm gates can
        # differ near threshold. Keep the exact callable value authoritative,
        # expose this mismatch, and do not call that term independently verified.
        shelf_i=names.index('shelf_collision')
        proximity=raw_check['shelf_collision']-a['shelf_moved_proxy']
        observed_shelf_moved=a['reward_raw'][...,shelf_i]-proximity
        shelf_residual_error=float(np.max(np.minimum(abs(observed_shelf_moved),abs(observed_shelf_moved-1))[mask]))
        shelf_proxy_mismatch=int((abs(observed_shelf_moved-a['shelf_moved_proxy'])>.5)[mask].sum())
        if sum_error>1e-5 or manager_error>1e-5 or shelf_residual_error>1e-5 or max(v for n,v in applied_errors.items() if n!='shelf_collision')>1e-4:raise ValueError(f'Reward reconstruction failed {run}: {applied_errors}')
        phase=classify(a);returns=[];successes=0;failure={};selected=[]
        np.savez_compressed(out/(run.name+'_derived.npz'),phase=phase,initial_episode_mask=mask,**gates)
        for e in range(meta['num_envs']):
            ids=np.flatnonzero(mask[:,e])
            if not len(ids) or not a['done'][ids[-1],e]:raise ValueError(f'Initial episode unfinished {run}, {e}')
            success=bool(a['success'][ids[-1],e]);successes+=success
            if success:cause='success'
            elif a['object_fall_or_tilt'][ids[-1],e]:cause='object_fall_or_tilt'
            elif a['shelf_failure'][ids[-1],e]:cause='shelf_failure'
            elif a['arm_joint_velocity'][ids[-1],e]:cause='arm_velocity'
            elif a['distance'][ids,e].min()<.03:cause='timeout_left_goal'
            else:cause='timeout_never_entered'
            failure[cause]=failure.get(cause,0)+1
            total=a['reward_contribution'][ids,e].sum(0);ret=float(total.sum());returns.append(ret)
            late=ids[-min(100,len(ids)):];idle=(a['distance'][late,e]>=.03)&(norm(a['velocity'][late,e,:2])<.005)
            reward=a['reward_contribution'][late,e]
            row=dict(iteration=iteration,env=e,region=int(a['region'][ids[0],e]),success=success,cause=cause,
                duration_s=len(ids)*dt,initial_distance_m=float(norm(a['initial_goal'][0,e,:2]-a['initial_target'][0,e,:2])),
                final_distance_m=float(a['distance'][ids[-1],e]),min_distance_m=float(a['distance'][ids,e].min()),
                return_total=ret,late_idle_outside_seconds=float(idle.sum()*dt),
                late_idle_outside_return=float(reward[idle].sum()),
                late_idle_outside_reach=float(reward[idle,names.index('reaching')].sum()),
                forward_distance_m=float(np.maximum(a['delta_distance'][ids,e],0).sum()),
                backward_distance_m=float(np.maximum(-a['delta_distance'][ids,e],0).sum()))
            for i,name in enumerate(names):row['return_'+name]=float(total[i])
            episodes.append(row)
            if iteration==15800:selected.append((row,ids,e))
        for i,name in enumerate(names):
            raw=a['reward_raw'][...,i][mask];value=a['reward_contribution'][...,i][mask]
            term_stats.append(dict(iteration=iteration,term=name,weight=meta['reward_weights'][i],active_fraction=float((abs(value)>1e-9).mean()),
                mean_episode_return=float(np.mean([r['return_'+name] for r in episodes if r['iteration']==iteration])),
                raw_p50=float(np.quantile(raw,.5)),raw_p95=float(np.quantile(raw,.95)),raw_p99=float(np.quantile(raw,.99)),raw_max=float(raw.max()),
                applied_p01=float(np.quantile(value,.01)),applied_p50=float(np.quantile(value,.5)),applied_p99=float(np.quantile(value,.99)),
                raw_reconstruction_max_error=raw_errors[name],applied_reconstruction_max_error=applied_errors[name]))
            for pi,pname in enumerate(phase_names):
                v=a['reward_contribution'][...,i][mask&(phase==pi)]
                phase_stats.append(dict(iteration=iteration,phase=pname,term=name,frames=len(v),mean_step_reward=float(v.mean()) if len(v) else None,total=float(v.sum())))
        for name,v in gates.items():gate_stats.append(dict(iteration=iteration,gate=name,fraction=float(v[mask].mean())))
        cohort=[r for r in episodes if r['iteration']==iteration]
        successful_returns=[r['return_total'] for r in cohort if r['success']]
        failed_returns=[r['return_total'] for r in cohort if not r['success']]
        summary=dict(iteration=iteration,episodes=len(cohort),successes=successes,success_rate=successes/len(cohort),success_wilson95=wilson(successes,len(cohort)),
            mean_return=float(np.mean(returns)),return_std=float(np.std(returns,ddof=1)),failures=failure,
            final_distance_median_m=float(np.median([r['final_distance_m'] for r in cohort])),
            mean_reach_return=float(np.mean([r['return_reaching'] for r in cohort])),mean_sweep_return=float(np.mean([r['return_sweep_progress'] for r in cohort])),
            success_return_mean=float(np.mean(successful_returns)) if successful_returns else None,
            failure_return_mean=float(np.mean(failed_returns)) if failed_returns else None,
            failures_at_or_above_success_median_return=int(sum(r>=np.median(successful_returns) for r in failed_returns)) if successful_returns else None,
            outside_idle_positive_episodes=sum(r['late_idle_outside_seconds']>=1 and r['late_idle_outside_return']>0 for r in cohort),
            reward_sum_max_error=sum_error,manager_step_max_error=manager_error,
            shelf_movement_proxy_mismatched_frames=shelf_proxy_mismatch,
            shelf_independent_reconstruction='Proximity inputs verified; movement gate reference differs from class default_root_state. Exact callable raw/applied value retained; not independently verified from this proxy.',
            applied_reconstruction_max_errors=applied_errors,scene_bank_identical=True,checkpoint_sha256=meta['checkpoint_sha256'])
        summaries.append(summary);data[iteration]=(a,meta,phase)
        if iteration==0:
            no_contact=[e for e in range(meta['num_envs']) if not a['contact'][mask[:,e],e].any()]
            if no_contact:
                e=sorted(no_contact,key=lambda e:float(a['distance'][0,e]))[len(no_contact)//2]
                ids=np.flatnonzero(mask[:,e]);timeline(a,ids,e,meta,out/f'00000_scene{e:03d}_no_contact.png',f'Iteration0 | scene{e} | no target contact')
        if iteration==15800:
            picks=[]
            for cause in sorted({r['cause'] for r,_,_ in selected}):
                candidates=[item for item in selected if item[0]['cause']==cause]
                candidates.sort(key=lambda item:item[0]['final_distance_m']);picks.append(candidates[len(candidates)//2])
            candidates=[item for item in selected if item[0]['region']==3 and not item[0]['success']]
            if candidates:picks.append(sorted(candidates,key=lambda item:item[0]['final_distance_m'])[len(candidates)//2])
            candidates=[item for item in selected if not item[0]['success'] and item[0]['late_idle_outside_seconds']>=1]
            if candidates:picks.append(max(candidates,key=lambda item:item[0]['return_total']))
            picks=list({item[2]:item for item in picks}.values())
            for row,ids,e in picks:
                label=f"15800_scene{e:03d}_{row['cause']}_R{row['region']}"
                timeline(a,ids,e,meta,out/(label+'.png'),label+f" | final {row['final_distance_m']*100:.1f}cm | return {row['return_total']:.3f}")
            write_csv(out/'timeline_representatives.csv',[r for r,_,_ in picks])
    write_csv(out/'episodes.csv',episodes);write_csv(out/'term_statistics.csv',term_stats);write_csv(out/'phase_rewards.csv',phase_stats);write_csv(out/'gate_statistics.csv',gate_stats)
    if len(summaries)<2:raise ValueError('Need at least two completed checkpoints')
    # Same scene IDs across iterations; pairing is about initial state, not policy-dependent later reset draws.
    rng=np.random.default_rng(20261121);reference={r['env']:r for r in episodes if r['iteration']==0}
    for s in summaries:
        cohort=sorted([r for r in episodes if r['iteration']==s['iteration']],key=lambda r:r['env'])
        delta=np.array([float(r['success'])-float(reference[r['env']]['success']) for r in cohort]);boots=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
        s['paired_success_difference_vs0']=float(delta.mean());s['paired_scene_bootstrap95']=np.quantile(boots,[.025,.975]).tolist()
    (out/'summary.json').write_text(json.dumps(dict(checkpoints=summaries,bank_size=len(reference),
        sampling='One completed initial episode per scene per checkpoint. Same physical/observation bank hashes. Original observation noise retained; later noise draws can differ after policy-dependent resets.',
        causality='Checkpoint differences are observational within one training run. Reward causality needs controlled reward/config changes and retraining.'),indent=2),encoding='utf-8')
    write_csv(out/'checkpoint_summary.csv',summaries)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True);x=np.arange(len(summaries));labels=[str(s['iteration']) for s in summaries]
    ax=axes[0,0];p=np.array([s['success_rate'] for s in summaries])*100;lo=np.array([s['success_wilson95'][0] for s in summaries])*100;hi=np.array([s['success_wilson95'][1] for s in summaries])*100
    ax.errorbar(x,p,yerr=[p-lo,hi-p],marker='o',capsize=4);ax.set(xticks=x,xticklabels=labels,ylim=(-2,102),ylabel='Success (%)',title='Same 128 initial scenes | episode Wilson95%')
    ax=axes[0,1]
    sweep=np.array([s['mean_sweep_return'] for s in summaries]);reach=np.array([s['mean_reach_return'] for s in summaries]);total=np.array([s['mean_return'] for s in summaries])
    ax.bar(x,sweep,label='Sweep progress',color='#326b9c');ax.bar(x,reach,bottom=sweep,label='Reach',color='#329b64')
    ax.bar(x,total-sweep-reach,label='All 13 penalties',color='#bf735a')
    ax.plot(x,total,color='black',marker='o',label='TOTAL');ax.set(xticks=x,xticklabels=labels,ylabel='Mean episode return',title='Actual applied contributions');ax.legend(ncol=2,fontsize=9,loc='upper left')
    ax=axes[1,0]
    for region in range(1,5):
        y=[np.mean([r['success'] for r in episodes if r['iteration']==s['iteration'] and r['region']==region])*100 for s in summaries];ax.plot(x,y,marker='o',label=f'Region {region}')
    ax.set(xticks=x,xticklabels=labels,ylim=(-2,102),ylabel='Success (%)',title='By intended target direction');ax.legend()
    ax=axes[1,1];ax.bar(x,[s['outside_idle_positive_episodes'] for s in summaries],color='#b76745');ax.set(xticks=x,xticklabels=labels,ylabel='Initial episodes',title='Last2s: >=1s idle outside goal with positive return')
    for ax in axes.flat:ax.set_xlabel('Training checkpoint iteration');ax.grid(alpha=.2)
    fig.savefig(out/'checkpoint_comparison.png',dpi=150);plt.close(fig)
    # Group timelines: terminated episodes contribute zero further rewards, never reset rewards.
    fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True,constrained_layout=True)
    for s in summaries:
        a,meta,_=data[s['iteration']];m=a['episode']==0;reward=np.where(m,a['reward_total'],0);t=np.arange(1,len(m)+1)*meta['dt']
        axes[0].plot(t,np.cumsum(reward,axis=0).mean(1),label=str(s['iteration']))
        axes[1].plot(t,np.where(m,a['reward_contribution'][...,meta['reward_terms'].index('reaching')],0).mean(1),label=str(s['iteration']))
        axes[2].plot(t,m.sum(1),label=str(s['iteration']))
    axes[0].set(ylabel='Mean cumulative return',title='Same initial scene bank; zero reward after each first termination')
    axes[1].set(ylabel='Mean Reach / step');axes[2].set(ylabel='Initial episodes still active',xlabel='Policy time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(ncol=6)
    fig.savefig(out/'checkpoint_reward_timelines.png',dpi=150);plt.close(fig)
    function_figure(out)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
