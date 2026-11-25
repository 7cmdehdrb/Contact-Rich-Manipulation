"""Audit trained reward logs and terminal-inclusive Play traces without Sim imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS = [('v0_1',13850),('v0_2',15200),('v0_3',7650)]
COLORS = ['#4878a8','#d98b31','#287f61']


def stats(values):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    if not len(x):
        return {'n':0}
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)),max=float(x.max()))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)


def ratio(a,b):
    return float(a/b) if b>1e-12 else None


def training(root):
    summaries={}; series={}; rows=[]
    for task,checkpoint in TASKS:
        raw=json.loads((root/f'{task}_training_raw.json').read_text())
        rewards={k.split('/',1)[1]:v for k,v in raw.items() if k.startswith('Episode_Reward/')}
        ref={v['step']:v['value'] for v in rewards['obstacle_displacement'] if v['step']<=checkpoint}
        last=max(ref); first=max(0,last-999)
        steps=np.array(sorted(ref))
        arrays={}
        for name,values in rewards.items():
            lookup={v['step']:v['value'] for v in values}
            arrays[name]=np.array([lookup.get(int(t),np.nan) for t in steps])
            assert np.isfinite(arrays[name]).all(),(task,name)
        positive=sum(np.maximum(v,0) for v in arrays.values())
        negative=sum(np.maximum(-v,0) for v in arrays.values())
        net=sum(arrays.values())
        selected=steps>=first
        cost=-arrays['obstacle_displacement']
        summaries[task]=dict(checkpoint_iteration=checkpoint,log_last_step=max(v['step'] for v in raw['Episode_Reward/obstacle_displacement']),
            window_first=int(steps[selected][0]),window_last=int(steps[-1]),window_points=int(selected.sum()),
            logged_displacement_per_second=float(cost[selected].mean()),
            episode_equivalent_displacement_cost=float(10*cost[selected].mean()),
            episode_equivalent_positive=float(10*positive[selected].mean()),
            episode_equivalent_total=float(10*net[selected].mean()),
            displacement_over_positive=ratio(cost[selected].sum(),positive[selected].sum()),
            displacement_over_all_negative=ratio(cost[selected].sum(),negative[selected].sum()),
            contact_cost_nonzero_points=int((arrays['obstacle_contact'][selected]<-1e-12).sum()),
            term_episode_equivalents={k:float(10*v[selected].mean()) for k,v in arrays.items()})
        for i,step in enumerate(steps):
            rows.append(dict(task=task,iteration=int(step),displacement_logged=float(cost[i]),
                displacement_episode_equivalent=float(10*cost[i]),positive_episode_equivalent=float(10*positive[i]),
                total_episode_equivalent=float(10*net[i])))
        series[task]=(steps,cost,positive)
    write_csv(root/'training_curves.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for (task,_),color in zip(TASKS,COLORS):
        steps,cost,pos=series[task]
        window=250
        sm=lambda x:np.convolve(x,np.ones(window)/window,mode='valid')
        axes[0].plot(steps[window-1:],10*sm(cost),label=task,color=color)
        axes[1].plot(steps[window-1:],100*sm(cost)/np.maximum(sm(pos),1e-10),label=task,color=color)
    axes[0].set(ylabel='Obstacle displacement cost (episode-equivalent)',xlabel='Training iteration',ylim=(0,None))
    axes[1].set(ylabel='Displacement / positive reward terms (%)',xlabel='Training iteration',ylim=(0,None))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Training logs: 250-iteration moving mean; clipped at evaluated checkpoints')
    fig.tight_layout();fig.savefig(root/'figures/training.png');plt.close(fig)
    return summaries


def episode_summary(rows):
    with_obstacles=[r for r in rows if r['obstacle_count']>0]
    moved=[r for r in with_obstacles if r['maximum_displacement_m']>=.01]
    def group(x):
        return dict(n=len(x),successes=sum(r['success'] for r in x),
            duration_s=stats([r['duration_s'] for r in x]),
            maximum_displacement_m=stats([r['maximum_displacement_m'] for r in x]),
            final_maximum_displacement_m=stats([r['final_maximum_displacement_m'] for r in x]),
            total_path_m=stats([r['obstacle_path_sum_m'] for r in x]),
            displacement_cost=stats([r['displacement_cost'] for r in x]),
            positive_reward=stats([r['positive_reward'] for r in x]),
            total_reward=stats([r['total_reward'] for r in x]),
            obstacle_contact_cost=stats([r['obstacle_contact_cost'] for r in x]),
            cost_over_positive=ratio(sum(r['displacement_cost'] for r in x),sum(r['positive_reward'] for r in x)),
            individual_cost_over_positive=stats([r['displacement_cost']/r['positive_reward'] for r in x if r['positive_reward']>1e-12]),
            cost_under_10pct_positive=sum(r['displacement_cost']<.1*r['positive_reward'] for r in x),
            positive_net_episodes=sum(r['total_reward']>0 for r in x),
            cost_over_negative=ratio(sum(r['displacement_cost'] for r in x),sum(r['negative_reward'] for r in x)),
            discounted_cost=stats([r['discounted_displacement_cost'] for r in x]),
            initial_first_second_positive=stats([r['first_second_positive'] for r in x]),
            first_crossing_s=stats([r['first_1cm_s'] for r in x if r['first_1cm_s']>=0]),
            crossing_to_end_s=stats([r['duration_s']-r['first_1cm_s'] for r in x if r['first_1cm_s']>=0]),
            cost_after_crossing_discounted=stats([r['cost_after_1cm_discounted'] for r in x if r['first_1cm_s']>=0]),
            thresholds={str(m):dict(count=sum(r['maximum_displacement_m']>=m for r in x),
                                    denominator=len(x),fraction=ratio(sum(r['maximum_displacement_m']>=m for r in x),len(x)))
                        for m in (.001,.005,.01,.02,.05,.10,.20)})
    return {'all':group(rows),'with_obstacles':group(with_obstacles),'moved_1cm':group(moved),
            'moved_5cm':group([r for r in moved if r['maximum_displacement_m']>=.05]),
            'success_moved_1cm':group([r for r in moved if r['success']]),
            'success_with_obstacles':group([r for r in with_obstacles if r['success']]),
            'failure_with_obstacles':group([r for r in with_obstacles if not r['success']]),
            'short_moved_1cm':group([r for r in moved if r['duration_s']<9.99]),
            'long_moved_1cm':group([r for r in moved if r['duration_s']>=9.99])}


def play(root,folder):
    meta=json.loads((folder/'metadata.json').read_text())
    assert meta['status']=='complete',folder
    with np.load(folder/'trace.npz') as z:
        d={k:z[k] for k in ('episode','step','reward_terms','reward_total','reward_raw','object_positions',
            'initial_object_positions','active_objects','object_velocity','target','goal','contact_recognized',
            'shelf_initial_position','done')}
        d.update({k:z[k] for k in z.files if k.startswith('termination_')})
    names=meta['reward_terms']; dt=meta['dt']; gamma=.98
    oi=names.index('obstacle_displacement'); ci=names.index('obstacle_contact')
    assert meta['reward_weights'][oi]==-150. and meta['reward_params'][oi]['displacement_max_cost']==.04
    assert np.isfinite(d['reward_total']).all()
    mask=d['active_objects'].copy();mask[...,0]=False
    difference=d['object_positions'].astype(np.float64)-d['initial_object_positions'].astype(np.float64)
    squared=np.square(difference).sum(-1)
    raw=np.minimum(np.where(mask,squared,0.).sum(-1),.04)
    err=float(np.abs(-150.*dt*raw-d['reward_terms'][...,oi]).max())
    raw_error=float(np.abs(raw-d['reward_raw'][...,oi]).max())
    assert err<2e-6 and raw_error<2e-6,(err,raw_error)
    total_error=float(np.abs(d['reward_terms'].sum(-1)-d['reward_total']).max())
    assert total_error<2e-5
    n=meta['num_envs'];limit=meta['episodes_per_env']; maximum_steps=round(10/dt)
    all_rows=[]; objects=[]; timeline_cost=[];timeline_positive=[]; alive=[]; representatives=[]
    for env in range(n):
        for episode in range(limit):
            idx=np.flatnonzero(d['episode'][:,env]==episode)
            assert len(idx) and d['done'][idx[-1],env] and d['step'][idx[0],env]==1,(env,episode)
            assert np.array_equal(d['step'][idx,env],np.arange(1,len(idx)+1))
            active=mask[idx[0],env]
            assert np.array_equal(mask[idx,env],np.broadcast_to(active,(len(idx),17)))
            initial=d['initial_object_positions'][idx[0],env]
            assert np.all(d['initial_object_positions'][idx,env]==initial)
            term=d['reward_terms'][idx,env].astype(float)
            pos=np.maximum(term,0).sum(-1); neg=np.maximum(-term,0).sum(-1)
            cost=-term[:,oi]
            per_obj=np.sqrt(squared[idx,env])[:,active]
            disp_max=per_obj.max(-1) if active.any() else np.zeros(len(idx))
            crossing=np.flatnonzero(disp_max>=.01)
            first=int(crossing[0]) if len(crossing) else -1
            actual_pos=d['object_positions'][idx,env].astype(float)
            path=np.linalg.norm(np.diff(np.concatenate((initial[None],actual_pos)),axis=0),axis=-1)
            totals=term.sum(0)
            last=idx[-1]; count=int(active.sum());duration=len(idx)*dt
            record=dict(task=meta['task'].replace('Inspire-Sweep-Prof-Revision-',''),seed=meta['seed'],env=env,episode=episode,
                steps=len(idx),duration_s=duration,obstacle_count=count,
                success=bool(d['termination_success'][last,env]),
                maximum_displacement_m=float(disp_max.max()),final_maximum_displacement_m=float(disp_max[-1]),
                summed_peak_displacement_m=float(per_obj.max(0).sum()) if count else 0.,
                summed_peak_excess_5mm_m=float(np.maximum(per_obj.max(0)-.005,0).sum()) if count else 0.,
                maximum_xy_displacement_m=float(np.linalg.norm(difference[idx,env,:,:2][:,active],axis=-1).max()) if count else 0.,
                maximum_z_displacement_m=float(np.abs(difference[idx,env,:,2][:,active]).max()) if count else 0.,
                obstacle_path_sum_m=float(path[:,active].sum()),
                displacement_cost=float(cost.sum()),obstacle_contact_cost=float(-totals[ci]),
                positive_reward=float(pos.sum()),negative_reward=float(neg.sum()),total_reward=float(term.sum()),
                discounted_displacement_cost=float((cost*gamma**np.arange(len(idx))).sum()),
                discounted_positive_reward=float((pos*gamma**np.arange(len(idx))).sum()),
                first_second_positive=float(pos[:round(1/dt)].sum()),
                first_1cm_s=(first+1)*dt if first>=0 else -1.,
                cost_after_1cm_discounted=float((cost[first:]*gamma**np.arange(len(idx)-first)).sum()) if first>=0 else 0.,
                cost_first_second_after_1cm=float(cost[first:first+round(1/dt)].sum()) if first>=0 else 0.,
                hypothetical_frozen_final_tail_cost=float(150.*raw[last,env]*(10-duration)),
                cap_steps=int((raw[idx,env]>=.04-1e-8).sum()),
                contact_penalty_steps=int((term[:,ci]<-1e-12).sum()))
            for name in names:record['reward_'+name]=float(totals[names.index(name)])
            for name in d:
                if name.startswith('termination_'):record[name]=bool(d[name][last,env])
            all_rows.append(record)
            for slot in np.flatnonzero(active):
                dis=np.sqrt(squared[idx,env,slot]); xy=np.linalg.norm(difference[idx,env,slot,:2],axis=-1)
                objects.append(dict(task=record['task'],seed=meta['seed'],env=env,episode=episode,slot=int(slot),
                    maximum_displacement_m=float(dis.max()),final_displacement_m=float(dis[-1]),
                    maximum_xy_displacement_m=float(xy.max()),path_length_m=float(path[:,slot].sum()),
                    position_quadratic_integral_m2_s=float((dis**2).sum()*dt),
                    pre_cap_displacement_cost=float(150*(dis**2).sum()*dt),duration_s=duration))
            for values,storage in ((cost,timeline_cost),(pos,timeline_positive)):
                cumulative=np.cumsum(values)
                storage.append(np.pad(cumulative,(0,maximum_steps-len(values)),mode='edge'))
            alive.append(np.arange(maximum_steps)<len(idx))
    summary=episode_summary(all_rows)
    summary['term_means']={name:float(np.mean([r['reward_'+name] for r in all_rows])) for name in names}
    summary['termination_counts']={name:sum(r[name] for r in all_rows) for name in all_rows[0] if name.startswith('termination_')}
    summary['cap_steps_total']=sum(r['cap_steps'] for r in all_rows)
    summary['contact_penalty_steps_total']=sum(r['contact_penalty_steps'] for r in all_rows)
    summary.update(object_distribution=stats([r['maximum_displacement_m'] for r in objects]),
        checkpoint=meta['checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],
        quality=dict(max_displacement_formula_error=err,max_raw_formula_error=raw_error,max_reward_sum_error=total_error,
                     complete_episodes=len(all_rows),eligible_samples=sum(r['steps'] for r in all_rows),
                     finite_positions=bool(np.isfinite(d['object_positions']).all())))
    assert len(all_rows)==n*limit
    timelines={'time_s':(np.arange(maximum_steps)+1)*dt,'cost':np.mean(timeline_cost,0),
               'positive':np.mean(timeline_positive,0),'alive_fraction':np.mean(alive,0)}
    # Highest cost and largest motion with an unusually short cost exposure.
    choices=[max(all_rows,key=lambda r:r['displacement_cost'])]
    moved=[r for r in all_rows if r['maximum_displacement_m']>=.01]
    if moved:choices.append(min(moved,key=lambda r:r['displacement_cost']/max(r['maximum_displacement_m']**2,1e-8)))
    successful=[r for r in moved if r['success']]
    if successful:choices.append(max(successful,key=lambda r:r['maximum_displacement_m']))
    if successful:choices.append(min(successful,key=lambda r:r['displacement_cost']/max(r['positive_reward'],1e-12)))
    seen=set()
    for choice in choices:
        env,episode=choice['env'],choice['episode']
        if (env,episode) in seen:continue
        seen.add((env,episode));representatives.append(choice)
        idx=np.flatnonzero(d['episode'][:,env]==episode);t=(np.arange(len(idx))+1)*dt
        active=mask[idx[0],env]; per_obj=np.sqrt(squared[idx,env])[:,active]
        terms=d['reward_terms'][idx,env]
        fig,axes=plt.subplots(1,3,figsize=(15,4.1))
        for i,slot in enumerate(np.flatnonzero(active)):
            axes[0].plot(t,100*per_obj[:,i],label=f'Obstacle {slot-1}')
        axes[0].axhline(1,color='black',ls=':',lw=1)
        axes[0].set(xlabel='Episode time (s)',ylabel='Displacement from reset (cm)')
        if active.any():axes[0].legend(fontsize=7)
        axes[1].plot(t,-np.cumsum(terms[:,oi]),label='Displacement cost',color='#bd493c')
        axes[1].plot(t,np.cumsum(np.maximum(terms,0).sum(-1)),label='Positive reward',color='#287f61')
        axes[1].set(xlabel='Episode time (s)',ylabel='Cumulative reward / cost');axes[1].legend(fontsize=8)
        center=d['shelf_initial_position'][idx[0],env,:2]
        for slot in np.flatnonzero(active):
            points=d['object_positions'][idx,env,slot,:2]-center
            axes[2].plot(100*points[:,0],100*points[:,1],lw=1)
            axes[2].scatter(*(100*points[0]),s=20,marker='o');axes[2].scatter(*(100*points[-1]),s=25,marker='x')
        points=d['target'][idx,env,:2]-center
        axes[2].plot(100*points[:,0],100*points[:,1],color='black',lw=2,label='Target')
        axes[2].scatter(*(100*(d['goal'][idx[0],env,:2]-center)),marker='*',s=90,color='black')
        axes[2].set(xlabel='Shelf X (cm)',ylabel='Shelf Y (cm)',xlim=(-22,22),ylim=(-52,52));axes[2].set_aspect('equal')
        fig.suptitle(f"{choice['task']}: env {env}, episode {episode}, T={choice['duration_s']:.2f}s, success={choice['success']}")
        for ax in axes:ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(root/f"figures/{choice['task']}_env{env}_ep{episode}.png");plt.close(fig)
    return summary,all_rows,objects,timelines,representatives


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':110,'savefig.dpi':160})
    summary={'training':training(root),'play':{}}; episodes=[];objects=[];curves={};cases=[]
    for task,_ in TASKS:
        folder=root/f'{task}_seed42'
        if not (folder/'trace.npz').exists():continue
        s,e,o,t,c=play(root,folder)
        summary['play'][task]=s;episodes+=e;objects+=o;curves[task]=t;cases+=c
    write_csv(root/'episodes.csv',episodes);write_csv(root/'obstacles.csv',objects);write_csv(root/'representatives.csv',cases)
    grouped=[];scenarios=[]
    for task,_ in TASKS:
        task_rows=[r for r in episodes if r['task']==task]
        for name,select in [('all',lambda r:True),('moved_1cm',lambda r:r['maximum_displacement_m']>=.01),
                            ('success',lambda r:r['success']),('success_moved_1cm',lambda r:r['success'] and r['maximum_displacement_m']>=.01)]:
            rows=[r for r in task_rows if select(r)]
            if not rows:continue
            for factor in (1,2,5,10):
                scenarios.append(dict(task=task,group=name,n=len(rows),weight_factor=factor,
                    mean_displacement_cost=factor*np.mean([r['displacement_cost'] for r in rows]),
                    mean_total_on_frozen_trajectory=np.mean([r['total_reward']-(factor-1)*r['displacement_cost'] for r in rows]),
                    previously_positive_to_nonpositive=sum(r['total_reward']>0 and r['total_reward']-(factor-1)*r['displacement_cost']<=0 for r in rows),
                    previously_positive=sum(r['total_reward']>0 for r in rows)))
        edges=[0,.001,.005,.01,.02,.05,.10,.20,float('inf')]
        for low,high in zip(edges[:-1],edges[1:]):
            rows=[r for r in task_rows if r['obstacle_count']>0 and low<=r['maximum_displacement_m']<high]
            grouped.append(dict(task=task,bin_low_m=low,bin_high_m=high,n=len(rows),
                mean_cost=np.mean([r['displacement_cost'] for r in rows]) if rows else None,
                mean_positive=np.mean([r['positive_reward'] for r in rows]) if rows else None,
                mean_duration_s=np.mean([r['duration_s'] for r in rows]) if rows else None,
                success_count=sum(r['success'] for r in rows)))
    write_csv(root/'magnitude_bins.csv',grouped);write_csv(root/'fixed_trajectory_weight_scenarios.csv',scenarios)
    counts=[]
    for task,_ in TASKS:
        for number in range(7):
            rows=[r for r in episodes if r['task']==task and r['obstacle_count']==number]
            if not rows:continue
            counts.append(dict(task=task,obstacle_count=number,n=len(rows),successes=sum(r['success'] for r in rows),
                moved_1cm=sum(r['maximum_displacement_m']>=.01 for r in rows),
                mean_cost=np.mean([r['displacement_cost'] for r in rows]),
                mean_positive=np.mean([r['positive_reward'] for r in rows])))
    write_csv(root/'obstacle_count_groups.csv',counts)
    peak_scenarios=[];success_checks=[]
    for task,_ in TASKS:
        rows=[r for r in episodes if r['task']==task]
        if not rows:continue
        for coefficient in (50,100):
            peak_scenarios.append(dict(task=task,k_reward_per_m=coefficient,tolerance_m=.005,n=len(rows),
                mean_extra_peak_cost=coefficient*np.mean([r['summed_peak_excess_5mm_m'] for r in rows]),
                mean_total_on_frozen_trajectory=np.mean([r['total_reward']-coefficient*r['summed_peak_excess_5mm_m'] for r in rows]),
                moved_1cm_mean_extra_peak_cost=coefficient*np.mean([r['summed_peak_excess_5mm_m'] for r in rows if r['maximum_displacement_m']>=.01])))
        for tolerance in (.005,.01,.02,.05):
            success_checks.append(dict(task=task,tolerance_m=tolerance,n=len(rows),
                original_successes=sum(r['success'] for r in rows),
                hypothetical_strict_successes=sum(r['success'] and r['maximum_displacement_m']<=tolerance for r in rows)))
    write_csv(root/'fixed_trajectory_peak_scenarios.csv',peak_scenarios)
    write_csv(root/'hypothetical_success_guardrails.csv',success_checks)
    write_csv(root/'penalty_scale.csv',[dict(displacement_cm=100*d,step_cost=150*d*d*.02,
        cost_per_second=150*d*d,ten_second_cost=150*d*d*10,
        discounted_infinite_cost_gamma_098=150*d*d*.02/(1-.98)) for d in (.001,.005,.01,.02,.05,.10,.20)])
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for (task,_),color in zip(TASKS,COLORS):
        rows=[r for r in episodes if r['task']==task and r['obstacle_count']>0]
        if not rows:continue
        x=np.sort([100*r['maximum_displacement_m'] for r in rows]); y=np.arange(1,len(x)+1)/len(x)*100
        axes[0,0].step(x,y,where='post',label=f'{task} (n={len(x)})',color=color)
        axes[0,1].scatter([100*r['maximum_displacement_m'] for r in rows],[r['displacement_cost'] for r in rows],
                          s=15,alpha=.5,label=task,color=color)
        t=curves[task];axes[1,0].plot(t['time_s'],t['cost'],label=task,color=color)
        axes[1,1].plot(t['time_s'],t['positive'],label=task,color=color)
    axes[0,0].set(xlabel='Maximum single-obstacle displacement / episode (cm)',ylabel='Cumulative episodes (%)',xlim=(0,None),ylim=(0,100))
    axes[0,1].set(xlabel='Maximum displacement (cm)',ylabel='Actual episode displacement cost')
    axes[1,0].set(xlabel='Episode time (s)',ylabel='Mean cumulative displacement cost')
    axes[1,1].set(xlabel='Episode time (s)',ylabel='Mean cumulative positive reward')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Deterministic Play, seed 42; first four COMPLETE episodes per environment\nCumulative curves retain final sums after termination (fixed original cohort)')
    fig.tight_layout();fig.savefig(root/'figures/play_comparison.png');plt.close(fig)
    # Theoretical calibration, separate from observed model behavior.
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    distance=np.linspace(0,.1,201)
    for seconds in (.1,1,5,10):
        axes[0].plot(distance*100,150*distance**2*seconds,label=f'{seconds:g}s maintained')
    axes[0].set(xlabel='One obstacle displaced (cm)',ylabel='Undiscounted cumulative cost')
    remaining=np.arange(1,501)*.02
    for distance in (.01,.02,.05):
        line,=axes[1].plot(remaining,150*distance**2*remaining,label=f'{distance*100:g}cm, undiscounted')
        axes[1].plot(remaining,150*distance**2*.02*(1-.98**np.arange(1,501))/(1-.98),
                      color=line.get_color(),ls='--',label=f'{distance*100:g}cm, gamma=0.98')
    axes[1].set(xlabel='Remaining duration (s)',ylabel='Cumulative cost from movement time')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Formula illustration: constant one-obstacle displacement; not a retrained policy')
    fig.tight_layout();fig.savefig(root/'figures/cost_calibration.png');plt.close(fig)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{'n':v['all']['n'],'with_obstacles':v['with_obstacles']['n'],
        'moved_1cm':v['moved_1cm']['n'],'mean_cost':v['all']['displacement_cost']['mean'],
        'mean_max_cm':100*v['with_obstacles']['maximum_displacement_m']['mean']} for k,v in summary['play'].items()},indent=2))


if __name__=='__main__':main()
