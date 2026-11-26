"""Aggregate complete Prof Revision play episodes and export scientific figures.

No simulator imports and no pandas dependency. Inputs are collector NPZ files.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def distribution(values):
    v=np.asarray(values,dtype=float).ravel(); v=v[np.isfinite(v)]
    if not len(v): return dict(n=0,mean=None,p10=None,median=None,p90=None,p95=None,max=None)
    return dict(n=len(v),mean=float(v.mean()),p10=float(np.quantile(v,.1)),
                median=float(np.median(v)),p90=float(np.quantile(v,.9)),
                p95=float(np.quantile(v,.95)),max=float(v.max()))


def wilson(success, count):
    if not count: return [None,None]
    z=1.95996398454; p=success/count; den=1+z*z/count
    center=(p+z*z/(2*count))/den
    half=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [float(center-half),float(center+half)]


def runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return np.flatnonzero(edges==1),np.flatnonzero(edges==-1)


def write_csv(path, rows):
    rows=list(rows)
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def angle(a,b):
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    cosine=np.divide((a*b).sum(-1),den,out=np.full(den.shape,np.nan),where=den>1e-10)
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runs',nargs='+',default=['seed42','seed43'])
    p.add_argument('--training-run',type=Path)
    args=p.parse_args();root=args.root.resolve();fig=root/'figures';fig.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':110,'savefig.dpi':180})
    episodes=[];curves=[];series=[];samples=[];sources=[];quality=[];representatives={};body_samples=[];schemas={}
    for run_name in args.runs:
        folder=root/run_name; meta=json.loads((folder/'metadata.json').read_text())
        assert meta['status']=='complete',folder
        with np.load(folder/'trace.npz') as archive:
            d={key:archive[key] for key in archive.files}
        schemas[run_name]={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in d.items()}
        dt=meta['dt'];names=meta['reward_terms'];steps=round(10/dt)
        n=meta['num_envs'];limit=meta['episodes_per_env']
        sources.append(dict(run=run_name,seed=meta['seed'],checkpoint_sha256=meta['checkpoint_sha256'],
                            elapsed_s=meta['elapsed_s'],num_envs=n,episodes_per_env=limit))
        eligible=d['episode']<limit
        q=dict(run=run_name,eligible_rows=int(eligible.sum()),finite=True,
               reward_sum_error=float(np.max(np.abs(d['reward_terms'].sum(-1)-d['reward_total']))),
               rate_error=float(np.max(np.abs(d['reward_rate']*dt-d['reward_terms']))))
        for key in d:
            if not np.isfinite(d[key]).all(): q['finite']=False
        # Independently derive key terms and success from their physical inputs.
        import yaml
        configuration=yaml.load((folder/'environment.yaml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
        formation_mode=configuration.get('formation_mode','distance_sum')
        distance_scale=float(configuration.get('formation_distance_scale',.05))
        gap=d['finger_gap']
        formation=((distance_scale/(distance_scale+gap))*np.clip(d['finger_normal_cosine'],-1,1)**2).sum(-1) if formation_mode=='distance_angle_sum' else distance_scale/(distance_scale+gap.sum(-1))
        q['formation_mode']=formation_mode
        q['formation_formula_error']=float(np.abs(formation-d['formation']).max())
        q['reach_formula_error']=float(np.abs(d['reach_delta']*4.9-d['reward_terms'][...,names.index('reaching')]).max())
        distance=np.linalg.norm(d['target'][...,:2]-d['goal'][...,:2],axis=-1)
        root_distance=np.linalg.norm(d['target_link'][...,:2]-d['goal'][...,:2],axis=-1)
        q['distance_formula_error']=float(np.abs(distance-d['distance']).max())
        failure=d['termination_object_fall_or_tilt']|d['termination_shelf_failure']|d['termination_tcp_linear_velocity']
        success=(d['step']>=steps)&(distance<=.03)&~failure
        q['success_mismatch']=int(np.count_nonzero(success!=d['termination_success']))
        tcp=np.linalg.norm(d['wrist_velocity'],axis=-1)>=4.
        q['tcp_mismatch']=int(np.count_nonzero(tcp!=d['termination_tcp_linear_velocity']))
        valid=d['contact_valid'];force=d['force'];force_xy=np.linalg.norm(force[...,:2],axis=-1)
        goal_vector=d['goal'][...,:2]-d['target'][...,:2]
        signed_xy=angle(force[...,:2],goal_vector)
        full_vector=np.concatenate((goal_vector,np.zeros((*goal_vector.shape[:-1],1))),axis=-1)
        signed_3d=angle(force,full_vector)
        speed=np.linalg.norm(d['object_velocity'][...,0,:2],axis=-1)
        usable=eligible&valid&d['contact']&(force_xy>=1.)&(distance>.03)
        moving=eligible&(speed>=.005)&(distance>.03)
        pushing=usable&(speed>=.005)
        # Original alignment includes vertical load in its denominator.
        dot=np.sum(force[...,:2]*goal_vector,axis=-1)
        denom=distance*np.linalg.norm(force,axis=-1)
        expected=np.clip(np.divide(dot,denom,out=np.zeros_like(dot),where=denom>1e-12),0,1)
        expected=np.where(d['contact']&valid&(force_xy>=1.)&(distance>1e-12),expected,0.)
        q['alignment_formula_error']=float(np.abs(expected-d['alignment']).max())
        q['sweep_formula_error']=float(np.abs(100*d['positive']-d['reward_terms'][...,names.index('sweep_progress')]).max())
        same_episode=d['step']>1
        target_step_speed=np.linalg.norm(d['target']-np.roll(d['target'],1,axis=0),axis=-1)/dt
        delta=np.roll(distance,1,axis=0)-distance
        independent_positive=np.maximum(delta,0)*d['contact_recognized']*expected
        q['sweep_increment_error_excluding_first_step']=float(np.abs(independent_positive-d['positive'])[same_episode].max())
        q['retreat_increment_error_excluding_first_step']=float(np.abs(np.maximum(-delta,0)-d['retreat'])[same_episode].max())
        quality.append(q)
        # Scalar/vector rows needed for pooled frame-level statistics.
        sample=dict(force_angle_xy=signed_xy[usable],force_angle_3d=signed_3d[usable],
                    push_angle_xy=signed_xy[pushing],force_n=np.linalg.norm(force,axis=-1)[usable],
                    vertical_fraction=np.abs(force[...,2][usable])/np.linalg.norm(force,axis=-1)[usable],
                    moving_contact=d['contact'][moving],moving_recognized=d['contact_recognized'][moving],
                    moving_continuous=(d['contact_substeps'][...,3]>=.01).all(-1)[moving],
                    contact=d['contact'][eligible],recognized=d['contact_recognized'][eligible],
                    valid=valid[eligible],wrist_speed=np.linalg.norm(d['wrist_velocity'],axis=-1)[eligible],
                    eef_speed=np.linalg.norm(d['eef_velocity'],axis=-1)[eligible],
                    arm_peak=np.abs(d['arm_joint_velocity']).max(-1)[eligible],
                    hand_speed=np.abs(d['hand_joint_velocity'])[eligible],
                    gap_mm=gap[eligible]*1000,normal_abs_cos=np.abs(d['finger_normal_cosine'][pushing]),
                    push_gap_mm=gap[pushing]*1000,
                    palm=d['palm_load'][pushing],dorsal=d['dorsal_load'][pushing],
                    push_mean_load=d['contact_substeps'][...,3].mean(-1)[pushing],
                    push_final_load=d['contact_substeps'][...,-1,3][pushing],
                    tactile=d['tactile'][eligible],wrench=d['wrench'][eligible],
                    actions=d['action'][eligible],hand_drive=d['hand_drive_position'][eligible],
                    net_reward=d['reward_total'][eligible])
        outside_static=eligible&(distance>.03)&(speed<.001)
        stalled=outside_static&same_episode&(target_step_speed<.001)&(np.linalg.norm(d['eef_velocity'],axis=-1)<.005)
        no_contact=eligible&~d['contact']
        sample.update(static_outside_reward=d['reward_total'][outside_static],
                      static_outside_formation=d['reward_terms'][...,names.index('contact_formation')][outside_static],
                      stalled_outside_reward=d['reward_total'][stalled],
                      stalled_outside_terms=d['reward_terms'][stalled],
                      no_contact_formation=d['reward_terms'][...,names.index('contact_formation')][no_contact])
        samples.append(sample)
        if 'target_normal_by_body' in d:
            body_samples.append(dict(run=run_name,names=meta['hand_contact_body_names'],
                force=np.linalg.norm(d['target_normal_by_body'][pushing],axis=-1)))
        for env in range(n):
            for ep in range(limit):
                ix=np.flatnonzero(d['episode'][:,env]==ep)
                assert len(ix)>0 and d['done'][ix[-1],env] and d['done'][ix,env].sum()==1,(run_name,env,ep)
                assert np.array_equal(d['step'][ix,env],np.arange(1,len(ix)+1))
                sl=(ix,env); last=ix[-1]; length=len(ix)
                dist=distance[sl]; con=d['contact'][sl];recogn=d['contact_recognized'][sl]
                mov=moving[sl];use=usable[sl];push=pushing[sl]
                reached=dist<=.03;ever=bool(reached.any());hit=int(np.flatnonzero(reached)[0]) if ever else length
                start_candidates=np.flatnonzero(con)
                start=int(start_candidates[0]) if len(start_candidates) else length
                transport=np.zeros(length,bool)
                if start<min(hit,length): transport[start:min(hit+1,length)]=True
                cb,ce=runs(con);gb,ge=runs(~con & transport)
                win=con[transport];counts=np.diff(np.r_[False,win].astype(int))
                rewards=d['reward_terms'][sl].sum(0)
                sweep_cumulative=np.cumsum(d['reward_terms'][sl][:,names.index('sweep_progress')])
                outcome='success' if success[last,env] else (
                    'object_fall_or_tilt' if d['termination_object_fall_or_tilt'][last,env] else
                    'shelf_failure' if d['termination_shelf_failure'][last,env] else
                    'tcp_linear_velocity' if tcp[last,env] else 'timeout_outside_goal')
                direction=d['sweep_direction'][ix[0],env]
                initial_angle=math.degrees(math.atan2(float(direction[1]),-float(direction[0])))
                region=1 if abs(initial_angle)<=30 else 3 if abs(initial_angle)>=150 else 2 if initial_angle>0 else 4
                active=d['active_objects'][sl]
                displacement=np.linalg.norm(d['object_positions'][sl]-d['initial_object_positions'][sl],axis=-1)
                obmask=active.copy();obmask[:,0]=False
                obstacle_disp=displacement[obmask]
                tilts=d['object_tilt'][sl][active]
                episode_row=dict(run=run_name,seed=meta['seed'],env=env,episode=ep,steps=length,
                    duration_s=length*dt,region=region,goal_angle_deg=initial_angle,
                    obstacles=int(active[0].sum()-1),outcome=outcome,success=int(success[last,env]),
                    final_distance_mm=float(dist[-1]*1000),minimum_distance_mm=float(dist.min()*1000),
                    root_center_final_distance_mm=float(root_distance[last,env]*1000),
                    ever_in_goal=int(ever),first_goal_s=(hit+1)*dt if ever else None,
                    goal_dwell_s=float(reached.sum()*dt),left_after_entry=int(ever and not reached[-1]),
                    final_1s_all_in_goal=int(length>=50 and reached[-50:].all()),
                    final_target_speed_mm_s=float(speed[last,env]*1000),
                    late2s_stalled_fraction=float(stalled[sl][d['step'][sl]>400].mean()) if length>400 else None,
                    late2s_reward=float(d['reward_total'][sl][d['step'][sl]>400].sum()),
                    total_reward=float(rewards.sum()),contact_fraction=float(con.mean()),
                    sweep_95percent_time_s=float((np.flatnonzero(sweep_cumulative>=.95*sweep_cumulative[-1])[0]+1)*dt) if sweep_cumulative[-1]>0 else None,
                    recognized_fraction=float(recogn.mean()),first_contact_s=(start+1)*dt if start<length else None,
                    transport_steps=int(transport.sum()),transport_contact_fraction=float(con[transport].mean()) if transport.any() else None,
                    transport_losses=int(np.count_nonzero(counts==-1)),longest_contact_s=float(max(ce-cb,default=0)*dt),
                    longest_transport_gap_s=float(max(ge-gb,default=0)*dt),
                    moving_steps=int(mov.sum()),moving_contact_fraction=float(con[mov].mean()) if mov.any() else None,
                    force_samples=int(use.sum()),force_angle_xy_deg=float(np.nanmean(signed_xy[sl][use])) if use.any() else None,
                    push_samples=int(push.sum()),push_angle_xy_deg=float(np.nanmean(signed_xy[sl][push])) if push.any() else None,
                    push_good_angle_fraction=float((signed_xy[sl][push]<=30).mean()) if push.any() else None,
                    finger_gap_sum_mm=float(gap[sl].sum(-1).mean()*1000),
                    formation_mean=float(d['formation'][sl].mean()),
                    wrist_speed_peak=float(np.linalg.norm(d['wrist_velocity'][sl],axis=-1).max()),
                    arm_speed_peak=float(np.abs(d['arm_joint_velocity'][sl]).max()),
                    obstacle_displacement_max_mm=float(obstacle_disp.max()*1000) if len(obstacle_disp) else 0.,
                    active_tilt_max_deg=float(np.degrees(tilts.max())),
                    shelf_displacement_max_mm=float(np.linalg.norm(d['shelf_position'][sl]-d['shelf_initial_position'][sl],axis=-1).max()*1000))
                for i,name in enumerate(names):episode_row['reward_'+name]=float(rewards[i])
                episodes.append(episode_row)
                cube=np.full((steps,len(names)),np.nan);cube[:length]=d['reward_terms'][sl];curves.append(cube)
                timeline=np.full((steps,8),np.nan)
                timeline[:length]=np.stack((dist,con,recogn,d['formation'][sl],speed[sl],signed_xy[sl],
                                          gap[sl].sum(-1),np.linalg.norm(d['wrist_velocity'][sl],axis=-1)),-1)
                series.append(timeline)
                # Retain representative real episodes, selected later by final distance.
                key=f'{run_name}:{env}:{ep}'
                representatives[key]=dict(time=np.arange(1,length+1)*dt,reward=d['reward_terms'][sl],
                    distance=dist,contact=con,recognized=recogn,force_angle=signed_xy[sl],usable=use,
                    finger_gap=gap[sl],finger_normal=d['finger_normal_cosine'][sl],force=force[sl],
                    target=d['target'][sl],goal=d['goal'][sl],eef=d['eef'][sl],finger=d['finger_position'][sl],
                    object=d['object_positions'][sl],active=active,
                    shelf_origin=d['shelf_initial_position'][ix[0],env],
                    action=d['action'][sl],joints=d['hand_drive_position'][sl],
                    tilt=d['object_tilt'][sl],wrist=np.linalg.norm(d['wrist_velocity'][sl],axis=-1))
        del d
    assert len({s['checkpoint_sha256'] for s in sources})==1
    write_csv(root/'episodes.csv',episodes)
    c=np.array(curves);s=np.array(series);successes=np.array([e['success'] for e in episodes],bool)
    merged={k:np.concatenate([samp[k] for samp in samples],axis=0) for k in samples[0]}
    total=len(episodes);count=int(successes.sum())
    summary=dict(episodes=total,successes=count,success_rate=count/total,success_wilson95=wilson(count,total),
                 observed_policy_steps=int(sum(e['steps'] for e in episodes)),
                 simulated_episode_seconds=sum(e['duration_s'] for e in episodes),sources=sources,quality=quality,
                 final_distance_mm=distribution([e['final_distance_mm'] for e in episodes]),
                 total_episode_reward=distribution([e['total_reward'] for e in episodes]),
                 ever_in_goal=sum(e['ever_in_goal'] for e in episodes),left_after_entry=sum(e['left_after_entry'] for e in episodes),
                 reached_but_failed=sum(e['ever_in_goal'] and not e['success'] for e in episodes),
                 termination_counts={outcome:sum(e['outcome']==outcome for e in episodes) for outcome in sorted({e['outcome'] for e in episodes})})
    for key in ('contact','recognized','valid','moving_contact','moving_recognized','moving_continuous'):
        summary[key]=dict(n=len(merged[key]),fraction=float(merged[key].mean()) if len(merged[key]) else None)
    for key in ('force_angle_xy','force_angle_3d','push_angle_xy','force_n','vertical_fraction',
                'wrist_speed','eef_speed','arm_peak','hand_speed','static_outside_reward','no_contact_formation'):
        summary[key]=distribution(merged[key])
    summary['force_angle_le30_fraction']=float((merged['force_angle_xy']<=30).mean())
    summary['force_wrong_hemisphere_fraction']=float((merged['force_angle_xy']>90).mean())
    summary['push_angle_le30_fraction']=float((merged['push_angle_xy']<=30).mean())
    summary['push_wrong_hemisphere_fraction']=float((merged['push_angle_xy']>90).mean())
    summary['pushing_frames']=len(merged['push_angle_xy'])
    summary['pushing_palm_load_sum']=float(merged['palm'].sum())
    summary['pushing_dorsal_load_sum']=float(merged['dorsal'].sum())
    summary['classified_palm_fraction']=float(merged['palm'].sum()/(merged['palm'].sum()+merged['dorsal'].sum()))
    summary['classified_vs_final_load_ratio']=float((merged['palm'].sum()+merged['dorsal'].sum())/merged['push_final_load'].sum())
    summary['goal_sensitivity']={str(mm):sum(e['final_distance_mm']<=mm and e['outcome'] in ('success','timeout_outside_goal') for e in episodes)/total for mm in (10,20,30,40,50)}
    summary['final_1s_all_in_goal_count']=sum(e['final_1s_all_in_goal'] for e in episodes)
    summary['success_and_final1s_in_goal']=sum(e['success'] and e['final_1s_all_in_goal'] for e in episodes)
    summary['success_and_final_speed_le10mm_s']=sum(e['success'] and e['final_target_speed_mm_s']<=10 for e in episodes)
    summary['visible_success_root_outside30mm']=sum(e['success'] and e['root_center_final_distance_mm']>30 for e in episodes)
    summary['success_root_final_distance_mm']=distribution([e['root_center_final_distance_mm'] for e in episodes if e['success']])
    for key in ('transport_contact_fraction','transport_losses','longest_contact_s','longest_transport_gap_s',
                'first_contact_s','first_goal_s','goal_dwell_s','obstacle_displacement_max_mm','active_tilt_max_deg',
                'shelf_displacement_max_mm','formation_mean'):
        summary[key]=distribution([np.nan if e[key] is None else e[key] for e in episodes])
    summary['sweep_95percent_time_s']=distribution([np.nan if e['sweep_95percent_time_s'] is None else e['sweep_95percent_time_s'] for e in episodes])
    summary['sweep_reward_after2s_fraction']=float(np.nansum(c[:,100:,names.index('sweep_progress')])/np.nansum(c[:,:,names.index('sweep_progress')]))
    summary['hand_action_outside_fraction']=float(((merged['actions'][:,6:]<0)|(merged['actions'][:,6:]>1)).mean())
    summary['arm_action_outside_fraction']=float((np.abs(merged['actions'][:,:6])>np.pi/2).mean())
    summary['arm_joint_above_old1_fraction']=float((merged['arm_peak']>1).mean())
    summary['static_outside_positive_fraction']=float((merged['static_outside_reward']>0).mean())
    summary['stalled_outside_reward']=distribution(merged['stalled_outside_reward'])
    summary['stalled_outside_positive_fraction']=float((merged['stalled_outside_reward']>0).mean())
    summary['stalled_outside_mean_terms']={name:float(merged['stalled_outside_terms'][:,i].mean()) for i,name in enumerate(names)}
    summary['timeout_late_stalled_majority']=sum(e['outcome']=='timeout_outside_goal' and e['late2s_stalled_fraction'] is not None and e['late2s_stalled_fraction']>.5 for e in episodes)
    summary['static_outside_formation_sum']=float(merged['static_outside_formation'].sum())
    summary['wrench_force_norm']=distribution(np.linalg.norm(merged['wrench'][:,:3],axis=-1))
    summary['wrench_moment_norm']=distribution(np.linalg.norm(merged['wrench'][:,3:],axis=-1))
    summary['wrench_axis_clip_fraction']=float((np.abs(merged['wrench'])>=np.array([150]*3+[8]*3)).mean())
    summary['tactile_active_fraction']=np.mean(merged['tactile']!=0,axis=0).tolist()
    summary['hand_drive_mean_rad']=np.mean(merged['hand_drive'],axis=0).tolist()
    summary['hand_drive_std_rad']=np.std(merged['hand_drive'],axis=0).tolist()
    finger_rows=[]
    for i,name in enumerate(('thumb','index','middle','ring','little')):
        finger_rows.append(dict(finger=name,gap_mean_mm=float(merged['gap_mm'][:,i].mean()),
             gap_p90_mm=float(np.quantile(merged['gap_mm'][:,i],.9)),
             pushing_gap_mean_mm=float(merged['push_gap_mm'][:,i].mean()),
             pushing_within5mm_fraction=float((merged['push_gap_mm'][:,i]<=5).mean()),
             pushing_within10mm_fraction=float((merged['push_gap_mm'][:,i]<=10).mean()),
             pushing_normal_abs_cos_mean=float(merged['normal_abs_cos'][:,i].mean()),
             pushing_normal_le30deg_fraction=float((merged['normal_abs_cos'][:,i]>=np.cos(np.pi/6)).mean())))
    summary['fingers']=finger_rows;write_csv(root/'fingers.csv',finger_rows)
    if body_samples:
        body_names=body_samples[0]['names'];body=np.concatenate([x['force'] for x in body_samples])
        body_rows=[]
        typical=body.sum(-1)<=np.quantile(body.sum(-1),.99)
        for group in ('thumb','index','middle','ring','little','palm_pad','hand_base'):
            ids=[i for i,n in enumerate(body_names) if ('palm' in n if group=='palm_pad' else
                not any(g in n for g in ('thumb','index','middle','ring','little','palm')) if group=='hand_base' else group in n)]
            loads=body[:,ids].sum(-1)
            body_rows.append(dict(group=group,frame_contact_fraction=float((loads>=.05).mean()),normal_load_sum=float(loads.sum()),load_share=float(loads.sum()/body.sum()),
                trimmed_load_share=float(loads[typical].sum()/body[typical].sum()),normal_load_p95=float(np.quantile(loads,.95)),normal_load_max=float(loads.max())))
        summary['target_body_contact_subset']=dict(runs=[x['run'] for x in body_samples],frames=len(body),rows=body_rows)
        write_csv(root/'target_body_contacts.csv',body_rows)
    reward_rows=[]
    for i,name in enumerate(names):
        totals=np.nansum(c[:,:,i],axis=1);values=c[:,:,i];flat=values[np.isfinite(values)]
        reward_rows.append(dict(term=name,weight=meta['reward_weights'][i],episode_mean=float(totals.mean()),
             episode_success_mean=float(totals[successes].mean()),episode_failure_mean=float(totals[~successes].mean()),
             episode_p10=float(np.quantile(totals,.1)),episode_p90=float(np.quantile(totals,.9)),
             step_mean=float(flat.mean()),step_min=float(flat.min()),step_max=float(flat.max()),
             nonzero_step_fraction=float((np.abs(flat)>1e-10).mean()),sum=float(totals.sum())))
    summary['rewards']=reward_rows;write_csv(root/'rewards.csv',reward_rows)
    grouped=[]
    for category in ('region','seed','episode','obstacles','outcome'):
        for value in sorted({e[category] for e in episodes}):
            group=[e for e in episodes if e[category]==value];nc=len(group);sc=sum(e['success'] for e in group)
            lo,hi=wilson(sc,nc)
            grouped.append(dict(category=category,value=value,n=nc,successes=sc,success_rate=sc/nc,ci95_low=lo,ci95_high=hi,
                final_distance_mean_mm=float(np.mean([e['final_distance_mm'] for e in group])),
                reward_mean=float(np.mean([e['total_reward'] for e in group])),
                contact_mean=float(np.mean([e['contact_fraction'] for e in group]))))
    summary['groups']=grouped;write_csv(root/'groups.csv',grouped)
    # Cluster bootstrap respects four repeated episodes per vector environment.
    clusters={}
    for e in episodes: clusters.setdefault((e['seed'],e['env']),[]).append(e['success'])
    cluster_rates=np.array([np.mean(v) for v in clusters.values()]);rng=np.random.default_rng(761)
    boot=cluster_rates[rng.integers(0,len(cluster_rates),(10000,len(cluster_rates)))].mean(1)
    summary['success_cluster_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
    summary['clusters']=len(clusters)
    if args.training_run:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        records=[];training={}
        for event in sorted(args.training_run.rglob('events.out.tfevents.*')):
            accumulator=EventAccumulator(str(event),size_guidance={'scalars':0});accumulator.Reload()
            for tag in accumulator.Tags()['scalars']:
                values=accumulator.Scalars(tag)
                training[tag]=values
                records.extend(dict(source=str(event),tag=tag,step=v.step,value=v.value,wall_time=v.wall_time) for v in values)
        write_csv(root/'training_scalars.csv',records)
        summary['training_scalar_tags']=len(training)
        summary['training_at_checkpoint']={tag:dict(step=values[-1].step,value=values[-1].value) for tag,allvalues in training.items()
            if (values:=[v for v in allvalues if v.step<=meta['checkpoint_iteration']]) and not tag.endswith('/time')}
        f,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        for tag,label in [('Episode_Termination/success','Training termination success'),('Metrics/timeout_success_rate','Timeout-only rolling rate')]:
            if tag in training:
                values=[v for v in training[tag] if v.step<=meta['checkpoint_iteration']]
                axes[0].plot([v.step for v in values],[v.value*100 for v in values],label=label,alpha=.65,lw=.8)
        axes[0].set(ylabel='Logged training rate (%)',ylim=(0,100));axes[0].legend()
        if 'Train/mean_reward' in training:
            v=[v for v in training['Train/mean_reward'] if v.step<=meta['checkpoint_iteration']]
            axes[1].plot([x.step for x in v],[x.value for x in v],alpha=.3,lw=.5,color='#477ca8')
            window=100
            axes[1].plot([x.step for x in v][window-1:],np.convolve([x.value for x in v],np.ones(window)/window,'valid'),color='#1f5b8f',label='100-iteration mean')
        axes[1].set(xlabel='Training iteration',ylabel='Logged training mean reward');axes[1].legend()
        f.suptitle('Training context only: stochastic policy, different logging denominators')
        f.tight_layout();plt.savefig(fig/'07_training_context.png',bbox_inches='tight');plt.savefig(fig/'07_training_context.svg',bbox_inches='tight');plt.close()
    # Pool direction data separately for success/failure and region, not average of percentages.
    force_episode=[e for e in episodes if e['force_samples']]
    summary['episode_force_angle_xy']=distribution([e['force_angle_xy_deg'] for e in force_episode])
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    time=np.arange(1,steps+1)*dt
    timeline_rows=[]
    for j,t in enumerate(time):
        for i,name in enumerate(names):
            vals=c[:,j,i];alive=np.isfinite(vals);v=vals[alive]
            timeline_rows.append(dict(time_s=t,term=name,alive_n=int(alive.sum()),cohort_n=total,
                step_mean_alive=float(v.mean()) if len(v) else None,
                step_p10_alive=float(np.quantile(v,.1)) if len(v) else None,
                step_p90_alive=float(np.quantile(v,.9)) if len(v) else None,
                step_mean_all_ended_zero=float(np.nan_to_num(vals).mean())))
    write_csv(root/'reward_timelines.csv',timeline_rows)
    phase_rows=[]
    for left,right in ((0,1),(1,3),(3,6),(6,10)):
        phase=(time>left)&(time<=right)
        for i,name in enumerate(names):
            vals=np.nansum(c[:,phase,i],axis=1)
            phase_rows.append(dict(phase_s=f'{left}-{right}',term=name,
                mean_contribution=float(vals.mean()),success_mean=float(vals[successes].mean()),failure_mean=float(vals[~successes].mean())))
    write_csv(root/'reward_phases.csv',phase_rows)
    np.savez_compressed(root/'episode_aligned.npz',rewards=c,diagnostics=s,time=time,success=successes,
        diagnostic_names=np.array(['distance','contact','recognized','formation','target_speed_xy','force_angle_xy','gap_sum','wrist_speed']))
    def save(name):
        plt.savefig(fig/(name+'.png'),bbox_inches='tight');plt.savefig(fig/(name+'.svg'),bbox_inches='tight');plt.close()
    fig1,axes=plt.subplots(4,4,figsize=(16,12),sharex=True)
    for i,ax in enumerate(axes.flat):
        v=c[:,:,i];ax.fill_between(time,np.nanquantile(v,.1,axis=0),np.nanquantile(v,.9,axis=0),color='#568abb',alpha=.25,label='P10-P90 alive')
        ax.plot(time,np.nanmean(v,axis=0),color='#1f5b8f',lw=1,label='Mean alive')
        ax.axhline(0,color='#999',lw=.5);ax.set_title(names[i],fontsize=11);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        if i>=12:ax.set_xlabel('Episode time (s)')
        if i%4==0:ax.set_ylabel('Reward / policy step')
    axes.flat[0].legend(fontsize=8);fig1.suptitle(f'All 16 reward terms | {total} complete episodes | step dt = {dt}s',fontsize=15)
    fig1.tight_layout();save('01_reward_timelines')
    f,axes=plt.subplots(4,4,figsize=(16,12),sharex=True)
    for i,ax in enumerate(axes.flat):
        for flag,label,color in ((True,'Success','#2c8b65'),(False,'Failure','#c56050')):
            vals=c[successes==flag,:,i];ax.plot(time,np.nanmean(vals,axis=0),label=label,color=color,lw=1)
        ax.set_title(names[i],fontsize=11);ax.axhline(0,color='#999',lw=.5)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        if i>=12:ax.set_xlabel('Episode time (s)')
        if i%4==0:ax.set_ylabel('Mean reward / policy step')
    axes.flat[0].legend(fontsize=8);f.suptitle('Reward timelines by final outcome (descriptive, not causal)');f.tight_layout();save('08_reward_by_outcome')
    f,axes=plt.subplots(1,2,figsize=(14,7))
    y=np.arange(len(names));axes[0].barh(y,[r['episode_mean'] for r in reward_rows],color=['#277b60' if r['episode_mean']>=0 else '#b65751' for r in reward_rows]);axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].axvline(0,color='k',lw=.7);axes[0].set_xlabel('Mean episode reward contribution')
    for success_flag,label,color in [(True,'Success','#2c8b65'),(False,'Failure','#c56050')]:
        cumul=np.cumsum(np.nan_to_num(c[successes==success_flag]).sum(-1),axis=1)
        axes[1].plot(time,cumul.mean(0),label=f'{label} (n={len(cumul)})',color=color)
        axes[1].fill_between(time,np.quantile(cumul,.1,axis=0),np.quantile(cumul,.9,axis=0),color=color,alpha=.12)
    axes[1].set(xlabel='Episode time (s)',ylabel='Cumulative total reward');axes[1].legend();f.tight_layout();save('02_reward_balance')
    f,axes=plt.subplots(2,2,figsize=(13,9))
    for flag,label,color in [(True,'Success','#2c8b65'),(False,'Failure','#c56050')]:
        data=s[successes==flag];axes[0,0].plot(time,np.nanmean(data[:,:,0],axis=0)*1000,label=label,color=color)
        axes[0,1].plot(time,np.nanmean(data[:,:,1],axis=0),label=label,color=color)
    axes[0,0].axhline(30,color='black',ls='--',label='Success radius');axes[0,0].set(ylabel='Goal distance (mm)');axes[0,0].legend()
    axes[0,1].set(ylabel='Raw target contact fraction',ylim=(0,1));axes[0,1].legend()
    groups=[g for g in grouped if g['category']=='region'];xs=[g['value'] for g in groups];rates=[g['success_rate']*100 for g in groups]
    axes[1,0].bar(xs,rates,color='#477ca8');axes[1,0].errorbar(xs,rates,yerr=[[100*(g['success_rate']-g['ci95_low']) for g in groups],[100*(g['ci95_high']-g['success_rate']) for g in groups]],fmt='none',color='black',capsize=4)
    for g in groups:axes[1,0].text(g['value'],g['success_rate']*100+5,f"{g['successes']}/{g['n']}",ha='center')
    axes[1,0].set(xlabel='Direction region (1: away, 3: toward robot)',ylabel='Success (%)',ylim=(0,110),xticks=xs)
    for flag,label,col in [(True,'Success','#2c8b65'),(False,'Failure','#c56050')]:
        vals=[e['final_distance_mm'] for e in episodes if bool(e['success'])==flag];axes[1,1].hist(vals,bins=np.arange(0,251,5),alpha=.5,label=label,color=col)
    axes[1,1].axvline(30,color='black',ls='--');axes[1,1].set(xlabel='Final goal distance (mm)',ylabel='Episodes');axes[1,1].legend()
    for ax in axes[0]:ax.set_xlabel('Episode time (s)')
    f.tight_layout();save('03_success_and_contact')
    f,axes=plt.subplots(1,3,figsize=(16,4.8))
    axes[0].hist(merged['force_angle_xy'],bins=np.arange(0,181,5),density=True,alpha=.45,label='Contact, >=1N, outside goal')
    axes[0].hist(merged['push_angle_xy'],bins=np.arange(0,181,5),density=True,histtype='step',lw=1.7,label='Also moving >=5mm/s');axes[0].axvline(30,color='black',ls='--');axes[0].axvline(90,color='grey',ls=':');axes[0].set(xlabel='Signed XY force error (degrees)',ylabel='Frame density');axes[0].legend(fontsize=8)
    axes[1].hist(merged['vertical_fraction'],bins=np.linspace(0,1,31),color='#477ca8');axes[1].set(xlabel='|Fz| / |F|',ylabel='Eligible contact frames')
    axes[2].hist([e['longest_transport_gap_s'] for e in episodes],bins=30,color='#b47750');axes[2].set(xlabel='Longest contact gap during transport (s)',ylabel='Episodes')
    f.tight_layout();save('04_force_and_maintenance')
    f,axes=plt.subplots(2,2,figsize=(12,8))
    labels=[r['finger'] for r in finger_rows]
    axes[0,0].boxplot([merged['push_gap_mm'][:,i] for i in range(5)],tick_labels=labels,showfliers=False);axes[0,0].axhline(5,color='k',ls='--');axes[0,0].set(ylabel='Pad to tracked OBB gap (mm)',title='Moving contact outside goal')
    axes[0,1].bar(labels,[r['pushing_within5mm_fraction']*100 for r in finger_rows],color='#477ca8');axes[0,1].set(ylabel='Within 5mm (%)',ylim=(0,100))
    axes[1,0].bar(labels,[r['pushing_normal_le30deg_fraction']*100 for r in finger_rows],color='#477ca8');axes[1,0].set(ylabel='Pad normal within 30deg (%)',ylim=(0,100),title='Two-sided pad-normal alignment')
    for flag,label,color in [(True,'Success','#2c8b65'),(False,'Failure','#c56050')]:axes[1,1].plot(time,np.nanmean(s[successes==flag,:,3],axis=0),label=label,color=color)
    axes[1,1].set(xlabel='Episode time (s)',ylabel='Formation raw [0,5]' if formation_mode=='distance_angle_sum' else 'Formation raw [0,1]');axes[1,1].legend();f.tight_layout();save('05_finger_posture')
    if body_samples:
        rows=summary['target_body_contact_subset']['rows'];f,axes=plt.subplots(1,2,figsize=(12,4))
        groups=[r['group'].replace('_','\n') for r in rows]
        axes[0].bar(groups,[r['frame_contact_fraction']*100 for r in rows],color='#477ca8');axes[0].set(ylabel='Frames with target normal load >= .05 N (%)',ylim=(0,100))
        axes[1].bar(groups,[r['load_share']*100 for r in rows],color='#477ca8');axes[1].set(ylabel='Share of body-wise normal-load sum (%)',ylim=(0,100))
        f.suptitle('Target-only physical contact | all recorded runs | moving outside goal');f.tight_layout();save('09_target_contact_body')
    f,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].hist(merged['wrist_speed'],bins=60,color='#477ca8');axes[0,0].axvline(4,color='#b65751',ls='--',label='v0_3 cutoff');axes[0,0].set(xlabel='wrist_3_link origin speed (m/s)',ylabel='Policy steps');axes[0,0].legend()
    axes[0,1].hist(merged['arm_peak'],bins=60,color='#477ca8');axes[0,1].axvline(1,color='#b65751',ls='--',label='Parent cutoff, removed in v0_3');axes[0,1].set(xlabel='Maximum absolute arm joint speed (rad/s)',ylabel='Policy steps');axes[0,1].legend(fontsize=8)
    axes[1,0].hist([e['obstacle_displacement_max_mm'] for e in episodes],bins=50,color='#b47750');axes[1,0].set(xlabel='Maximum active obstacle displacement (mm)',ylabel='Episodes')
    axes[1,1].scatter([e['final_distance_mm'] for e in episodes],[e['total_reward'] for e in episodes],c=np.where(successes,'#2c8b65','#c56050'),s=12,alpha=.6);axes[1,1].axvline(30,color='k',ls='--');axes[1,1].set(xlabel='Final goal distance (mm)',ylabel='Total episode reward')
    f.tight_layout();save('06_safety_and_reward_alignment')
    selected=[]
    for label,predicate in [('success',lambda e:e['success']),('timeout',lambda e:e['outcome']=='timeout_outside_goal'),('lost_goal',lambda e:e['ever_in_goal'] and not e['success']),('physical_failure',lambda e:e['outcome'] not in ('success','timeout_outside_goal'))]:
        pool=[e for e in episodes if predicate(e)]
        if not pool:continue
        pool.sort(key=lambda e:e['final_distance_mm']);e=pool[len(pool)//2];key=f"{e['run']}:{e['env']}:{e['episode']}";r=representatives[key]
        selected.append(dict(label=label,**e));f,axes=plt.subplots(3,2,figsize=(13,11));t=r['time']
        axes[0,0].plot(t,r['distance']*1000);axes[0,0].axhline(30,color='k',ls='--');axes[0,0].set(ylabel='Goal distance (mm)')
        axes[0,1].plot(t,np.cumsum(r['reward'],axis=0)[:,names.index('sweep_progress')],label='Sweep');axes[0,1].plot(t,np.cumsum(r['reward'],axis=0)[:,names.index('contact_formation')],label='Formation');axes[0,1].plot(t,np.cumsum(r['reward'].sum(-1)),label='Total');axes[0,1].set(ylabel='Cumulative reward');axes[0,1].legend()
        axes[1,0].step(t,r['contact'],label='Raw contact',where='post');axes[1,0].step(t,r['recognized'],label='2 of 3 recognized',where='post',alpha=.7);axes[1,0].set(ylabel='Contact state');axes[1,0].legend()
        axes[1,1].plot(t,np.where(r['usable'],r['force_angle'],np.nan));axes[1,1].axhline(30,color='k',ls='--');axes[1,1].set(ylabel='Force XY error (deg)',ylim=(0,180))
        axes[2,0].plot(t,r['finger_gap']*1000);axes[2,0].set(ylabel='Finger gap (mm)');axes[2,0].legend(labels,fontsize=7)
        # Top view matches the existing camera: screen right=-Y, up=+X.
        pos=r['target']-r['shelf_origin'];eef=r['eef']-r['shelf_origin'];axes[2,1].plot(-pos[:,1],pos[:,0],label='Tracked visible OBB');axes[2,1].plot(-eef[:,1],eef[:,0],alpha=.7,label='EEF')
        g=r['goal'][0,:3]-r['shelf_origin'];axes[2,1].add_patch(plt.Circle((-g[1],g[0]),.03,fill=False,color='green'));axes[2,1].scatter(-g[1],g[0],marker='x',color='green');axes[2,1].axis('equal');axes[2,1].set(xlabel='-Y from shelf origin (m)',ylabel='X from shelf origin (m)');axes[2,1].legend(fontsize=8)
        for ax in axes.flat:
            if ax!=axes[2,1]:
                ax.set_xlabel('Episode time (s)')
                ax.set_xlim(0,float(t[-1]))
        f.suptitle(f"{label}: seed {e['seed']}, env {e['env']}, episode {e['episode']} | {e['outcome']}");f.tight_layout();save('case_'+label)
        np.savez_compressed(root/('case_'+label+'.npz'),**r)
    (root/'selected_cases.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    dictionary=dict(runs=schemas,coordinates='world axes unless explicitly named; SI units; terminal state before reset',
        time='step*dt is episode age at the post-physics sample; dt=.02s; substep dt=.01s',
        contact_substeps='[world Fx,Fy,Fz in N, scalar normal-load sum in N, valid=1/0]',
        finger_gap='meters to tracked visible OBB, not necessarily whole physical cylinder surface',
        reward_raw='original callable output; reward_rate=raw*weight; reward_terms=raw*weight*dt',
        hand_drive_position='six physical actuator joints in HAND_DRIVE_JOINT_NAMES order; radians',
        target_normal_by_body='when present: final physics substep, target normal force in N by metadata hand_contact_body_names',
        tactile='17 original paired surface states, all objects/shelf: none0,palm1,dorsal-1,both2',
        wrench='physical compensated wrist support reaction, EEF axes, N and Nm; not external force on target')
    (root/'data_dictionary.json').write_text(json.dumps(dictionary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('episodes','successes','success_rate','success_wilson95','termination_counts','quality')},indent=2))


if __name__=='__main__':main()
