"""Audit impact/displacement costs from terminal-inclusive, physics-substep Play traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stats(x):
    x = np.asarray(x, dtype=float).ravel()
    assert np.isfinite(x).all()
    if not len(x):
        return {'n': 0}
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.median(x)),
                p90=float(np.quantile(x, .9)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), max=float(x.max()))


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def contact_cost(force, threshold):
    return .04*.02*np.minimum(.5*np.expm1(2*np.clip(force-threshold, 0, .5*np.log(201))), 100)


def wrench_cost(df, dm, ft, mt, weight=.1):
    severity = np.maximum(np.maximum(df-ft, 0)/(1.5*ft), np.maximum(dm-mt, 0)/(1.5*mt))
    return weight*.02*np.expm1(3*np.clip(severity, 0, 1))/np.expm1(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', default=['seed42_live', 'seed43_live'])
    args = parser.parse_args()
    root = args.root
    (root/'figures').mkdir(exist_ok=True)
    episodes, events, sensitivities, quality, cases, objects_rows = [], [], [], {}, [], []
    large_contacts=[]
    pools = {}
    def pool(name, value):
        pools.setdefault(name, []).append(np.asarray(value).ravel())
    for run in args.runs:
        folder = root/run
        meta = json.loads((folder/'metadata.json').read_text())
        assert meta['status'] == 'complete'
        cfg = yaml.load((folder/'environment.yaml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
        trained = yaml.load((Path(meta['checkpoint']).parent/'params/env.yaml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
        managers = ['actions', 'observations', 'events', 'rewards', 'terminations', 'commands', 'curriculum']
        matches = {k: cfg.get(k) == trained.get(k) for k in managers}
        assert all(matches.values()), matches
        assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest() == meta['checkpoint_sha256']
        keys = ['episode', 'step', 'done', 'reward_terms', 'reward_raw', 'reward_total', 'object_positions',
                'initial_object_positions', 'active_objects', 'object_velocity', 'wrench', 'sensor_fresh',
                'distance', 'wrist_velocity', 'contact_substeps', 'hand_net_normal', 'wrench_pure_world', 'substep_object_hand_normal_vector',
                'substep_object_hand_normal_load', 'substep_whole_hand_normal_load', 'substep_pure_wrench',
                'substep_raw_wrench', 'substep_support_wrench']
        with np.load(folder/'trace.npz') as z:
            d = {k: z[k] for k in keys}
            d.update({k: z[k] for k in z.files if k.startswith('termination_')})
        names = meta['reward_terms']; dt = meta['dt']; h = meta['physics_dt']
        oi, ci, wi = [names.index(n) for n in ['obstacle_displacement','obstacle_contact','wrench_change']]
        active = d['active_objects'].copy(); active[..., 0] = False
        displacement = np.linalg.norm(d['object_positions'].astype(float)-d['initial_object_positions'].astype(float), axis=-1)
        displacement = np.where(active, displacement, 0)
        obvec = np.where(active[:, :, None, :, None], d['substep_object_hand_normal_vector'], 0).sum(-2)
        obforce = np.linalg.norm(obvec, axis=-1)
        obload = np.where(active[:, :, None, :], d['substep_object_hand_normal_load'], 0).sum(-1)
        targetload = d['substep_object_hand_normal_load'][..., 0]
        norm_f = np.linalg.norm(d['wrench'][..., :3], axis=-1)
        norm_m = np.linalg.norm(d['wrench'][..., 3:], axis=-1)
        same = np.zeros_like(d['done']); same[1:] = d['episode'][1:] == d['episode'][:-1]
        df = np.zeros_like(norm_f); dm = np.zeros_like(norm_m)
        df[1:] = np.abs(np.diff(norm_f, axis=0))*same[1:]
        dm[1:] = np.abs(np.diff(norm_m, axis=0))*same[1:]
        expected_ob = 300*dt*np.minimum((displacement**2).sum(-1), .04)
        errors = dict(displacement=float(np.abs(expected_ob+d['reward_terms'][...,oi]).max()),
                      obstacle_contact=float(np.abs(contact_cost(obforce[...,-1],50)+d['reward_terms'][...,ci]).max()),
                      wrench=float(np.abs(wrench_cost(df,dm,30,3)+d['reward_terms'][...,wi]).max()),
                      reward_sum=float(np.abs(d['reward_terms'].sum(-1)-d['reward_total']).max()))
        assert max(errors.values()) < 3e-5, errors
        quality[run] = dict(manager_matches=matches, errors=errors, checkpoint_sha256=meta['checkpoint_sha256'])
        eligible=d['episode']<meta['episodes_per_env']
        hand_norm=np.linalg.norm(d['hand_net_normal'],axis=-1)
        pure_norm=np.linalg.norm(d['wrench_pure_world'][...,:3],axis=-1)
        contact_rows=eligible&(hand_norm>1)
        quality[run]['wrench_contact_comparison']=dict(n=int(contact_rows.sum()),
            magnitude_correlation=float(np.corrcoef(hand_norm[contact_rows],pure_norm[contact_rows])[0,1]),
            wrist_over_hand=stats(pure_norm[contact_rows]/hand_norm[contact_rows]),
            scope='normal resultant >1 N; wrist includes tangential force and compensation; consistency check, not calibration')
        large=np.argwhere((d['substep_whole_hand_normal_load']>100)&eligible[:,:,None])
        for tick,env,sub in large:
            large_contacts.append(dict(run=run,tick=int(tick),env=int(env),substep=int(sub),
                episode=int(d['episode'][tick,env]),episode_time_s=(int(d['step'][tick,env])-1)*dt+(sub+1)*h,
                target_load_N=float(targetload[tick,env,sub]),
                target_normal_resultant_N=float(np.linalg.norm(d['substep_object_hand_normal_vector'][tick,env,sub,0])),
                target_with_friction_N=float(np.linalg.norm(d['contact_substeps'][tick,env,sub,:3])),
                wrist_pure_N=float(np.linalg.norm(d['substep_pure_wrench'][tick,env,sub,:3])),
                wrist_raw_N=float(np.linalg.norm(d['substep_raw_wrench'][tick,env,sub,:3])),
                wrist_support_N=float(np.linalg.norm(d['substep_support_wrench'][tick,env,sub,:3]))))
        for env in range(meta['num_envs']):
            for ep in range(meta['episodes_per_env']):
                t = np.flatnonzero(d['episode'][:,env] == ep)
                assert len(t) and d['done'][t[-1],env] and not d['done'][t[:-1],env].any()
                assert np.array_equal(d['step'][t,env], np.arange(1,len(t)+1))
                assert np.all(d['active_objects'][t,env] == d['active_objects'][t[0],env])
                assert np.all(d['initial_object_positions'][t,env] == d['initial_object_positions'][t[0],env])
                rewards = d['reward_terms'][t,env]
                disp = displacement[t,env]
                force = obforce[t,env]; load = obload[t,env]
                raw = d['substep_raw_wrench'][t,env].reshape(-1,6)
                pure = d['substep_pure_wrench'][t,env].reshape(-1,6)
                support = d['substep_support_wrench'][t,env].reshape(-1,6)
                whole = d['substep_whole_hand_normal_load'][t,env].ravel()
                trg = targetload[t,env].ravel()
                pforce, pmoment = np.linalg.norm(pure[:,:3],axis=-1), np.linalg.norm(pure[:,3:],axis=-1)
                p_df = np.abs(np.diff(pforce)); p_dm = np.abs(np.diff(pmoment))
                vec_df = np.linalg.norm(np.diff(pure[:,:3],axis=0),axis=-1)
                vec_dm = np.linalg.norm(np.diff(pure[:,3:],axis=0),axis=-1)
                positive = np.maximum(rewards,0).sum()
                row = dict(run=run,seed=meta['seed'],env=env,episode=ep,n_steps=len(t),duration_s=len(t)*dt,
                           success=bool(d['termination_success'][t[-1],env]),obstacle_count=int(active[t[0],env].sum()),
                           max_displacement_m=float(disp.max()),final_displacement_m=float(disp[-1].max()),
                           displacement_cost=float(-rewards[:,oi].sum()),contact_cost=float(-rewards[:,ci].sum()),
                           wrench_cost=float(-rewards[:,wi].sum()),positive_reward=float(positive),total_reward=float(rewards.sum()),
                           obstacle_peak_N=float(force.max()),obstacle_load_peak_N=float(load.max()),
                           target_peak_N=float(trg.max()),whole_hand_peak_N=float(whole.max()),
                           wrist_force_peak_N=float(pforce.max()),wrist_moment_peak_Nm=float(pmoment.max()),
                           wrist_force_delta_peak_N=float(df[t,env].max()),wrist_moment_delta_peak_Nm=float(dm[t,env].max()),
                           physics_force_delta_peak_N=float(p_df.max()),physics_moment_delta_peak_Nm=float(p_dm.max()),
                           contact_steps=int((rewards[:,ci]<0).sum()),wrench_steps=int((rewards[:,wi]<0).sum()),
                           first_1cm_s=float((np.flatnonzero(disp.max(-1)>=.01)[0]+1)*dt) if disp.max()>=.01 else None)
                for weight in (300,600,900,1500):
                    row[f'displacement_cost_w{weight}'] = float(weight/300*row['displacement_cost'])
                for threshold in (5,10,20,30,50):
                    row[f'contact_last_F{threshold}'] = float(contact_cost(force[:,-1],threshold).sum())
                    row[f'contact_peak_F{threshold}'] = float(contact_cost(force.max(-1),threshold).sum())
                    row[f'load_peak_F{threshold}'] = float(contact_cost(load.max(-1),threshold).sum())
                for f,m in ((30,3),(10,1),(5,.5),(5,.3),(5,.2),(2,.2)):
                    row[f'wrench_F{f}_M{m}'] = float(wrench_cost(df[t,env],dm[t,env],f,m).sum())
                row['proposed_contact_5to20N_peak_w10'] = float((10*dt*np.clip((force.max(-1)-5)/15,0,1)**2).sum())
                row['proposed_wrench_F5_M03_w10'] = float(wrench_cost(df[t,env],dm[t,env],5,.3,10).sum())
                for tolerance in (.001,.002,.005):
                    peak_excess = np.maximum(disp.max(0)-tolerance,0).sum()
                    for k in (50,100,200):
                        row[f'peak_cost_tol{tolerance}_k{k}'] = float(k*peak_excess)
                row['displacement_raw_peak'] = float(np.minimum((disp**2).sum(-1),.04).max())
                row['displacement_cap_steps'] = int(((disp**2).sum(-1)>=.04).sum())
                row['remaining_after_1cm_s'] = row['duration_s']-row['first_1cm_s'] if row['first_1cm_s'] is not None else None
                row['discounted_displacement_cost'] = float((-rewards[:,oi]*.98**np.arange(len(t))).sum())
                for obj in np.flatnonzero(active[t[0],env]):
                    objload = d['substep_object_hand_normal_load'][t,env,:,obj]
                    objects_rows.append(dict(run=run,env=env,episode=ep,object=int(obj),
                        max_displacement_m=float(disp[:,obj].max()),
                        hand_contact_peak_N=float(objload.max()),
                        hand_contact_samples=int((objload>.1).sum())))
                for key in ('obstacle_contact','wrench_change','obstacle_displacement','sweep_progress','sweep_retreat','contact_formation'):
                    row['reward_'+key] = float(rewards[:,names.index(key)].sum())
                episodes.append(row)
                for name,values in dict(obstacle_force_all=force,obstacle_load_all=load,
                    obstacle_force_contact=force[force>.1],obstacle_load_contact=load[load>.1],
                    target_load_contact=trg[trg>.1],whole_hand_load_contact=whole[whole>.1],
                    wrist_force=pforce,wrist_moment=pmoment,wrist_force_delta=df[t,env][1:],wrist_moment_delta=dm[t,env][1:],
                    physics_force_delta=p_df,physics_moment_delta=p_dm,physics_vector_force_delta=vec_df,
                    physics_vector_moment_delta=vec_dm,raw_force=np.linalg.norm(raw[:,:3],axis=-1),
                    support_force=np.linalg.norm(support[:,:3],axis=-1)).items():
                    pool(name,values)
                pool('missed_50N', (force[:,0]>50)&(force[:,1]<=50))
                pool('missed_5N', (force[:,0]>5)&(force[:,1]<=5))
                pool('missed_10N', (force[:,0]>10)&(force[:,1]<=10))
                pool('above_5N_substeps',force>5)
                pool('above_10N_substeps',force>10)
                pool('above_5N_policy_delta',df[t,env][1:]>5)
                pool('above_03Nm_policy_delta',dm[t,env][1:]>.3)
                pool('above_05Nm_policy_delta',dm[t,env][1:]>.5)
                pool('cancellation_50N', (load>50)&(force<=50))
                pool('large_contact_tiny_wrench',(whole>100)&(pforce<1))
                # Contiguous obstacle contacts from the 100 Hz load proxy.
                continuous = load.ravel(); on=continuous>.1
                edges=np.diff(np.r_[False,on,False].astype(int)); starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)
                for start,end in zip(starts,ends):
                    events.append(dict(run=run,env=env,episode=ep,start_s=(start+1)*h,duration_s=(end-start)*h,
                                       peak_load_N=float(continuous[start:end].max()),normal_impulse_proxy_Ns=float(continuous[start:end].sum()*h),
                                       episode_max_displacement_m=row['max_displacement_m']))
                cases.append((row,dict(time=np.arange(1,len(t)+1)*dt,force=force,load=load,target=targetload[t,env],
                    wrist_f=norm_f[t,env],wrist_m=norm_m[t,env],df=df[t,env],dm=dm[t,env],
                    disp=disp.max(-1),distance=d['distance'][t,env],rewards=rewards)))
        del d
    csv_write(root/'episodes.csv',episodes); csv_write(root/'contact_events.csv',events)
    csv_write(root/'objects.csv',objects_rows)
    csv_write(root/'large_contact_samples.csv',large_contacts)
    def group(rows):
        return dict(n=len(rows),successes=sum(r['success'] for r in rows),
                    metrics={k:stats([r[k] for r in rows]) for k in ('max_displacement_m','displacement_cost','contact_cost','wrench_cost','positive_reward','total_reward','obstacle_peak_N','target_peak_N')},
                    moved_counts={str(v):sum(r['max_displacement_m']>=v for r in rows) for v in (.001,.002,.005,.01,.02,.05)},
                    contact_nonzero_episodes=sum(r['contact_steps']>0 for r in rows),wrench_nonzero_episodes=sum(r['wrench_steps']>0 for r in rows),
                    contact_nonzero_steps=sum(r['contact_steps'] for r in rows),wrench_nonzero_steps=sum(r['wrench_steps'] for r in rows),
                    cap_steps=sum(r['displacement_cap_steps'] for r in rows),
                    cost_over_positive=sum(r['displacement_cost'] for r in rows)/max(sum(r['positive_reward'] for r in rows),1e-12))
    groups={'all':episodes,'with_obstacles':[r for r in episodes if r['obstacle_count']],
            'moved_1cm':[r for r in episodes if r['max_displacement_m']>=.01],
            'moved_1_to_5cm':[r for r in episodes if .01<=r['max_displacement_m']<.05],
            'moved_5cm':[r for r in episodes if r['max_displacement_m']>=.05],
            'quiet_success':[r for r in episodes if r['success'] and r['max_displacement_m']<.002]}
    for run in args.runs: groups[run]=[r for r in episodes if r['run']==run]
    distributions={name:stats(np.concatenate(values)) for name,values in pools.items()}
    for label,group_rows in groups.items():
        for key in episodes[0]:
            if key.startswith(('contact_last_','contact_peak_','load_peak_','wrench_F','peak_cost_','displacement_cost_w','proposed_')):
                values=[r[key] for r in group_rows]
                sensitivities.append(dict(group=label,candidate=key,n=len(values),mean=float(np.mean(values)) if values else 0,
                                          sum=float(np.sum(values)),nonzero_episodes=sum(v>1e-10 for v in values)))
    csv_write(root/'sensitivity.csv',sensitivities)
    summary=dict(groups={name:group(rows) for name,rows in groups.items()},distributions=distributions,quality=quality,
                 contact_events=dict(all=stats([r['peak_load_N'] for r in events]),
                    short_le_30ms=stats([r['peak_load_N'] for r in events if r['duration_s']<=.03]),
                    sustained_ge_100ms=stats([r['peak_load_N'] for r in events if r['duration_s']>=.1])),
                 moved_objects=dict(n=sum(r['max_displacement_m']>=.01 for r in objects_rows),
                    without_hand_contact=sum(r['max_displacement_m']>=.01 and r['hand_contact_peak_N']<=.1 for r in objects_rows)),
                 total_policy_steps=sum(r['n_steps'] for r in episodes))
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Select observed examples, not synthetic trajectories.
    selectors={'largest_displacement':lambda r:r['max_displacement_m'], 'largest_contact':lambda r:r['obstacle_load_peak_N'],
               'largest_target_contact':lambda r:r['target_peak_N']}
    selected=[]
    for label,key in selectors.items():
        row,trace=max(cases,key=lambda pair:key(pair[0])); selected.append(dict(label=label,**row))
        fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
        axes[0].plot(trace['time'],trace['load'].max(-1),label='Obstacle normal-load proxy (substep peak)')
        axes[0].plot(trace['time'],trace['force'][:,-1],label='Obstacle resultant (reward sample)',alpha=.7)
        axes[0].axhline(50,color='r',ls='--',label='Current 50 N threshold');axes[0].set_ylabel('N');axes[0].legend(fontsize=8)
        axes[1].plot(trace['time'],trace['target'].max(-1),label='Target normal load')
        axes[1].plot(trace['time'],trace['wrist_f'],label='Compensated wrist force');axes[1].set_ylabel('N');axes[1].legend()
        axes[2].plot(trace['time'],1000*trace['disp'],label='Largest obstacle displacement')
        axes[2].plot(trace['time'],1000*trace['distance'],label='Target goal error');axes[2].set_ylabel('mm');axes[2].legend()
        for name in ('obstacle_displacement','obstacle_contact','wrench_change','sweep_progress','sweep_retreat','contact_formation'):
            axes[3].plot(trace['time'],trace['rewards'][:,names.index(name)].cumsum(),label=name)
        axes[3].set(xlabel='Episode time (s)',ylabel='Cumulative reward');axes[3].legend(ncol=2,fontsize=8)
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f"{label}: {row['run']} / env {row['env']} / episode {row['episode']} / success {row['success']}")
        fig.tight_layout();fig.savefig(root/'figures'/f'{label}.png',dpi=140);plt.close(fig)
    csv_write(root/'selected_cases.csv',selected)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for success,color in ((True,'#187b65'),(False,'#bd5947')):
        rr=[r for r in episodes if r['obstacle_count'] and r['success']==success]
        axes[0].scatter([r['max_displacement_m']*1000 for r in rr],[r['displacement_cost'] for r in rr],s=18,alpha=.6,label='Success' if success else 'Failure',color=color)
    axes[0].set(xlabel='Maximum obstacle displacement (mm)',ylabel='Episode displacement cost');axes[0].legend()
    for name,label in (('obstacle_force_contact','Obstacle resultant'),('obstacle_load_contact','Obstacle load proxy'),('target_load_contact','Target load')):
        x=np.sort(np.concatenate(pools[name])); axes[1].plot(x,np.arange(1,len(x)+1)/len(x),label=label)
    axes[1].set(xscale='log',xlabel='Contact force > 0.1 N (N)',ylabel='Empirical CDF');axes[1].axvline(50,ls='--',color='gray');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'figures/overview.png',dpi=140);plt.close(fig)
    print(json.dumps({'episodes':len(episodes),'groups':summary['groups'],'quality':quality},indent=2))


if __name__=='__main__':
    main()
