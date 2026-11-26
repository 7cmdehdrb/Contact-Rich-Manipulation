"""Direction-resolved physical behavior and reward audits from completed Play traces."""
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
from report_prof_revision_play import angle, distribution, write_csv, wilson

REGIONS={1:'Away (-X)',2:'+Y side',3:'Toward robot (+X)',4:'-Y side'}
FINGERS=('thumb','index','middle','ring','little')


def frac(n,d):
    return float(n/d) if d else None


def bootstrap(rows,field='success'):
    clusters={}
    for e in rows:clusters.setdefault((e['seed'],e['env']),[]).append(e[field])
    values=np.array([(np.sum(v),len(v)) for v in clusters.values()])
    rng=np.random.default_rng(26)
    picked=values[rng.integers(0,len(values),(5000,len(values)))].sum(1)
    return np.quantile(picked[:,0]/picked[:,1],[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runs',nargs='+',default=['seed42','seed43'])
    args=parser.parse_args();root=args.root.resolve();fig=root/'figures';fig.mkdir(exist_ok=True)
    base=list(csv.DictReader((root/'episodes.csv').open(encoding='utf-8-sig')))
    lookup={(e['run'],int(e['env']),int(e['episode'])):e for e in base}
    episodes=[];fingers=[];objects=[];quality=[];pools={k:{} for k in range(5)}
    reward_cubes=[];case_paths={};training_checks={}
    for run in args.runs:
        folder=root/run;meta=json.loads((folder/'metadata.json').read_text())
        assert meta['status']=='complete'
        config=yaml.load((folder/'environment.yaml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
        trained=yaml.load((Path(meta['checkpoint']).parent/'params/env.yaml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
        managers=['actions','observations','commands','events','rewards','terminations','curriculum']
        matches={k:config.get(k)==trained.get(k) for k in managers}
        assert all(matches.values()),matches
        assert config['formation_mode']=='distance_angle_sum'
        assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
        training_checks[run]=dict(manager_matches=matches,checkpoint_sha256=meta['checkpoint_sha256'])
        with np.load(folder/'trace.npz') as z:
            keys=['episode','step','done','target','target_link','goal','distance','sweep_direction','force','contact',
                  'contact_valid','contact_recognized','contact_substeps','finger_gap','finger_normal_cosine','formation',
                  'target_normal_by_body','object_positions','initial_object_positions','active_objects','object_velocity',
                  'reward_terms','reward_raw','reward_total','alignment','positive','retreat','reach_delta','wrist_velocity',
                  'hand_joint_position','hand_joint_velocity','action','previous_action','object_tilt']
            d={k:z[k] for k in keys}
            d.update({k:z[k] for k in z.files if k.startswith('termination_')})
            if 'wrench_raw_world' in z.files:
                d.update({k:z[k] for k in ('wrench_raw_world','wrench_pure_world','wrench_support_world')})
        dt=meta['dt'];terms=meta['reward_terms'];weights=np.array(meta['reward_weights'])
        scale=float(config['formation_distance_scale']);cos=np.clip(d['finger_normal_cosine'],-1,1)
        proximity=scale/(scale+d['finger_gap']);contribution=proximity*cos**2
        mask=d['episode']<meta['episodes_per_env']
        distance=np.linalg.norm(d['target'][...,:2]-d['goal'][...,:2],axis=-1)
        speed=np.linalg.norm(d['object_velocity'][...,0,:2],axis=-1)
        force_xy=np.linalg.norm(d['force'][...,:2],axis=-1)
        moving=(speed>=.005)&(distance>.03)
        pushing=moving&d['contact']&d['contact_valid']&(force_xy>=1.)
        force_error=angle(d['force'][...,:2],d['goal'][...,:2]-d['target'][...,:2])
        velocity_error=angle(d['object_velocity'][...,0,:2],d['goal'][...,:2]-d['target'][...,:2])
        obmask=d['active_objects'].copy();obmask[...,0]=False
        displacement=np.linalg.norm(d['object_positions'].astype(float)-d['initial_object_positions'].astype(float),axis=-1)
        displacement_raw=np.minimum(np.where(obmask,displacement**2,0).sum(-1),.04)
        independent_reward=d['reward_raw']*weights*dt
        errors=dict(run=run,formation=float(abs(contribution.sum(-1)-d['formation']).max()),
            formation_reward=float(abs(contribution.sum(-1)*.245*dt-d['reward_terms'][...,terms.index('contact_formation')]).max()),
            obstacle_cost=float(abs(-150*dt*displacement_raw-d['reward_terms'][...,terms.index('obstacle_displacement')]).max()),
            all_term_weight_dt=float(abs(independent_reward-d['reward_terms']).max()),
            reward_sum=float(abs(d['reward_terms'].sum(-1)-d['reward_total']).max()),
            finite=all(np.isfinite(v).all() for v in d.values()),
            retained_frames=int(mask.sum()),contact_valid_fraction=float(d['contact_valid'][mask].mean()))
        assert errors['formation']<2e-5 and errors['obstacle_cost']<2e-6 and errors['all_term_weight_dt']<2e-5 and errors['finite'],errors
        for env in range(meta['num_envs']):
            for ep in range(meta['episodes_per_env']):
                ix=np.flatnonzero(d['episode'][:,env]==ep);sl=(ix,env);last=ix[-1];length=len(ix)
                assert d['done'][last,env] and np.array_equal(d['step'][sl],np.arange(1,length+1))
                assert np.all(d['active_objects'][sl]==d['active_objects'][ix[0],env])
                assert np.all(d['initial_object_positions'][sl]==d['initial_object_positions'][ix[0],env])
                old=lookup[(run,env,ep)]
                e={k:(None if v=='' else float(v)) for k,v in old.items() if k not in ('run','outcome')}
                e.update(run=run,outcome=old['outcome'])
                for k in ('seed','env','episode','region','steps','success','obstacles','ever_in_goal','left_after_entry'):
                    e[k]=int(e[k])
                u=d['sweep_direction'][ix[0],env,:2].astype(float);u/=np.linalg.norm(u)
                side=np.array([-u[1],u[0]])
                initial=d['initial_object_positions'][ix[0],env,0,:2].astype(float)
                actual=d['object_positions'][last,env,0,:2].astype(float)-initial
                length_m=float(np.linalg.norm(actual))
                heading=float(np.degrees(np.arctan2(actual[1],-actual[0]))) if length_m>=.005 else None
                endpoint_error=float(angle(actual,u)) if length_m>=.005 else None
                final_error=d['target'][last,env,:2]-d['goal'][last,env,:2]
                angle_deg=float(e['goal_angle_deg'])
                sector=int(np.floor((angle_deg+180)%360/30))
                rewards=d['reward_terms'][sl].astype(float)
                no_contact=~d['contact'][sl]
                static=(distance[sl]>.03)&(speed[sl]<.001)
                obs_disp=displacement[sl][:,obmask[ix[0],env]]
                final_obs=obs_disp[-1] if obs_disp.shape[-1] else np.array([])
                force=d['force'][sl];m=moving[sl];p=pushing[sl]
                last1=d['step'][sl]>max(0,500-50)
                e.update(sector=sector,sector_low_deg=sector*30-180,sector_high_deg=(sector+1)*30-180,
                    final_error_along_mm=float(final_error@u*1000),final_error_cross_mm=float(final_error@side*1000),
                    final_error_x_mm=float(final_error[0]*1000),final_error_y_mm=float(final_error[1]*1000),
                    final_tracked_x_m=float(d['target'][last,env,0]),final_tracked_y_m=float(d['target'][last,env,1]),
                    root_travel_mm=length_m*1000,root_progress_mm=float(actual@u*1000),
                    movement_heading_deg=heading,movement_goal_error_deg=endpoint_error,
                    positive_reward=float(np.maximum(rewards,0).sum()),negative_reward=float(np.maximum(-rewards,0).sum()),
                    formation_no_contact=float(rewards[no_contact,terms.index('contact_formation')].sum()),
                    formation_static_outside=float(rewards[static,terms.index('contact_formation')].sum()),
                    sweep_inside_goal=float(rewards[distance[sl]<=.03,terms.index('sweep_progress')].sum()),
                    sweep_after2s=float(rewards[d['step'][sl]>100,terms.index('sweep_progress')].sum()),
                    final_obstacle_max_mm=float(final_obs.max()*1000) if len(final_obs) else 0.,
                    obstacle_sum_peak_mm=float(obs_disp.max(0).sum()*1000) if len(final_obs) else 0.,
                    obstacle_contact_nonzero_steps=int((rewards[:,terms.index('obstacle_contact')]<-1e-12).sum()),
                    moving_contact_steps=int((m&d['contact'][sl]).sum()),moving_n=int(m.sum()),
                    moving_recognized_steps=int((m&d['contact_recognized'][sl]).sum()),
                    moving_both_substeps_contact=int((m&(d['contact_substeps'][sl][...,3]>=.01).all(-1)).sum()),
                    pushing_n=int(p.sum()),velocity_error_n=int(m.sum()),
                    final1s_contact_fraction=float(d['contact'][sl][last1].mean()) if last1.any() else None,
                    formation_mean=float(contribution[sl].sum(-1).mean()))
                assert abs(e['total_reward']-float(rewards.sum()))<2e-5
                episodes.append(e)
                cube=np.zeros((500,len(terms)));cube[:length]=rewards;reward_cubes.append(cube)
                case_paths[(run,env,ep)]=dict(xy=d['target'][sl][:,:2].copy(),goal=d['goal'][last,env,:2].copy(),
                    start=d['target'][ix[0],env,:2].copy(),initial=initial.copy(),time=d['step'][sl]*dt,
                    reward=rewards,formation=contribution[sl].sum(-1),distance=distance[sl],contact=d['contact'][sl])
                loads=np.linalg.norm(d['target_normal_by_body'][sl],axis=-1)
                for j,name in enumerate(FINGERS):
                    bodies=[k for k,body in enumerate(meta['hand_contact_body_names']) if name in body]
                    load=loads[:,bodies].sum(-1)
                    values=dict(run=run,seed=e['seed'],env=env,episode=ep,region=e['region'],finger=name,
                        pushing_frames=int(p.sum()),all_frames=length,
                        all_contribution_sum=float(contribution[sl][:,j].sum()),
                        pushing_contribution_sum=float(contribution[sl][:,j][p].sum()),
                        pushing_gap_sum_mm=float(d['finger_gap'][sl][:,j][p].sum()*1000),
                        pushing_angle_error_sum_deg=float(np.degrees(np.arccos(abs(cos[sl][:,j][p]))).sum()),
                        pushing_within5mm=int((d['finger_gap'][sl][:,j][p]<=.005).sum()),
                        pushing_aligned30=int((abs(cos[sl][:,j][p])>=np.cos(np.pi/6)).sum()),
                        pushing_close_and_aligned=int(((d['finger_gap'][sl][:,j][p]<=.005)&(abs(cos[sl][:,j][p])>=np.cos(np.pi/6))).sum()),
                        pushing_target_contact=int((load[p]>=.05).sum()),pushing_load_sum=float(load[p].sum()))
                    fingers.append(values)
                for j in np.flatnonzero(obmask[ix[0],env]):
                    dd=displacement[sl][:,j];path=d['object_positions'][sl][:,j]
                    path=np.concatenate((d['initial_object_positions'][ix[0],env,j][None],path))
                    objects.append(dict(run=run,env=env,episode=ep,region=e['region'],slot=int(j),
                        peak_mm=float(dd.max()*1000),final_mm=float(dd[-1]*1000),path_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=-1).sum()*1000),
                        pre_cap_cost=float(150*dt*(dd**2).sum())))
                for group in (0,e['region']):
                    values=dict(force_error=force_error[sl][p],velocity_error=velocity_error[sl][m],
                        force_early_error=force_error[sl][p&(d['step'][sl]<=50)],
                        force_far_error=force_error[sl][p&(distance[sl]>.06)],
                        force_near_error=force_error[sl][p&(distance[sl]<=.06)],
                        force_n=np.linalg.norm(force,axis=-1)[p],
                        pure_wrench_n=np.linalg.norm(d['wrench_pure_world'][sl][:,:3],axis=-1) if 'wrench_pure_world' in d else np.array([]),
                        raw_wrench_n=np.linalg.norm(d['wrench_raw_world'][sl][:,:3],axis=-1) if 'wrench_raw_world' in d else np.array([]),
                        formation=contribution[sl].sum(-1))
                    for k,v in values.items():pools[group].setdefault(k,[]).append(v)
        quality.append(errors)
        del d
    write_csv(root/'direction_episodes.csv',episodes);write_csv(root/'direction_finger_episodes.csv',fingers);write_csv(root/'direction_obstacles.csv',objects)
    curves=np.array(reward_cubes)
    grouped=[];rewards_by_group=[];finger_groups=[];curves_by_group={}
    for dimension,values in [('all',[0]),('region',[1,2,3,4]),('sector',list(range(12))),('seed',sorted({e['seed'] for e in episodes}))]:
        for value in values:
            ix=[i for i,e in enumerate(episodes) if dimension=='all' or e[dimension]==value]
            if not ix:continue
            g=[episodes[i] for i in ix];n=len(g);success=sum(e['success'] for e in g)
            row=dict(dimension=dimension,value=value,n=n,successes=success,success_rate=success/n,
                wilson95=wilson(success,n),cluster_bootstrap95=bootstrap(g),
                seed_counts={str(seed):sum(e['seed']==seed for e in g) for seed in sorted({e['seed'] for e in g})},
                reached=sum(e['ever_in_goal'] for e in g),reached_but_failed=sum(e['ever_in_goal'] and not e['success'] for e in g),
                final_distance_mm=distribution([e['final_distance_mm'] for e in g]),
                final_error_along_mm=distribution([e['final_error_along_mm'] for e in g]),
                final_error_cross_mm=distribution([e['final_error_cross_mm'] for e in g]),
                movement_goal_error_deg=distribution([e['movement_goal_error_deg'] for e in g]),
                moving_contact=frac(sum(e['moving_contact_steps'] for e in g),sum(e['moving_n'] for e in g)),
                moving_recognized=frac(sum(e['moving_recognized_steps'] for e in g),sum(e['moving_n'] for e in g)),
                moving_continuous=frac(sum(e['moving_both_substeps_contact'] for e in g),sum(e['moving_n'] for e in g)),
                moving_frames=sum(e['moving_n'] for e in g),pushing_frames=sum(e['pushing_n'] for e in g),
                transport_contact_episode_mean=distribution([e['transport_contact_fraction'] for e in g]),
                positive_mean=float(np.mean([e['positive_reward'] for e in g])),negative_mean=float(np.mean([e['negative_reward'] for e in g])),
                net_mean=float(np.mean([e['total_reward'] for e in g])),
                formation_no_contact_share=frac(sum(e['formation_no_contact'] for e in g),sum(e['reward_contact_formation'] for e in g)),
                formation_static_outside_share=frac(sum(e['formation_static_outside'] for e in g),sum(e['reward_contact_formation'] for e in g)),
                sweep_inside_goal_share=frac(sum(e['sweep_inside_goal'] for e in g),sum(e['reward_sweep_progress'] for e in g)),
                sweep_after2s_share=frac(sum(e['sweep_after2s'] for e in g),sum(e['reward_sweep_progress'] for e in g)),
                with_obstacles=sum(e['obstacles']>0 for e in g),
                obstacle_peak_mm=distribution([e['obstacle_displacement_max_mm'] for e in g if e['obstacles']>0]),
                obstacle_moved_1cm=sum(e['obstacle_displacement_max_mm']>=10 for e in g),
                obstacle_moved_5cm=sum(e['obstacle_displacement_max_mm']>=50 for e in g),
                success_with_obstacle_moved_1cm=sum(e['success'] and e['obstacle_displacement_max_mm']>=10 for e in g),
                obstacle_cost_mean=float(-np.mean([e['reward_obstacle_displacement'] for e in g])),
                termination_counts={k:sum(e['outcome']==k for e in g) for k in sorted({e['outcome'] for e in g})})
            if dimension in ('all','region'):
                pool={k:np.concatenate(v) for k,v in pools[value].items()}
                row.update({k:distribution(v) for k,v in pool.items()})
                row['force_good_30']=frac(int((pool['force_error']<=30).sum()),len(pool['force_error']))
                row['force_wrong_90']=frac(int((pool['force_error']>90).sum()),len(pool['force_error']))
                row['velocity_good_30']=frac(int((pool['velocity_error']<=30).sum()),len(pool['velocity_error']))
                curves_by_group[value]=curves[ix].mean(0)
                fg=[f for f in fingers if dimension=='all' or f['region']==value]
                for name in FINGERS:
                    ff=[f for f in fg if f['finger']==name];frames=sum(f['pushing_frames'] for f in ff)
                    numerator=lambda key:sum(f[key] for f in ff)
                    finger_groups.append(dict(dimension=dimension,value=value,finger=name,pushing_frames=frames,
                        mean_pushing_gap_mm=frac(numerator('pushing_gap_sum_mm'),frames),
                        mean_pushing_normal_error_deg=frac(numerator('pushing_angle_error_sum_deg'),frames),
                        within5mm=frac(numerator('pushing_within5mm'),frames),aligned30=frac(numerator('pushing_aligned30'),frames),
                        close_aligned=frac(numerator('pushing_close_and_aligned'),frames),
                        target_contact=frac(numerator('pushing_target_contact'),frames),
                        all_reward_mean=.245*.02*numerator('all_contribution_sum')/n,
                        all_contribution_mean=frac(numerator('all_contribution_sum'),sum(f['all_frames'] for f in ff)),
                        pushing_contribution_mean=frac(numerator('pushing_contribution_sum'),frames)))
            grouped.append(row)
            positive=sum(e['positive_reward'] for e in g);negative=sum(e['negative_reward'] for e in g)
            for term in terms:
                v=np.array([e['reward_'+term] for e in g])
                contributions=curves[ix,:,terms.index(term)]
                rewards_by_group.append(dict(dimension=dimension,value=value,term=term,n=n,mean=float(v.mean()),
                    p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),
                    share_positive=frac(float(np.maximum(contributions,0).sum()),positive),
                    share_negative=frac(float(np.maximum(-contributions,0).sum()),negative)))
    result=dict(task='Inspire-Sweep-Prof-Revision-v0_3_1',checkpoint=meta['checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],
        formation_mode='distance_angle_sum',groups=grouped,quality=quality,training_config=training_checks)
    (root/'direction_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(root/'direction_rewards.csv',rewards_by_group);write_csv(root/'direction_fingers.csv',finger_groups)
    draw_figures(fig,episodes,grouped,rewards_by_group,finger_groups,curves_by_group,terms)
    print(json.dumps([g for g in grouped if g['dimension']=='all'],indent=2))


def draw_figures(folder,episodes,groups,rewards,fingers,curves,terms):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':110,'savefig.dpi':160})
    colors=['#3d78a5','#c58a37','#9c5387','#36896d']
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    for region,ax in zip(range(1,5),axes.flat):
        rows=[e for e in episodes if e['region']==region]
        g=next(g for g in groups if g['dimension']=='region' and g['value']==region)
        for flag,color,label in ((1,'#2f8b62','Success'),(0,'#bd5d55','Failure')):
            x=[e for e in rows if e['success']==flag]
            ax.scatter([e['final_error_cross_mm'] for e in x],[e['final_error_along_mm'] for e in x],c=color,label=label,s=13,alpha=.6)
        ax.add_patch(plt.Circle((0,0),30,fill=False,ls='--',color='black'))
        ax.axhline(0,color='gray',lw=.4);ax.axvline(0,color='gray',lw=.4)
        ax.set(title=f'{REGIONS[region]} | {g["successes"]}/{g["n"]} success',xlabel='Cross-track final error (mm)',ylabel='Along-goal final error (mm)',aspect='equal')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Final tracked-point errors | origin = goal | dashed radius = 30 mm')
    fig.tight_layout();fig.savefig(folder/'direction_final_positions.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    g=[g for g in groups if g['dimension']=='region'];x=np.arange(4)
    rates=np.array([r['success_rate'] for r in g]);ci=np.array([r['cluster_bootstrap95'] for r in g])
    axes[0,0].bar(x,100*rates,color=colors);axes[0,0].errorbar(x,100*rates,yerr=100*np.maximum(0,np.stack((rates-ci[:,0],ci[:,1]-rates))),fmt='none',color='black',capsize=4)
    for i,r in enumerate(g):axes[0,0].text(i,max(r['success_rate'],r['cluster_bootstrap95'][1])*100+3,f'{r["successes"]}/{r["n"]}',ha='center')
    axes[0,0].set(ylabel='Success (%)',ylim=(0,110),title='95% cluster bootstrap interval')
    axes[0,1].bar(x,[r['moving_contact']*100 for r in g],color=colors);axes[0,1].set(ylabel='Moving frames with target contact (%)',ylim=(0,100))
    axes[1,0].boxplot([[e['movement_goal_error_deg'] for e in episodes if e['region']==r and e['movement_goal_error_deg'] is not None] for r in range(1,5)],positions=x,showfliers=True)
    axes[1,0].set(ylabel='Net root-displacement error (deg)',ylim=(0,180))
    axes[1,1].bar(x,[r['force_error']['mean'] for r in g],color=colors);axes[1,1].set(ylabel='Moving-contact force error (deg)',ylim=(0,180))
    for ax in axes.flat:ax.set_xticks(x,[REGIONS[i] for i in range(1,5)],rotation=12);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'direction_performance.png');plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(16,11),sharex=True)
    t=np.arange(1,501)*.02
    for j,term in enumerate(terms):
        ax=axes.flat[j]
        for region,color in zip(range(1,5),colors):ax.plot(t,curves[region][:,j],color=color,lw=.9,label=REGIONS[region])
        ax.axhline(0,color='gray',lw=.4);ax.set_title(term)
        if j>=12:ax.set_xlabel('Episode time (s)')
        if j%4==0:ax.set_ylabel('Reward per policy step')
    axes.flat[0].legend(fontsize=7);fig.suptitle('Direction-specific reward timelines | fixed cohort, zero contribution after termination')
    fig.tight_layout();fig.savefig(folder/'direction_reward_timelines.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    g=[g for g in groups if g['dimension']=='region']
    for i,r in enumerate(g):
        rows=[e['obstacle_displacement_max_mm'] for e in episodes if e['region']==r['value'] and e['obstacles']>0]
        rows=np.sort(rows);axes[0].step(rows,np.arange(1,len(rows)+1)/len(rows)*100,label=REGIONS[r['value']],color=colors[i])
    axes[0].set(xlabel='Maximum obstacle displacement / episode (mm)',ylabel='Cumulative episodes with obstacles (%)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for j,metric in enumerate(('mean_pushing_gap_mm','close_aligned')):
        matrix=np.array([[next(f for f in fingers if f['dimension']=='region' and f['value']==r and f['finger']==name)[metric] for name in FINGERS] for r in range(1,5)])
        if metric=='close_aligned':matrix*=100
        image=axes[j+1].imshow(matrix,aspect='auto',cmap='viridis',vmin=0,vmax=100 if metric=='close_aligned' else None)
        axes[j+1].set(xticks=range(5),xticklabels=FINGERS,yticks=range(4),yticklabels=[REGIONS[r] for r in range(1,5)],title='Gap (mm)' if metric=='mean_pushing_gap_mm' else 'Gap <=5mm & pad error <=30deg (%)')
        axes[j+1].tick_params(axis='x',rotation=25)
        for a in range(4):
            for b in range(5):axes[j+1].text(b,a,f'{matrix[a,b]:.1f}',ha='center',va='center',color='white' if matrix[a,b]<np.nanmax(matrix)*.65 else 'black',fontsize=8)
        fig.colorbar(image,ax=axes[j+1],shrink=.7)
    fig.tight_layout();fig.savefig(folder/'direction_obstacles_fingers.png');plt.close(fig)


if __name__=='__main__':main()
