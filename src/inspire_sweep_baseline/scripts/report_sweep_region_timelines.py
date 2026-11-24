"""Produce a source-backed regional checkpoint report from physical rollout traces."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import warnings
import yaml

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = [('sweep_v0', 'Sweep v0'), ('single_v0', 'Single Contact v0'), ('single_v1', 'Single Contact v1')]
COLORS = ['#2563eb', '#ea580c', '#0f766e']
BIN = 10  # .2 s; each source sample is .02 s


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stat(x, fn=np.mean):
    x = np.asarray(x)
    return float(fn(x)) if x.size else float('nan')


def fmt(x, places=2):
    return f'{x:.{places}f}' if np.isfinite(x) else '—'


def masked_bins(x, mask):
    x = np.asarray(x)
    expanded = mask[(...,)+(None,)*(x.ndim-mask.ndim)]
    v = np.where(expanded, x, np.nan)
    v = v.reshape(len(v)//BIN, BIN*v.shape[1], *v.shape[2:])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(v, axis=1), np.nanquantile(v, [.1,.9], axis=1)


def setup_time(ax, ylabel):
    ax.set(xlim=(0,10), xlabel='Episode time (s)', ylabel=ylabel)
    ax.grid(alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':9, 'axes.titlesize':11, 'figure.facecolor':'white', 'axes.spines.top':False, 'axes.spines.right':False})
    data, metadata, checks, config_checks = {}, {}, [], {}
    for key, label in MODELS:
        traces, metas = [], []
        for seed in (42,43):
            folder = args.input/f'{key}_seed{seed}'
            m = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            assert m['status'] == 'complete' and m['steps'] == 500 and m['num_envs'] == 256
            with np.load(folder/'trace.npz') as z:
                a = {k:z[k] for k in z.files}
            assert all(len(v) == 500 for v in a.values())
            assert np.isfinite(a['reward_terms']).all() and np.isfinite(a['force_w']).all() and np.isfinite(a['wrench_eef']).all()
            assert set(np.unique(a['tactile_raw'])).issubset({-1,0,1,2})
            assert np.max(np.abs(a['policy_wrench'])) <= 1.000001
            assert np.max(np.abs(a['reward_terms'].sum(-1)-a['reward_total'])) < 2e-5
            active = a['episode'] == 0
            assert ((a['done'] & active).sum(0) == 1).all(), 'Every initial episode must end in 10 s'
            assert not (active[1:] & ~active[:-1]).any()
            checks.append(dict(model=key, seed=seed, reward_sum_error=m['max_reward_sum_error'],
                               sensor_fresh_fraction=float(a['sensor_fresh'][active].mean()),
                               contact_valid_fraction=float(a['contact_valid'][active].mean()),
                               checkpoint_sha256=m['checkpoint_sha256'],
                               trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest()))
            if seed == 42:
                current = yaml.load((folder/'environment.yaml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
                trained = yaml.load((Path(m['checkpoint']).parent/'params/env.yaml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
                fields = ('actions','rewards','observations','commands','terminations','events','curriculum','episode_length_s','decimation','goal_success_radius','wrist_wrench_observation_limits')
                config_checks[key] = {field:current[field] == trained[field] for field in fields}
                assert all(config_checks[key].values()), 'Current evaluation config differs from trained task contract'
            if key == 'single_v1':
                terms = m['reward_terms']
                np.testing.assert_allclose(a['reward_terms'][...,terms.index('sweep_progress')],
                    50*(a['sweep_positive']+a['sweep_direction_negative']),atol=2e-6,rtol=2e-5)
                np.testing.assert_allclose(a['reward_terms'][...,terms.index('reaching')],4.9*a['reach_delta'],atol=2e-6,rtol=2e-5)
            traces.append(a); metas.append(m)
        assert metas[0]['reward_terms'] == metas[1]['reward_terms']
        data[key] = {k:np.concatenate([a[k] for a in traces],axis=1) for k in traces[0]}
        metadata[key] = metas
    # A common seed is a paired scene comparison only if physical snapshots agree.
    paired = {}
    for seed_index,seed in enumerate((42,43)):
        paired[str(seed)] = {field:len({metadata[k][seed_index]['initial_hashes'][field] for k,_ in MODELS}) == 1
                             for field in ('joint_pos','object_state','goal','sweep_dir','bounds','active_objects')}
    provenance = dict(checks=checks, paired_initial_scenes=paired, trained_config_comparison=config_checks,
                      script_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ('collect_sweep_region_timelines.py','report_sweep_region_timelines.py')})
    (args.assets/'validation.json').write_text(json.dumps(provenance, indent=2),encoding='utf-8')
    summary, alignments, sensor_summary, term_summary, timeline_rows = [], [], [], [], []
    time = (np.arange(50)+.5)*.2
    regional = {}
    for region in range(1,5):
        regional[region] = {}
        for key,label in MODELS:
            full = data[key]
            ids = full['region'][0] == region
            a = {k:v[:,ids] for k,v in full.items()}
            n = int(ids.sum())
            active = a['episode'] == 0
            valid = active & a['sensor_fresh']
            policy_valid = valid & a['policy_valid']
            contributions = np.where(active[...,None], a['reward_terms'],0)
            total = contributions.sum(-1)
            names = metadata[key][0]['reward_terms']
            sweep = contributions[...,names.index('sweep_progress')]
            reach = contributions[...,names.index('reaching')]
            positive = np.maximum(sweep,0)
            negative = np.minimum(sweep,0)
            returns = total.sum(0)
            success = (a['success'] & active).any(0)
            duration = active.sum(0)*.02
            row = dict(model=label,region=region,n=n,successes=int(success.sum()),
                       success_pct=float(success.mean()*100),return_mean=float(returns.mean()),
                       return_sd=float(returns.std(ddof=1)),duration_mean_s=float(duration.mean()),
                       sweep_positive_return=float(positive.sum(0).mean()),
                       sweep_negative_return=float(negative.sum(0).mean()),
                       reaching_return=float(reach.sum(0).mean()),
                       other_cost_return=float((total-sweep-reach).sum(0).mean()))
            for name in [k for k in a if k.startswith('termination_')]:
                row[name] = int((a[name] & active).any(0).sum())
            summary.append(row)
            for j,name in enumerate(names):
                term_summary.append(dict(model=label,region=region,term=name,mean_episode_return=float(contributions[...,j].sum(0).mean())))
            delta = a['goal'][...,:2]-a['target'][...,:2]
            force = a['force_w']
            fxy = np.linalg.norm(force[...,:2],axis=-1)
            f3 = np.linalg.norm(force,axis=-1)
            distance = np.linalg.norm(delta,axis=-1)
            cosine = (delta*force[...,:2]).sum(-1)/np.maximum(distance*fxy,1e-12)
            angle = np.degrees(np.arccos(np.clip(cosine,-1,1)))
            cosine3 = (delta*force[...,:2]).sum(-1)/np.maximum(distance*f3,1e-12)
            events = valid & a['contact_valid'] & (sweep>0) & (fxy>1e-9) & (distance>1e-9)
            broad = valid & a['contact_valid'] & a['contact'] & (fxy>=1) & (distance>1e-9)
            event_count = events.sum(0)
            angle_by_episode = np.divide(np.where(events,angle,0).sum(0),event_count,
                                         out=np.full(n,np.nan),where=event_count>0)
            ar = dict(model=label,region=region,positive_reward_steps=int(events.sum()),
                      positive_reward_episodes=int((event_count>0).sum()),
                      positive_time_s_per_episode=float(events.sum()*.02/n),
                      angle_mean_deg=stat(angle[events]),angle_median_deg=stat(angle[events],np.median),
                      angle_p90_deg=stat(angle[events],lambda x:np.quantile(x,.9)),
                      within15_pct=stat(angle[events]<=15)*100,within30_pct=stat(angle[events]<=30)*100,
                      cosine_xy_mean=stat(cosine[events]),cosine_3d_mean=stat(cosine3[events]),
                      force_mean_N=stat(f3[events]),
                      equal_episode_angle_mean_deg=stat(angle_by_episode[np.isfinite(angle_by_episode)]),
                      all_contact_steps=int(broad.sum()),all_contact_angle_mean_deg=stat(angle[broad]),
                      all_contact_within30_pct=stat(angle[broad]<=30)*100)
            alignments.append(ar)
            if key == 'single_v1':
                assert (angle[events] < 30.002).all(), 'v1 positive direction coefficient requires angle<30 degrees'
            raw = a['tactile_raw']
            palm = (raw==1)|(raw==2)
            dorsal = (raw==-1)|(raw==2)
            both = raw==2
            force_norm = np.linalg.norm(a['wrench_eef'][...,:3],axis=-1)
            torque_norm = np.linalg.norm(a['wrench_eef'][...,3:],axis=-1)
            sr = dict(model=label,region=region,active_sensor_steps=int(valid.sum()),
                      target_contact_pct=stat(a['contact'][valid])*100,
                      tactile_any_pct=stat((raw!=0).any(-1)[valid])*100,
                      palm_channels_mean=stat(palm.sum(-1)[valid]),dorsal_channels_mean=stat(dorsal.sum(-1)[valid]),
                      both_channels_mean=stat(both.sum(-1)[valid]),
                      target_force_mean_N=stat(f3[valid]),target_force_p95_N=stat(f3[valid],lambda x:np.quantile(x,.95)),
                      target_normal_resultant_mean_N=stat(np.linalg.norm(a['normal_w'],axis=-1)[valid]),
                      target_normal_load_mean_N=stat(a['normal_load'][valid]),
                      ft_force_mean_N=stat(force_norm[valid]),ft_force_p95_N=stat(force_norm[valid],lambda x:np.quantile(x,.95)),
                      ft_torque_mean_Nm=stat(torque_norm[valid]),ft_torque_p95_Nm=stat(torque_norm[valid],lambda x:np.quantile(x,.95)),
                      policy_ft_any_saturation_pct=stat((np.abs(a['policy_wrench'])>=.999999).any(-1)[policy_valid])*100)
            sr['policy_tactile_any_pct'] = stat((a['policy_tactile']!=0).any(-1)[policy_valid])*100
            sr['policy_tactile_active_channels_mean'] = stat((a['policy_tactile']!=0).sum(-1)[policy_valid])
            sr['raw_tactile_any_pct_on_policy_valid_steps'] = stat((raw!=0).any(-1)[policy_valid])*100
            observed = a['policy_tactile']
            observed_bits = np.concatenate([(observed==1)|(observed==2),(observed==-1)|(observed==2)],axis=-1)
            raw_bits = np.concatenate([palm,dorsal],axis=-1)
            sr['observed_tactile_bit_flip_pct'] = stat((observed_bits!=raw_bits)[policy_valid])*100
            pooled_positive = positive.sum(1)
            row['sweep_positive_90pct_time_s'] = float((np.searchsorted(pooled_positive.cumsum(),.9*pooled_positive.sum())+1)*.02) if pooled_positive.sum()>0 else float('nan')
            row['sweep_positive_after3s_pct'] = float(pooled_positive[150:].sum()/pooled_positive.sum()*100) if pooled_positive.sum()>0 else float('nan')
            sensor_summary.append(sr)
            regional[region][key] = dict(a=a,n=n,active=active,valid=valid,policy_valid=policy_valid,
                contributions=contributions,total=total,sweep=sweep,reach=reach,positive=positive,negative=negative,
                angle=angle,events=events,broad=broad,palm=palm,dorsal=dorsal,both=both,f3=f3,
                ft_force=force_norm,ft_torque=torque_norm,row=row,ar=ar,sr=sr)
            for b,t in enumerate(time):
                s=slice(b*BIN,(b+1)*BIN)
                m=valid[s]
                timeline_rows.append(dict(model=label,region=region,time_s=float(t),
                    at_risk_mean=float(active[s].sum()/BIN),
                    reward_per_step=float(total[s].mean()),sweep_positive_per_step=float(positive[s].mean()),
                    sweep_negative_per_step=float(negative[s].mean()),reach_per_step=float(reach[s].mean()),
                    other_cost_per_step=float((total-sweep-reach)[s].mean()),
                    target_contact_pct=stat(a['contact'][s][m])*100,
                    tactile_palm_channels=stat(palm.sum(-1)[s][m]),
                    tactile_dorsal_channels=stat(dorsal.sum(-1)[s][m]),
                    ft_force_mean_N=stat(force_norm[s][m]),ft_torque_mean_Nm=stat(torque_norm[s][m]),
                    rewarded_force_angle_mean_deg=stat(angle[s][events[s]])))
        # Average reward per control step in .2 s bins; fixed initial cohort.
        fig,axs=plt.subplots(3,2,figsize=(13,10),constrained_layout=True,sharey='col')
        for i,(key,label) in enumerate(MODELS):
            d=regional[region][key]; ax=axs[i,0]
            series=[(d['total'],'Total','#111827','-'),(d['positive'],'Sweep +','#0f766e','-'),
                    (d['negative'],'Sweep direction -','#dc2626',':'),(d['reach'],'Reaching','#2563eb','--'),
                    (d['total']-d['sweep']-d['reach'],'Other costs','#a16207','-.')]
            for val,name,color,style in series:
                per_episode=val.reshape(50,BIN,d['n']).mean(1)
                mean=per_episode.mean(1)
                ax.plot(time,mean,label=name,color=color,ls=style)
                if name=='Total':
                    sem=per_episode.std(1,ddof=1)/np.sqrt(d['n'])
                    ax.fill_between(time,mean-sem,mean+sem,color=color,alpha=.12)
            ax.axhline(0,color='#aaa',lw=.7);setup_time(ax,'Reward / policy step')
            ax.set_title(f'{label} | initial n={d["n"]}');ax.legend(ncol=3,fontsize=7)
            ax=axs[i,1]
            for val,name,color,style in series:
                ax.plot(np.arange(1,501)*.02,val.mean(1).cumsum(),label=name,color=color,ls=style)
            setup_time(ax,'Mean cumulative reward');ax.set_title('Fixed cohort; zero reward after episode end')
        fig.suptitle(f'Region {region} | reward timeline | shaded total: +/- 1 episode SEM')
        fig.savefig(args.assets/f'region{region}_rewards.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(3,1,figsize=(13,11),constrained_layout=True)
        limit=max(float(np.abs(d['contributions'].reshape(50,BIN,d['n'],-1).mean((1,2))).max()) for d in regional[region].values())
        limit=max(limit,1e-8)
        for i,(key,label) in enumerate(MODELS):
            d=regional[region][key]
            matrix=d['contributions'].reshape(50,BIN,d['n'],-1).mean((1,2)).T
            im=axs[i].imshow(matrix,origin='upper',aspect='auto',extent=[0,10,len(matrix)-.5,-.5],cmap='RdBu',vmin=-limit,vmax=limit)
            axs[i].set(yticks=np.arange(len(matrix)),yticklabels=metadata[key][0]['reward_terms'],xlabel='Episode time (s)',title=label)
        fig.colorbar(im,ax=axs,label='Mean contribution / policy step; fixed initial cohort',shrink=.8)
        fig.suptitle(f'Region {region} | all reward terms | identical color scale across policies')
        fig.savefig(args.assets/f'region{region}_reward_terms.png',dpi=160);plt.close(fig)
        # Conditioned alignment with an unconditioned contact reference.
        fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
        for i,(key,label) in enumerate(MODELS):
            d=regional[region][key]
            for mask,ls,suffix in [(d['events'],'-','reward > 0'),(d['broad'],'--','all target contacts >=1 N')]:
                values=np.sort(d['angle'][mask])
                if len(values):axs[0,0].plot(values,np.arange(1,len(values)+1)/len(values),ls=ls,color=COLORS[i],label=f'{label}: {suffix}')
            mean,_=masked_bins(d['angle'],d['events'])
            axs[0,1].plot(time,mean,color=COLORS[i],label=label)
            count=d['events'].reshape(50,BIN,d['n']).sum((1,2))
            axs[1,0].plot(time,count,color=COLORS[i],label=label)
            alive=d['active'].reshape(50,BIN,d['n']).sum((1,2))/BIN
            axs[1,1].plot(time,alive,color=COLORS[i],label=label)
        axs[0,0].axvline(30,color='#aaa',ls=':');axs[0,0].set(xlim=(0,180),ylim=(0,1),xlabel='Horizontal force-goal angle (deg)',ylabel='Cumulative fraction')
        axs[0,0].legend(fontsize=7)
        setup_time(axs[0,1],'Mean rewarded angle (deg)');axs[0,1].set_ylim(0,90)
        setup_time(axs[1,0],'Rewarded env-steps / .2 s bin')
        setup_time(axs[1,1],'Episodes still active (mean / bin)')
        for ax in axs.flat:ax.grid(alpha=.2)
        axs[1,1].legend()
        fig.suptitle(f'Region {region} | force alignment | solid CDF: positive sweep reward only')
        fig.savefig(args.assets/f'region{region}_alignment.png',dpi=160);plt.close(fig)
        # Never average the categorical state encoding: separate both surfaces.
        fig,axs=plt.subplots(3,3,figsize=(16,10),constrained_layout=True,sharey='col')
        heatmaps = {k:masked_bins((d['a']['tactile_raw']!=0).astype(float),d['valid'])[0] for k,d in regional[region].items()}
        heat_limit = max(.01,max(float(np.nanmax(h)) for h in heatmaps.values()))
        for i,(key,label) in enumerate(MODELS):
            d=regional[region][key];a=d['a'];ax=axs[i,0]
            for x,name,color,style in [(d['palm'].sum(-1),'Palm','#2563eb','-'),(d['dorsal'].sum(-1),'Dorsal','#ea580c','--'),(d['both'].sum(-1),'Both','#0f766e',':')]:
                mean,_=masked_bins(x,d['valid']);ax.plot(time,mean,label=name,color=color,ls=style)
            p=a['policy_tactile'];mean,_=masked_bins((p!=0).sum(-1),d['policy_valid'])
            ax.plot(time,mean,label='Observed any',color='#6b7280',ls='-.')
            setup_time(ax,'Mean active channels (of 17)');ax.set_ylim(bottom=0);ax.legend(fontsize=7,ncol=2);ax.set_title(label)
            heat=heatmaps[key]
            im=axs[i,1].imshow(heat.T,aspect='auto',origin='lower',extent=[0,10,.5,17.5],vmin=0,vmax=heat_limit,cmap='viridis')
            axs[i,1].set(xlabel='Episode time (s)',ylabel='Tactile channel',yticks=[1,5,9,13,17],title='Raw activation probability; all contacted bodies')
            ax=axs[i,2]
            mean,q=masked_bins(d['f3'],d['valid']);ax.plot(time,mean,color=COLORS[i],label='Mean target force')
            ax.fill_between(time,q[0],q[1],color=COLORS[i],alpha=.15,label='10-90% sample range')
            mean,_=masked_bins(np.linalg.norm(a['normal_w'],axis=-1),d['valid'])
            ax.plot(time,mean,color='#6b7280',ls='--',label='Normal resultant norm')
            mean,_=masked_bins(a['normal_load'],d['valid'])
            ax.plot(time,mean,color='#9333ea',ls=':',label='Sum of normal point loads')
            setup_time(ax,'Target contact force / load (N)');ax.set_title('Hand contacts on target; physics-substep mean');ax.legend(fontsize=6)
        fig.colorbar(im,ax=axs[:,1],label='Fraction active; common scale within region',shrink=.7)
        fig.suptitle(f'Region {region} | contact/tactile timeline | active initial episodes only')
        fig.savefig(args.assets/f'region{region}_contact.png',dpi=160);plt.close(fig)
        # Signed physical axes plus non-cancelling norms. Separate N and Nm.
        fig,axs=plt.subplots(3,4,figsize=(18,10),constrained_layout=True,sharey='col')
        for i,(key,label) in enumerate(MODELS):
            d=regional[region][key];w=d['a']['wrench_eef']
            for col,offset,ylabel in [(0,0,'Force (N), EEF axes'),(1,3,'Torque (Nm), EEF axes')]:
                for j,(axis,color) in enumerate(zip('XYZ',COLORS)):
                    mean,_=masked_bins(w[...,offset+j],d['valid'])
                    axs[i,col].plot(time,mean,label=axis,color=color,ls=['-','--',':'][j])
                setup_time(axs[i,col],ylabel);axs[i,col].legend(fontsize=7);axs[i,col].set_title(label)
            for col,x,ylabel in [(2,d['ft_force'],'Force norm (N)'),(3,d['ft_torque'],'Torque norm (Nm)')]:
                mean,q=masked_bins(x,d['valid']);axs[i,col].plot(time,mean,color=COLORS[i])
                axs[i,col].fill_between(time,q[0],q[1],color=COLORS[i],alpha=.15)
                setup_time(axs[i,col],ylabel);axs[i,col].set_title('Mean and 10-90% sample range')
        fig.suptitle(f'Region {region} | compensated wrist F/T at EEF | active initial episodes only')
        fig.savefig(args.assets/f'region{region}_ft.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(3,2,figsize=(13,8),constrained_layout=True)
        for i,(key,label) in enumerate(MODELS):
            d=regional[region][key]
            for col,offset in [(0,0),(1,3)]:
                for j,(axis,color) in enumerate(zip('XYZ',COLORS)):
                    mean,_=masked_bins(d['a']['policy_wrench'][...,offset+j],d['policy_valid'])
                    axs[i,col].plot(time,mean,label=axis,color=color,ls=['-','--',':'][j])
                setup_time(axs[i,col],'Normalized observed '+('force' if col==0 else 'torque'))
                axs[i,col].set_ylim(-1,1);axs[i,col].legend(fontsize=7);axs[i,col].set_title(label)
        fig.suptitle(f'Region {region} | actual policy F/T after noise, clipping and normalization')
        fig.savefig(args.assets/f'region{region}_ft_policy.png',dpi=160);plt.close(fig)
    for filename,rows in [('summary.csv',summary),('alignment.csv',alignments),('sensors.csv',sensor_summary),('reward_terms.csv',term_summary),('timeline.csv',timeline_rows)]:
        csv_write(args.assets/filename,rows)
    def rel(path):return Path(os.path.relpath(path,args.report.parent)).as_posix()
    def image(name):return f'![{name}]({rel(args.assets/name)})'
    text=['# Sweep 세 체크포인트의 영역별 보상·접촉·F/T Timeline', '',
          '**실측 평가 보고서 · 2026-11-23 · 초기 episode 0~10초**','',
          '## 핵심 결과','']
    for key,label in MODELS:
        rs=[r for r in summary if r['model']==label]
        n=sum(r['n'] for r in rs);success=sum(r['successes'] for r in rs)
        text.append(f'- **{label}**: {n}개 초기 episode 중 {success}개 성공({success/n*100:.1f}%). 평균 episode return은 {sum(r["return_mean"]*r["n"] for r in rs)/n:.3f}.')
    text.extend(['', '**가장 큰 차이는 영역4다.** 성공률이 Sweep v0 80.5%, Single Contact v0 81.6%, v1 3.4%였다. v1의 양의 Sweep 누적 보상은 0.188로 두 v0의 3.619·3.974보다 작았고, 전체 유효 타겟 접촉의 평균 방향 오차는 85.54°였다.', '',
                 '**영역3에서는 v1이 더 자주 성공했다.** 성공률은 2.5%·13.8%·32.5%였다. 다만 v1의 양의 보상 구간 평균 각도 10.03°와 전체 접촉 평균 106.75°는 크게 다르다. 양의 보상이 들어오는 짧은 구간만 보고 전체 접촉이 잘 정렬됐다고 해석할 수 없다.'])
    text.extend(['', '**보상 크기는 환경 간 동일 척도의 성능 점수가 아니다.** v1은 Reaching을 거리 증분으로 바꾸고 방향 불일치에 음의 Sweep을 주므로, v0와 return의 직접 순위 비교는 피한다. 영역별 성공·접촉·힘 방향을 함께 본다.', '',
        '## 평가 범위와 그래프 읽기','',
        '- 각 체크포인트를 256개 병렬 환경에서 시드 42·43으로 실행했다. 체크포인트당 초기 episode 512개, 총 1,536개다. 체크포인트 간 실행은 GPU 메모리를 고려해 순차 진행했다. 이 환경의 성공은 10초 timeout 시점의 타겟–목표 XY 거리 <3 cm이며, 그 전에 물체·선반·팔 속도 실패가 발생하면 실패 종료다.',
        '- 정책은 deterministic inference이며 학습 환경의 관측 noise를 유지했다. 정책 간 초기 물리 장면 일치 여부는 아래 검증 항목에 기록했다.',
        '- 정책 주기는 0.02초, 물리 주기는 0.01초다. 그래프는 0.2초 구간 평균이며, x축은 episode 시작부터의 시간이다. 0초는 첫 동작 전 상태이며 첫 측정은 0.02초다.',
        '- 보상 y축은 **실제 weight × raw × dt가 적용된 정책 step당 Reward**다. 조기 종료 이후에는 보상 0을 넣어 초기 표본 수를 고정한다. 누적 보상은 실제 step 보상 합이다. Total 음영은 episode 간 표준오차 ±1 SEM이며 95% 신뢰구간이 아니다.',
        '- 센서는 **해당 시각까지 살아 있는 초기 episode**에서 평균한다. 종료 이후 센서값은 존재하지 않으므로 0으로 채우지 않는다. 장면 구성은 시간이 갈수록 달라질 수 있으며, 방향 그래프의 생존 episode 수를 함께 확인해야 한다. 센서 음영은 개별 env-step의 10~90% 범위이며 평균의 신뢰구간이 아니다.',
        '- 힘 방향 일치도는 **양의 `sweep_progress`가 실제 지급된 step**에 한정한다. 현재 타겟→목표 XY 방향과 타겟에 작용하는 손의 접촉력 XY 사이 각도 θ를 사용한다. 0°는 일치, 90°는 수직, 180°는 반대다. 접촉력은 법선+마찰을 물리 substep 평균한 값이다.',
        '- 이 조건은 선택 편향을 갖는다. v0도 양의 정렬을 요구하고, v1은 방향 보상식 자체가 **θ<30°에서만 양수**다. 따라서 v1의 지급 구간에서 30° 이내 비율이 높다는 사실만으로 방향 제어 개선을 입증하지 않는다. 점선 CDF와 표의 ‘전체 접촉’(수평 합력≥1 N) 결과를 함께 제시한다.',
        '- 촉각 17채널은 없음 0·손바닥 1·손등 −1·양면 2의 범주형이다. 범주 코드를 직접 평균하지 않고 면별 활성 수·채널별 활성 확률을 제시한다. 원시 촉각 임계값은 0.05 N이며 타겟뿐 아니라 선반·장애물 접촉도 포함한다. `Observed any`는 실제 정책 입력의 noise 적용 후 활성 채널 수다.',
        '- 손목 F/T는 손의 중력·관성을 보상한 EEF 기준점·EEF 좌표계의 반력이다. N과 Nm을 분리했다. 타겟 접촉력과 좌표계·부호·측정 대상이 다르며, F/T만으로 타겟에 가한 힘을 분리할 수 없다. 노름은 **각 표본에서 먼저 계산한 뒤 평균**해 방향 상쇄를 막았다.',
        '- 공통 각도 지표는 XY 힘만으로 계산한다. 기존 v0 보상식은 같은 내적을 3D 힘 크기로 나누므로 수직력이 크면 정렬 계수가 추가로 낮아진다. alignment.csv의 `cosine_xy_mean`·`cosine_3d_mean`에 두 값을 구분해 보존했다.',
        '- 정책 F/T는 힘 축별 ±150 N·모멘트 축별 ±8 Nm로 clip한 뒤 [-1,1]로 정규화한다. 별도 정책 그래프는 실제 noise·clip·scale 적용 후 입력이며 terminal step의 자동 reset 관측은 제외했다.', '',
        '| 영역 | reset 시 지정 방향 | 선호 접촉면 |','| --- | --- | --- |',
        '| 1 | 12시 중심 ±30°, 선반 −X | 손등 |','| 2 | 3시 방향, +30°~+150°, 선반 +Y | 손등 |',
        '| 3 | 6시 중심 ±30°, 선반 +X | 손바닥 |','| 4 | 9시 방향, −150°~−30°, 선반 −Y | 손바닥 |',
        '', '영역은 reset 시의 방향으로 고정했다. 정확히 ±30°는 영역1, ±150°는 영역3이다. 시간 중 목표를 지나쳐도 영역을 다시 분류하지 않는다.',''])
    for region in range(1,5):
        text.extend([f'## 영역 {region}', '', '### 보상 분포', '',
            '| 정책 | episode 수 | 성공률 | 평균 return | Sweep + | Sweep − | Reaching | 기타 비용 | 평균 길이(s) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for key,label in MODELS:
            d=regional[region][key];r=d['row']
            text.append(f'| {label} | {r["n"]} | {r["success_pct"]:.1f}% | {r["return_mean"]:.3f} | {r["sweep_positive_return"]:.3f} | {r["sweep_negative_return"]:.3f} | {r["reaching_return"]:.3f} | {r["other_cost_return"]:.3f} | {r["duration_mean_s"]:.2f} |')
        timing=' / '.join(f'{label} {regional[region][key]["row"]["sweep_positive_90pct_time_s"]:.2f}초' for key,label in MODELS)
        text.extend(['',f'**양의 Sweep 지급 시점:** 전체 양의 Sweep 보상 중 90%가 지급된 시각은 {timing}다. 이는 영역 내 episode들의 양의 보상을 합산한 분포의 누적 90% 시각이다.','',image(f'region{region}_rewards.png'),'','표의 보상 항은 episode 전체 누적값의 평균이다. 그래프 왼쪽은 step당 보상, 오른쪽은 초기 표본 전체의 평균 누적 보상이다. Sweep −는 v1의 방향 불일치 음수분이며, 목표에서 멀어지는 `sweep_retreat`는 기타 비용에 포함했다.','',
            '<details>','<summary>모든 개별 Reward 항의 평균 Timeline</summary>','',image(f'region{region}_reward_terms.png'),'','</details>','',
            '### 양의 Sweep 지급 구간의 힘 방향', '',
            '| 정책 | 지급 episode / 전체 | 지급 step | 지급 시간/episode(s) | 평균 θ | 중앙 θ | P90 θ | ≤30° | 전체 접촉 평균 θ | 전체 접촉 ≤30° |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for key,label in MODELS:
            d=regional[region][key];r=d['ar']
            text.append(f'| {label} | {r["positive_reward_episodes"]}/{d["n"]} | {r["positive_reward_steps"]} | {r["positive_time_s_per_episode"]:.3f} | {fmt(r["angle_mean_deg"])}° | {fmt(r["angle_median_deg"])}° | {fmt(r["angle_p90_deg"])}° | {fmt(r["within30_pct"],1)}% | {fmt(r["all_contact_angle_mean_deg"])}° | {fmt(r["all_contact_within30_pct"],1)}% |')
        text.extend(['',image(f'region{region}_alignment.png'),'','각도 통계는 지급 step을 동일 가중치로 집계했다. 오래 접촉하는 episode가 더 많이 기여한다. episode별 각도를 먼저 평균한 결과와 XY·3D 코사인, 표본 수는 alignment.csv에 보존했다. 지급 step이 없으면 각도는 미정(—)이다.','',
            '### Contact Sensor Timeline','',image(f'region{region}_contact.png'),'',
            '오른쪽은 타겟에 작용한 접촉력이다. 실선은 법선+마찰 합력의 크기, 회색 점선은 법선 합력의 크기, 보라 점선은 개별 법선 접촉 하중 크기의 합이다. 마지막 값은 서로 반대 방향으로 누르는 접촉이 상쇄되지 않는다. 음영은 실선의 10~90% 표본 범위다.','',
            '### F/T Sensor Timeline','',image(f'region{region}_ft.png'),'',
            '| 정책 | 타겟 접촉 비율 | 촉각 활성 비율 | 평균 활성 손바닥/손등 채널 | 타겟 힘 평균/P95(N) | F/T 힘 평균/P95(N) | F/T 모멘트 평균/P95(Nm) | 정책 F/T 포화 비율 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for key,label in MODELS:
            r=regional[region][key]['sr']
            text.append(f'| {label} | {r["target_contact_pct"]:.1f}% | {r["tactile_any_pct"]:.1f}% | {r["palm_channels_mean"]:.2f}/{r["dorsal_channels_mean"]:.2f} | {r["target_force_mean_N"]:.2f}/{r["target_force_p95_N"]:.2f} | {r["ft_force_mean_N"]:.2f}/{r["ft_force_p95_N"]:.2f} | {r["ft_torque_mean_Nm"]:.3f}/{r["ft_torque_p95_Nm"]:.3f} | {r["policy_ft_any_saturation_pct"]:.2f}% |')
        raw_noise=' / '.join(f'{label} {regional[region][key]["sr"]["raw_tactile_any_pct_on_policy_valid_steps"]:.2f}%→{regional[region][key]["sr"]["policy_tactile_any_pct"]:.2f}%' for key,label in MODELS)
        text.extend(['',f'**원시 촉각과 정책 입력:** 동일한 비종료 env-step에서 한 채널 이상 활성인 비율은 원시→noise 적용 후 {raw_noise}였다. 정책에는 34개 면별 bit에 각각 1% flip noise가 적용되므로 실제 접촉이 없는 표본에도 활성 신호가 들어갈 수 있다. 타겟 접촉 비율과 촉각 활성 비율은 측정 대상·표면 분류·임계값이 달라 직접 같아야 하는 값이 아니다.','',
            '비율·센서 요약은 생존 초기 episode의 env-step을 합산한 값이다. 촉각 활성 비율은 17채널 중 하나라도 활성인 비율이며 양면은 손바닥·손등 양쪽에 포함한다. 포화는 실제 정책 관측의 6축 중 하나라도 정규화 절댓값이 1인 비율이다.','',
            '<details>','<summary>실제 정책 입력의 정규화 F/T 6축</summary>','',image(f'region{region}_ft_policy.png'),'','</details>',''])
    text.extend(['## 체크포인트·근거·검증','', '| 정책 | 학습 run | checkpoint |','| --- | --- | --- |'])
    for key,label in MODELS:
        m=metadata[key][0];p=Path(m['checkpoint'])
        text.append(f'| {label} | {p.parent.name} | [{p.name}]({rel(p)}) |')
    all_paired=all(all(v.values()) for v in paired.values())
    text.extend(['',f'- 초기 물리 장면 대조: 두 시드 모두 robot joint·물체 상태·목표·방향·bounds·활성 물체의 hash가 정책 간 '+('**모두 일치**했다. 동일 장면에서 각 정책을 비교한 결과다.' if all_paired else '**일부 불일치**했다. 정책 간 결과를 완전한 paired 비교로 해석할 수 없다. validation.json의 항목별 결과를 확인한다.'),
        f'- 6개 실행 모두 500step을 완료했고 1,536개 초기 episode의 종료를 각 1회 확인했다. 보상항 합과 실제 RewardManager total의 최대 차이는 {max(c["reward_sum_error"] for c in checks):.2e}다.',
        f'- 초기 episode에서 접촉 샘플 유효 비율은 최소 {min(c["contact_valid_fraction"] for c in checks)*100:.4f}%, 센서 fresh 비율은 최소 {min(c["sensor_fresh_fraction"] for c in checks)*100:.4f}%다. 실제 정책 F/T 범위 [-1,1]과 촉각 범주, v1 양의 방향 보상 θ<30°를 원시 배열로 확인했다.',
        '- 각 체크포인트의 SHA-256을 수집 전·후 대조했다. 원본 학습 로그에는 쓰지 않았다. 환경·보상·정책 입력 구성은 변경하지 않았고 평가 스크립트만 추가했다. PhysX aggregate-pair capacity만 평가에 필요한 메모리 크기 2²¹로 축소했다.',
        '- 저장된 학습 env.yaml과 현재 평가 설정의 Action·Reward·Observation·Command·Termination·Event·Curriculum, episode 길이·decimation·목표 반경·F/T 제한값을 대조해 세 정책 모두 일치를 확인했다. v1 Sweep 및 Reaching 항은 수집한 내부 구성요소로 별도 재계산해 실제 지급값과 대조했다.',
        '- **미검증 범위:** 실물 센서·실물 조작, 학습 seed 간 재학습 분산, 장기 성능 및 reward 조건을 제거한 인과적 방향 제어 개선. 같은 정책에서 관측 noise를 제거했을 때의 성능도 측정하지 않았다.', '',
        '### 원시 데이터와 재계산', '',
        f'- [실행별 원시 trace·초기 장면·설정·metadata]({rel(args.input)})',
        f'- [영역별 성과·보상 요약]({rel(args.assets/"summary.csv")}) · [전체 보상항]({rel(args.assets/"reward_terms.csv")})',
        f'- [방향 일치도와 분모]({rel(args.assets/"alignment.csv")}) · [센서 통계]({rel(args.assets/"sensors.csv")}) · [평균 Timeline 수치]({rel(args.assets/"timeline.csv")})',
        f'- [검증·장면 hash 대조]({rel(args.assets/"validation.json")})',
        f'- [수집 스크립트]({rel(Path(__file__).with_name("collect_sweep_region_timelines.py"))}) · [보고서 재계산 스크립트]({rel(Path(__file__))})',
        '- 환경 정의: [Sweep v0 계약](../environments/SWEEP_V0.md). 센서 채널 이름과 보상별 callable·params·weight, 체크포인트 hash는 실행별 metadata.json에 저장했다.', ''])
    text.extend(['### 촉각 채널 번호','', '| 번호 | 센서 body |','| ---: | --- |'])
    for i,name in enumerate(metadata['sweep_v0'][0]['tactile_channels'],1):
        text.append(f'| {i} | `{name}` |')
    text.append('')
    args.report.write_text('\n'.join(text),encoding='utf-8')
    print(args.report)


if __name__=='__main__':
    main()
