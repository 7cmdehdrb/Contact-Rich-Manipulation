"""Paired counterfactual Sweep rewards on recorded, unchanged checkpoint trajectories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def cosine(a, b):
    den = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return np.clip(np.divide((a*b).sum(-1), den, out=np.zeros_like(den), where=den>1e-12), 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root=args.root.resolve(); out=args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base=list(csv.DictReader((root/'episodes.csv').open(encoding='utf-8-sig')))
    lookup={(e['run'],int(e['env']),int(e['episode'])):e for e in base}
    episodes=[]; phases=[]; diagnostics=[]; timelines=[]; provenance=[]; cases=[]
    variants=['force','motion_same_force_gate','motion_xy','motion_3d','motion_pre_goal',
              'motion_mid_goal','motion_speed_1mm','motion_speed_5mm','distance_only']
    for run in ('seed42','seed43'):
        folder=root/run; meta=json.loads((folder/'metadata.json').read_text())
        assert meta['status']=='complete' and meta['reward_weights'][0]==100
        checkpoint=Path(meta['checkpoint']); digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest==meta['checkpoint_sha256']
        keys=['target','goal','distance','episode','step','done','reward_terms','reward_total',
              'force','contact','contact_valid','contact_recognized','alignment','positive','retreat']
        with np.load(folder/'trace.npz') as z: d={k:z[k] for k in keys}
        eligible=d['episode']<4; first=d['step']==1
        assert not d['contact_recognized'][first].any()
        assert np.max(np.abs(d['reward_terms'][...,0][first]))==0
        assert all(np.isfinite(a).all() for a in d.values())
        p=d['target'].astype(float); goal=d['goal'][...,:3].astype(float)
        prev=np.concatenate([p[:1],p[:-1]],axis=0)
        same=np.concatenate([np.zeros_like(eligible[:1]),d['episode'][1:]==d['episode'][:-1]],axis=0)
        pair=eligible & ~first & same
        assert np.array_equal(pair, eligible & ~first)
        assert np.max(np.abs((goal[1:]-goal[:-1])[same[1:]]))==0
        movement=p-prev; movement[~pair]=0
        previous_distance=np.concatenate([d['distance'][:1],d['distance'][:-1]],axis=0)
        progress=(previous_distance-d['distance']).astype(float); progress[~pair]=0
        gain=np.maximum(progress,0)
        current_vector=goal-p; before_vector=goal-prev
        cxy=cosine(movement[...,:2],current_vector[...,:2])
        c3d=cosine(movement,np.concatenate([current_vector[...,:2],np.zeros_like(current_vector[...,:1])],-1))
        speed=np.linalg.norm(movement[...,:2],axis=-1)/meta['dt']
        gate=d['contact_recognized'] & d['contact'] & d['contact_valid']
        force_gate=gate & (np.linalg.norm(d['force'][...,:2],axis=-1)>=1)
        force=d['force'].astype(float)
        u=current_vector[...,:2]/np.maximum(np.linalg.norm(current_vector[...,:2],axis=-1)[...,None],1e-12)
        cf=np.clip((force[...,:2]*u).sum(-1)/np.maximum(np.linalg.norm(force,axis=-1),1e-12),0,1)
        original=100*gain*cf*force_gate
        rewards=dict(force=d['reward_terms'][...,0].astype(float),
            motion_same_force_gate=100*gain*cxy*force_gate,
            motion_xy=100*gain*cxy*gate,
            motion_3d=100*gain*c3d*gate,
            motion_pre_goal=100*gain*cosine(movement[...,:2],before_vector[...,:2])*gate,
            motion_mid_goal=100*gain*cosine(movement[...,:2],(goal-(p+prev)/2)[...,:2])*gate,
            motion_speed_1mm=100*gain*cxy*gate*(speed>=.001),
            motion_speed_5mm=100*gain*cxy*gate*(speed>=.005),
            distance_only=100*gain*gate)
        # Independent geometric identity, without reusing dot-product cosine.
        dp=np.linalg.norm(before_vector[...,:2],axis=-1)
        dc=np.linalg.norm(current_vector[...,:2],axis=-1)
        length=np.linalg.norm(movement[...,:2],axis=-1)
        den=2*length*dc
        geometric=np.clip(np.divide(dp**2-dc**2-length**2,den,
            out=np.zeros_like(den),where=den>2e-12),0,1)
        checked=pair & (length>1e-7) & (dc>1e-6)
        geom_error=float(np.max(np.abs(geometric[checked]-cxy[checked])))
        original_error=float(np.max(np.abs(original[pair]-rewards['force'][pair])))
        assert geom_error<1e-5 and original_error<2e-5
        assert np.max(np.abs(20*np.maximum(-progress[pair],0)+d['reward_terms'][...,1][pair]))<2e-6
        assert np.all(rewards['motion_xy']<=rewards['distance_only']+1e-12)
        assert not rewards['motion_xy'][progress<=0].any()
        assert not rewards['motion_xy'][~gate].any()
        # First step has no 2-of-3 contact recognition: missing reset center cannot affect reward.
        quality=dict(run=run,frames=int(eligible.sum()),paired_positions=int(pair.sum()),
            first_steps=int((eligible&first).sum()),first_recognized=0,first_sweep=0,
            original_reward_max_error=original_error,geometric_cosine_max_error=geom_error,
            reward_sum_max_error=float(np.max(np.abs(d['reward_terms'].sum(-1)-d['reward_total']))),
            inward_contact_frames=int((eligible&gate&(gain>0)).sum()),
            old_zero_new_positive_frames=int((eligible&(rewards['force']<=1e-12)&(rewards['motion_xy']>1e-12)).sum()),
            old_zero_new_positive_sum=float(rewards['motion_xy'][eligible&(rewards['force']<=1e-12)].sum()),
            positive_progress_nonpositive_movement_angle_frames=int((pair&gate&(gain>0)&(cxy<=0)).sum()),
            force_less_1N_new_sum=float(rewards['motion_xy'][eligible&~force_gate].sum()))
        diagnostics.append(quality)
        provenance.append(dict(run=run,checkpoint=str(checkpoint),checkpoint_sha256=digest,
            trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest(),dt=meta['dt']))
        for env in range(meta['num_envs']):
            for ep in range(4):
                ix=np.flatnonzero(d['episode'][:,env]==ep); sl=(ix,env); n=len(ix)
                assert d['done'][ix[-1],env] and np.array_equal(d['step'][sl],np.arange(1,n+1))
                old=lookup[(run,env,ep)]
                terms=d['reward_terms'][sl].astype(float)
                totals={v:float(rewards[v][sl].sum()) for v in variants}
                net=float(terms.sum()); pos=float(np.maximum(terms,0).sum()); neg=float(np.maximum(-terms,0).sum())
                e=dict(run=run,seed=meta['seed'],env=env,episode=ep,region=int(old['region']),
                    success=int(old['success']),outcome=old['outcome'],ever_in_goal=int(old['ever_in_goal']),
                    final_distance_mm=float(old['final_distance_mm']),steps=n,old_total=net,
                    old_positive=pos,negative=neg,formation=float(terms[:,-1].sum()),
                    retreat=float(terms[:,1].sum()),**totals)
                e.update(delta=totals['motion_xy']-totals['force'],
                    new_total=net+totals['motion_xy']-totals['force'],
                    new_positive=pos+totals['motion_xy']-totals['force'])
                episodes.append(e)
                distance=d['distance'][sl]; time=d['step'][sl]*meta['dt']; hit=np.flatnonzero(distance<=.03)
                after=np.arange(n)>=hit[0] if len(hit) else np.zeros(n,dtype=bool)
                pmasks={'all':np.ones(n,dtype=bool),'far_gt60mm':distance>.06,
                    'near_30to60mm':(distance>.03)&(distance<=.06),'inside_30mm':distance<=.03,
                    'first_1s':time<=1,'after_2s':time>2,'after_first_entry':after,
                    'outside_after_entry':after&(distance>.03),
                    'force_nonpositive_alignment':d['alignment'][sl]<=0,
                    'movement_below1mm_s':speed[sl]<.001}
                for name,m in pmasks.items():
                    phases.append(dict(run=run,env=env,episode=ep,phase=name,frames=int(m.sum()),
                        **{v:float(rewards[v][sl][m].sum()) for v in variants}))
                cube=np.zeros((500,3)); cube[:n,0]=rewards['force'][sl]
                cube[:n,1]=rewards['motion_xy'][sl]; cube[:n,2]=terms[:,-1]
                timelines.append(cube)
                if (run,env,ep)==('seed43',41,1):
                    cases=[dict(step=int(d['step'][i,env]),time_s=float(d['step'][i,env]*meta['dt']),
                        distance_mm=float(d['distance'][i,env]*1000),speed_mm_s=float(speed[i,env]*1000),
                        force_alignment=float(d['alignment'][i,env]),motion_alignment=float(cxy[i,env]),
                        force_reward=float(rewards['force'][i,env]),motion_reward=float(rewards['motion_xy'][i,env]),
                        formation_reward=float(d['reward_terms'][i,env,-1])) for i in ix]
    assert len(episodes)==512
    groups={ 'all':episodes, **{f'region_{i}':[e for e in episodes if e['region']==i] for i in range(1,5)},
        'success':[e for e in episodes if e['success']], 'failure':[e for e in episodes if not e['success']],
        'entered_then_failed':[e for e in episodes if e['ever_in_goal'] and not e['success']],
        **{f'seed_{seed}':[e for e in episodes if e['seed']==seed] for seed in (42,43)}}
    fields=variants+['old_total','new_total','old_positive','new_positive','negative','formation','retreat','delta']
    rows=[]; rng=np.random.default_rng(26)
    for name,es in groups.items():
        row=dict(group=name,n=len(es),**{f:float(np.mean([e[f] for e in es])) for f in fields})
        clusters={}
        for e in es: clusters.setdefault((e['seed'],e['env']),[]).append(e['delta'])
        vals=np.array([(np.sum(v),len(v)) for v in clusters.values()])
        sampled=vals[rng.integers(0,len(vals),(5000,len(vals)))].sum(1)
        row.update(delta_ci_low=float(np.quantile(sampled[:,0]/sampled[:,1],.025)),
                   delta_ci_high=float(np.quantile(sampled[:,0]/sampled[:,1],.975)),
                   sweep_positive_share=row['motion_xy']/row['new_positive'],
                   formation_positive_share=row['formation']/row['new_positive'])
        rows.append(row)
    phase_rows=[]
    for name in dict.fromkeys(p['phase'] for p in phases):
        pp=[p for p in phases if p['phase']==name]
        phase_rows.append(dict(phase=name,frames=sum(p['frames'] for p in pp),
            **{v:sum(p[v] for p in pp)/512 for v in variants}))
    write_csv(out/'episodes.csv',episodes); write_csv(out/'groups.csv',rows)
    write_csv(out/'phase_episodes.csv',phases); write_csv(out/'phases.csv',phase_rows)
    write_csv(out/'lost_goal_case.csv',cases)
    avg=np.mean(timelines,axis=0)
    write_csv(out/'timeline.csv',[dict(time_s=(i+1)*.02,force=avg[i,0],motion=avg[i,1],formation=avg[i,2]) for i in range(500)])
    summary=dict(method='paired offline reward re-scoring; unchanged model_15550 trajectories; no retraining',
        source=str(root),provenance=provenance,validation=diagnostics,groups=rows,phases=phase_rows,
        variants=variants,episodes=512,steps=sum(e['steps'] for e in episodes),
        success_count=sum(e['success'] for e in episodes),
        improved_episodes=sum(e['delta']>1e-6 for e in episodes),
        reduced_episodes=sum(e['delta']< -1e-6 for e in episodes),
        unchanged_episodes=sum(abs(e['delta'])<=1e-6 for e in episodes),
        delta_quantiles=np.quantile([e['delta'] for e in episodes],[0,.1,.5,.9,1]).tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    rr=rows[1:5]; x=np.arange(4)
    for offset,v,label in [(-.24,'force','Force (current)'),(0,'motion_same_force_gate','Motion + force threshold'),(.24,'motion_xy','Motion (no force threshold)')]:
        ax[0].bar(x+offset,[r[v] for r in rr],.24,label=label)
    ax[0].set_xticks(x,['Away -X','Side +Y','Toward +X','Side -Y']); ax[0].set_ylabel('Mean Sweep return / episode')
    ax[0].set_ylim(0,10)
    ax[0].legend(fontsize=8); ax[0].set_title('Same checkpoint, same trajectories')
    for j,label in enumerate(['Force Sweep','Motion Sweep','Formation (unchanged)']):
        ax[1].plot(np.arange(1,501)*.02,np.cumsum(avg[:,j]),label=label)
    ax[1].set(xlabel='Episode time (s)',ylabel='Mean cumulative reward',title='Fixed cohort of 512 episodes; zero after termination')
    ax[1].legend(); fig.savefig(out/'reward_comparison.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for success,color in [(1,'tab:blue'),(0,'tab:orange')]:
        es=[e for e in episodes if e['success']==success]
        ax[0].scatter([e['old_total'] for e in es],[e['new_total'] for e in es],s=12,alpha=.55,
            color=color,label='Success' if success else 'Failure')
    low=min(min(e['old_total'],e['new_total']) for e in episodes); high=max(max(e['old_total'],e['new_total']) for e in episodes)
    ax[0].plot([low,high],[low,high],'k--',lw=1); ax[0].set(xlabel='Original total return',ylabel='Re-scored total return',title='Failure trajectories can also gain reward'); ax[0].legend()
    time=[c['time_s'] for c in cases]
    for key,label in [('force_reward','Force Sweep'),('motion_reward','Motion Sweep'),('formation_reward','Formation')]:
        ax[1].plot(time,np.cumsum([c[key] for c in cases]),label=label)
    ax[1].set(xlabel='Episode time (s)',ylabel='Cumulative reward',title='Previously selected lost-goal case: seed43 / env41 / ep1'); ax[1].legend()
    fig.savefig(out/'outcomes_and_case.png',dpi=160); plt.close(fig)
    print(json.dumps(dict(groups=rows,phases=phase_rows,validation=diagnostics,
                         improved=summary['improved_episodes'],reduced=summary['reduced_episodes']),indent=2))


if __name__=='__main__':
    main()
