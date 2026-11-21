"""Summarize fixed episode cohorts from Single Contact read-only Play traces."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalized(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def wilson(success, count):
    if not count:
        return [None, None]
    z = 1.95996398454
    p = success / count
    center = (p + z*z/(2*count)) / (1+z*z/count)
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count)) / (1+z*z/count)
    return [float(center-radius), float(center+radius)]


def extract(directory, limit):
    meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if meta['status'] != 'complete':
        raise ValueError(f'Incomplete run: {directory}')
    a = dict(np.load(directory/'trace.npz'))
    if len({v.shape[:2] for v in a.values()}) != 1:
        raise ValueError('Inconsistent trace column dimensions')
    for name, value in a.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite values: {name}')
    decomposition = float(np.max(np.abs(a['force']-a['normal']-a['friction'])))
    goal_delta = a['goal'][..., :2]-a['target'][..., :2]
    alignment = ((a['force'][..., :2]*normalized(goal_delta)).sum(-1)
                 / np.maximum(np.linalg.norm(a['force'], axis=-1), 1e-12)).clip(0, 1)
    usable = a['contact'] & a['valid'] & (np.linalg.norm(a['force'][..., :2], axis=-1)>=1.) & (a['distance']>1e-12)
    reconstructed = np.maximum(a['delta_distance'],0)*a['contact_recognized']*alignment*usable
    positive_error = float(np.max(np.abs(reconstructed-a['positive'])))
    outside_error = float(np.max(np.abs(reconstructed-a['positive'])[a['distance']>.03]))
    # Separate float32 local-coordinate subtraction loses angle precision very
    # near zero distance. The evaluated pushing interval is outside 3cm.
    if decomposition > 1e-3 or outside_error > 1e-5:
        raise ValueError(f'Physical trace consistency failed: {decomposition}, {outside_error}')
    rows, trajectories = [], {}
    for e in range(meta['num_envs']):
        for ep in range(limit):
            ids = np.flatnonzero(a['episode'][:, e] == ep)
            if not len(ids) or not a['done'][ids[-1], e]:
                raise ValueError(f'Fixed cohort episode unfinished: {directory}, {e}, {ep}')
            r = {k: v[ids, e] for k, v in a.items()}
            if np.any(r['done'][:-1]) or len(np.unique(r['region'])) != 1:
                raise ValueError('Episode boundary or region changed')
            moving = (r['contact'] & r['valid'] & (r['load'] >= .01)
                      & (np.linalg.norm(r['velocity'][:, :2], axis=-1) >= .005)
                      & (r['distance'] > .03))
            direction = normalized(r['goal'][:, :2]-r['target'][:, :2])
            cos_xy = (normalized(r['force'][:, :2])*direction).sum(-1).clip(-1, 1)
            cos_3d = (r['force'][:, :2]*direction).sum(-1) / np.maximum(np.linalg.norm(r['force'], axis=-1), 1e-12)
            original_cos = (normalized(r['force'][:, :2])*normalized(r['sweep_direction'][:, :2])).sum(-1).clip(-1, 1)
            classified = r['palm_load']+r['dorsal_load'] >= .01
            use = moving & classified
            region = int(r['region'][0])
            preferred = r['dorsal_load'] if region <= 2 else r['palm_load']
            other = r['palm_load'] if region <= 2 else r['dorsal_load']
            flags = {k: bool(r[k][-1]) for k in ('object_fall_or_tilt', 'shelf_failure', 'arm_joint_velocity', 'timeout')}
            success = bool(r['success'][-1])
            if success:
                cause = 'success'
            elif flags['object_fall_or_tilt']:
                cause = 'object_fall_or_tilt'
            elif flags['shelf_failure']:
                cause = 'shelf_failure'
            elif flags['arm_joint_velocity']:
                cause = 'arm_joint_velocity'
            elif flags['timeout']:
                if r['distance'].min() < .03:
                    cause = 'timeout_entered_then_left_goal'
                elif not r['contact'].any():
                    cause = 'timeout_no_target_contact'
                elif not moving.any():
                    cause = 'timeout_no_moving_contact'
                elif r['distance'][-1] <= .05:
                    cause = 'timeout_near_goal_3_to_5cm'
                elif r['distance'][0]-r['distance'].min() < .01:
                    cause = 'timeout_less_than_1cm_progress'
                else:
                    cause = 'timeout_incomplete_progress'
            else:
                raise ValueError('Unknown termination')
            failed_objects = r['active_objects'][-1] & ((r['object_bottom'][-1] < 1.04)
                | (r['object_tilt'][-1] >= .9))
            row = dict(seed=meta['seed'], env=e, episode=ep, region=region, success=success,
                       preferred_side='dorsal' if region <= 2 else 'palm', cause=cause,
                       final_distance_m=float(r['distance'][-1]), min_distance_m=float(r['distance'].min()),
                       initial_distance_m=float(np.linalg.norm(r['initial_goal'][0,:2]-r['initial_target'][0,:2])),
                       duration_s=float(len(ids)*meta['dt']), moving_frames=int(moving.sum()),
                       terminal_speed_xy_m_s=float(np.linalg.norm(r['velocity'][-1, :2])),
                       last_second_mean_speed_xy_m_s=float(np.linalg.norm(r['velocity'][-min(50,len(ids)):, :2],axis=-1).mean()),
                       last_second_progress_m=float(r['distance'][-min(50,len(ids))]-r['distance'][-1]),
                       contact_seconds=float(r['contact'].sum()*meta['dt']),
                       moving_seconds=float(moving.sum()*meta['dt']), classified_frames=int(use.sum()),
                       target_failed=bool(failed_objects[0]), obstacle_failed=bool(failed_objects[1:].any()),
                       target_tilt_rad=float(r['object_tilt'][-1, 0]),
                       max_active_tilt_rad=float(r['object_tilt'][-1][r['active_objects'][-1]].max()),
                       min_active_bottom_m=float(r['object_bottom'][-1][r['active_objects'][-1]].min()),
                       terminal_arm_peak_rad_s=float(r['arm_peak_velocity'][-1]),
                       terminal_shelf_velocity=float(r['shelf_velocity'][-1]),
                       terminal_shelf_displacement_m=float(r['shelf_displacement'][-1]),
                       terminal_wrist_clearance_m=float(r['wrist_clearance'][-1]), **flags)
            if use.any():
                preferred_share = float(preferred[use].sum()/(preferred[use]+other[use]).sum())
                row.update(preferred_load_share=preferred_share,
                           palm_load_share=float(r['palm_load'][use].sum()/(preferred[use]+other[use]).sum()),
                           preferred_dominant_frame_fraction=float((preferred[use]>other[use]).mean()),
                           opposite_dominant_frame_fraction=float((preferred[use]<other[use]).mean()),
                           preferred_dominant_episode=bool(preferred_share>.5))
            if moving.any():
                palm_facing = (normalized(r['palm'][:, :2])*normalized(r['target'][:, :2]-r['eef'][:, :2])).sum(-1)
                row.update(classified_moving_frame_fraction=float(use.sum()/moving.sum()),
                           palm_facing_target_fraction=float((palm_facing[moving]>0).mean()),
                           intended_pose_fraction=float(((palm_facing[moving]<0) if region<=2 else (palm_facing[moving]>0)).mean()),
                           preferred_all_moving_frame_fraction=float((use & (preferred>other)).sum()/moving.sum()),
                           opposite_all_moving_frame_fraction=float((use & (preferred<other)).sum()/moving.sum()),
                           tied_all_moving_frame_fraction=float((use & (preferred==other)).sum()/moving.sum()),
                           unclassified_moving_frame_fraction=float((moving & ~classified).sum()/moving.sum()),
                           force_cos_xy=float(cos_xy[moving].mean()),
                           force_cos_3d=float(cos_3d[moving].mean()),
                           force_angle_deg=float(np.degrees(np.arccos(cos_xy[moving])).mean()),
                           force_within30_fraction=float((cos_xy[moving]>=np.cos(np.pi/6)).mean()),
                           force_within60_fraction=float((cos_xy[moving]>=.5).mean()),
                           force_opposite_fraction=float((cos_xy[moving]<0).mean()),
                           force_original_cos_xy=float(original_cos[moving].mean()),
                           force_mean_n=float(np.linalg.norm(r['force'][moving], axis=-1).mean()))
            rows.append(row)
            trajectories[(meta['seed'],e,ep)] = r
    return rows, trajectories, dict(seed=meta['seed'], decomposition_max_error_n=decomposition,
                                    reconstructed_positive_max_error_m=positive_error,
                                    reconstructed_positive_outside3cm_max_error_m=outside_error,
                                    normal_matrix_max_error_n=meta['normal_matrix_max_abs_error_n'],
                                    checkpoint_sha256=meta['checkpoint_sha256'])


def aggregate(rows):
    success = [r for r in rows if r['success']]
    result = dict(episodes=len(rows), successes=len(success), success_rate=len(success)/len(rows) if rows else None,
                  success_wilson95=wilson(len(success),len(rows)),
                  success_no_moving_contact=sum(r['moving_frames']==0 for r in success),
                  success_no_classified_moving_contact=sum('preferred_load_share' not in r for r in success))
    keys = ('preferred_load_share','palm_load_share','preferred_dominant_frame_fraction',
            'opposite_dominant_frame_fraction','preferred_dominant_episode','classified_moving_frame_fraction',
            'force_cos_xy','force_cos_3d','force_angle_deg','force_within30_fraction',
            'force_within60_fraction','force_opposite_fraction','force_original_cos_xy','force_mean_n','moving_seconds',
            'preferred_all_moving_frame_fraction','opposite_all_moving_frame_fraction',
            'tied_all_moving_frame_fraction','unclassified_moving_frame_fraction',
            'palm_facing_target_fraction','intended_pose_fraction')
    for key in keys:
        values = [r[key] for r in success if key in r]
        result[key] = float(np.mean(values)) if values else None
        result[key+'_n'] = len(values)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--episodes-per-env', default=5, type=int)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, trajectories, checks = [], {}, []
    for directory in args.runs:
        r,t,c=extract(directory,args.episodes_per_env)
        rows.extend(r); trajectories.update(t); checks.append(c)
    if len({c['checkpoint_sha256'] for c in checks}) != 1:
        raise ValueError('Different checkpoints')
    causes={name:sum(r['cause']==name for r in rows) for name in sorted({r['cause'] for r in rows})}
    failure_details={}
    for cause in causes:
        if cause=='success':
            continue
        cohort=[r for r in rows if r['cause']==cause]
        numeric=('final_distance_m','min_distance_m','duration_s','contact_seconds','terminal_arm_peak_rad_s',
                 'last_second_mean_speed_xy_m_s','last_second_progress_m')
        failure_details[cause]={'count':len(cohort),
            'medians':{key:float(np.median([r[key] for r in cohort])) for key in numeric},
            'no_contact':sum(r['contact_seconds']==0 for r in cohort),
            'last_second_under_1mm_progress':sum(abs(r['last_second_progress_m'])<.001 for r in cohort)}
    summary=dict(overall=aggregate(rows), by_region={str(k):aggregate([r for r in rows if r['region']==k]) for k in range(1,5)},
                 by_seed={str(k):aggregate([r for r in rows if r['seed']==k]) for k in sorted({r['seed'] for r in rows})},
                 causes=causes, failure_details=failure_details, checks=checks,
                 causes_by_region={str(i):{name:sum(r['cause']==name and r['region']==i for r in rows) for name in causes} for i in range(1,5)},
                 object_failure=dict(target=sum(r['object_fall_or_tilt'] and r['target_failed'] for r in rows),
                                     obstacle=sum(r['object_fall_or_tilt'] and r['obstacle_failed'] for r in rows)),
                 termination_flags={k:sum(r[k] and not r['success'] for r in rows) for k in ('object_fall_or_tilt','shelf_failure','arm_joint_velocity','timeout')},
                 methodology='First five completed episodes per environment; success metrics are episode-equal means on moving target-contact frames outside 3cm; actual reward-side classified surface normal loads; region uses episode sweep_dir; CI is episode Wilson, not training-seed uncertainty.')
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (args.output_dir/'episodes.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    groups=list(summary['by_region'].values()); x=np.arange(4)
    labels=['1: 12h / dorsal','2: 3h / dorsal','3: 6h / palm','4: 9h / palm']
    ax=axes[0,0];ax.bar(x,[100*g['success_rate'] if g['episodes'] else 0 for g in groups],color='#356ba5')
    for i,g in enumerate(groups):
        if g['episodes']:
            lo,hi=g['success_wilson95'];ax.errorbar(i,100*g['success_rate'],yerr=[[100*(g['success_rate']-lo)],[100*(hi-g['success_rate'])]],color='black',capsize=4)
        ax.text(i,4,f"{g['successes']}/{g['episodes']}",ha='center',color='white')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,105),ylabel='Success (%)',title='Play outcomes, Wilson 95% intervals')
    ax=axes[0,1];bottom=np.zeros(4)
    for key,label,color in [('preferred_all_moving_frame_fraction','Intended dominant','#239269'),
                            ('opposite_all_moving_frame_fraction','Opposite dominant','#c36d59'),
                            ('tied_all_moving_frame_fraction','Equal loads','#d8b74c'),
                            ('unclassified_moving_frame_fraction','Unclassified','#b9bdc4')]:
        height=np.array([100*(g[key] or 0) for g in groups]);ax.bar(x,height,bottom=bottom,label=label,color=color);bottom+=height
    ax.set(xticks=x,xticklabels=labels,ylim=(0,105),ylabel='Moving-contact frames (%)',title='Successful episodes: surface dominance and coverage');ax.legend(fontsize=8,loc='lower left')
    ax=axes[1,0]
    values=[[r['force_angle_deg'] for r in rows if r['success'] and r['region']==i and 'force_angle_deg' in r] for i in range(1,5)]
    ax.boxplot([v or [np.nan] for v in values],tick_labels=labels,showmeans=True);ax.axhline(30,color='gray',ls='--')
    ax.set(ylabel='Mean XY force-to-goal angle (deg)',title='Successful moving-contact episodes')
    ax=axes[1,1];names=[k for k in causes if k!='success'];counts=[causes[k] for k in names]
    ax.barh([n.replace('timeout_','T: ').replace('_',' ') for n in names],counts,color='#c36d59');ax.set(xlabel='Episodes',title='Failure classification (exclusive, ordered)')
    fig.suptitle(f"Single Contact model_15800 | {summary['overall']['successes']}/{len(rows)} successful",fontsize=16)
    fig.savefig(args.output_dir/'evaluation_summary.png',dpi=160);plt.close(fig)
    selected=[]
    for cause in causes:
        if cause=='success':
            continue
        candidates=[r for r in rows if r['cause']==cause]
        selected.append(sorted(candidates,key=lambda r:r['final_distance_m'])[len(candidates)//2])
    for region in range(1,5):
        candidates=[r for r in rows if r['success'] and r['region']==region and 'preferred_load_share' in r]
        if candidates:
            selected.append(sorted(candidates,key=lambda r:r['preferred_load_share'])[len(candidates)//2])
    (args.output_dir/'representatives.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(int(np.ceil(len(selected)/3)),3,figsize=(15,4*int(np.ceil(len(selected)/3))),squeeze=False,constrained_layout=True)
    for ax,row in zip(axes.flat,selected):
        r=trajectories[(row['seed'],row['env'],row['episode'])]
        def screen(p): return np.stack((-p[...,1],p[...,0]),axis=-1)*100
        p=screen(r['target']);goal=screen(r['goal'][0]);eef=screen(r['eef'])
        ax.plot(p[:,0],p[:,1],label='Target');ax.plot(eef[:,0],eef[:,1],alpha=.4,label='EEF')
        ax.scatter(*p[0],color='black',s=20);ax.add_patch(plt.Circle(goal,3,color='green',fill=False));ax.scatter(*goal,marker='*',s=80,color='green')
        ix=np.flatnonzero(r['contact'] & (r['distance']>.03))[::15]
        if len(ix):
            f=screen(normalized(r['force'][ix,:2]))*.04
            ax.quiver(p[ix,0],p[ix,1],f[:,0],f[:,1],angles='xy',scale_units='xy',scale=1,color='#ce4545')
        ax.set(aspect='equal',xlabel='-shelf Y (cm)',ylabel='+shelf X (cm)',title=f"s{row['seed']}/e{row['env']}/ep{row['episode']} R{row['region']}\n{row['cause']} | final {100*row['final_distance_m']:.1f}cm")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(selected):]: ax.set_visible(False)
    fig.savefig(args.output_dir/'representative_trajectories.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
