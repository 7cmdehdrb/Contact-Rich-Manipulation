"""Scientific figures and a contact-vector animation from saved rollout traces."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D


COLORS = {'force': '#d33c38', 'normal': '#2264be', 'friction': '#e29122', 'palm': '#20946b'}


def screen(v):
    return np.stack((-v[..., 1], v[..., 0]), axis=-1)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def calculate(directory):
    a = dict(np.load(directory / 'trace.npz'))
    meta = json.loads((directory / 'metadata.json').read_text())
    episodes = []
    for e in range(a['done'].shape[1]):
        for ep in np.unique(a['episode'][:, e]):
            if meta.get('conditional_initial_distribution') and ep != 0:
                continue
            ids = np.flatnonzero(a['episode'][:, e] == ep)
            if not a['done'][ids[-1], e]:
                continue
            row = {k: v[ids, e] for k, v in a.items()}
            row['time'] = row['step'] * meta['dt']
            direction = unit(row['goal'][:, :2] - row['target'][:, :2])
            palm_to_object = (unit(row['palm'][:, :2]) * unit(row['target'][:, :2] - row['eef'][:, :2])).sum(-1)
            pushing = (row['contact'] & row['valid'] & (row['load'] >= .01)
                       & (np.linalg.norm(row['velocity'][:, :2], axis=-1) >= .005)
                       & (row['distance'] > .03))
            d0 = unit(screen(row['initial_goal'][0] - row['initial_target'][0]))
            angle = float(np.degrees(np.arctan2(d0[0], d0[1])))
            forward = {k: (row[k][:, :2] * direction).sum(-1) for k in COLORS if k != 'palm'}
            net_to_goal = (row['force'][:, :2] * direction).sum(-1) / np.maximum(np.linalg.norm(row['force'], axis=-1), 1e-12)
            hand_side = ((row['eef'][:, :2] - row['target'][:, :2]) * direction).sum(-1)
            f_hand = (unit(row['force'][:, :2]) * unit(row['target'][:, :2] - row['eef'][:, :2])).sum(-1)
            n_inward = (unit(row['normal']) * unit(row['target'] - row['contact_point'])).sum(-1)
            m = pushing
            stats = dict(env=int(e), episode=int(ep), success=bool(row['success'][-1]),
                         initial_direction_degrees=angle, moving_contact_frames=int(m.sum()),
                         final_distance_m=float(row['distance'][-1]), duration_s=float(row['time'][-1]),
                         push_seconds=float(m.sum()*meta['dt']))
            if m.any():
                stats.update(mean_palm_to_target_cos=float(palm_to_object[m].mean()),
                             palm_away_fraction=float((palm_to_object[m] < 0).mean()),
                             hand_behind_fraction=float((hand_side[m] < 0).mean()),
                             force_toward_target_from_hand_fraction=float((f_hand[m] > 0).mean()),
                             mean_force_goal_cos_3d=float(net_to_goal[m].mean()),
                             positive_reward_m=float(row['positive'][m].sum()),
                             mean_force_z_n=float(row['force'][m, 2].mean()),
                             mean_normal_z_n=float(row['normal'][m, 2].mean()),
                             mean_friction_z_n=float(row['friction'][m, 2].mean()),
                             mean_load_n=float(row['load'][m].mean()),
                             mean_net_norm_n=float(np.linalg.norm(row['force'][m], axis=-1).mean()),
                             normal_toward_target_center_fraction=float((n_inward[m]>0).mean()),
                             **{f'mean_{k}_goal_n': float(v[m].mean()) for k, v in forward.items()})
                positive_friction = np.maximum(forward['friction'][m], 0).sum()
                positive_normal = np.maximum(forward['normal'][m], 0).sum()
                stats['friction_positive_forward_share'] = float(positive_friction / max(positive_friction+positive_normal, 1e-12))
                stats['mean_pair_load_n'] = row['pair_load'][m].mean(0).tolist()
            episodes.append(dict(stats=stats, row=row, moving=m, direction=direction,
                                 d0=d0, palm_cos=palm_to_object, forward=forward))
    # Reconstruct the existing positive reward independently, including its 3D norm.
    direction = unit(a['goal'][..., :2] - a['target'][..., :2])
    align = np.maximum((a['force'][..., :2]*direction).sum(-1) /
                       np.maximum(np.linalg.norm(a['force'], axis=-1), 1e-12), 0)
    align = np.where(a['contact'] & a['valid'] & (np.linalg.norm(a['force'][..., :2], axis=-1)>=1), align, 0)
    expected = np.maximum(a['delta_distance'], 0) * a['contact_count']/20 * a['recognized'] * align
    checks = dict(force_decomposition_max_abs_error_n=float(np.abs(a['force']-a['normal']-a['friction']).max()),
                  positive_reward_reconstruction_max_abs_error_m=float(np.abs(a['positive']-expected).max()),
                  finite_all=all(np.isfinite(v).all() for v in a.values() if v.dtype.kind == 'f'))
    return episodes, meta, checks


def draw_arrow(ax, origin, value, color, scale=1., width=.003, alpha=1.):
    return ax.quiver(*origin, *(value*scale), angles='xy', scale_units='xy', scale=1,
                     color=color, width=width, alpha=alpha, zorder=6)


def glyph(ax, points, names, alpha=.5):
    base = names.index('inspire_base_link')
    for finger, count in [('thumb', 4), ('index', 2), ('middle', 2), ('ring', 2), ('little', 2)]:
        ids = [base] + [names.index(f'inspire_left_{finger}_{i}') for i in range(1, count+1)]
        ax.plot(points[ids, 0], points[ids, 1], '.-', color='#647184', lw=2, ms=3, alpha=alpha)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='+', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, metadata, checks = [], [], []
    for directory in args.directories:
        part, meta, check = calculate(directory)
        for ep in part:
            ep['stats']['run'] = str(directory)
        episodes.extend(part); metadata.append(meta); checks.append(check)
    successful = [ep for ep in episodes if ep['stats']['success']]
    usable = [ep for ep in successful if ep['moving'].any()]
    if not usable:
        raise RuntimeError('No successful moving-contact episodes')
    keys = [k for k, v in usable[0]['stats'].items() if isinstance(v, (float, int)) and not isinstance(v, bool)
            and k not in ('env','episode','initial_direction_degrees')]
    means = {k: float(np.mean([ep['stats'][k] for ep in usable])) for k in keys}
    rng=np.random.default_rng(20261120)
    bootstrap_ids=rng.integers(0,len(usable),size=(2000,len(usable)))
    confidence={k:np.quantile(np.array([ep['stats'][k] for ep in usable])[bootstrap_ids].mean(1),[.025,.975]).tolist()
                for k in keys if k not in ('env','episode','initial_direction_degrees')}
    report = dict(completed=len(episodes), successes=len(successful), success_rate=len(successful)/len(episodes),
                  success_with_moving_contact=len(usable), episode_equal_weight_mean=means,
                  episode_bootstrap_95pct_intervals=confidence,
                  moving_contact_definition='valid target-hand contact; load>=0.01N; target XY speed>=0.005m/s; goal distance>0.03m',
                  seeds=[m['seed'] for m in metadata], checks=checks,
                  episodes=[ep['stats'] for ep in episodes])
    report['conditional_initial_distribution'] = any(m.get('conditional_initial_distribution',False) for m in metadata)
    eleven = [ep for ep in usable if -45 <= ep['stats']['initial_direction_degrees'] < -15]
    # Representative is closest to the median palm-facing value, not the most extreme trace.
    candidates = eleven or usable
    median = np.median([ep['stats']['mean_palm_to_target_cos'] for ep in candidates])
    selected = min(candidates, key=lambda ep: abs(ep['stats']['mean_palm_to_target_cos']-median))
    report['eleven_oclock_successes'] = len(eleven)
    report['eleven_oclock_mean'] = {k: float(np.mean([ep['stats'][k] for ep in eleven])) for k in keys} if eleven else None
    report['representative'] = selected['stats']
    report['selection'] = 'median per-episode palm-to-target cosine among successful 10:30-11:30 directions; fallback all successes'
    loads=np.asarray([ep['stats']['mean_pair_load_n'] for ep in usable])
    shares=loads/np.maximum(loads.sum(1,keepdims=True),1e-12)
    report['episode_equal_weight_normal_load_share_by_body'] = dict(zip(metadata[0]['filter_body_names'],shares.mean(0).tolist()))
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False,
                         'axes.spines.top':False, 'axes.spines.right':False, 'font.size':10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    ax = axes[0, 0]
    aligned = {k: [] for k in ['eef', 'contact_point', 'force', 'normal', 'friction', 'palm']}
    for ep in usable:
        row, m, d = ep['row'], ep['moving'], ep['d0']
        right = np.array([d[1], -d[0]])
        for k in aligned:
            value = screen(row[k] - row['target']) if k in ('eef', 'contact_point') else screen(row[k])
            aligned[k].append(np.column_stack((value@right, value@d))[m].mean(0))
    for k, color, label in [('eef','#677287','EEF 평균 위치'),('contact_point','#d33c38','접촉 중심 평균')]:
        v=np.asarray(aligned[k]); ax.scatter(v[:,0]*100,v[:,1]*100,s=15,alpha=.3,color=color,label=label)
        ax.scatter(*v.mean(0)*100,s=85,marker='X',color=color)
    ax.scatter(0,0,s=180,color='#212d40',label='타겟 중심')
    ax.annotate('목표 방향',xy=(0,5.5),xytext=(0,2),ha='center',arrowprops={'arrowstyle':'->','lw':2})
    for k in ('force','normal','friction'):
        draw_arrow(ax, np.asarray(aligned['contact_point']).mean(0)*100,
                   np.asarray(aligned[k]).mean(0), COLORS[k], scale=2., width=.009)
    ax.set(xlabel='목표 기준 좌우 (cm)',ylabel='목표 기준 전후 (cm)',title='성공 사례: 목표 방향을 위로 정렬\n점 1개 = 에피소드 1개의 이동 접촉 평균')
    ax.axhline(0,color='#ccc',lw=1);ax.axvline(0,color='#ccc',lw=1);ax.set_aspect('equal');ax.set_ylim(-11,6);ax.legend(loc='lower left',fontsize=8)
    ax=axes[0,1]
    labels=['수직항력','마찰력','합력']; fkeys=['normal','friction','force']
    values=np.array([[ep['stats'][f'mean_{k}_goal_n'] for k in fkeys] for ep in usable])
    ax.bar(np.arange(3),values.mean(0),color=[COLORS[k] for k in fkeys],alpha=.8)
    for i in range(3):ax.scatter(np.full(len(values),i)+np.linspace(-.15,.15,len(values)),values[:,i],s=12,color='#27364a',alpha=.3)
    ax.axhline(0,color='#999');ax.set(xticks=np.arange(3),xticklabels=labels,ylabel='목표 방향 힘 성분 (N)',title='목표 쪽으로 미는 힘은 어디에서 오는가?\n막대 = 성공 에피소드 평균, 점 = 각 에피소드')
    ax=axes[1,0]
    angles=np.array([ep['stats']['initial_direction_degrees'] for ep in usable])
    mids=np.arange(-150,181,30);bins=np.arange(-165,196,30)
    wrap=lambda x:np.where(np.asarray(x)<-165,np.asarray(x)+360,np.asarray(x))
    total,_=np.histogram(wrap(angles),bins)
    away,_=np.histogram(wrap([ep['stats']['initial_direction_degrees'] for ep in usable if ep['stats']['mean_palm_to_target_cos']<0]),bins)
    ax.bar(mids,total,width=26,color='#20946b',alpha=.75,label='손바닥이 타겟 쪽')
    ax.bar(mids,away,width=26,color='#8c63bd',label='손바닥이 타겟 반대쪽')
    ax.set(xticks=[-150,-90,-30,30,90,150],xticklabels=['7시','9시','11시','1시','3시','5시'],ylabel='성공 에피소드 수',title='초기 목표 방향별 손 자세\n실제 손등 접촉 분류가 아닌 EEF +Z 방향 지표');ax.legend(fontsize=9)
    ax=axes[1,1]
    xs=[ep['stats']['mean_palm_to_target_cos'] for ep in usable]
    ys=[ep['stats']['friction_positive_forward_share']*100 for ep in usable]
    sc=ax.scatter(xs,ys,c=angles,cmap='twilight',vmin=-180,vmax=180,s=35,alpha=.8)
    ax.axvline(0,color='#999',ls='--');ax.set(xlabel='손바닥 방향 · 손→타겟 방향 (cos)',ylabel='전방 양의 힘 중 마찰력 비중 (%)',title='손바닥이 반대여도 전진하는가?\n음수: 손바닥이 타겟 반대쪽');fig.colorbar(sc,ax=ax,label='초기 목표 방향 (°; 11시=-30°)')
    condition=' | 초기 목표 10:30–11:30 조건부 표본' if report['conditional_initial_distribution'] else ''
    fig.suptitle(f'model_1800 | 완료 {len(episodes)}회 중 성공 {len(successful)}회 | seed '+','.join(str(m['seed']) for m in metadata)+condition,fontsize=15)
    fig.savefig(args.output_dir/'success_contact_summary.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5.5),constrained_layout=True)
    order=np.argsort(shares.mean(0))[::-1][:8]
    name_map={'inspire_base_link':'손바닥/손등 몸체', 'inspire_left_middle_2':'중지 두 번째 링크',
              'inspire_left_ring_2':'약지 두 번째 링크','inspire_left_middle_1':'중지 첫 번째 링크',
              'inspire_left_ring_1':'약지 첫 번째 링크','inspire_left_index_2':'검지 두 번째 링크',
              'inspire_left_little_2':'소지 두 번째 링크'}
    names=[metadata[0]['filter_body_names'][i] for i in order]
    ax.barh(np.arange(len(order)),shares.mean(0)[order]*100,color='#526f9b')
    for j,i in enumerate(order):ax.text(shares.mean(0)[i]*100+.3,j,f'{shares.mean(0)[i]*100:.1f}%',va='center')
    ax.set(yticks=np.arange(len(order)),yticklabels=[name_map.get(n,n.replace('inspire_','')) for n in names],xlabel='타겟에 가해진 수직 접촉하중 비중 (%)',title='성공 사례의 실제 접촉 부위\n각 에피소드의 이동 접촉 구간에서 하중 비중을 구한 뒤 평균')
    ax.invert_yaxis();fig.savefig(args.output_dir/'contact_body_summary.png',dpi=160);plt.close(fig)

    row=selected['row']; time=row['time']; origin=screen(row['initial_target'][0])
    pos={k:screen(row[k])-origin for k in ['target','eef','contact_point','goal']}
    hand=screen(row['hand_points'])-origin
    goal=pos['goal'][0]
    active=np.flatnonzero(selected['moving'])
    indices=active[np.linspace(0,len(active)-1,4).astype(int)]
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    allpos=np.concatenate([pos['target'],pos['eef'],hand.reshape(-1,2),goal[None]])
    low=np.nanmin(allpos,axis=0)-.035;high=np.nanmax(allpos,axis=0)+.035
    def scene(ax, t):
        ax.plot(pos['target'][:t+1,0]*100,pos['target'][:t+1,1]*100,color='#303c4e',lw=2)
        ax.plot(pos['eef'][:t+1,0]*100,pos['eef'][:t+1,1]*100,color='#9ca7b7',lw=1,alpha=.6)
        ax.add_patch(Circle(goal*100,3,fill=False,color='#299769',ls='--'))
        ax.scatter(*goal*100,marker='*',s=150,color='#299769')
        ax.scatter(*pos['target'][t]*100,s=170,color='#303c4e',zorder=5)
        ax.plot([pos['target'][t,0]*100,goal[0]*100],[pos['target'][t,1]*100,goal[1]*100],ls=':',color='#303c4e',alpha=.5)
        glyph(ax,hand[t]*100,metadata[0]['hand_body_names'])
        draw_arrow(ax,pos['eef'][t]*100,screen(row['palm'][t]),COLORS['palm'],scale=4,width=.009)
        if row['load'][t]>=.01:
            ax.scatter(*pos['contact_point'][t]*100,s=40,color=COLORS['force'],zorder=7)
            for k in ('normal','friction','force'):
                draw_arrow(ax,pos['contact_point'][t]*100,screen(row[k][t]),COLORS[k],scale=2.,width=.012 if k=='normal' else .007)
        ax.set(xlim=(low[0]*100,high[0]*100),ylim=(low[1]*100,high[1]*100),aspect='equal',xlabel='화면 오른쪽 = -선반 Y (cm)',ylabel='화면 위 = +선반 X (cm)')
        ax.grid(alpha=.15)
    for ax,t in zip(axes.flat[:4],indices):
        scene(ax,t);ax.set_title(f'{time[t]:.2f}s | 목표 거리 {row["distance"][t]*100:.1f}cm\n손바닥-타겟 cos={selected["palm_cos"][t]:+.2f}')
    ax=axes.flat[4]
    for k in ('normal','friction','force'):ax.plot(time,selected['forward'][k],color=COLORS[k],label={'normal':'수직항력','friction':'마찰력','force':'합력'}[k])
    ax.axhline(0,color='#aaa');ax.set(xlabel='시간 (s)',ylabel='목표 방향 성분 (N)',title='동일 시점의 힘 분해');ax.legend()
    ax=axes.flat[5];ax.plot(time,selected['palm_cos'],color=COLORS['palm'],label='손바닥→타겟 cos');ax.axhline(0,color='#999',ls='--');ax.set(ylim=(-1.1,1.1),xlabel='시간 (s)',ylabel='cos',title='손바닥 방향과 목표 거리')
    other=ax.twinx();other.plot(time,row['distance']*100,color='#303c4e',label='목표 거리');other.axhline(3,color='#303c4e',ls=':');other.set_ylabel('목표 거리 (cm)')
    legend=[Line2D([0],[0],color=COLORS[k],lw=3,label=v) for k,v in [('force','합력'),('normal','수직항력'),('friction','마찰력'),('palm','손바닥 바깥 방향 (EEF +Z)')]]
    fig.legend(handles=legend,loc='outside lower center',ncol=4)
    fig.suptitle(f'성공 대표 사례 | 초기 목표 {selected["stats"]["initial_direction_degrees"]:.1f}° | env {selected["stats"]["env"]}, episode {selected["stats"]["episode"]}\n힘 화살표: 1N=2cm (접촉 중심에서 확대 표시), 회색 선: 실제 손 링크 중심 연결',fontsize=14)
    fig.savefig(args.output_dir/'representative_contact_vectors.png',dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,7.5),constrained_layout=True)
    frame_ids=np.arange(0,len(time),5)
    def animate(i):
        ax.clear();t=frame_ids[i];scene(ax,t)
        ax.set_title(f'model_1800 | {time[t]:.2f}s / {time[-1]:.2f}s\n거리 {row["distance"][t]*100:.1f}cm | 손바닥 cos {selected["palm_cos"][t]:+.2f}\n빨강=합력, 파랑=수직항력, 주황=마찰력, 초록=손바닥')
    FuncAnimation(fig,animate,frames=len(frame_ids),interval=100).save(args.output_dir/'representative_contact_vectors.gif',writer=PillowWriter(fps=10))
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))


if __name__=='__main__':
    main()
