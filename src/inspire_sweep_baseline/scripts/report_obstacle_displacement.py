"""Build the repository report from reviewed obstacle-analysis artifacts."""
import argparse
import csv
import json
import os
import statistics
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); root=args.root.resolve(); output=args.output.resolve()
    s=json.loads((root/'summary.json').read_text())
    tasks=['v0_1','v0_2','v0_3'];assert set(s['play'])==set(tasks)
    episodes=read_csv(root/'episodes.csv'); cases=read_csv(root/'representatives.csv')
    rel=os.path.relpath(root,output.parent).replace('\\','/')
    link=lambda name:f'{rel}/{name}'
    pct=lambda x:f'{100*x:.2f}%' if x is not None else '—'
    cm=lambda x:f'{100*x:.2f}'
    report=[]
    report.append('# Prof Revision 3개 체크포인트의 장애물 변위 비용 평가')
    report.append('작성일: 2026-11-25 KST. 대상: v0_1/model_13850, v0_2/model_15200, v0_3/model_7650.')
    report.append('## 핵심 판단')
    v3=s['play']['v0_3']; moving=v3['moved_1cm']; active=v3['with_obstacles']
    report.append('**현재 비용은 정상적으로 적용되며, 큰 변위를 오래 유지하면 상당히 강하다. 작은 이동·짧은 충돌·종료 직전 이동까지 강하게 억제하려는 목적에는 설계상 빈틈이 있다.** 초기 위치 이탈의 제곱을 시간에 따라 부과하므로, 같은 최대 이동 거리에도 발생 시점·지속시간·episode 종료에 따라 부담이 크게 달라진다. 단순히 전체 평균이나 weight −150만으로 적절성을 판단하기 어렵다.')
    report.append(f"세 정책을 새로 Play한 **각256episode, 총768episode**에서 확인했다. v0_3는 장애물이 있는 {active['n']}episode 중 {moving['n']}건({pct(moving['n']/active['n'])})에서 한 장애물이1cm 이상 이동했고, 그중 {moving['successes']}건이 성공 판정을 받았다. 이 집단의 비용/양의 보상 합계 비율은 **{pct(moving['cost_over_positive'])}**지만 episode별 비율의 중앙값은 **{pct(statistics.median(float(x['displacement_cost'])/float(x['positive_reward']) for x in episodes if x['task']=='v0_3' and float(x['maximum_displacement_m'])>=.01))}**다. 큰 이동 몇 건이 합계 비율을 크게 올린다.")
    medium=[x for x in episodes if x['task']=='v0_3' and .01<=float(x['maximum_displacement_m'])<.05]
    large=[x for x in episodes if x['task']=='v0_3' and float(x['maximum_displacement_m'])>=.05]
    mean_value=lambda group,key:statistics.mean(float(x[key]) for x in group)
    report.append(f"v0_3의 **1cm 이상·5cm 미만 {len(medium)}건**은 평균 비용 **−{mean_value(medium,'displacement_cost'):.3f}**, 양의 보상 **+{mean_value(medium,'positive_reward'):.3f}**로 비용 비중이 **{pct(mean_value(medium,'displacement_cost')/mean_value(medium,'positive_reward'))}**였다. 반면 **5cm 이상 {len(large)}건**은 비용 **−{mean_value(large,'displacement_cost'):.3f}**가 양의 보상 **+{mean_value(large,'positive_reward'):.3f}**보다 컸다. 따라서 ‘작은 장애물 이동에도 너그럽다’는 인상은 근거가 있지만, 큰 변위까지 일률적으로 비용이 약하다고 보기는 어렵다.")
    report.append('**권고는 현재의 지속 변위 비용을 유지하면서, 허용 변위를 넘긴 최대 이탈 증가에 즉시 비용을 추가하는 방향이다.** 몇 cm까지 허용할지와 성공 판정에 장애물 조건을 넣을지를 먼저 정하면 계수의 의미가 명확해진다. weight만 크게 올리는 안은 아래 고정 trajectory 민감도로 비교하되 정책 개선이 입증된 값으로 해석하지 않는다.')
    report.append(f"수치 근거는 [Play 요약]({link('summary.json')})·[episode별 데이터]({link('episodes.csv')})·[장애물별 데이터]({link('obstacles.csv')})이며, 상세 재현 조건은 마지막 절에 있다. [1]")

    report.append('## 학습 로그에서도 변위 비용의 비중은 정책별로 다르다')
    report.append('평가 checkpoint iteration까지의 최근1,000개 logged iteration 평균이다. `Episode_Reward/*`는 RewardManager가 누적항을 **고정 최대 episode 길이10초로 나눈 값**이므로, 표는 이를×10한 로그 환산값이다. 각 iteration에 보고된 episode 집단의 평균을 다시 평균한 값이며, 실제 episode 수로 재가중한 평균은 아니다. 양의 보상 열은 양수인 항별 logged 평균의 합이다. 학습 로그는 확률적 행동을 포함하고 Play는 행동을 결정적으로 추론하므로 절대값이 같을 필요가 없다. [2]')
    report.append('| 환경·로그 구간 | 변위 비용 환산 | 양의 항 합 환산 | 순보상 환산 | 변위/양의 항 합 |\n| --- | ---: | ---: | ---: | ---: |')
    for task in tasks:
        t=s['training'][task]
        report.append(f"| {task} · {t['window_first']}–{t['window_last']} | −{t['episode_equivalent_displacement_cost']:.4f} | +{t['episode_equivalent_positive']:.3f} | {t['episode_equivalent_total']:+.3f} | {pct(t['displacement_over_positive'])} |")
    report.append(f"![학습 로그의 변위 비용과 양의 보상 대비 비중]({link('figures/training.png')})")
    report.append('로그에서 변위 비용이 작다는 사실만으로 계수가 약하다고 단정하지 않는다. 장애물이 없는 배치·움직이지 않은 장애물·짧게 끝난 episode가 섞인 결과이며, 로그만으로 최대 이동 거리를 복원할 수 없다. 그래서 다음 Play에서 실제 위치를 확인했다.')

    report.append('## 실제 Play: 움직인 빈도와 비용을 분리해서 본다')
    report.append('각 정책은 seed42·64환경에서 각 환경의 **처음4개 완료 episode**를 사용했다. 최대 단일 장애물 변위는 episode 중 초기 위치에서 가장 멀어진3D 거리다. 이하 이동 빈도·거리 분포의 분모는 장애물이1개 이상인 episode이며,0개 배치는 제외했다.1cm·5cm는 분석 구간이지 사용자가 이미 정한 허용 기준이 아니다. [1]')
    report.append('| 환경 | 장애물 있는 episode | ≥1cm | ≥5cm | 최대 변위 P95 | 최대 관측 |\n| --- | ---: | ---: | ---: | ---: | ---: |')
    for task in tasks:
        a=s['play'][task]['with_obstacles'];h=a['thresholds'];d=a['maximum_displacement_m']
        report.append(f"| {task} | {a['n']} | {h['0.01']['count']}/{a['n']} ({pct(h['0.01']['fraction'])}) | {h['0.05']['count']}/{a['n']} ({pct(h['0.05']['fraction'])}) | {cm(d['p95'])}cm | {cm(d['max'])}cm |")
    report.append('v0_1은 이동을 거의 만들지 않았다. 이 정책의 낮은 변위 비용을 근거로 세 정책 모두 장애물 충돌에 너그럽다고 결론 내리면 안 된다. v0_2·v0_3는 실제 이동이 있는 집단의 부담을 따로 비교해야 한다.')
    report.append('### 전체 episode의 보상 규모')
    report.append('각 행의 분모는0개 장애물 배치도 포함한256episode다. “양의 보상”은 실제 각 step·각 항의 양수 기여를 모두 더한 예산이다. 순보상이0 근처이거나 음수이면 비율이 불안정해져, 비용 비중은 양의 보상 합을 기준으로 표시한다.')
    report.append('| 환경 | 평균 변위 비용 | 중앙값 비용 | 평균 양의 보상 | 평균 순보상 | 변위/양의 보상 |\n| --- | ---: | ---: | ---: | ---: | ---: |')
    for task in tasks:
        a=s['play'][task]['all']
        report.append(f"| {task} | −{a['displacement_cost']['mean']:.4f} | −{a['displacement_cost']['median']:.5f} | +{a['positive_reward']['mean']:.3f} | {a['total_reward']['mean']:+.3f} | {pct(a['cost_over_positive'])} |")
    report.append('### 1cm 이상 움직인 episode만 비교')
    report.append('| 환경 | episode·성공 수 | 평균 최대 변위 | 평균 지속시간 | 평균 변위 비용 | 평균 양의 보상 | 비용/양의 보상 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for task in tasks:
        a=s['play'][task]['moved_1cm']
        report.append(f"| {task} | {a['n']}건·{a['successes']}성공 | {cm(a['maximum_displacement_m']['mean'])}cm | {a['duration_s']['mean']:.2f}s | −{a['displacement_cost']['mean']:.3f} | +{a['positive_reward']['mean']:.3f} | {pct(a['cost_over_positive'])} |")
    report.append('v0_1의 해당 표본은1건뿐이다. 집단의 비용/양의 보상 비율은 합계 비율이며 개인 episode 비율의 평균과 다르다. 변위가 큰 소수 사례가 평균을 끌어올릴 수 있으므로 [구간별 데이터]('+link('magnitude_bins.csv')+')와 함께 본다.')
    report.append('### 작은 이동과 큰 이동을 나눈 보상 부담')
    report.append('구간은 episode의 최대 단일 장애물 변위로 분류한다. 비용은 해당 episode의 모든 활성 장애물에 대한 합계이며, 각 열은 그 구간의 episode 평균이다.')
    report.append('| 환경·최대 변위 구간 | 건수·성공 수 | 평균 변위 비용 | 평균 양의 보상 | 비용/양의 보상 합계 | episode별 비율 중앙값 |\n| --- | ---: | ---: | ---: | ---: | ---: |')
    for task in tasks:
        for low,high,label in ((.01,.05,'1cm 이상·5cm 미만'),(.05,float('inf'),'5cm 이상')):
            group=[x for x in episodes if x['task']==task and low<=float(x['maximum_displacement_m'])<high]
            if not group:continue
            cost=mean_value(group,'displacement_cost');positive=mean_value(group,'positive_reward')
            median=statistics.median(float(x['displacement_cost'])/float(x['positive_reward']) for x in group)
            report.append(f"| {task} · {label} | {len(group)}건·{sum(x['success']=='True' for x in group)}성공 | −{cost:.3f} | +{positive:.3f} | {pct(cost/positive)} | {pct(median)} |")
    report.append(f"![실제 이동 분포·비용·보상 timeline 비교]({link('figures/play_comparison.png')})")
    report.append('그림 하단의 두 누적 곡선은 최초256episode를 고정 분모로 삼으며, 조기 종료 후에는 마지막 누적 합계를 유지한다. 종료 후에도 비용이 새로 발생한 것으로 계산하지 않는다.')

    report.append('## 같은 몇 cm 이동에도 부담이 크게 달라진 실제 사례')
    report.append('전체 결과를 대표하는 평균 사례가 아니라, 최대 비용·짧은 노출·장애물을 움직인 성공 사례를 드러내기 위한 선택이다. 선택 규칙은 분석 스크립트와 [사례 목록]('+link('representatives.csv')+')에 보존했다. [1]')
    report.append('| 환경·env/episode | 최대 변위 | episode 길이 | 변위 비용 | 양의 보상 | 순보상 | 성공 |\n| --- | ---: | ---: | ---: | ---: | ---: | --- |')
    for c in cases:
        report.append(f"| {c['task']} · {c['env']}/{c['episode']} | {100*float(c['maximum_displacement_m']):.2f}cm | {float(c['duration_s']):.2f}s | −{float(c['displacement_cost']):.4f} | +{float(c['positive_reward']):.3f} | {float(c['total_reward']):+.3f} | {'예' if c['success']=='True' else '아니오'} |")
    report.append('**조기 종료·늦은 이동:** v0_1의 env18/episode2는 최대2.61cm 이동했지만0.70초에 끝나 비용은−0.037이었다. v0_2의 env29/episode1은0.80초에1cm를 넘고0.82초에 종료돼 최대1.41cm 이동에도−0.00148만 누적됐다. 해당 episode들은 다른 실패 비용 때문에 순보상이 음수였다. 작은 변위 비용만 보고 행동 전체가 이득이었다고 해석하지 않는다.')
    report.append('**비용을 내고도 남는 보상:** v0_2의 env6/episode1은 장애물을4.80cm 옮긴 뒤10초까지 유지했고 변위 비용−3.049를 받았다. 양의 보상+8.089·다른 비용까지 합친 순보상+3.126이 남았고 성공 판정도 받았다. 현재 목적함수는 이런 교환을 허용한다.')
    report.append(f"![v0_2에서 장애물을4.8cm 옮기고도 성공한 실제 trajectory]({link('figures/v0_2_env6_ep1.png')})")
    v3cases=[c for c in cases if c['task']=='v0_3' and c['success']=='True']
    if v3cases:
        c=min(v3cases,key=lambda x:float(x['displacement_cost'])/max(float(x['positive_reward']),1e-12))
        report.append(f"**v0_3 사례:** env{c['env']}/episode{c['episode']}는 최대{100*float(c['maximum_displacement_m']):.2f}cm 이동에 변위 비용−{float(c['displacement_cost']):.3f}, 양의 보상+{float(c['positive_reward']):.3f}, 순보상{float(c['total_reward']):+.3f}이며 성공이었다.")
        report.append(f"![v0_3의 이동·보상·물체 경로]({link('figures/'+c['task']+'_env'+c['env']+'_ep'+c['episode']+'.png')})")
    report.append('전체 후보와 peak가 큰 성공 사례도 사례 CSV에 포함했다. 그림은 policy50Hz에서 관측한 위치 경로이며 substep 사이의 순간 이동까지 복원한 영상은 아니다.')

    report.append('## 현재 비용식이 허용하는 행동과 포착하지 않는 부분')
    report.append('현재 세 환경의 변위 항은 동일하다. 활성 비타겟 장애물만 합산하며, 목표 물체와 비활성 주차 물체는 제외한다. 거리 단위는 m, cap의 단위는 m²다. [3]')
    report.append(r'''```math
S_t=\sum_{i\in\mathcal{O}_{\mathrm{active}}}\|\mathbf{p}_{i,t}-\mathbf{p}_{i,0}\|_2^2,
\qquad \Delta R_t=-150\min(S_t,0.04)\,0.02
```''')
    report.append('**`displacement_max_cost=0.04`는4cm가 아니다.** 합산 제곱거리0.04m²의 cap이고, 한 물체만 움직이면20cm에서 포화된다. 여러 물체는 root-sum-square 변위20cm에서 포화된다. 최대 rate는−6/s·최대 step 비용은−0.12다.')
    caps=sum(s['play'][task]['cap_steps_total'] for task in tasks)
    report.append(f'이번768episode의 cap 도달 policy step은 **{caps}개**였다. 관측된 비용의 크기를 cap 탓으로 설명하기 전에 이 실제 도달 빈도를 보아야 한다.')
    report.append('| 한 장애물 변위 유지 | 1 policy step | 1초 유지 | 10초 유지 | γ=0.98 무한 유지 할인 비용 |\n| --- | ---: | ---: | ---: | ---: |')
    for d in (.005,.01,.02,.05,.10,.20):
        report.append(f'| {d*100:g}cm | −{150*d*d*.02:.5f} | −{150*d*d:.4f} | −{150*d*d*10:.3f} | −{150*d*d:.4f} |')
    report.append('표는 한 물체가 일정 변위에 머문다는 수식 예시다. 실제 이동 중인 trajectory에 같은 값을 대입한 예측이 아니다. 선반 밖 낙하나 종료가 있으면 실제 비용 노출은 더 짧아진다.')
    report.append(f"![수식의 변위·지속시간·할인 효과]({link('figures/cost_calibration.png')})")
    report.append('세 학습 설정의 gamma는0.98이다. policy dt0.02초에서 동일한 미래 비용의 할인 질량은 약1초분이며,1cm를 계속 유지해도 이동 시점 기준 할인 비용은 최대 약0.015다. v0_3에서 목표를1cm 전진시키는 Sweep의 기여는 접촉·힘 정렬이 충분할 때 최대 약+1이므로,1cm 장애물 이동은 작은 교환 비용이 될 수 있다. 이는 수식 비교이며 특정 정책이 그 의도를 학습했다는 증명은 아니다.')
    report.append('변위 항은 **충돌 충격·충돌 횟수·yaw 회전량·누적 이동 경로**를 직접 측정하지 않는다. 원위치로 돌아오면 이후 변위 비용도 감소한다. 기울기·낙하 비용은 별도 존재한다. 성공 판정에는 목표3cm 이내·timeout·물리 실패 제외가 있지만 “장애물을 몇 cm 이하로 유지”하는 조건은 없다. [3]')
    contacts=[s['play'][t]['contact_penalty_steps_total'] for t in tasks]
    report.append(f"보완 항인 `obstacle_contact`의 비영(非零) policy step은 v0_1/v0_2/v0_3에서 각각 **{contacts}**개였다. 학습 로그의 최근1,000iteration 환산 평균 접촉 비용도 변위 비용보다 훨씬 작다. 이 항은 손–장애물 직접 접촉의 **벡터 합력**에서50N 초과를 평가한다. 모든 로봇 링크 충돌이나 타겟–장애물 간접 충돌을 포괄하지 않는다. 비용0을 곧 충돌0으로 해석하지 않는다. [2][3]")

    report.append('## 변경안을 고정 trajectory에서 계산해 보면')
    report.append('다음은 **재학습·새 정책 Play가 아니다.** 같은 기록에서 변위 weight만 바꿨을 때 총보상이 어떻게 바뀌는지 계산한 민감도다. 성공 flag·물체 경로는 그대로 둔다. [1]')
    scenarios=read_csv(root/'fixed_trajectory_weight_scenarios.csv')
    report.append('| 환경·1cm 이상 이동 집단 | 현재 평균 순보상 | weight×2 | weight×5 | weight×10 |\n| --- | ---: | ---: | ---: | ---: |')
    for task in tasks:
        by={int(r['weight_factor']):r for r in scenarios if r['task']==task and r['group']=='moved_1cm'}
        report.append('| '+task+' · n='+by[1]['n']+' | '+' | '.join(f"{float(by[k]['mean_total_on_frozen_trajectory']):+.3f}" for k in (1,2,5,10))+' |')
    report.append('weight 증가는 큰 지속 변위를 더 강하게 누르지만, 짧거나 종료 직전인 이동에 대해서는 노출시간 문제를 유지한다. 큰 이동 실패 사례에 이미 강한 비용이 있는지도 함께 확인해야 한다.')
    report.append('### 우선 검토할 방향')
    report.append('1. **지속 변위 비용 유지 + 새로운 최대 이탈 증가에 즉시 비용.** 허용오차 ε를 넘긴 과거 최대 변위를 `q_i`로 두고, 증가분에 비용을 주면 늦은 이동과 조기 종료에도 이동한 양에 대한 부담이 즉시 남는다. 같은 최대 변위를 매 step 중복 청구하지 않는다.')
    report.append(r'''```math
q_i(t)=\max_{s\le t}\left(\|\mathbf{p}_{i,s}-\mathbf{p}_{i,0}\|_2-\epsilon\right)_+,
\qquad \Delta R_{\mathrm{peak},t}=-k\sum_i\left(q_i(t)-q_i(t-1)\right)
```''')
    report.append('2. **장애물 안정성이 성공 요구사항이면 성공 조건·별도 지표에 반영.** 예를 들어 최대 변위1cm 조건을 추가할 때 기존 성공 중 몇 건이 제외되는지 아래처럼 확인할 수 있다. 허용오차5mm·1cm는 민감도 예시이며 최종 과업 기준으로 확정한 값이 아니다.')
    guards=read_csv(root/'hypothetical_success_guardrails.csv')
    report.append('| 환경 | 기존 성공 | 동일 기록에 최대 변위≤1cm를 요구 |\n| --- | ---: | ---: |')
    for task in tasks:
        r=next(r for r in guards if r['task']==task and float(r['tolerance_m'])==.01)
        report.append(f"| {task} | {r['original_successes']}/{r['n']} | {r['hypothetical_strict_successes']}/{r['n']} |")
    report.append('3. **충돌 자체를 억제하려면 접촉 항의 범위를 별도로 점검.** 손–장애물 합력50N 문턱이 사용자의 “치는 것” 기준과 맞는지, 팔·타겟에 의한 접촉도 포함할지 분리해서 정한다. 변위 weight 하나로 그 의미를 모두 대신하기 어렵다.')
    report.append('peak 증가 비용은 k=50/100 reward/m·허용오차5mm를 예시로 [별도 계산]('+link('fixed_trajectory_peak_scenarios.csv')+')했다. 과업 전진 보상의 규모와 비교하면 k=100 reward/m는 허용오차를 넘긴 새로운 최대 이탈1cm당−1에 해당한다. 적용 후 장애물 안정성·목표 성공·당김 구간 행동을 함께 비교하는 재학습이 필요하며, 여기서는 수치나 환경을 변경하지 않았다.')

    report.append('## 보조 seed와 검증 범위')
    sensitivity=root/'v03_seed43_sensitivity/summary.json'
    if sensitivity.exists():
        other=json.loads(sensitivity.read_text());a=other['with_obstacles'];m=other['moved_1cm']
        report.append(f"같은 v0_3/model_7650의 이전 Play seed43·256episode도 독립 보조 자료로 재분석했다. 장애물이 있는 {a['n']}건 중1cm 이상 이동은 {m['n']}건({pct(m['n']/a['n'])}), 전체 평균 변위 비용은−{other['all']['displacement_cost']['mean']:.3f}였다. seed42의 빈도{pct(moving['n']/active['n'])}·평균 비용−{v3['all']['displacement_cost']['mean']:.3f}와 비교하면 이동 빈도보다 비용의 차이가 크다. 드문 큰 이동의 크기·지속시간에 평균 비용이 민감하므로 단일 seed 평균을 일반화하지 않는다. 주 비교표는 새로 수집한 seed42만 사용하며, 이 보조 자료를 다른 두 모델과 합쳐 동일 표본인 것처럼 비교하지 않았다. [보조 결과]({link('v03_seed43_sensitivity/summary.json')}).")
    errors=[s['play'][task]['quality']['max_displacement_formula_error'] for task in tasks]
    report.append(f"실제 위치·초기 위치·활성 mask에서 비용을 독립 재계산했다. 세 trace의 최대 step 오차는 **{max(errors):.3g}**였다. terminal step을 포함하고 자동 reset 이후 위치를 섞지 않았다. 타겟 제외·활성 mask·3D 거리·합산 cap은 실제 함수를8개 경계 사례로 별도 실행해 확인했다. [검사]({link('formula_check.json')}).")
    report.append('세 정책은 서로 다른 학습 이력·iteration의 checkpoint이므로 관찰된 차이를 종료 방식 변경 하나의 인과 효과라고 단정하지 않는다. 원래 관측 noise는 유지했고 행동만 결정적으로 추론했다. seed42의64환경을 각각4번 사용한 기술통계이며 모든 초기 상태로 일반화한 확률은 아니다. 학습은2048환경, 이번 Play는64환경으로 같은 분포를 재샘플링했다. 초기 장면이 같은지와 이후 episode가 같은지는 구분한다. 이번 분석은 기존 v0_1/v0_2/v0_3 대상이며 Hard/v0_3_1 정책 성능을 평가하지 않는다.')
    report.append('세 실행 모두 학습 시 저장한 설정과 현재 Play 설정의 actions·observations·events·rewards·terminations·commands·curriculum7개 Manager 구성이 일치했다. 수집 후 checkpoint SHA256도 다시 일치했다. [입력 최종 대조]('+link('final_input_validation.json')+'). GPU found/lost aggregate pair capacity만 평가 장치 메모리에 맞춰2^25→2^21로 설정했고, 보상·제어·물리 수치는 변경하지 않았다. 원시 위치로는 충돌의 가해 링크·접촉 충격량·관측 사이의 순간 회전량까지 확정할 수 없다.')

    report.append('## 근거·재현 파일')
    report.append(f"[1] [요약·수식 대조]({link('summary.json')}), [episode별 누적 보상·거리·종료]({link('episodes.csv')}), [장애물별 peak·최종 변위·50Hz 경로]({link('obstacles.csv')}), [변위 구간별]({link('magnitude_bins.csv')}), [장애물 수별]({link('obstacle_count_groups.csv')}). 각 `v0_*_seed42` 폴더에 `trace.npz`, `metadata.json`, 실제 Env/PPO 설정·소스 snapshot·checkpoint SHA256가 있다.")
    report.append(f"[2] [학습 로그 목록]({link('training_inventory.json')}), [로그 설정·파일 해시 대조]({link('training_config_checks.json')}), [전체 학습 곡선]({link('training_curves.csv')}). 로그가 checkpoint보다 이후 iteration까지 있어도 주 집계는 checkpoint iteration에서 잘랐다.")
    source_rel=os.path.relpath(Path('src/inspire_sweep_baseline/inspire_sweep_baseline').resolve(),output.parent).replace('\\','/')
    report.append(f"[3] [변위·접촉 비용 구현]({source_rel}/mdp/reward_functions/contact_sweep_clutter_fusion.py), [보상 계수]({source_rel}/mdp/rewards/prof_revision.py), [v0_3 성공 조건]({source_rel}/mdp/termination_functions/prof_revision.py). RewardManager의10초 정규화·dt 처리는 로컬 Isaac Lab `managers/reward_manager.py`를 기준으로 확인했다.")
    report.append('분석 재실행:')
    report.append('```powershell\npython src/inspire_sweep_baseline/scripts/analyze_obstacle_displacement.py --root outputs/prof_revision_obstacle_analysis_20261125\npython src/inspire_sweep_baseline/scripts/report_obstacle_displacement.py --root outputs/prof_revision_obstacle_analysis_20261125 --output example/Contact-Rich-Manipulation-Context/02_implementation/experiments/2026-11-25_prof_revision_obstacle_displacement_analysis.md\n```')
    output.parent.mkdir(parents=True,exist_ok=True)
    rendered=report[0]
    for previous,current in zip(report,report[1:]):
        separator='\n' if previous.startswith('|') and current.startswith('|') else '\n\n'
        rendered+=separator+current
    output.write_text(rendered+'\n',encoding='utf-8')
    print('Wrote',output)


if __name__=='__main__':main()
