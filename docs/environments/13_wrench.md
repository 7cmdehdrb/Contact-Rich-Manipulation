# Wrench 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Inspire-Wrench-v0

`WrenchEnvCfg`는 `ManagerBasedRLEnvCfg`를 직접 상속한다. Revision-Test-v2를 기준으로 독립 구성했다. **v0의 Arm은 pose OSC**이며 실제 wrench 제어는 v1부터다. Gym ID에는 `Isaac-` 접두사가 없다.

### 관측

직접 **57D**: Arm 실제 위치·속도 12 + Hand 2 + EEF pose 7 + 초기 타겟 XYZ 3 + 목표 극좌표 2 + 이전 행동 8 + 촉각 17 + 보정 F/T 6. Actor 초기 장애물 64D, Critic 현재 타겟 3D·현재 활성 장애물 64D. **원시 Actor/Critic 121/124D, MLP 73/76D**. 모든 Wrench 버전에서 차원은 같다.

### 환경 구성

원통 타겟 + 다양한 장애물, 물리 10개(최대 Gray 4/White 5), 기본 2048개·10초·50Hz. 생성 시 종류/크기를 정하고 startup에서 단계별 **layout bank(기본 16개)**와 물성을 만들어 둔다. Reset에서는 bank 배치를 선택하며 물성을 다시 뽑지 않는다. OBB Gray padding 65mm, RED margin 23mm, 뒤쪽 Gray offset 20mm. 행동은 상대 pose OSC 6 + Hand 2, scale 0.03m/0.15rad다.

### 보상

15개 term: 접근 최단 기록 30, 거리 `3.7×exp(−d/0.05)`, Hold 10, 실제 접촉 1.45, 최초 진입 최대 +25, 안정 성공 +75와 비용 9개. 비용은 행동 변화·관절 속도·물체 기울기·장애물 변위·물체 가속도·선반·장애물 접촉력·wrench 변화·타겟 속도다.

거리/Hold/접촉은 **EEF 120mm 미만** 조건. Hold는 2.5cm까지 1, 3.5cm까지 Smoothstep으로 0이 되며 속도 감쇠는 없다. 최초 3cm 내부 진입 보너스에는 `exp(−(max(v−0.01,0)/0.02)²)` 속도 감쇠가 있고 높이 조건은 없다. wrench 비용은 자유 변화 20N/2Nm, weight −0.1이며 양의 보상 예산 제한은 없다.

### 터미네이션

실패는 Revision과 동일한 낙하/기울기·타겟 1m/s·Arm 5rad/s·선반 실패. 성공은 **XY 거리 3cm 이하·속도 0.005m/s 이하를 연속 3초** 유지하면 조기 종료한다. 성공은 terminated, 10초 미성공은 truncated다. +75는 timeout이 아니라 이 안정 성공 이벤트에 지급한다.

### 커리큘럼

장애물 3/3→3/4(10k)→4/4(12k)→4/5(14k). 타겟 자유 속도 0.1→0.02m/s. 목표 제외 반각 30°는 **10k에서 처음 5° 감소**, 이후 3k마다 감소한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Inspire-Wrench-v0_1 | **Wrench-v0** | 독립 접촉 보상 제거. 거리 점수에 초기/현재 목표 벡터의 signed cosine과 연속 2step 타겟 접촉 조건을 곱한다. d<3cm에서는 cosine=1. EEF 높이 조건 유지. 목표 제외 반각 40° 고정, 허용 목표 거리 4–7.5cm. |
| Inspire-Wrench-v0_2 | **Wrench-v0** | 거리×signed cosine은 쓰되 연속 접촉 조건 없음, d<3cm에서 cosine=1. 접촉 보상은 최근 3step 중 2회 인정 또는 목표 3cm 내부일 때 누적 count(최대 10)를 지급, 연속 불인정 5회면 초기화. 장애물 기준 −1/−1에서 2k에 White 먼저 증가, 이후 3k마다 교대 증가·reset ±1 샘플 후 0~4/0~5로 제한. |
| Inspire-Wrench-v1 | **Wrench-v0** | Arm을 Axia80 센서 중심 6축 closed-loop wrench 제어로 변경. `wrench_abs`, 목표 ±100N/±5Nm, P gain 0.2, 보정 명령 cap ±120N/±6Nm, 중력 보상. Arm action-rate를 목표 범위로 정규화한 변화량으로 계산. EEF 관측/보상 frame과 Hand는 유지. |
| Inspire-Wrench-v2 | **Wrench-v1** | 목표 범위를 ±10N/±0.5Nm, 보정 cap을 ±12N/±0.6Nm로 1/10 축소. 정규화 action-rate와 나머지 구성은 상속. |

v0_2 접촉 count는 불인정 step에는 지급하지 않되 4회까지 보존한다. 목표 내부 판정은 접촉력이 없어도 인정되며, 실제 접촉 경로에는 높이 조건이 적용된다. 50Hz에서 최대 지급은 `1.45×10×0.02=0.29/step`이다. v0_1/v0_2는 v1의 wrench 제어를 상속하지 않는다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/wrench_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/wrench.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/wrench.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/wrench.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/wrench.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/wrench.py) · [제어](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/actions/wrench.py) · [거리·접촉 함수](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/wrench.py)
