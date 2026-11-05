# Sweep Omni Renewal 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Sweep-Omni-Renewal-v0

`InspireSweepOmniRenewalEnvCfg`는 [Sweep-Baseline-v0](01_sweep.md)를 직접 상속한다. Omni의 목표 기반 보상과 완화된 실패 한계를 새로 조합한 구성이다.

### 관측

기본 Actor/Critic **41D**: Arm 위치·속도 12 + 이전 행동 8 + 현재 타겟 XYZ·폭 4 + EEF pose 7 + 목표 XYZ 3 + 손바닥 접촉 1 + F/T 6. Tactile 파생은 접촉 1 → 17로 **57D**가 된다.

### 환경 구성

원통 1개, 시작 반경 2cm·목표 거리 15cm·가장자리 여유 2cm, 매 reset 질량·마찰 랜덤화. 기본 8D 행동은 Arm 관절 위치 6 + Hand 개방 2다. 4096개 환경, 10초·50Hz.

### 보상

`30×높이 조건부 XY 접근 진전 + 40×목표 진전 − 행동 변화/관절 속도 − 물체 기울기 − 선반 충돌 − 물체/Arm 초과속도 비용`. 접근 높이는 offset 9cm ±2cm, 근접 반경 10cm, 목표 허용 거리 1cm다. 속도 비용은 허용 속도를 넘은 부분에 지수적으로 증가한다(기본 물체 0.3m/s·Arm 1rad/s).

### 터미네이션

10초, 낙하/기울기, 물체 속도 0.8m/s 초과, Arm 5rad/s 초과, 선반 충돌/변위 5cm. 선반 충돌 종료에서 손가락 검사는 제외한다. 선반 밖 위치 자체와 성공 도달은 종료 조건이 아니다.

### 커리큘럼

없음. v1/v3의 값 변경은 고정 설정이다.

## 파생 환경

아래 `Renewal`은 `Sweep-Omni-Renewal`을 뜻한다.

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Dex-Sweep-Omni-Renewal-v0 | Renewal-v0 | 접근/목표를 Dex 새 최단 거리 보상으로 변경, 계수 30/300. |
| Sweep-Omni-Renewal-OSC-v0 | Renewal-v0 | Arm을 상대 pose OSC로 교체. stiffness 200, 위치/회전 scale 1/1, Arm PD 해제. |
| Tactile-Sweep-Omni-Renewal-v0 | Renewal-v0 | 촉각 17bit, 57D 관측. |
| Dex-Tactile-Sweep-Omni-Renewal-v0 | Dex-Sweep-Omni-Renewal-v0 | 촉각 17bit, 57D 관측. |
| Dex-Tactile…-v1 | Dex-Tactile…-v0 | Arm 물리 속도 제한 2/2/2/4/4/4rad/s, 두 지수 속도 비용 cap 100, 손목 선반 여유 3cm, 목표 ±X 방향 제외 반각 15°. |
| Dex-Tactile…-v2 | Dex-Tactile…-v1 | 1cm·2cm/s·10°·0.5초 안정 유지 시 +100 일회 지급, goal_hold weight 20. 성공 후에도 실패/timeout까지 계속한다. |
| Dex-Tactile…-OSC-v2 | Dex-Tactile…-v2 | bounded OSC, stiffness 100, 위치 0.005m·회전 0.025rad scale 및 명령 제한. |
| Dex-Tactile…-v3 | **Dex-Tactile…-v1** | v2의 Hold 실험을 상속하지 않음. 물체 질량 1.5배, 물체 자유 속도 0.2m/s·weight −2, 목표 진전 weight 94. |

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_renewal_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep_omni_renewal.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/sweep_omni_renewal.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/sweep_omni_renewal.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/sweep_omni_renewal.py) · [기본 관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep.py)
