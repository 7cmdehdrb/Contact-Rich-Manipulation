# Sweep Omni 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Sweep-Omni-v0

`InspireSweepOmniEnvCfg`는 [Sweep-Baseline-v0](01_sweep.md)의 `InspireSweepEnvCfg`를 직접 상속한다. 좌우 밀기에서 평면의 다양한 방향으로 목표에 접근하는 과제로 바꾼 계열이다.

### 관측

Actor/Critic **41D**: Arm 위치·속도 12 + 이전 행동 8 + 현재 타겟 XYZ·폭 4 + EEF pose 7 + 목표 XYZ 3 + 손바닥 접촉 1 + F/T 6. Baseline의 실시간 타겟 관측을 유지한다.

### 환경 구성

선반 위 원통 1개, Arm 위치 6 + Hand 개방 2의 8D 행동, 10초·50Hz. Reset 시 반경 2cm 범위에서 시작점을 정하고 타겟으로부터 15cm 거리의 평면 목표를 생성한다. 가장자리 여유 2cm를 두며 Realistic 질량·마찰 랜덤화를 적용한다.

### 보상

`접근 진전 + 6×(exp(−d/0.10)−1) + 성공 30 − 실패 80`에 Arm/Hand 행동 변화·관절 속도, 물체 기울기, 선반 충돌, 시간 비용(−0.1)을 더한다. 접근은 초기 타겟 위치까지의 새 최단 거리 진전을 초기 거리로 정규화한 값이다. 거리 항은 목표에서 0이고 멀어질수록 음수가 되는 지속 비용이다. 성공/실패 이벤트 보상은 dt와 무관한 일회 지급이다.

### 터미네이션

Baseline의 낙하·과속·충돌·Arm 속도 제한에 선반 밖 이탈, 선반 변위 5cm 초과, 안정 목표 도달을 추가한다. 성공은 **1cm 이내·속도 2cm/s 이하·기울기 10° 이내·0.3초 유지**와 선반 지지 조건을 사용한다.

### 커리큘럼

없음.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Sweep-Omni-v1 | Omni-v0 | Arm 종료 속도를 **1 → 1.5rad/s**로 변경. 클래스 설명의 3rad/s보다 실제 대입값을 따른다. |
| Sweep-Omni-v2 | Omni-v1 | 현재 물체 접근 진전 weight 30, 목표 진전 weight 40. 시간·성공·실패 보상과 성공 종료를 제거하여 실패/timeout까지 진행한다. |
| Dex-Omni-v0 | Omni-v2 | 접근/목표를 새 최단 거리 갱신량 보상으로 교체. 계수 30/300, 접근 반경 10cm·목표 허용 1cm. |
| Sweep-Omni-v3 | Omni-v2 | 접근 보상을 높이 조건부 XY 진전으로 변경. 타겟 기준 z offset 9cm, 허용 높이 오차 2cm. |
| Sweep-Omni-v4 | Omni-v3 | 물체·Arm 초과속도 비용 추가. Arm 종료 5rad/s, 물체 종료 속도 0.5m/s, 선반 밖 종료 제거. 보상 Cfg는 v2에서 파생하지만 EnvCfg의 v3 post-init이 높이 조건부 접근을 다시 적용한다. |

Renewal은 이름과 달리 Omni-v4의 자식이 아니라 Baseline의 별도 직계 파생이다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/sweep_omni.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/sweep_omni.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/sweep_omni.py)
