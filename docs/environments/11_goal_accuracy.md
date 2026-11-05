# Final GoalAccuracy 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Final-GoalAccuracy-v0

`FinalGoalAccuracyEnvCfg`는 [Final-v1](10_final.md)을 직접 상속한다. 목표 정밀도 평가를 위해 성공 반경을 고정하고 전용 runtime에서 정확도 지표를 기록한다.

### 관측

Final-v1과 동일한 직접 **57D**: Arm 상대 위치·속도 12 + 이전 명령 8 + 초기 타겟 3 + EEF pose 7 + 목표 극좌표 2 + 촉각 17 + 보정 F/T 6 + Hand 구성 2. 초기 장애물 20×4=80D, Critic 현재 타겟 3D. **원시 Actor/Critic 137/140D, MLP 73/76D**. 모든 Accuracy 파생이 같은 차원을 유지한다.

### 환경 구성

Final-v1의 다양한 타겟/장애물 형상, 생성 시 ±10% 크기, reset 배치·질량·마찰 랜덤화. 물리 용량은 기본 21개(타겟 1 + 장애물 20), 활성 개수는 최대 Gray 4/White 5다. 8D 상대 pose OSC/Hand, 4096개·10초·50Hz.

### 보상

부모의 접근 30·목표 최단 기록 100·접촉 OR 목표 점유 1.45와 행동/관절/물체/선반/접촉력/wrench 비용을 유지한다. **Hold는 weight 8, 반경 1cm 안에서만** 거리·속도 Gaussian과 중심 가중으로 지급하며 외곽 보상은 없다. 속도 scale 5cm/s 고정. 물체 속도 비용의 이름은 `target_speed_penalty`로 바뀐다.

### 터미네이션

부모의 물체 낙하/기울기·타겟 과속·Arm 과속·선반 실패를 유지한다. 10초 시점 XY 거리 **1cm 이하**면 성공 truncation, 아니면 실패 timeout이다. 이 판정에는 정지 속도·연속 유지 시간 조건이 없다.

### 커리큘럼

목표 반경·Hold 속도 scale 일정은 제거한다. 장애물은 1/1→**4/5**(White 2000·Gray 2500 시작, 각 1000 간격, 5000 이후 추가 증가 없음). 부모의 타겟 자유 속도, 가속도, Gray 안쪽 경계 침범 비율(0→15%), 접촉력·wrench, 목표 방향 일정은 유지한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| GoalAccuracy-v1 | GoalAccuracy-v0 | 1cm 반경은 고정, Hold 외곽 2r 범위와 속도 scale 5→2cm/s 일정을 복원. |
| GoalAccuracy-v0_1 | GoalAccuracy-v0 | 반경 2cm·Hold 외곽 6cm, 거리 보상 `3.7×exp(−d/0.05)`, 접촉 대상을 타겟으로 한정(목표 점유 OR 유지). Reaching은 현재 코드 **10cm 고정**, 목표 점유 완화 제거. 큐브 타겟 후보 크기를 6×8×10cm로 변경하며 다른 형상 후보는 유지. Manager 이름을 대상별로 정리. |
| GoalAccuracy-v0_2 | v0_1 | timeout 때 2cm 내부 도달 보너스 +30 추가. 이 보너스 함수는 `< 2cm`, 성공 종료 분류는 `≤ 2cm`라 경계가 다르다. |
| GoalAccuracy-v0_3 | v0_2 | 거리 항을 최단 기록 갱신(weight 100·허용 거리 0)으로 복원, 목표 반경 3cm, 접촉 weight 1.45→0.3625, 성공 timeout 보너스 +100. |
| GoalAccuracy-v0_2_1 | **v0_2** | 실제 타겟 접촉만 보상, 성공 timeout +75(2cm 경계 포함). reset마다 타겟 후보 재선택, OBB 배치·최소 간격 20mm·Gray padding 60→69mm. 후보 풀로 물리 21→36개. |
| GoalAccuracy-v0_2_2 | v0_2_1 | 물리 36→12개, 공유 물성/prototype cache와 텐서 기반 역할 재할당. 보상·관측·활성 최대 4/5·OBB 규칙은 유지. 환경별 보유 형상이 고정되어 reset 간 형상 분포의 상관은 달라진다. |

v0_1의 예전 설명에 있는 Reaching 20cm보다 **현재 코드의 10cm**를 기준으로 읽는다. v0_2_1과 v0_3은 각각 v0_2에서 갈라지는 형제다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/final.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/final.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/final.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/final.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/final.py) · [보상 함수](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py)
