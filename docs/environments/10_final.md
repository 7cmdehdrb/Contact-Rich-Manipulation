# Final 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Final-v0

`FinalEnvCfg`는 `ManagerBasedRLEnvCfg`를 직접 상속한다. Holding-OSC-ObjectRandom-v3의 동작을 기준으로 Scene·Manager·runtime·PPO를 독립 구성했다. Python 상속으로 그 환경에 연결되지는 않는다.

### 관측

직접 **55D**: Arm 상대 위치·속도 12 + 이전 행동 8 + 초기 타겟 XYZ 3 + EEF pose 7 + 목표 극좌표 2 + 촉각 17 + 보정 F/T 6. 초기 장애물 16×4=64D, Critic 현재 타겟 XYZ 3D. **원시 Actor/Critic 119/122D, MLP 71/74D**. 두 모델 모두 초기 장애물을 쓰며 Critic에 실시간 장애물은 아직 없다.

### 환경 구성

원통 타겟 1 + 다양한 형상의 장애물 최대 8/8, 물리 용량 17개 중 초기 활성 1/1. Reset에서 배치와 물성을 재추출한다. 기본 4096개·10초·50Hz. 행동은 root-frame 상대 OSC pose 6 + Hand 2, 위치/회전 scale 0.03m/0.15rad, stiffness 병진 100·회전 50, Arm PD 해제다.

### 보상

`30×접근 최단 기록 + 100×목표 최단 기록 + 4×중심 가중 Hold + 1.45×(접촉 OR 목표 점유) − 비용`이다. 목표 점유 시 Reaching 반경을 10cm→1m로 완화한다. Hold는 거리·속도 Gaussian에 중심 가중을 곱하고 목표 반경 r의 2배까지 지급한다. 비용은 행동 변화, 관절 속도, 기울기/장애물 변위, 선반 충돌/실패, 물체·Arm·EEF 과속, 장애물 접촉력·물체 가속도·wrench 변화다.

### 터미네이션

10초, 활성 물체 낙하/기울기, 타겟 1m/s 초과, Arm 5rad/s 초과, 선반 충돌/변위. **성공 조기 종료 없음**.

### 커리큘럼

장애물 1/1→6/6(White 3k·Gray 4k 시작, 각 2k 간격), Gray 안쪽 경계 침범 비율 0→15%(6k→12k). Hold 반경 2→1→0.5cm(5k/10k/20k), 속도 scale 5→2cm/s. 물체/Arm/EEF 자유 속도와 접촉력·가속도·wrench 비용/문턱을 점진적으로 강화한다. 목표 제외 반각은 30°에서 3000 이후 500마다 5°씩 감소한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Final-v1 | Final-v0 | 생성 시 타겟도 형상 랜덤화·크기 ±10%, 물리 용량 10/10·관측 20슬롯. Hand 실제 구성 2D와 촉각/F/T 노이즈, 실행 명령 기준 이전 행동을 사용. 비용을 11개 보상 term으로 재구성하고 Arm/EEF 초과속도 항 제거. 10초 시점 목표 반경 내/외를 success/time_out으로 분리(둘 다 truncation). |
| Final-v1-1 | Final-v1 | 목표 최단 거리 갱신 보상을 `1.5×exp(−d/0.05)`의 지속 거리 보상으로 교체. |
| Final-Test-v1 | Final-v1 | 통합 object 비용을 기울기·장애물 변위·가속도로 분리해 기록. 가속도 일정의 대상 term도 변경. 총 보상 의미는 유지. |
| Final-v2 | Final-v1 | reset에서 `wide_target=True`: 타겟 시작점을 중앙 반경 2cm에서 선반 중앙 80%와 물체 크기/여유 조건의 교집합으로 확장. 보상·관측·종료는 v1 유지. |
| Final-Test-v2 | Final-v2 | 첫 reset부터 최종 장애물 요청 수 8/8을 사용하고 개수 커리큘럼만 제거. |

v1 및 위 파생의 직접 관측은 57D(기존 55 + Hand 2), 장애물 80D이므로 **원시 137/140D, MLP 73/76D**다. v1 개수 일정은 1/1→8/8(White 2k·Gray 3k 시작·각 2k 간격), Hold 반경은 2cm→1cm(4k→10k), Gray 안쪽 경계 침범 비율은 0→15%(4k→10k)로 변경된다. 물체 자유 속도는 0.1→0.02m/s로 유지하고 물체 가속도·장애물 접촉·wrench 비용 일정을 적용한다.

`FinalGoalAccuracyEnvCfg`도 **Final-v1의 직계 자식**이며 [GoalAccuracy](11_goal_accuracy.md) 문서에 별도로 정리한다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/final.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/final.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/final.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/final.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/final.py)
