# Sweep Baseline 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Sweep-Baseline-v0

`InspireSweepEnvCfg`는 `ManagerBasedRLEnvCfg`를 직접 상속한다. UR5e + Inspire Hand + Axia80 F/T, 선반, 원통 타겟 1개로 구성된 기본 밀기 환경이다. 장애물은 없다. 기본 4096개 환경, 10초 에피소드, 물리 100Hz·정책 50Hz다.

### 관측

Actor/Critic 공통 **41D**: Arm 상대 관절 위치 6 + 속도 6 + 이전 행동 8 + 현재 타겟 XYZ 3 + 폭 1 + EEF pose 7 + 목표 XYZ 3 + 손바닥 접촉 1 + 손목 wrench 6. 타겟 위치·폭에 ±1cm 노이즈가 있다.

### 환경 구성

Arm 6D는 초기 관절각 기준 위치 명령(scale 0.5), Hand 2D는 손 개방 명령으로 **총 8D 행동**이다. Reset마다 기본 자세를 복원하고 원통 초기 위치에 ±2cm 범위 잡음을 주며 양방향 sweep 목표를 생성한다.

### 보상

구성은 `3×접근 + 2×EEF 자세 정렬 + 6×밀기 + 9×복귀 − 0.03×행동 변화 − 0.03×관절 속도 − 0.5×선반 충돌 − 0.5×물체 운동 비용`이다. 접근은 거리 지수함수, 밀기는 EEF 근접 조건과 목표 방향 이동을 사용하고, 복귀는 밀기 진행 후 초기 자세로 돌아오는 항이다.

### 터미네이션

10초 timeout, 물체 낙하/기울어짐(기준 높이 1.04m·0.9rad), 물체 속도 0.3m/s 초과, 선반 충돌, Arm 관절 속도 1rad/s 초과. `hand_velocity`라는 term 이름이지만 실제 검사 대상은 Arm이다. 성공 조기 종료는 없다.

### 커리큘럼

없음. 물성 랜덤화는 아래 파생 환경에서 추가된다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Sweep-Realistic-Physics-v0 | Sweep-Baseline-v0 | 매 reset 질량·마찰 랜덤화. 질량 평균 0.365kg, 표준편차 0.0675kg, 범위 0.23–0.50kg. 손·선반·물체 마찰을 역할별로 샘플링한다. |
| Sweep-Realistic-Physics-Initial-Privileged-v0 | Sweep-Realistic-Physics-v0 | Actor 타겟 XYZ를 에피소드 초기값으로 고정. Critic만 현재 타겟 위치·회전·선속도·각속도 13D 추가. **Actor 41D / Critic 54D**. |
| Sweep-Baseline-Tactile-v0 | Sweep-Baseline-v0 | 손바닥 1bit를 부위별 촉각 17bit로 교체. **Actor/Critic 57D**. 물성은 Realistic 부모를 상속하지 않는다. |

[Omni](02_omni.md)와 [Renewal](03_renewal.md)도 Baseline의 직접 파생이며 별도 문서로 정리한다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/sweep.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/sweep.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/sweep.py)
