# Sweep Baseline 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

베이스는 전체 항목을, 각 파생 환경은 **직접 부모 대비 변경점만** 적는다. 보상 weight는 커리큘럼 적용 전 값이며, 일정의 단위는 학습 iteration이다.

**환경 바로가기:** [Sweep-Baseline-v0](01_sweep.md#env-inspiresweepenvcfg) · [Sweep-Realistic-Physics-v0](01_sweep.md#env-inspiresweeprealisticphysicsenvcfg) · [Sweep-Realistic-Physics-Initial-Privileged-v0](01_sweep.md#env-inspiresweeprealisticinitialprivilegedenvcfg) · [Sweep-Baseline-Tactile-v0](01_sweep.md#env-inspiresweeptactileenvcfg)

<a id="env-inspiresweepenvcfg"></a>

## 베이스: Sweep-Baseline-v0

`InspireSweepEnvCfg`는 `ManagerBasedRLEnvCfg`를 직접 상속한다. UR5e + Inspire Hand + Axia80 F/T, 선반, 원통 타겟 1개로 구성된 기본 밀기 환경이다. 장애물은 없다. 기본 4096개 환경, 10초 에피소드, 물리 100Hz·정책 50Hz다.

### 관측

Actor/Critic 공통 **41D**: Arm 상대 관절 위치 6 + 속도 6 + 이전 행동 8 + 현재 타겟 XYZ 3 + 폭 1 + EEF pose 7 + 목표 XYZ 3 + 손바닥 접촉 1 + 손목 wrench 6. 타겟 위치·폭에 ±1cm 노이즈가 있다.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 기본 자세 대비 상대 각도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_pos_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 기본 속도 대비 상대 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_vel_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.actions` | 8 | 직전 정책 행동 8개(Arm 6 + Hand 2). | 추가 ObservationTerm 노이즈 없음 | [last_action](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obs_state` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 매 관측 균등 노이즈 -0.01~0.01 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obj_width` | 1 | 타겟의 폭/크기 대표값(target_width). | 매 관측 균등 노이즈 -0.01~0.01 | [object_width](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.goal_pos` | 3 | 목표 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [goal_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.tactile` | 1 | 손바닥 접촉 유무를 나타내는 이진값. | 추가 ObservationTerm 노이즈 없음 | [palm_contact_bit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.wrist_wrench` | 6 | 손목 센서의 힘 XYZ·모멘트 XYZ를 EEF 기준으로 변환한 wrench. | 추가 ObservationTerm 노이즈 없음 | [wrist_wrench_eef](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |

### 환경 구성

Arm 6D는 초기 관절각 기준 위치 명령(scale 0.5), Hand 2D는 손 개방 명령으로 **총 8D 행동**이다. Reset마다 기본 자세를 복원하고 원통 초기 위치에 ±2cm 범위 잡음을 주며 양방향 sweep 목표를 생성한다.

### 보상

구성은 `3×접근 + 2×EEF 자세 정렬 + 6×밀기 + 9×복귀 − 0.03×행동 변화 − 0.03×관절 속도 − 0.5×선반 충돌 − 0.5×물체 운동 비용`이다. 접근은 거리 지수함수, 밀기는 EEF 근접 조건과 목표 방향 이동을 사용하고, 복귀는 밀기 진행 후 초기 자세로 돌아오는 항이다.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `joint_vel` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `object_collision` | -0.5 | 물체 속도 절댓값 합을 tanh로 제한한 운동 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [object_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `reaching` | 3 | EEF와 타겟 접근점 사이 거리에 대한 exp(−10d) 접근 보상. | 높이 offset(m): `0.09` | [reaching_target](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `orientation` | 2 | EEF Y축과 선반 Z축 정렬 점수(sign(cos)×cos²). | 추가 수치 설정 없음(생략값은 함수 기본값) | [align_eef_y_with_shelf_z](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `sweeping_object` | 6 | EEF가 타겟 가까이에 있을 때 목표 방향 밀기 진행·속도에 보상. | 높이 offset(m): `0.09` | [pushing_target](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `homing_after_sweep` | 9 | 밀기 진행 후 Arm이 초기 관절 자세로 돌아오도록 유도. | 추가 수치 설정 없음(생략값은 함수 기본값) | [homing_reward](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |

### 터미네이션

10초 timeout, 물체 낙하/기울어짐(기준 높이 1.04m·0.9rad), 물체 속도 0.3m/s 초과, 선반 충돌, Arm 관절 속도 1rad/s 초과. `hand_velocity`라는 term 이름이지만 실제 검사 대상은 Arm이다. 성공 조기 종료는 없다.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 타겟이 기준 높이 아래로 떨어지거나 회전 조건을 초과하면 실패. | 높이 기준(m): `1.04`<br>회전 문턱: `0.9` | [object_drop](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.3` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `1` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |

### 커리큘럼

없음. 물성 랜덤화는 아래 파생 환경에서 추가된다.

설정된 항목 없음.

## 관련 상속 관계

[Omni](02_omni.md)와 [Renewal](03_renewal.md)는 Baseline에서 분기하며 각 문서에서 설명한다.

<a id="env-inspiresweeprealisticphysicsenvcfg"></a>

## Sweep-Realistic-Physics-v0

**직접 부모:** [Sweep-Baseline-v0](01_sweep.md#env-inspiresweepenvcfg) (`InspireSweepEnvCfg`). **환경 설정:** [InspireSweepRealisticPhysicsEnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_envs.py).

### 이 환경에서 바뀐 점

매 reset 질량·마찰 랜덤화. 질량 평균 0.365kg, 표준편차 0.0675kg, 범위 0.23–0.50kg. 손·선반·물체 마찰을 역할별로 샘플링한다.

**직접 부모와 동일:** 관측 · 보상 · 터미네이션 · 커리큘럼.

<a id="env-inspiresweeprealisticinitialprivilegedenvcfg"></a>

## Sweep-Realistic-Physics-Initial-Privileged-v0

**직접 부모:** [Sweep-Realistic-Physics-v0](01_sweep.md#env-inspiresweeprealisticphysicsenvcfg) (`InspireSweepRealisticPhysicsEnvCfg`). **환경 설정:** [InspireSweepRealisticInitialPrivilegedEnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_envs.py).

### 이 환경에서 바뀐 점

Actor 타겟 XYZ를 에피소드 초기값으로 고정. Critic만 현재 타겟 위치·회전·선속도·각속도 13D 추가. **Actor 41D / Critic 54D**.

### 관측 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `privileged.target_state` | 추가 | 없음 | 차원: 13<br>추가 ObservationTerm 노이즈 없음; Critic 전용<br>구현: [live_object_state_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/asymmetric_observations.py) | Critic 전용 현재 타겟 위치 3·quaternion 4·선속도 3·각속도 3, robot-root 기준. |

**직접 부모와 동일:** 보상 · 터미네이션 · 커리큘럼.

<a id="env-inspiresweeptactileenvcfg"></a>

## Sweep-Baseline-Tactile-v0

**직접 부모:** [Sweep-Baseline-v0](01_sweep.md#env-inspiresweepenvcfg) (`InspireSweepEnvCfg`). **환경 설정:** [InspireSweepTactileEnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_envs.py).

### 이 환경에서 바뀐 점

손바닥 1bit를 부위별 촉각 17bit로 교체. **Actor/Critic 57D**. 물성은 Realistic 부모를 상속하지 않는다.

### 관측 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `policy.tactile` | 수정 | 차원: 1<br>손바닥 접촉 유무를 나타내는 이진값.<br>구현: [palm_contact_bit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py)<br>sensor_name: `함수 기본값`<br>접촉 문턱(N): `함수 기본값` | 차원: 17<br>손의 17개 촉각 부위별 접촉 유무. 힘을 부위 방향에 투영해 문턱과 비교.<br>구현: [ProjectedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/tactile_observations.py)<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.05` | 손의 17개 촉각 부위별 접촉 유무. 힘을 부위 방향에 투영해 문턱과 비교. |

**직접 부모와 동일:** 보상 · 터미네이션 · 커리큘럼.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/sweep.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/sweep.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/sweep.py)
