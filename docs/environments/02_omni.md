# Sweep Omni 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

베이스는 전체 항목을, 각 파생 환경은 **직접 부모 대비 변경점만** 적는다. 보상 weight는 커리큘럼 적용 전 값이며, 일정의 단위는 학습 iteration이다.

**환경 바로가기:** [Sweep-Omni-v0](02_omni.md#env-inspiresweepomnienvcfg) · [Sweep-Omni-v1](02_omni.md#env-inspiresweepomniv1envcfg) · [Sweep-Omni-v2](02_omni.md#env-inspiresweepomniv2envcfg) · [Dex-Omni-v0](02_omni.md#env-inspiredexomnienvcfg) · [Sweep-Omni-v3](02_omni.md#env-inspiresweepomniv3envcfg) · [Sweep-Omni-v4](02_omni.md#env-inspiresweepomniv4envcfg)

<a id="env-inspiresweepomnienvcfg"></a>

## 베이스: Sweep-Omni-v0

`InspireSweepOmniEnvCfg`는 [Sweep-Baseline-v0](01_sweep.md)의 `InspireSweepEnvCfg`를 직접 상속한다. 좌우 밀기에서 평면의 다양한 방향으로 목표에 접근하는 과제로 바꾼 계열이다.

### 관측

Actor/Critic **41D**: Arm 위치·속도 12 + 이전 행동 8 + 현재 타겟 XYZ·폭 4 + EEF pose 7 + 목표 XYZ 3 + 손바닥 접촉 1 + F/T 6. Baseline의 실시간 타겟 관측을 유지한다.

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

선반 위 원통 1개, Arm 위치 6 + Hand 개방 2의 8D 행동, 10초·50Hz. Reset 시 반경 2cm 범위에서 시작점을 정하고 타겟으로부터 15cm 거리의 평면 목표를 생성한다. 가장자리 여유 2cm를 두며 Realistic 질량·마찰 랜덤화를 적용한다.

### 보상

`접근 진전 + 6×(exp(−d/0.10)−1) + 성공 30 − 실패 80`에 Arm/Hand 행동 변화·관절 속도, 물체 기울기, 선반 충돌, 시간 비용(−0.1)을 더한다. 접근은 초기 타겟 위치까지의 새 최단 거리 진전을 초기 거리로 정규화한 값이다. 거리 항은 목표에서 0이고 멀어질수록 음수가 되는 지속 비용이다. 성공/실패 이벤트 보상은 dt와 무관한 일회 지급이다.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 1 | 초기 타겟 위치까지의 새 최단 접근 진전을 초기 접근 거리로 정규화. dt를 상쇄하는 진전 보상. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [ReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `goal_distance` | 6 | exp(−d/sigma)−1의 지속 거리 비용. 목표에서 0, 멀어지면 음수. | 거리 scale(m): `0.1` | [planar_goal_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object_tilt` | -2 | 타겟 기울기가 free_angle을 넘은 부분의 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `time` | -0.1 | 매 step 경과 시간에 비례하는 일정 시간 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [time_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `success` | 30 | 지정 성공 termination이 발생한 step의 일회 보너스. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [success_event](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `failure` | -80 | 지정 실패 termination 중 하나가 발생한 step의 일회 비용. dt 상쇄. | 제외/검사할 실패 term: `(object_drop, push_fast, shelf_collision, hand_velocity, object_outside_shelf, shelf_displacement)` | [failure_event](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

### 터미네이션

Baseline의 낙하·과속·충돌·Arm 속도 제한에 선반 밖 이탈, 선반 변위 5cm 초과, 안정 목표 도달을 추가한다. 성공은 **1cm 이내·속도 2cm/s 이하·기울기 10° 이내·0.3초 유지**와 선반 지지 조건을 사용한다.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 타겟이 기준 높이 아래로 떨어지거나 기울기 한계를 초과하면 실패. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.3` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `1` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `object_outside_shelf` | terminated | 타겟이 선반 평면 경계 밖으로 나가면 실패. | half_extents: `(0.2, 0.5)` | [object_outside_shelf](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `success` | terminated | 목표 거리·속도·기울기·선반 지지 조건을 hold_time 동안 연속 만족하면 성공. 동시 실패 제외. | 목표 반경(m): `0.01`<br>속도 한계(m/s): `0.02`<br>기울기 한계(rad): `0.174533`<br>연속 유지(s): `0.3`<br>surface_height: `1.05`<br>선반 지지 허용 오차(m): `0.01`<br>half_extents: `(0.2, 0.5)`<br>object_radius: `0.04`<br>제외/검사할 실패 term: `(object_drop, push_fast, shelf_collision, hand_velocity, object_outside_shelf, shelf_displacement)` | [StableGoalSuccess](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

### 커리큘럼

없음.

설정된 항목 없음.

## 관련 상속 관계

[Renewal](03_renewal.md)의 직접 부모는 Baseline이며 Omni-v4가 아니다.

<a id="env-inspiresweepomniv1envcfg"></a>

## Sweep-Omni-v1

**직접 부모:** [Sweep-Omni-v0](02_omni.md#env-inspiresweepomnienvcfg) (`InspireSweepOmniEnvCfg`). **환경 설정:** [InspireSweepOmniV1EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_envs.py).

### 이 환경에서 바뀐 점

Arm 종료 속도를 **1 → 1.5rad/s**로 변경. 클래스 설명의 3rad/s보다 실제 대입값을 따른다.

### 터미네이션 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `hand_velocity` | 수정 | threshold: `1` | threshold: `1.5` | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. |

**직접 부모와 동일:** 관측 · 보상 · 커리큘럼.

<a id="env-inspiresweepomniv2envcfg"></a>

## Sweep-Omni-v2

**직접 부모:** [Sweep-Omni-v1](02_omni.md#env-inspiresweepomniv1envcfg) (`InspireSweepOmniV1EnvCfg`). **환경 설정:** [InspireSweepOmniV2EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_envs.py).

### 이 환경에서 바뀐 점

현재 물체 접근 진전 weight 30, 목표 진전 weight 40. 시간·성공·실패 보상과 성공 종료를 제거하여 실패/timeout까지 진행한다.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `reaching` | 수정 | weight: 1<br>초기 타겟 위치까지의 새 최단 접근 진전을 초기 접근 거리로 정규화. dt를 상쇄하는 진전 보상.<br>구현: [ReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | weight: 30<br>현재 움직이는 타겟까지의 새 최단 접근 진전. 초기 거리로 정규화하고 dt 상쇄.<br>구현: [CurrentObjectReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | 현재 움직이는 타겟까지의 새 최단 접근 진전. 초기 거리로 정규화하고 dt 상쇄. |
| `goal_distance` | 수정 | weight: 6<br>exp(−d/sigma)−1의 지속 거리 비용. 목표에서 0, 멀어지면 음수.<br>구현: [planar_goal_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py)<br>거리 scale(m): `0.1`<br>목표 반경(m): `함수 기본값` | weight: 40<br>타겟→목표 XY 최단 거리의 새 개선량을 초기 거리로 정규화. 에피소드 진전 예산형 보상.<br>구현: [PlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py)<br>거리 scale(m): `함수 기본값`<br>목표 반경(m): `0.01` | 타겟→목표 XY 최단 거리의 새 개선량을 초기 거리로 정규화. 에피소드 진전 예산형 보상. |
| `time` | 삭제 | weight: -0.1<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [time_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 매 step 경과 시간에 비례하는 일정 시간 비용. → 이 환경에서는 제거. |
| `success` | 삭제 | weight: 30<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [success_event](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 지정 성공 termination이 발생한 step의 일회 보너스. dt 상쇄. → 이 환경에서는 제거. |
| `failure` | 삭제 | weight: -80<br>제외/검사할 실패 term: `(object_drop, push_fast, shelf_collision, hand_velocity, object_outside_shelf, shelf_displacement)`<br>구현: [failure_event](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 지정 실패 termination 중 하나가 발생한 step의 일회 비용. dt 상쇄. → 이 환경에서는 제거. |

### 터미네이션 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `success` | 삭제 | 종료 유형: terminated<br>목표 반경(m): `0.01`<br>속도 한계(m/s): `0.02`<br>기울기 한계(rad): `0.174533`<br>연속 유지(s): `0.3`<br>surface_height: `1.05`<br>선반 지지 허용 오차(m): `0.01`<br>half_extents: `(0.2, 0.5)`<br>object_radius: `0.04`<br>제외/검사할 실패 term: `(object_drop, push_fast, shelf_collision, hand_velocity, object_outside_shelf, shelf_displacement)`<br>구현: [StableGoalSuccess](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) | `None` / 미사용 | 목표 거리·속도·기울기·선반 지지 조건을 hold_time 동안 연속 만족하면 성공. 동시 실패 제외. → 이 환경에서는 제거. |

**직접 부모와 동일:** 관측 · 커리큘럼.

<a id="env-inspiredexomnienvcfg"></a>

## Dex-Omni-v0

**직접 부모:** [Sweep-Omni-v2](02_omni.md#env-inspiresweepomniv2envcfg) (`InspireSweepOmniV2EnvCfg`). **환경 설정:** [InspireDexOmniEnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_envs.py).

### 이 환경에서 바뀐 점

접근/목표를 새 최단 거리 갱신량 보상으로 교체. 계수 30/300, 접근 반경 10cm·목표 허용 1cm.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `reaching` | 수정 | 현재 움직이는 타겟까지의 새 최단 접근 진전. 초기 거리로 정규화하고 dt 상쇄.<br>구현: [CurrentObjectReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음.<br>구현: [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. |
| `goal_distance` | 수정 | weight: 40<br>타겟→목표 XY 최단 거리의 새 개선량을 초기 거리로 정규화. 에피소드 진전 예산형 보상.<br>구현: [PlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | weight: 300<br>허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상.<br>구현: [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. |

**직접 부모와 동일:** 관측 · 터미네이션 · 커리큘럼.

<a id="env-inspiresweepomniv3envcfg"></a>

## Sweep-Omni-v3

**직접 부모:** [Sweep-Omni-v2](02_omni.md#env-inspiresweepomniv2envcfg) (`InspireSweepOmniV2EnvCfg`). **환경 설정:** [InspireSweepOmniV3EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_envs.py).

### 이 환경에서 바뀐 점

접근 보상을 높이 조건부 XY 진전으로 변경. 타겟 기준 z offset 9cm, 허용 높이 오차 2cm.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `reaching` | 수정 | 현재 움직이는 타겟까지의 새 최단 접근 진전. 초기 거리로 정규화하고 dt 상쇄.<br>구현: [CurrentObjectReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py)<br>높이 offset(m): `함수 기본값`<br>높이 오차(m): `함수 기본값` | 지정 높이 오차 내에서만 XY 최단 접근 진전을 지급. dt 상쇄.<br>구현: [HeightGatedPlanarReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py)<br>높이 offset(m): `0.09`<br>높이 오차(m): `0.02` | 지정 높이 오차 내에서만 XY 최단 접근 진전을 지급. dt 상쇄. |

**직접 부모와 동일:** 관측 · 터미네이션 · 커리큘럼.

<a id="env-inspiresweepomniv4envcfg"></a>

## Sweep-Omni-v4

**직접 부모:** [Sweep-Omni-v3](02_omni.md#env-inspiresweepomniv3envcfg) (`InspireSweepOmniV3EnvCfg`). **환경 설정:** [InspireSweepOmniV4EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_envs.py).

### 이 환경에서 바뀐 점

물체·Arm 초과속도 비용 추가. Arm 종료 5rad/s, 물체 종료 속도 0.5m/s, 선반 밖 종료 제거. 보상 Cfg는 v2에서 파생하지만 EnvCfg의 v3 post-init이 높이 조건부 접근을 다시 적용한다.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `pushing_speed` | 추가 | 없음 | weight: -0.05<br>자유 속도(m/s 또는 관절 rad/s): `0.3`<br>정규화 기준 속도: `0.5`<br>구현: [pushing_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | 타겟 선속도가 자유 속도를 넘은 부분의 정규화 제곱 비용. |
| `arm_speed_excess` | 추가 | 없음 | weight: -0.03<br>자유 속도(m/s 또는 관절 rad/s): `1`<br>정규화 기준 속도: `2`<br>구현: [joint_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | 선택 관절 중 최대 절대 속도의 자유 속도 초과 제곱 비용. |

### 터미네이션 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `push_fast` | 수정 | 타겟 종료 속도(m/s): `0.3` | 타겟 종료 속도(m/s): `0.5` | 타겟 3D 선속도가 허용 속도를 초과하면 실패. |
| `hand_velocity` | 수정 | threshold: `1.5` | threshold: `5` | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. |
| `object_outside_shelf` | 삭제 | 종료 유형: terminated<br>half_extents: `(0.2, 0.5)`<br>구현: [object_outside_shelf](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) | `None` / 미사용 | 타겟이 선반 평면 경계 밖으로 나가면 실패. → 이 환경에서는 제거. |

**직접 부모와 동일:** 관측 · 커리큘럼.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/sweep_omni.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/sweep_omni.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/sweep_omni.py)
