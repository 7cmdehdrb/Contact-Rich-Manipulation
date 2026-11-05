# Sweep Omni Renewal 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

[전체 관측표](#전체-관측-항목) · [전체 보상표](#전체-보상-항목) · [전체 터미네이션표](#전체-터미네이션-항목) · [전체 커리큘럼표](#전체-커리큘럼-항목)

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








<!-- TERM_TABLES_START -->
## 버전별 전체 Manager 항목

이하 표는 **부모 상속과 EnvCfg의 후처리까지 반영한 전체 구성**이다. 동일한 구성을 쓰는 버전은 표 하나를 공유한다. 표의 항목명은 코드/학습 로그의 이름이며 구현 함수 링크로 계산을 확인할 수 있다. 보상 weight는 커리큘럼 적용 전 설정값이다. **weight 0은 설정된 항목이고, `None`은 제거된 항목**이다. 일정은 학습 iteration 기준이다.

[관측](#전체-관측-항목) · [보상](#전체-보상-항목) · [터미네이션](#전체-터미네이션-항목) · [커리큘럼](#전체-커리큘럼-항목)

### 전체 관측 항목

#### 관측 구성 1

적용: `Sweep-Omni-Renewal-v0`, `Sweep-Omni-Renewal-OSC-v0`, `Dex-Sweep-Omni-Renewal-v0`.

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

#### 관측 구성 2

적용: `Tactile-Sweep-Omni-Renewal-v0`, `Dex-Tactile-Sweep-Omni-Renewal-v0`, `Dex-Tactile-Sweep-Omni-Renewal-v1`, `Dex-Tactile-Sweep-Omni-Renewal-v2`, `Dex-Tactile-Sweep-Omni-Renewal-v3`, `Dex-Tactile-Sweep-Omni-Renewal-OSC-v2`.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 기본 자세 대비 상대 각도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_pos_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 기본 속도 대비 상대 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_vel_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.actions` | 8 | 직전 정책 행동 8개(Arm 6 + Hand 2). | 추가 ObservationTerm 노이즈 없음 | [last_action](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obs_state` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 매 관측 균등 노이즈 -0.01~0.01 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obj_width` | 1 | 타겟의 폭/크기 대표값(target_width). | 매 관측 균등 노이즈 -0.01~0.01 | [object_width](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.goal_pos` | 3 | 목표 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [goal_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.tactile` | 17 | 손의 17개 촉각 부위별 접촉 유무. 힘을 부위 방향에 투영해 문턱과 비교. | 추가 ObservationTerm 노이즈 없음<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [ProjectedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/tactile_observations.py) |
| `policy.wrist_wrench` | 6 | 손목 센서의 힘 XYZ·모멘트 XYZ를 EEF 기준으로 변환한 wrench. | 추가 ObservationTerm 노이즈 없음 | [wrist_wrench_eef](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |

### 전체 보상 항목

#### 보상 구성 1

적용: `Sweep-Omni-Renewal-v0`, `Sweep-Omni-Renewal-OSC-v0`, `Tactile-Sweep-Omni-Renewal-v0`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 지정 높이 오차 내에서만 XY 최단 접근 진전을 지급. dt 상쇄. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame`<br>높이 offset(m): `0.09`<br>높이 오차(m): `0.02` | [HeightGatedPlanarReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `goal_distance` | 40 | 타겟→목표 XY 최단 거리의 새 개선량을 초기 거리로 정규화. 에피소드 진전 예산형 보상. | 목표 반경(m): `0.01` | [PlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object_tilt` | -2 | 타겟 기울기가 free_angle을 넘은 부분의 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -1 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용. | 자유 속도(m/s 또는 관절 rad/s): `0.3` | [exponential_pushing_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용. | 자유 속도(m/s 또는 관절 rad/s): `1` | [exponential_joint_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

#### 보상 구성 2

적용: `Dex-Sweep-Omni-Renewal-v0`, `Dex-Tactile-Sweep-Omni-Renewal-v0`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 300 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object_tilt` | -2 | 타겟 기울기가 free_angle을 넘은 부분의 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -1 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용. | 자유 속도(m/s 또는 관절 rad/s): `0.3` | [exponential_pushing_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용. | 자유 속도(m/s 또는 관절 rad/s): `1` | [exponential_joint_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

#### 보상 구성 3

적용: `Dex-Tactile-Sweep-Omni-Renewal-v1`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 300 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object_tilt` | -2 | 타겟 기울기가 free_angle을 넘은 부분의 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -1 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.3`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

#### 보상 구성 4

적용: `Dex-Tactile-Sweep-Omni-Renewal-v2`, `Dex-Tactile-Sweep-Omni-Renewal-OSC-v2`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 300 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object_tilt` | -2 | 타겟 기울기가 free_angle을 넘은 부분의 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -1 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.3`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `success` | 100 | 거리·속도·기울기·지지 조건을 연속 유지하면 성공 보너스를 한 번 지급. 성공 자체로 reset하지 않음. | 목표 반경(m): `0.01`<br>속도 한계(m/s): `0.02`<br>기울기 한계(rad): `0.174533`<br>연속 유지(s): `0.5`<br>surface_height: `1.05`<br>선반 지지 허용 오차(m): `0.01`<br>half_extents: `(0.2, 0.5)`<br>object_radius: `0.04`<br>제외/검사할 실패 term: `(object_drop, push_fast, shelf_collision, hand_velocity, shelf_displacement)` | [StableGoalAchievement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_hold_rewards.py) |
| `goal_hold` | 20 | 안정 목표 점유의 연속 유지 시간 / hold_time을 [0, 1]로 제한한 지속 보상. | 추가 수치 설정 없음(생략값은 함수 기본값) | [stable_goal_hold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

#### 보상 구성 5

적용: `Dex-Tactile-Sweep-Omni-Renewal-v3`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 94 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object_tilt` | -2 | 타겟 기울기가 free_angle을 넘은 부분의 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -2 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.2`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

### 전체 터미네이션 항목

#### 터미네이션 구성 1

적용: `Sweep-Omni-Renewal-v0`, `Sweep-Omni-Renewal-OSC-v0`, `Dex-Sweep-Omni-Renewal-v0`, `Tactile-Sweep-Omni-Renewal-v0`, `Dex-Tactile-Sweep-Omni-Renewal-v0`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 타겟이 기준 높이 아래로 떨어지거나 기울기 한계를 초과하면 실패. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.8` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

`None`으로 비활성: `object_outside_shelf`.

#### 터미네이션 구성 2

적용: `Dex-Tactile-Sweep-Omni-Renewal-v1`, `Dex-Tactile-Sweep-Omni-Renewal-v2`, `Dex-Tactile-Sweep-Omni-Renewal-v3`, `Dex-Tactile-Sweep-Omni-Renewal-OSC-v2`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 타겟이 기준 높이 아래로 떨어지거나 기울기 한계를 초과하면 실패. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.8` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

`None`으로 비활성: `object_outside_shelf`.

`None`으로 비활성: `object_outside_shelf`, `success`.

### 전체 커리큘럼 항목

#### 커리큘럼 구성 1

적용: `Sweep-Omni-Renewal-v0`, `Sweep-Omni-Renewal-OSC-v0`, `Dex-Sweep-Omni-Renewal-v0`, `Tactile-Sweep-Omni-Renewal-v0`, `Dex-Tactile-Sweep-Omni-Renewal-v0`, `Dex-Tactile-Sweep-Omni-Renewal-v1`, `Dex-Tactile-Sweep-Omni-Renewal-v2`, `Dex-Tactile-Sweep-Omni-Renewal-v3`, `Dex-Tactile-Sweep-Omni-Renewal-OSC-v2`.

설정된 항목 없음.

<!-- TERM_TABLES_END -->

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/sweep_omni_renewal_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep_omni_renewal.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/sweep_omni_renewal.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/sweep_omni_renewal.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/sweep_omni_renewal.py) · [기본 관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/sweep.py)
