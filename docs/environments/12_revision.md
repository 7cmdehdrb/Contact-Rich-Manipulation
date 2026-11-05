# Revision 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

[전체 관측표](#전체-관측-항목) · [전체 보상표](#전체-보상-항목) · [전체 터미네이션표](#전체-터미네이션-항목) · [전체 커리큘럼표](#전체-커리큘럼-항목)

## 베이스: Revision-v0

`RevisionEnvCfg`는 `ManagerBasedRLEnvCfg`의 직접 자식이다. GoalAccuracy-v0_2를 기능 기준으로 재구성했지만 Final/Holding EnvCfg·runtime을 상속하지 않는다.

### 관측

직접 **57D**, 순서는 Arm 실제 위치·속도 12 → Hand 구성 2 → EEF pose 7 → 초기 타겟 XYZ 3 → 목표 극좌표 2 → 이전 행동 8 → 촉각 17 → 보정 F/T 6. Actor는 초기 장애물 16×4=64D를 받아 **121D**, Critic은 직접 57 + 현재 타겟 3 + **현재 활성 장애물 64**로 **124D**다. 인코딩 후 **73/76D**. Critic 장애물은 가시성 누락 없이 실제 활성 물체만 쓰며 좌표 노이즈 없이 매 step 갱신한다(정렬·10% 인접 교환 유지).

### 환경 구성

타겟 1 + Gray 최대 4/White 최대 5, **물리 물체 10개**. 타겟/장애물 모두 8종을 쓰고 중앙 기준 크기에 U(0.9,1.1)을 한 번 적용한다. 종류·크기는 생성 시 고정, reset에서는 활성 구성·배치·물성을 바꾼다. OBB 배치, Gray padding 65mm 고정, 크기를 반영해 바닥을 정렬한다. 기본 **2048개**, 10초·50Hz, 상대 pose OSC 6 + Hand 2(scale 0.03m/0.15rad).

### 보상

접근 최단 기록 30 + 거리 `3.7×exp(−d/0.05)` + Hold 8 + 접촉 1.45 + 성공 timeout 30 − 행동/관절/물체/선반/접촉력/wrench/타겟 속도 비용. Hold는 목표 반경 3cm, 외곽 8cm까지의 위치 점수에 속도 Gaussian(scale 5cm/s)을 곱한다. 접촉은 **실제 타겟 접촉 AND EEF 높이 120mm 미만**(선반 기준)이다.

### 터미네이션

활성 물체 바닥이 선반보다 1cm 이상 낮아짐/기울기 0.9rad 초과, 타겟 속도 1m/s 초과, Arm 5rad/s 초과, 선반 실패. 10초 시점 목표 **3cm 이하**를 성공으로 분류하며 양쪽 timeout 분류 모두 truncation이다. 성공 조기 종료 없음.

### 커리큘럼

장애물 1/1→4/5(White 2k·Gray 2.5k 시작·각 1k 간격). 목표 반경 3cm와 Gray padding 65mm는 고정. 타겟 자유 속도 0.1→0.02m/s, 가속도·wrench·장애물 접촉 비용/문턱은 점차 강화한다. 목표 제외 반각 30°를 3k 이후 500마다 5°씩 감소한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Revision-v1 | Revision-v0 | object 비용을 기울기 −30·변위 −150·가속도 −0.05로 분리. 가속도 허용 20m/s², 장애물 힘 50N/weight −0.05, wrench 변화 20N/2Nm/weight −0.1 등 **7개 일정의 10k 값 고정**. wrench 양의 보상 예산 제거. Hold는 3cm 내부 Gaussian, 속도 5cm/s까지 감쇠 없음. 최초 진입 +30, timeout +40. |
| Revision-Test-v1 | Revision-v1 | 타겟 종류만 원통으로 고정. 크기 및 장애물 랜덤화 유지. |
| Revision-v2 | **Revision-v1** | 거리·Hold·접촉에 EEF 95mm 미만 조건. Hold weight 10, 2.5cm까지 1·3.5cm까지 Smoothstep 감소, 속도 감쇠 없음. 최초 진입 최대 +25×속도 감쇠, timeout +75. 가속도 자유 문턱 15m/s². |
| Revision-Test-v2 | Revision-v2 | 타겟만 원통 고정. 현재 runtime/종료 함수에 Test-v2 전용 임시 종료 계측 print가 있다. |
| Revision-Test-v2-GT | Revision-Test-v2 | EnvCfg 변화 없이 전용 PPO 설정이 Actor에 현재 타겟 XYZ 3D를 추가. **Actor 원시 124D·MLP 76D**, Critic 유지. Test-v2 전용 exact-type print 대상에서는 제외된다. |

v1 이후 개수 일정은 **3/3 → 10k에 3/4 → 12k에 4/4 → 14k에 4/5**이며 v2도 유지한다. v2 최초 진입 보너스는 높이 조건이 없고, 속도 감쇠로 지급액이 작아도 최초 진입 기회는 소진한다. 일반 v0/v1/v2와 Test는 관측 차원이 같다.





<!-- TERM_TABLES_START -->
## 버전별 전체 Manager 항목

이하 표는 **부모 상속과 EnvCfg의 후처리까지 반영한 전체 구성**이다. 동일한 구성을 쓰는 버전은 표 하나를 공유한다. 표의 항목명은 코드/학습 로그의 이름이며 구현 함수 링크로 계산을 확인할 수 있다. 보상 weight는 커리큘럼 적용 전 설정값이다. **weight 0은 설정된 항목이고, `None`은 제거된 항목**이다. 일정은 학습 iteration 기준이다.

[관측](#전체-관측-항목) · [보상](#전체-보상-항목) · [터미네이션](#전체-터미네이션-항목) · [커리큘럼](#전체-커리큘럼-항목)

### 전체 관측 항목

#### 관측 구성 1

적용: `Revision-v0`, `Revision-v1`, `Revision-Test-v1`, `Revision-v2`, `Revision-Test-v2`.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 실제 각도. 기본 자세를 빼지 않는다. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 실제 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_velocities](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `policy.hand_configuration` | 2 | 실제 손 관절에서 역산한 굽힘·대립 개방도. 각 채널을 [0,1]로 제한. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [hand_configuration](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obs_state` | 3 | reset 직후 타겟 XYZ를 robot-root 좌표계로 저장하여 에피소드 동안 고정. | 초기 ±0.01m 오차를 에피소드 동안 고정<br>추가 수치 설정 없음(생략값은 함수 기본값) | [InitialObjectPositionInRoot](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/asymmetric_observations.py) |
| `policy.goal_pos` | 2 | 초기 타겟에서 목표까지의 평면 거리(m)와 로봇 root 기준 시계 방향각(rad). | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [initial_goal_polar_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.actions` | 8 | 직전 raw 정책 행동 8개에 tanh를 적용한 관측. 실제 행동·action-rate 계산용 raw 값은 변경하지 않는다. | 추가 ObservationTerm 노이즈 없음 | [previous_action_command](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `policy.tactile` | 17 | 정책과 보상이 공유하는 최신 촉각 샘플의 17개 접촉 bit. | 각 bit 1% 확률 반전<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [SharedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `policy.wrist_wrench` | 6 | 중력·관성 등을 보정하고 EEF 기준으로 표현한 힘 3 + 모멘트 3. | 힘 σ=0.05N, 모멘트 σ=0.005Nm Gaussian | [PureWristWrench](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `obstacles.initial_positions` | 64 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. | 추가 ObservationTerm 노이즈 없음; 16D로 인코딩 | [initial_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `privileged.target_position` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음; Critic 전용 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `critic_obstacles.current_positions` | 64 | 매 step 현재 활성 장애물 [존재 여부, x, y, z]. 주차/비활성 물체 제외, 가시성 누락 없이 제공. | 추가 ObservationTerm 노이즈 없음<br>adjacent_swap_probability: `0.1`; Critic 전용 | [current_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |

#### 관측 구성 2

적용: `Revision-Test-v2-GT`.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 실제 각도. 기본 자세를 빼지 않는다. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 실제 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_velocities](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `policy.hand_configuration` | 2 | 실제 손 관절에서 역산한 굽힘·대립 개방도. 각 채널을 [0,1]로 제한. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [hand_configuration](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obs_state` | 3 | reset 직후 타겟 XYZ를 robot-root 좌표계로 저장하여 에피소드 동안 고정. | 초기 ±0.01m 오차를 에피소드 동안 고정<br>추가 수치 설정 없음(생략값은 함수 기본값) | [InitialObjectPositionInRoot](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/asymmetric_observations.py) |
| `policy.goal_pos` | 2 | 초기 타겟에서 목표까지의 평면 거리(m)와 로봇 root 기준 시계 방향각(rad). | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [initial_goal_polar_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.actions` | 8 | 직전 raw 정책 행동 8개에 tanh를 적용한 관측. 실제 행동·action-rate 계산용 raw 값은 변경하지 않는다. | 추가 ObservationTerm 노이즈 없음 | [previous_action_command](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `policy.tactile` | 17 | 정책과 보상이 공유하는 최신 촉각 샘플의 17개 접촉 bit. | 각 bit 1% 확률 반전<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [SharedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `policy.wrist_wrench` | 6 | 중력·관성 등을 보정하고 EEF 기준으로 표현한 힘 3 + 모멘트 3. | 힘 σ=0.05N, 모멘트 σ=0.005Nm Gaussian | [PureWristWrench](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `obstacles.initial_positions` | 64 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. | 추가 ObservationTerm 노이즈 없음; 16D로 인코딩 | [initial_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `privileged.target_position` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음; Critic 전용 → GT 버전에서는 Actor도 사용 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `critic_obstacles.current_positions` | 64 | 매 step 현재 활성 장애물 [존재 여부, x, y, z]. 주차/비활성 물체 제외, 가시성 누락 없이 제공. | 추가 ObservationTerm 노이즈 없음<br>adjacent_swap_probability: `0.1`; Critic 전용 | [current_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |

### 전체 보상 항목

#### 보상 구성 1

적용: `Revision-v0`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 3.7 | exp(−XY 목표 거리/distance_scale)의 지속 근접 보상. | 거리 scale(m): `0.05` | [exponential_planar_goal_proximity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_goal_hold` | 8 | 목표 반경 내부 중심 가중 Gaussian + 외곽 이차 꼬리 + 속도 Gaussian. 유효한 물체 상태에서 지급. | 목표 반경(m): `0.03`<br>외곽 반경(m): `0.08`<br>속도 scale(m/s): `0.05` | [goal_hold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_contact` | 1.45 | 실제 손-타겟 접촉 AND 초기 선반 기준 EEF 높이 제한 미만에서만 보상. EEF 높이 제한 120mm. | 접촉 문턱(N): `0.01` | [target_contact_below_height](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_penalty` | -1 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100` | [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_goal_timeout_bonus` | 30 | 목표 내부 성공 timeout에 한 번 지급하는 보너스. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [successful_timeout_bonus](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `wrist_wrench_change_penalty` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

#### 보상 구성 2

적용: `Revision-v1`, `Revision-Test-v1`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 3.7 | exp(−XY 목표 거리/distance_scale)의 지속 근접 보상. | 거리 scale(m): `0.05` | [exponential_planar_goal_proximity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_goal_hold` | 8 | 목표 반경 미만의 중심 Gaussian Hold. free_speed까지 속도 감쇠 없음. | 목표 반경(m): `0.03`<br>자유 속도(m/s 또는 관절 rad/s): `0.05`<br>속도 scale(m/s): `0.05` | [goal_hold_v1](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_contact` | 1.45 | 실제 손-타겟 접촉 AND 초기 선반 기준 EEF 높이 제한 미만에서만 보상. EEF 높이 제한 120mm. | 접촉 문턱(N): `0.01` | [target_contact_below_height](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_tilt_penalty` | -30 | 활성 타겟/장애물만 대상으로 기울기 초과의 제한된 제곱 비용 합. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [active_object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_displacement_penalty` | -150 | 활성 비타겟 장애물의 초기 위치 대비 3D 변위 제곱합. 주차된 물체 제외. | 변위 제곱합 상한: `0.04` | [active_obstacle_displacement_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration_penalty` | -0.05 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `20`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | -0.05 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `50`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_goal_first_entry_bonus` | 30 | 한 에피소드에서 처음 목표 반경 미만으로 들어갈 때 일회 보상. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [FirstGoalEntryBonus](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_goal_timeout_bonus` | 40 | 목표 내부 성공 timeout에 한 번 지급하는 보너스. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [successful_timeout_bonus](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `wrist_wrench_change_penalty` | -0.1 | 힘·모멘트 변화량 크기의 제한된 비용. 양의 보상 예산 제한은 없음. | 허용 힘 변화(N): `20`<br>허용 모멘트 변화(Nm): `2`<br>포화 비율: `2.5`<br>곡선 기울기: `3` | [IndependentWrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `object_penalty`.

#### 보상 구성 3

적용: `Revision-v2`, `Revision-Test-v2`, `Revision-Test-v2-GT`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 3.7 | EEF 높이 제한 미만일 때만 exp(−d/distance_scale) 거리 보상. EEF 높이 제한 95mm. | 거리 scale(m): `0.05` | [goal_distance_v2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_goal_hold` | 10 | EEF 높이 제한 미만에서 full_reward_radius까지 1, outer_radius까지 Smoothstep으로 0으로 감소. 속도 감쇠 없음. EEF 높이 제한 95mm. | 최대 Hold 반경(m): `0.025`<br>외곽 반경(m): `0.035` | [goal_hold_v2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_contact` | 1.45 | 실제 손-타겟 접촉 AND 초기 선반 기준 EEF 높이 제한 미만에서만 보상. EEF 높이 제한 95mm. | 접촉 문턱(N): `0.01` | [target_contact_below_height](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_tilt_penalty` | -30 | 활성 타겟/장애물만 대상으로 기울기 초과의 제한된 제곱 비용 합. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [active_object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_displacement_penalty` | -150 | 활성 비타겟 장애물의 초기 위치 대비 3D 변위 제곱합. 주차된 물체 제외. | 변위 제곱합 상한: `0.04` | [active_obstacle_displacement_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration_penalty` | -0.05 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `15`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | -0.05 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `50`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_goal_first_entry_bonus` | 25 | 최초 목표 내부 진입에 속도 감쇠를 곱한 일회 보상. 높이 조건 없음, 지급액과 무관하게 기회 소진. | 자유 속도(m/s 또는 관절 rad/s): `0.01`<br>속도 scale(m/s): `0.02` | [FirstGoalEntryBonusV2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_goal_timeout_bonus` | 75 | 목표 내부 성공 timeout에 한 번 지급하는 보너스. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [successful_timeout_bonus](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `wrist_wrench_change_penalty` | -0.1 | 힘·모멘트 변화량 크기의 제한된 비용. 양의 보상 예산 제한은 없음. | 허용 힘 변화(N): `20`<br>허용 모멘트 변화(Nm): `2`<br>포화 비율: `2.5`<br>곡선 기울기: `3` | [IndependentWrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `object_penalty`.

### 전체 터미네이션 항목

#### 터미네이션 구성 1

적용: `Revision-v0`, `Revision-v1`, `Revision-Test-v1`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `object_fall_or_tilt` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. 크기/회전을 반영한 물체 바닥 높이로 검사. | bottom_clearance: `0.01`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/revision.py) |
| `target_speed` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_failure` | terminated | 선반 충돌/손목 여유 위반 OR 선반 변위 초과를 하나로 합친 실패. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_termination](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `arm_joint_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `target_goal_success_at_timeout` | truncated | 시간 제한 도달 시 타겟 XY 거리가 현재 목표 반경 이하인 성공 timeout. 초기 반경 3cm(반경 일정이 있으면 갱신값 사용). | 추가 수치 설정 없음(생략값은 함수 기본값) | [success_at_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/revision.py) |
| `target_goal_miss_at_timeout` | truncated | 시간 제한 도달 시 성공 조건이 성립하지 않는 timeout. 초기 반경 3cm(반경 일정이 있으면 갱신값 사용). | 추가 수치 설정 없음(생략값은 함수 기본값) | [unsuccessful_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/revision.py) |

#### 터미네이션 구성 2

적용: `Revision-v2`, `Revision-Test-v2`, `Revision-Test-v2-GT`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `object_fall_or_tilt` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. 크기/회전을 반영한 물체 바닥 높이로 검사. | bottom_clearance: `0.01`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/revision.py) |
| `target_speed` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_failure` | terminated | 선반 충돌/손목 여유 위반 OR 선반 변위 초과를 하나로 합친 실패. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_termination](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `arm_joint_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `target_goal_success_at_timeout` | truncated | 시간 제한 도달 시 타겟 XY 거리가 현재 목표 반경 이하인 성공 timeout. 초기 반경 3cm(반경 일정이 있으면 갱신값 사용). | 추가 수치 설정 없음(생략값은 함수 기본값) | [success_at_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/revision.py) |
| `target_goal_miss_at_timeout` | truncated | 시간 제한 도달 시 성공 조건이 성립하지 않는 timeout. 초기 반경 3cm(반경 일정이 있으면 갱신값 사용). | 추가 수치 설정 없음(생략값은 함수 기본값) | [unsuccessful_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/revision.py) |

### 전체 커리큘럼 항목

#### 커리큘럼 구성 1

적용: `Revision-v0`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→4, 2500부터 / White 1→5, 2000부터 / 각 1000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `target_free_speed` | `target_speed_penalty`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_free_acceleration` | `object_penalty`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_acceleration_weight` | `object_penalty`의 가속도 내부 계수 (`acceleration_weight`)를 변경. | 3000: 0 → 10000: -0.05; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `target_goal_direction_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrist_wrench_free_force_change` | `wrist_wrench_change_penalty`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrist_wrench_free_moment_change` | `wrist_wrench_change_penalty`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 3000: 5 → 10000: 2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrist_wrench_change_weight` | `wrist_wrench_change_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_contact_free_force` | `obstacle_contact_penalty`의 허용 접촉력(N) (`free_force`)를 변경. | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_contact_weight` | `obstacle_contact_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |

#### 커리큘럼 구성 2

적용: `Revision-v1`, `Revision-Test-v1`, `Revision-v2`, `Revision-Test-v2`, `Revision-Test-v2-GT`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 3→4, 12000부터 / White 3→5, 10000부터 / 각 4000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `target_free_speed` | `target_speed_penalty`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `target_goal_direction_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `object_free_acceleration`, `object_acceleration_weight`, `wrist_wrench_free_force_change`, `wrist_wrench_free_moment_change`, `wrist_wrench_change_weight`, `obstacle_contact_free_force`, `obstacle_contact_weight`.

<!-- TERM_TABLES_END -->

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/revision_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/revision.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/revision.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/revision.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/revision.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/revision.py) · [Actor/Critic 입력](../../src/inspire_sweep_baseline/inspire_sweep_baseline/agent/revision_ppo_cfg.py)
