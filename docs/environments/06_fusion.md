# Contact Sweep Clutter Fusion 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

[전체 관측표](#전체-관측-항목) · [전체 보상표](#전체-보상-항목) · [전체 터미네이션표](#전체-터미네이션-항목) · [전체 커리큘럼표](#전체-커리큘럼-항목)

## 베이스: Contact-Sweep-Clutter-Fusion-v0

`ContactSweepClutterFusionEnvCfg`는 `ManagerBasedRLEnvCfg`의 직접 자식이다. Clutter의 설계를 참고한 독립 조합이며 Clutter EnvCfg를 상속하지 않는다.

### 관측

직접 관측 **55D**: Arm 상대 위치·속도 12 + 이전 행동 8 + 초기 타겟 XYZ 3 + EEF pose 7 + 초기 목표 극좌표(거리·각도) 2 + 촉각 17 + 보정 F/T 6. 물체 폭과 목표 XYZ 대신 위 구성을 쓴다. 초기 장애물 40D(10×4), Critic 현재 타겟 XYZ 3D를 별도 제공한다. **Actor/Critic 원시 95/98D, 인코딩 후 71/74D**.

### 환경 구성

원통 타겟 1개와 장애물 3개, Arm 위치 6 + Hand 개방 2, 4096개·10초·50Hz. Reset에서 시작 반경 2cm·목표 거리 15cm의 clutter 배치와 물성을 샘플링한다. 초기 장애물 관측은 가시성 문턱 0.35, 가시성 반전 확률 1%, 위치 ±1cm 오차를 반영한다. 초기 ±X 목표 제외 반각은 60°다.

### 보상

접근 최단 기록 30, 목표 최단 기록 100, 목표 Hold 0.9, 접촉 유무/개수 각 0.4. 여기에 행동 변화·관절 속도, 물체 기울기/장애물 변위, 선반 충돌/실패, 물체·Arm·EEF 과속, EEF 방향, wrench 변화율 비용을 합친다. wrench 비용은 양의 누적 보상 5% 예산을 가진다.

### 터미네이션

10초, 물체 낙하/기울기, 타겟 1m/s 초과, Arm 5rad/s 초과, 선반 충돌/변위. 성공 종료 없음. v3 계보에서는 타겟 과속 한계가 0.8m/s로 바뀐다.

### 커리큘럼

물체 자유 속도 0.1→0.05m/s, Arm 1→0.8rad/s, EEF 1.5→0.7m/s, 행동 변화 weight −0.03→−0.06. 목표 제외 반각 60°를 500 iter 유지 후 200 iter마다 1°씩 줄인다.

## 파생 환경

아래 이름은 모두 `Contact-Sweep-Clutter-Fusion-` 뒤의 버전이다. **번호 순서가 상속 순서가 아니다.**

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| v2 | v0 | EEF 방향 비용만 제거. |
| v3 | v2 | 속도/행동 변화 일정을 500 iter 시작·200 간격으로 변경. 목표 deadzone은 1000 유지·100마다 1° 감소. 접촉 보상 weight 증가와 wrench 변화율 문턱 감소 일정 추가. 타겟 종료 속도 0.8m/s. |
| v1_0 | **v3** | 가변 장애물 Gray/White 각 1→3, 비활성 물체 제외, 인접 관측 교환 10%. 접촉을 hand_contact 0.29로 통합하고 Arm/Hand 비용 분리. 장애물 접촉력·물체 가속도 비용 추가, wrench를 변화율 대신 변화량 기반으로 교체. 목표 제외 반각 30°, 3000 이후 500마다 −5°. |
| v1_1 | v1_0 | 장애물 힘 허용 2000→100N, wrench 허용 5000/500→500N/50Nm. 힘·가속도·wrench 문턱을 10k/20k까지 줄이는 staged 일정. |
| v1_1_1 | v1_1 | 가속도 실험: 장애물 힘 2000N, wrench 5000N/500Nm로 완화하고 해당 일정 제거. 가속도 일정 유지. |
| v1_1_2 | v1_1 | 힘 실험: 가속도 허용을 1,000,000m/s²로 올리고 가속도 일정 제거. 장애물 힘·wrench 일정 유지. |
| v1_2 | v1_1 | Hold 반경 2cm→1cm→0.5cm 일정, 목표 부근 접촉 보상 fade 폭 1cm. |
| v1_3 | v1_1 | 힘·wrench·가속도 문턱을 여러 절점의 선형 일정으로 변경. 초기 장애물 힘 200N, wrench 1000N/100Nm. |
| v1_3_1 | v1_3 | Hold 반경 2→1→0.5cm(5k/10k/20k), 접촉 fade 폭 1mm→1cm(5k→10k). |
| v1_4 | **v1_0** | 장애물 각 5개까지, Hold weight 9·접촉 1.45. Hold 초기 비활성 후 반경 축소, 충격/속도 일정 재구성. 목표 3초 유지 성공 종료 추가, 타겟 종료 속도 1m/s. |

[Holding-v0](07_holding.md)는 **v1_0**의 별도 직계 파생이다.





<!-- TERM_TABLES_START -->
## 버전별 전체 Manager 항목

이하 표는 **부모 상속과 EnvCfg의 후처리까지 반영한 전체 구성**이다. 동일한 구성을 쓰는 버전은 표 하나를 공유한다. 표의 항목명은 코드/학습 로그의 이름이며 구현 함수 링크로 계산을 확인할 수 있다. 보상 weight는 커리큘럼 적용 전 설정값이다. **weight 0은 설정된 항목이고, `None`은 제거된 항목**이다. 일정은 학습 iteration 기준이다.

[관측](#전체-관측-항목) · [보상](#전체-보상-항목) · [터미네이션](#전체-터미네이션-항목) · [커리큘럼](#전체-커리큘럼-항목)

### 전체 관측 항목

#### 관측 구성 1

적용: `Contact-Sweep-Clutter-Fusion-v1_1_1`, `Contact-Sweep-Clutter-Fusion-v1_1_2`, `Contact-Sweep-Clutter-Fusion-v1_4`, `Contact-Sweep-Clutter-Fusion-v1_3`, `Contact-Sweep-Clutter-Fusion-v1_3_1`, `Contact-Sweep-Clutter-Fusion-v1_1`, `Contact-Sweep-Clutter-Fusion-v1_2`, `Contact-Sweep-Clutter-Fusion-v1_0`, `Contact-Sweep-Clutter-Fusion-v2`, `Contact-Sweep-Clutter-Fusion-v3`, `Contact-Sweep-Clutter-Fusion-v0`.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 기본 자세 대비 상대 각도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_pos_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 기본 속도 대비 상대 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_vel_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.actions` | 8 | 직전 행동을 [−1, 1]로 clamp한 Arm 6 + Hand 2. | 추가 ObservationTerm 노이즈 없음 | [bounded_last_action](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.target_obs_state` | 3 | reset 직후 타겟 XYZ를 robot-root 좌표계로 저장하여 에피소드 동안 고정. | 초기 ±0.01m 오차를 에피소드 동안 고정<br>추가 수치 설정 없음(생략값은 함수 기본값) | [InitialObjectPositionInRoot](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/asymmetric_observations.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.goal_pos` | 2 | 초기 타겟에서 목표까지의 평면 거리(m)와 로봇 root 기준 시계 방향각(rad). | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [initial_goal_polar_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.tactile` | 17 | 정책과 보상이 공유하는 최신 촉각 샘플의 17개 접촉 bit. | 추가 ObservationTerm 노이즈 없음<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [SharedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `policy.wrist_wrench` | 6 | 중력·관성 등을 보정하고 EEF 기준으로 표현한 힘 3 + 모멘트 3. | 추가 ObservationTerm 노이즈 없음 | [PureWristWrench](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `obstacles.initial_positions` | 40 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. | 추가 ObservationTerm 노이즈 없음; 16D로 인코딩 | [initial_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `privileged.target_position` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음; Critic 전용 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |

### 전체 보상 항목

#### 보상 구성 1

적용: `Contact-Sweep-Clutter-Fusion-v1_1_1`, `Contact-Sweep-Clutter-Fusion-v1_0`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.01` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `hand_contact` | 0.29 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | -1 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `2000`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | -1 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `100`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `5000`<br>허용 모멘트 변화(Nm): `500`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 2

적용: `Contact-Sweep-Clutter-Fusion-v1_1_2`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.01` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `hand_contact` | 0.29 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | -1 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `100`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | -1 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `1e+06`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `500`<br>허용 모멘트 변화(Nm): `50`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 3

적용: `Contact-Sweep-Clutter-Fusion-v1_4`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 9 | 커리큘럼이 정한 반경 내 점유 보상. 반경 0일 때 비활성. | 목표 반경(m): `0` | [scheduled_goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `hand_contact` | 1.45 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | -1 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `2000`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | -1 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `100`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `5000`<br>허용 모멘트 변화(Nm): `500`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 4

적용: `Contact-Sweep-Clutter-Fusion-v1_3`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.01` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `hand_contact` | 0.29 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | -1 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `200`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | -1 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `100`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `1000`<br>허용 모멘트 변화(Nm): `100`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 5

적용: `Contact-Sweep-Clutter-Fusion-v1_3_1`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.02` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `hand_contact` | 0.29 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0.001` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | -1 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `200`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | -1 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `100`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `1000`<br>허용 모멘트 변화(Nm): `100`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 6

적용: `Contact-Sweep-Clutter-Fusion-v1_1`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.01` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `hand_contact` | 0.29 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | -1 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `100`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | -1 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `100`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `500`<br>허용 모멘트 변화(Nm): `50`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 7

적용: `Contact-Sweep-Clutter-Fusion-v1_2`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.02` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `hand_contact` | 0.29 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0.01` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | -1 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `100`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | -1 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `100`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `500`<br>허용 모멘트 변화(Nm): `50`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 8

적용: `Contact-Sweep-Clutter-Fusion-v2`, `Contact-Sweep-Clutter-Fusion-v3`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.01` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `contact_any` | 0.4 | 촉각에서 하나 이상 접촉하면 보상. | 추가 수치 설정 없음(생략값은 함수 기본값) | [contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `contact_count` | 0.4 | 촉각 접촉 개수를 max_contacts까지 제한·정규화한 보상. | 접촉 개수 상한: `5` | [contact_count](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `action_rate` | -0.03 | Arm/Hand 행동 변화 제곱합을 각 그룹별 cap 후 합산. | raw 상한: `100` | [combined_action_rate](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `joint_velocity` | -0.03 | Arm 속도 비용 + hand_ratio×Hand 속도 비용. | Hand 상대 비중: `0.333333`<br>raw 상한: `100` | [combined_joint_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object` | -30 | 모든 물체 기울기 + displacement_ratio×장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 보정 wrench 변화율(힘/초·모멘트/초)의 초과 비용. 포화 및 양의 보상 예산 적용. | 허용 힘 변화율(N/s): `2000`<br>허용 모멘트 변화율(Nm/s): `100`<br>포화 힘 변화율(N/s): `5000`<br>포화 모멘트 변화율(Nm/s): `250`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchRatePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `eef_orientation`.

#### 보상 구성 9

적용: `Contact-Sweep-Clutter-Fusion-v0`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 0.9 | 유효한 타겟이 목표 XY 반경 내부에 있을 때 매 step 점유 보상. | 목표 반경(m): `0.01` | [goal_hold_xy](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `contact_any` | 0.4 | 촉각에서 하나 이상 접촉하면 보상. | 추가 수치 설정 없음(생략값은 함수 기본값) | [contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `contact_count` | 0.4 | 촉각 접촉 개수를 max_contacts까지 제한·정규화한 보상. | 접촉 개수 상한: `5` | [contact_count](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `action_rate` | -0.03 | Arm/Hand 행동 변화 제곱합을 각 그룹별 cap 후 합산. | raw 상한: `100` | [combined_action_rate](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `joint_velocity` | -0.03 | Arm 속도 비용 + hand_ratio×Hand 속도 비용. | Hand 상대 비중: `0.333333`<br>raw 상한: `100` | [combined_joint_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object` | -30 | 모든 물체 기울기 + displacement_ratio×장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `2.5`<br>변위 제곱합 상한: `0.04` | [combined_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `eef_orientation` | -0.02 | EEF +Z가 위쪽 원뿔 안에 들어갈 때의 자세 비용(손바닥이 아래로 향하는 자세 억제). | 비용 시작 각도(rad): `1.0472` | [eef_upward_z_orientation_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | -0.1 | 보정 wrench 변화율(힘/초·모멘트/초)의 초과 비용. 포화 및 양의 보상 예산 적용. | 허용 힘 변화율(N/s): `2000`<br>허용 모멘트 변화율(Nm/s): `100`<br>포화 힘 변화율(N/s): `5000`<br>포화 모멘트 변화율(Nm/s): `250`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchRatePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

### 전체 터미네이션 항목

#### 터미네이션 구성 1

적용: `Contact-Sweep-Clutter-Fusion-v1_1_1`, `Contact-Sweep-Clutter-Fusion-v1_1_2`, `Contact-Sweep-Clutter-Fusion-v1_3`, `Contact-Sweep-Clutter-Fusion-v1_1`, `Contact-Sweep-Clutter-Fusion-v1_0`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.8` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

`None`으로 비활성: `object_outside_shelf`.

#### 터미네이션 구성 2

적용: `Contact-Sweep-Clutter-Fusion-v1_4`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `success` | terminated | 현재 goal_hold 반경을 연속 hold_time 동안 점유하면 성공. 동시 실패 제외. 초기 반경 0cm(반경 일정이 있으면 갱신값 사용). | 연속 유지(s): `3`<br>제외/검사할 실패 term: `(object_drop, push_fast, shelf_collision, hand_velocity, shelf_displacement)` | [GoalHoldSuccess](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |

`None`으로 비활성: `object_outside_shelf`.

#### 터미네이션 구성 3

적용: `Contact-Sweep-Clutter-Fusion-v1_3_1`, `Contact-Sweep-Clutter-Fusion-v1_2`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.8` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

`None`으로 비활성: `object_outside_shelf`.

#### 터미네이션 구성 4

적용: `Contact-Sweep-Clutter-Fusion-v2`, `Contact-Sweep-Clutter-Fusion-v0`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 타겟/장애물 중 어느 물체든 낙하 또는 기울기 한계를 넘으면 실패. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

`None`으로 비활성: `object_outside_shelf`.

#### 터미네이션 구성 5

적용: `Contact-Sweep-Clutter-Fusion-v3`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 타겟/장애물 중 어느 물체든 낙하 또는 기울기 한계를 넘으면 실패. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.8` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

`None`으로 비활성: `object_outside_shelf`.

### 전체 커리큘럼 항목

#### 커리큘럼 구성 1

적용: `Contact-Sweep-Clutter-Fusion-v1_1_1`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→3, 4500부터 / White 1→3, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 초기 100; 1000~20000: 5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 2

적용: `Contact-Sweep-Clutter-Fusion-v1_1_2`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→3, 4500부터 / White 1→3, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_free_force` | `obstacle_contact`의 허용 접촉력(N) (`free_force`)를 변경. | 초기 100; 1000~10000: 50까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 초기 500; 1000~20000: 5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 초기 50; 1000~20000: 0.5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 3

적용: `Contact-Sweep-Clutter-Fusion-v1_4`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→5, 4500부터 / White 1→5, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1; 500~1900: 0.09→0.02; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1; 500~3300: 0.98→0.7; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5; 500~13300: 1.48→0.2; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 초기 5000; 10000~10000: 200→200; 12000~18000: 200→100; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 초기 500; 10000~10000: 30→30; 12000~18000: 30→20; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 초기 100; 10000~10000: 50→50; 12000~18000: 50→10; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py) |
| `goal_hold_tolerance` | `goal_hold`의 목표 반경(m) (`position_tolerance`)를 변경. | 초기 0; 5000~10000: 0.01→0.005; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_quantized_ramps.py) |

#### 커리큘럼 구성 4

적용: `Contact-Sweep-Clutter-Fusion-v1_3`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→3, 4500부터 / White 1→3, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_free_force` | `obstacle_contact`의 허용 접촉력(N) (`free_force`)를 변경. | 1000: 200 → 5000: 100 → 10000: 50; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 1000: 1000 → 5000: 500 → 10000: 100 → 20000: 5; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 1000: 100 → 5000: 50 → 10000: 10 → 20000: 0.5; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 2000: 100 → 5000: 20 → 10000: 10 → 20000: 5; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 5

적용: `Contact-Sweep-Clutter-Fusion-v1_3_1`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→3, 4500부터 / White 1→3, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_free_force` | `obstacle_contact`의 허용 접촉력(N) (`free_force`)를 변경. | 1000: 200 → 5000: 100 → 10000: 50; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 1000: 1000 → 5000: 500 → 10000: 100 → 20000: 5; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 1000: 100 → 5000: 50 → 10000: 10 → 20000: 0.5; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 2000: 100 → 5000: 20 → 10000: 10 → 20000: 5; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_hold_tolerance` | `goal_hold`의 목표 반경(m) (`position_tolerance`)를 변경. | 5000: 0.02 → 10000: 0.01 → 20000: 0.005; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `hand_contact_fade_width` | `hand_contact`의 접촉 감쇠 폭(m) (`goal_contact_fade_width`)를 변경. | 5000: 0.001 → 10000: 0.01; 갱신 간격 1 iter | [piecewise_linear_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 6

적용: `Contact-Sweep-Clutter-Fusion-v1_1`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→3, 4500부터 / White 1→3, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_free_force` | `obstacle_contact`의 허용 접촉력(N) (`free_force`)를 변경. | 초기 100; 1000~10000: 50까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 초기 100; 1000~20000: 5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 초기 500; 1000~20000: 5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 초기 50; 1000~20000: 0.5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 7

적용: `Contact-Sweep-Clutter-Fusion-v1_2`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→3, 4500부터 / White 1→3, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_free_force` | `obstacle_contact`의 허용 접촉력(N) (`free_force`)를 변경. | 초기 100; 1000~10000: 50까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 초기 100; 1000~20000: 5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 초기 500; 1000~20000: 5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 초기 50; 1000~20000: 0.5까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_hold_tolerance` | `goal_hold`의 목표 반경(m) (`position_tolerance`)를 변경. | 초기 0.02; 1000~5000: 0.01까지; 5250~10000: 0.005까지; 250 iter 격자 | [staged_reward_threshold](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 8

적용: `Contact-Sweep-Clutter-Fusion-v1_0`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→3, 4500부터 / White 1→3, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 초기 5000, 1000부터 250마다 -1, 최종 10 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 초기 500, 1000부터 250마다 -0.1, 최종 1 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 초기 100, 1000부터 250마다 -0.02, 최종 0.1 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 9

적용: `Contact-Sweep-Clutter-Fusion-v2`, `Contact-Sweep-Clutter-Fusion-v0`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 300마다 −0.01, 최종 0.05 | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 300마다 −0.02, 최종 0.8 | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter.py) |
| `action_rate_weight` | `action_rate`의 보상 weight를 변경. | 초기 -0.03, 300마다 −0.005, 최종 -0.06 | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 300마다 −0.02, 최종 0.7 | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 60°; 500까지 유지, 이후 200마다 −1° (첫 감소 700), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

#### 커리큘럼 구성 10

적용: `Contact-Sweep-Clutter-Fusion-v3`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 0.1, 500부터 200마다 -0.01, 최종 0.03 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1, 500부터 200마다 -0.02, 최종 0.7 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `action_rate_weight` | `action_rate`의 보상 weight를 변경. | 초기 -0.03, 500부터 200마다 -0.005, 최종 -0.06 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 초기 1.5, 500부터 200마다 -0.02, 최종 0.3 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 60°; 1000까지 유지, 이후 100마다 −1° (첫 감소 1100), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `contact_any_weight` | `contact_any`의 보상 weight를 변경. | 초기 0.4, 500부터 200마다 0.02, 최종 2 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `contact_count_weight` | `contact_count`의 보상 weight를 변경. | 초기 0.4, 500부터 200마다 0.02, 최종 2 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_rate` | `wrench_change`의 허용 힘 변화율(N/s) (`free_force_rate`)를 변경. | 초기 2000, 500부터 200마다 -50, 최종 1000 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_moment_rate` | `wrench_change`의 허용 모멘트 변화율(Nm/s) (`free_moment_rate`)를 변경. | 초기 100, 500부터 200마다 -2.5, 최종 50 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_saturation_force_rate` | `wrench_change`의 포화 힘 변화율(N/s) (`saturation_force_rate`)를 변경. | 초기 5000, 500부터 200마다 -125, 최종 2500 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_saturation_moment_rate` | `wrench_change`의 포화 모멘트 변화율(Nm/s) (`saturation_moment_rate`)를 변경. | 초기 250, 500부터 200마다 -6.25, 최종 125 | [delayed_reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |

<!-- TERM_TABLES_END -->

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep_clutter_fusion.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep_clutter_fusion.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep_clutter_fusion.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/contact_sweep_clutter_fusion.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/contact_sweep_clutter_fusion.py)
