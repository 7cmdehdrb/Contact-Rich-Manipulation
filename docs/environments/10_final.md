# Final 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

베이스는 전체 항목을, 각 파생 환경은 **직접 부모 대비 변경점만** 적는다. 보상 weight는 커리큘럼 적용 전 값이며, 일정의 단위는 학습 iteration이다.

**환경 바로가기:** [Final-v0](10_final.md#env-finalenvcfg) · [Final-v1](10_final.md#env-finalv1envcfg) · [Final-v1-1](10_final.md#env-finalv1_1envcfg) · [Final-Test-v1](10_final.md#env-finalv1testenvcfg) · [Final-v2](10_final.md#env-finalv2envcfg) · [Final-Test-v2](10_final.md#env-finalv2testenvcfg)

<a id="env-finalenvcfg"></a>

## 베이스: Final-v0

`FinalEnvCfg`는 `ManagerBasedRLEnvCfg`를 직접 상속한다. Holding-OSC-ObjectRandom-v3의 동작을 기준으로 Scene·Manager·runtime·PPO를 독립 구성했다. Python 상속으로 그 환경에 연결되지는 않는다.

### 관측

직접 **55D**: Arm 상대 위치·속도 12 + 이전 행동 8 + 초기 타겟 XYZ 3 + EEF pose 7 + 목표 극좌표 2 + 촉각 17 + 보정 F/T 6. 초기 장애물 16×4=64D, Critic 현재 타겟 XYZ 3D. **원시 Actor/Critic 119/122D, MLP 71/74D**. 두 모델 모두 초기 장애물을 쓰며 Critic에 실시간 장애물은 아직 없다.

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
| `obstacles.initial_positions` | 64 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. | 추가 ObservationTerm 노이즈 없음; 16D로 인코딩 | [initial_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `privileged.target_position` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음; Critic 전용 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |

### 환경 구성

원통 타겟 1 + 다양한 형상의 장애물 최대 8/8, 물리 용량 17개 중 초기 활성 1/1. Reset에서 배치와 물성을 재추출한다. 기본 4096개·10초·50Hz. 행동은 root-frame 상대 OSC pose 6 + Hand 2, 위치/회전 scale 0.03m/0.15rad, stiffness 병진 100·회전 50, Arm PD 해제다.

### 보상

`30×접근 최단 기록 + 100×목표 최단 기록 + 4×중심 가중 Hold + 1.45×(접촉 OR 목표 점유) − 비용`이다. 목표 점유 시 Reaching 반경을 10cm→1m로 완화한다. Hold는 거리·속도 Gaussian에 중심 가중을 곱하고 목표 반경 r의 2배까지 지급한다. 비용은 행동 변화, 관절 속도, 기울기/장애물 변위, 선반 충돌/실패, 물체·Arm·EEF 과속, 장애물 접촉력·물체 가속도·wrench 변화다.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 새 최단 접근 진전. 타겟이 목표를 점유하면 접근 허용 반경을 goal_near_radius로 완화. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame`<br>목표 점유 시 접근 반경(m): `1` | [ReachingOrGoal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 4 | 거리·속도 Gaussian, 반경 2r까지 지급하고 r 내부는 중심에서 최대 2배 가중. | 목표 반경(m): `0.02`<br>속도 scale(m/s): `0.05` | [goal_settling_centered](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `hand_contact` | 1.45 | 손의 실제 접촉 또는 유효한 목표 점유 중 하나라도 성립하면 1회분 접촉 보상. | 접촉 문턱(N): `0.01` | [hand_contact_or_goal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | 0 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `50`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

### 터미네이션

10초, 활성 물체 낙하/기울기, 타겟 1m/s 초과, Arm 5rad/s 초과, 선반 충돌/변위. **성공 조기 종료 없음**.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

### 커리큘럼

장애물 1/1→6/6(White 3k·Gray 4k 시작, 각 2k 간격), Gray 안쪽 경계 침범 비율 0→15%(6k→12k). Hold 반경 2→1→0.5cm(5k/10k/20k), 속도 scale 5→2cm/s. 물체/Arm/EEF 자유 속도와 접촉력·가속도·wrench 비용/문턱을 점진적으로 강화한다. 목표 제외 반각은 30°에서 3000 이후 500마다 5°씩 감소한다.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→6, 4000부터 / White 1→6, 3000부터 / 각 2000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 1 → 499: 1 → 500: 0.98 → 3300: 0.7; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 1.5 → 499: 1.5 → 500: 1.48 → 13300: 0.2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 0: 50 → 3000: 50 → 6000: 30 → 10000: 30 → 12000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 0: 5 → 3000: 5 → 6000: 3 → 10000: 3 → 12000: 2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 0: 50 → 3000: 50 → 6000: 30 → 10000: 30 → 12000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_free_force` | `obstacle_contact`의 허용 접촉력(N) (`free_force`)를 변경. | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `acceleration_weight` | `object_acceleration`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.01 → 6000: -0.03 → 10000: -0.03 → 12000: -0.05; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_weight` | `wrench_change`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_weight` | `obstacle_contact`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_speed_scale` | `goal_hold`의 속도 scale(m/s) (`speed_scale`)를 변경. | 0: 0.05 → 6000: 0.05 → 10000: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_position_tolerance` | `goal_hold`의 목표 반경(m) (`position_tolerance`)를 변경. | 0: 0.02 → 5000: 0.02 → 10000: 0.01 → 20000: 0.005; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `gray_zone_padding` | 다음 reset의 Gray 안쪽 경계 침범 비율을 변경. 절대 거리(m)가 아닌 비율. | 6000: 0 → 12000: 0.15; 갱신 간격 250 iter | [gray_zone_padding_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding_osc_object_random.py) |

## 관련 상속 관계

[GoalAccuracy-v0](11_goal_accuracy.md)도 Final-v1의 직접 자식이다.

<a id="env-finalv1envcfg"></a>

## Final-v1

**직접 부모:** [Final-v0](10_final.md#env-finalenvcfg) (`FinalEnvCfg`). **환경 설정:** [FinalV1EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py).

### 이 환경에서 바뀐 점

생성 시 타겟도 형상 랜덤화·크기 ±10%, 물리 용량 10/10·관측 20슬롯. Hand 실제 구성 2D와 촉각/F/T 노이즈, tanh(raw) 기준 이전 행동을 사용. 비용을 11개 보상 term으로 재구성하고 Arm/EEF 초과속도 항 제거. 10초 시점 목표 반경 내/외를 success/time_out으로 분리(둘 다 truncation).

직접 관측 55→57D(Hand 2D 추가), 장애물 64→80D로 원시 Actor/Critic 119/122→137/140D, MLP 71/74→73/76D다. 장애물 개수 일정은 1/1→8/8(White 2000·Gray 3000 시작, 각 2000 간격), Hold 반경은 2→1cm(4000~10000), Gray 안쪽 경계 침범 비율은 0→15%(4000~10000)로 변경한다.

### 관측 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `policy.actions` | 수정 | 직전 행동을 [−1, 1]로 clamp한 Arm 6 + Hand 2.<br>구현: [bounded_last_action](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) | 직전 raw 정책 행동 8개에 tanh를 적용한 관측. 실제 행동·action-rate 계산용 raw 값은 변경하지 않는다.<br>구현: [previous_action_command](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) | 직전 raw 정책 행동 8개에 tanh를 적용한 관측. 실제 행동·action-rate 계산용 raw 값은 변경하지 않는다. |
| `policy.tactile` | 수정 | 추가 ObservationTerm 노이즈 없음 | 각 bit 1% 확률 반전 | 정책과 보상이 공유하는 최신 촉각 샘플의 17개 접촉 bit. |
| `policy.wrist_wrench` | 수정 | 추가 ObservationTerm 노이즈 없음 | 힘 σ=0.05N, 모멘트 σ=0.005Nm Gaussian | 중력·관성 등을 보정하고 EEF 기준으로 표현한 힘 3 + 모멘트 3. |
| `obstacles.initial_positions` | 수정 | 차원: 64 | 차원: 80 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. |
| `policy.hand_configuration` | 추가 | 없음 | 차원: 2<br>추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [hand_configuration](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) | 실제 손 관절에서 역산한 굽힘·대립 개방도. 각 채널을 [0,1]로 제한. |

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `reaching` | 삭제 | weight: 30<br>접근 반경(m): `0.1`<br>기준 frame: `ee_frame`<br>목표 점유 시 접근 반경(m): `1`<br>구현: [ReachingOrGoal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) | `None` / 미사용 | 새 최단 접근 진전. 타겟이 목표를 점유하면 접근 허용 반경을 goal_near_radius로 완화. → 이 환경에서는 제거. |
| `arm_action_rate` | 삭제 | weight: -0.03<br>행동 그룹: `arm`<br>raw 상한: `100`<br>구현: [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. → 이 환경에서는 제거. |
| `hand_action_rate` | 삭제 | weight: -0.03<br>행동 그룹: `hand`<br>raw 상한: `100`<br>구현: [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. → 이 환경에서는 제거. |
| `arm_joint_velocity` | 삭제 | weight: -0.03<br>raw 상한: `100`<br>구현: [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 지정 관절 각속도의 제곱합 비용. → 이 환경에서는 제거. |
| `hand_joint_velocity` | 삭제 | weight: -0.01<br>raw 상한: `100`<br>구현: [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 지정 관절 각속도의 제곱합 비용. → 이 환경에서는 제거. |
| `object` | 삭제 | weight: -30<br>무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>구현: [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | `None` / 미사용 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. → 이 환경에서는 제거. |
| `shelf_collision` | 삭제 | weight: -0.5<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) | `None` / 미사용 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. → 이 환경에서는 제거. |
| `pushing_speed` | 삭제 | weight: -5<br>자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100`<br>구현: [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. → 이 환경에서는 제거. |
| `arm_speed_excess` | 삭제 | weight: -1<br>자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100`<br>구현: [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | `None` / 미사용 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. → 이 환경에서는 제거. |
| `shelf_failure` | 삭제 | weight: -3<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) | `None` / 미사용 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. → 이 환경에서는 제거. |
| `eef_linear_speed` | 삭제 | weight: -1<br>자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100`<br>구현: [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | `None` / 미사용 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. → 이 환경에서는 제거. |
| `obstacle_contact` | 삭제 | weight: 0<br>허용 접촉력(N): `80`<br>raw 상한: `100`<br>구현: [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | `None` / 미사용 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. → 이 환경에서는 제거. |
| `object_acceleration` | 삭제 | weight: 0<br>자유 가속도(m/s²): `50`<br>raw 상한: `100`<br>구현: [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | `None` / 미사용 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. → 이 환경에서는 제거. |
| `wrench_change` | 삭제 | weight: 0<br>허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05`<br>구현: [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | `None` / 미사용 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. → 이 환경에서는 제거. |
| `hand_reaching` | 추가 | 없음 | weight: 30<br>접근 반경(m): `0.1`<br>기준 frame: `ee_frame`<br>목표 점유 시 접근 반경(m): `1`<br>구현: [ReachingOrGoal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) | 새 최단 접근 진전. 타겟이 목표를 점유하면 접근 허용 반경을 goal_near_radius로 완화. |
| `action_rate_penalty` | 추가 | 없음 | weight: -1<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. |
| `joint_velocity_penalty` | 추가 | 없음 | weight: -1<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. |
| `object_penalty` | 추가 | 없음 | weight: -1<br>무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100`<br>구현: [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. |
| `object_speed_penalty` | 추가 | 없음 | weight: -5<br>자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100`<br>구현: [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. |
| `shelf_penalty` | 추가 | 없음 | weight: -1<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. |
| `obstacle_contact_penalty` | 추가 | 없음 | weight: 0<br>허용 접촉력(N): `80`<br>raw 상한: `100`<br>구현: [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. |
| `wrench_change_penalty` | 추가 | 없음 | weight: 0<br>허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05`<br>구현: [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. |

### 터미네이션 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `time_out` | 수정 | 에피소드 제한 시간(기본 10초) 도달.<br>구현: `time_out` | 시간 제한 도달 시 성공 조건이 성립하지 않는 timeout. 초기 반경 2cm(반경 일정이 있으면 갱신값 사용).<br>구현: [unsuccessful_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) | 시간 제한 도달 시 성공 조건이 성립하지 않는 timeout. 초기 반경 2cm(반경 일정이 있으면 갱신값 사용). |
| `push_fast` | 삭제 | 종료 유형: terminated<br>타겟 종료 속도(m/s): `1`<br>구현: [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) | `None` / 미사용 | 타겟 3D 선속도가 허용 속도를 초과하면 실패. → 이 환경에서는 제거. |
| `shelf_collision` | 삭제 | 종료 유형: terminated<br>threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03`<br>구현: [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) | `None` / 미사용 | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. → 이 환경에서는 제거. |
| `hand_velocity` | 삭제 | 종료 유형: terminated<br>threshold: `5`<br>구현: [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) | `None` / 미사용 | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. → 이 환경에서는 제거. |
| `shelf_displacement` | 삭제 | 종료 유형: terminated<br>threshold: `0.05`<br>구현: [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) | `None` / 미사용 | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. → 이 환경에서는 제거. |
| `object_speed` | 추가 | 없음 | 종료 유형: terminated<br>타겟 종료 속도(m/s): `1`<br>구현: [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) | 타겟 3D 선속도가 허용 속도를 초과하면 실패. |
| `shelf_termination` | 추가 | 없음 | 종료 유형: terminated<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [shelf_termination](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) | 선반 충돌/손목 여유 위반 OR 선반 변위 초과를 하나로 합친 실패. |
| `joint_velocity` | 추가 | 없음 | 종료 유형: terminated<br>threshold: `5`<br>구현: [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. |
| `success` | 추가 | 없음 | 종료 유형: truncated<br>추가 수치 설정 없음(생략값은 함수 기본값)<br>구현: [success_at_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) | 시간 제한 도달 시 타겟 XY 거리가 현재 목표 반경 이하인 성공 timeout. 초기 반경 2cm(반경 일정이 있으면 갱신값 사용). |

### 커리큘럼 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `obstacle_population` | 수정 | Gray 1→6, 4000부터 / White 1→6, 3000부터 / 각 2000 iter마다 +1<br>구현: [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) | Gray 1→8, 3000부터 / White 1→8, 2000부터 / 각 2000 iter마다 +1<br>구현: [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. |
| `pushing_free_speed` | 삭제 | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `None` / 미사용 | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. → 이 환경에서는 제거. |
| `arm_free_speed` | 삭제 | 0: 1 → 499: 1 → 500: 0.98 → 3300: 0.7; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `None` / 미사용 | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. → 이 환경에서는 제거. |
| `eef_free_speed` | 삭제 | 0: 1.5 → 499: 1.5 → 500: 1.48 → 13300: 0.2; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `None` / 미사용 | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. → 이 환경에서는 제거. |
| `wrench_free_force_change` | 수정 | 0: 50 → 3000: 50 → 6000: 30 → 10000: 30 → 12000: 20; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 3000: 50 → 10000: 20; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `wrench_change_penalty`의 허용 힘 변화(N) (`free_force_change`)를 변경. |
| `wrench_free_moment_change` | 수정 | 0: 5 → 3000: 5 → 6000: 3 → 10000: 3 → 12000: 2; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 3000: 5 → 10000: 2; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `wrench_change_penalty`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. |
| `object_free_acceleration` | 수정 | 0: 50 → 3000: 50 → 6000: 30 → 10000: 30 → 12000: 20; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 3000: 50 → 10000: 20; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `object_penalty`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. |
| `obstacle_free_force` | 수정 | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `obstacle_contact_penalty`의 허용 접촉력(N) (`free_force`)를 변경. |
| `acceleration_weight` | 삭제 | 0: 0 → 1000: 0 → 3000: -0.01 → 6000: -0.03 → 10000: -0.03 → 12000: -0.05; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `None` / 미사용 | `object_acceleration`의 보상 weight를 변경. → 이 환경에서는 제거. |
| `wrench_weight` | 수정 | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `wrench_change_penalty`의 보상 weight를 변경. |
| `obstacle_weight` | 수정 | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `obstacle_contact_penalty`의 보상 weight를 변경. |
| `goal_position_tolerance` | 수정 | 0: 0.02 → 5000: 0.02 → 10000: 0.01 → 20000: 0.005; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 4000: 0.02 → 10000: 0.01; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `goal_hold`의 목표 반경(m) (`position_tolerance`)를 변경. |
| `gray_zone_padding` | 삭제 | 6000: 0 → 12000: 0.15; 갱신 간격 250 iter<br>구현: [gray_zone_padding_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding_osc_object_random.py) | `None` / 미사용 | 다음 reset의 Gray 안쪽 경계 침범 비율을 변경. 절대 거리(m)가 아닌 비율. → 이 환경에서는 제거. |
| `obstacle_gray_zone_padding` | 추가 | 없음 | 4000: 0 → 10000: 0.15; 갱신 간격 1000 iter<br>구현: [gray_zone_padding_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding_osc_object_random.py) | 다음 reset의 Gray 안쪽 경계 침범 비율을 변경. 절대 거리(m)가 아닌 비율. |
| `object_free_speed` | 추가 | 없음 | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `object_speed_penalty`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. |
| `object_acceleration_weight` | 추가 | 없음 | 3000: 0 → 10000: -0.05; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `object_penalty`의 가속도 내부 계수 (`acceleration_weight`)를 변경. |

<a id="env-finalv1_1envcfg"></a>

## Final-v1-1

**직접 부모:** [Final-v1](10_final.md#env-finalv1envcfg) (`FinalV1EnvCfg`). **환경 설정:** [FinalV1_1EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py).

### 이 환경에서 바뀐 점

목표 최단 거리 갱신 보상을 `1.5×exp(−d/0.05)`의 지속 거리 보상으로 교체.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `goal_distance` | 수정 | weight: 100<br>허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상.<br>구현: [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py)<br>목표 반경(m): `0`<br>거리 scale(m): `함수 기본값` | weight: 1.5<br>exp(−XY 목표 거리/distance_scale)의 지속 근접 보상.<br>구현: [exponential_planar_goal_proximity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py)<br>목표 반경(m): `함수 기본값`<br>거리 scale(m): `0.05` | exp(−XY 목표 거리/distance_scale)의 지속 근접 보상. |

**직접 부모와 동일:** 관측 · 터미네이션 · 커리큘럼.

<a id="env-finalv1testenvcfg"></a>

## Final-Test-v1

**직접 부모:** [Final-v1](10_final.md#env-finalv1envcfg) (`FinalV1EnvCfg`). **환경 설정:** [FinalV1TestEnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py).

### 이 환경에서 바뀐 점

통합 object 비용을 기울기·장애물 변위·가속도로 분리해 기록. 가속도 일정의 대상 term도 변경. 총 보상 의미는 유지.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `object_penalty` | 삭제 | weight: -1<br>무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100`<br>구현: [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) | `None` / 미사용 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. → 이 환경에서는 제거. |
| `object_tilt_penalty` | 추가 | 없음 | weight: -30<br>무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>구현: [active_object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | 활성 타겟/장애물만 대상으로 기울기 초과의 제한된 제곱 비용 합. |
| `object_displacement_penalty` | 추가 | 없음 | weight: -150<br>변위 제곱합 상한: `0.04`<br>구현: [active_obstacle_displacement_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | 활성 비타겟 장애물의 초기 위치 대비 3D 변위 제곱합. 주차된 물체 제외. |
| `object_acceleration_penalty` | 추가 | 없음 | weight: 0<br>자유 가속도(m/s²): `50`<br>raw 상한: `100`<br>구현: [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. |

### 커리큘럼 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `object_free_acceleration` | 수정 | 3000: 50 → 10000: 20; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 3000: 50 → 10000: 20; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `object_acceleration_penalty`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. |
| `object_acceleration_weight` | 수정 | 3000: 0 → 10000: -0.05; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | 3000: 0 → 10000: -0.05; 갱신 간격 1 iter<br>구현: [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) | `object_acceleration_penalty`의 보상 weight를 변경. |

**직접 부모와 동일:** 관측 · 터미네이션.

<a id="env-finalv2envcfg"></a>

## Final-v2

**직접 부모:** [Final-v1](10_final.md#env-finalv1envcfg) (`FinalV1EnvCfg`). **환경 설정:** [FinalV2EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py).

### 이 환경에서 바뀐 점

reset에서 `wide_target=True`: 타겟 시작점을 중앙 반경 2cm에서 선반 중앙 80%와 물체 크기/여유 조건의 교집합으로 확장. 보상·관측·종료는 v1 유지.

**직접 부모와 동일:** 관측 · 보상 · 터미네이션 · 커리큘럼.

<a id="env-finalv2testenvcfg"></a>

## Final-Test-v2

**직접 부모:** [Final-v2](10_final.md#env-finalv2envcfg) (`FinalV2EnvCfg`). **환경 설정:** [FinalV2TestEnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py).

### 이 환경에서 바뀐 점

첫 reset부터 최종 장애물 요청 수 8/8을 사용하고 개수 커리큘럼만 제거.

### 커리큘럼 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `obstacle_population` | 삭제 | Gray 1→8, 3000부터 / White 1→8, 2000부터 / 각 2000 iter마다 +1<br>구현: [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) | `None` / 미사용 | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. → 이 환경에서는 제거. |

**직접 부모와 동일:** 관측 · 보상 · 터미네이션.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/final.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/final.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/final.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/final.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/final.py)
